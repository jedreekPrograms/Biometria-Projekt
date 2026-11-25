"""Interfejs wiersza poleceń do odtworzenia eksperymentu i klasyfikacji audio."""

import argparse
import json
import os
from pathlib import Path

# Ogranicza nadmiarowe wątki BLAS podczas równoległej ekstrakcji.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import pandas as pd
import soundfile as sf

from cechy import AudioRejected, MIN_SECONDS, SileroVAD, VARIANTS, extract_features
from eksperyment import default_workers, run_experiment
from zbior import extract_dataset


ROOT = Path(__file__).resolve().parent
VAD_PATH = ROOT / "zasoby" / "silero_vad.onnx"


def predict(path, variant, models):
    values = extract_features(path, SileroVAD(VAD_PATH))
    if not values["n_segments"]:
        raise AudioRejected("no_speech", values["duration_s"])
    bundle = joblib.load(models / f"{variant}.joblib")
    if bundle["variant"] != variant or bundle["columns"] != VARIANTS[variant]:
        raise ValueError("Model jest niezgodny z aktualnym zestawem cech.")
    row = pd.DataFrame([values], columns=bundle["columns"])
    score = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = bundle["threshold"]
    return {
        "plik": str(path), "wariant": variant,
        "decyzja": "syntetyczna" if score >= threshold else "naturalna",
        "score_syntetyczna": round(score, 6), "prog": threshold,
        "czas_s": round(values["duration_s"], 3),
        "segmenty_mowy": int(values["n_segments"]),
        "mikropauzy": int(values["n_micro"]), "makropauzy": int(values["n_macro"]),
    }


def cli():
    parser = argparse.ArgumentParser(description="Miniprojekt ASVspoof 2019 LA")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("ekstrakcja", help="Zbuduj tabelę cech z oficjalnego zbioru.")
    extract.add_argument("dane", type=Path, help="LA.zip lub wypakowany folder LA.")
    extract.add_argument("--wyniki", type=Path, default=ROOT / "wyniki")
    extract.add_argument("--procesy", type=int, default=default_workers())
    train = commands.add_parser("trening", help="Odtwórz trening i ocenę wariantów A–E.")
    train.add_argument("--wyniki", type=Path, default=ROOT / "wyniki")
    train.add_argument("--modele", type=Path, default=ROOT / "modele")
    train.add_argument("--procesy", type=int, default=default_workers())
    detect = commands.add_parser("ocena", help="Oceń jeden plik WAV lub FLAC.")
    detect.add_argument("audio", type=Path)
    detect.add_argument("--wariant", choices=VARIANTS, default="E")
    detect.add_argument("--modele", type=Path, default=ROOT / "modele")
    args = parser.parse_args()
    if hasattr(args, "procesy") and args.procesy < 1:
        parser.error("--procesy musi być dodatnią liczbą całkowitą.")

    if args.command == "ekstrakcja":
        rows = extract_dataset(args.dane, VAD_PATH, args.wyniki, args.procesy)
        print(f"Przyjęto {len(rows):,} nagrań. Wyniki zapisano w {args.wyniki}.")
    elif args.command == "trening":
        features = pd.read_csv(args.wyniki / "cechy.csv.gz", float_precision="round_trip")
        run_experiment(features, args.wyniki, args.modele, args.procesy)
    else:
        try:
            output = predict(args.audio, args.wariant, args.modele)
        except AudioRejected as error:
            reasons = {
                "too_short": f"Nagranie musi trwać co najmniej {MIN_SECONDS:g} s.",
                "invalid_or_silent": "Nagranie jest ciche lub zawiera nieprawidłowe próbki.",
                "no_speech": "Silero VAD nie wykrył mowy.",
            }
            output = {
                "plik": str(args.audio), "decyzja": "brak oceny",
                "powod": reasons[error.reason], "czas_s": round(error.duration, 3),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 2
        except (OSError, sf.LibsndfileError, ValueError) as error:
            parser.exit(2, f"Błąd oceny: {error}\n")
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
