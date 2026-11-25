"""Wczytanie protokołów ASVspoof oraz utworzenie tabeli cech."""

import hashlib
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

from cechy import AudioRejected, QUALITY_COLUMNS, SileroVAD, VARIANTS, extract_features


PARTS = ("train", "dev", "eval")
PROTOCOL_SIZES = {"train": 25_380, "dev": 24_844, "eval": 71_237}
SEEN = {f"A{i:02}" for i in range(1, 7)}
REFERENCE = {"A16", "A19"}
UNSEEN = {f"A{i:02}" for i in range(7, 20)} - REFERENCE
METADATA = ["audio_id", "path", "label", "speaker_id", "generator_id", "split"]


class Dataset:
    def __init__(self, source):
        path = Path(source) if isinstance(source, (str, Path)) else None
        if path is not None and path.is_dir():
            self.root = path / "LA" if (path / "LA").is_dir() else path
            self.archive = None
        else:
            self.archive = ZipFile(source)
            self.root = PurePosixPath("LA")

    def read(self, name):
        path = self.root / name
        return self.archive.read(str(path)) if self.archive else path.read_bytes()

    def manifest(self):
        records = []
        for split in PARTS:
            kind = "trn" if split == "train" else "trl"
            filename = f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{kind}.txt"
            for line in self.read(filename).decode("ascii").splitlines():
                if not line.strip():
                    continue
                speaker, audio_id, _, generator, label = line.split()
                valid_generators = SEEN if split != "eval" else REFERENCE | UNSEEN
                if (
                    not re.fullmatch(r"LA_[TDE]_\d{7}", audio_id)
                    or audio_id[3] != split[0].upper()
                    or (label, generator) != ("bonafide", "-")
                    and (label != "spoof" or generator not in valid_generators)
                ):
                    raise ValueError(f"Nieprawidłowy wiersz protokołu {split}: {line}")
                path = f"ASVspoof2019_LA_{split}/flac/{audio_id}.flac"
                records.append((audio_id, path, int(label == "spoof"), speaker, generator, split))
        manifest = pd.DataFrame(records, columns=METADATA)
        counts = manifest["split"].value_counts().to_dict()
        if counts != PROTOCOL_SIZES:
            raise ValueError(f"Nieoczekiwane liczności protokołów: {counts}")
        assert_disjoint(manifest)
        return manifest


def assert_disjoint(manifest):
    if manifest["audio_id"].duplicated().any():
        raise ValueError("Powtórzony identyfikator nagrania.")
    if manifest.groupby("speaker_id")["split"].nunique().gt(1).any():
        raise ValueError("Identyfikator mówcy występuje w więcej niż jednym podziale.")


def pcm_fingerprint(data):
    # STREAMINFO FLAC zawiera sumę kontrolną nieskompresowanych próbek PCM.
    if data[:4] == b"fLaC" and data[4] & 0x7F == 0 and len(data) >= 42:
        pcm_md5 = data[26:42]
        if any(pcm_md5):
            return "pcm:" + pcm_md5.hex()
    return "file:" + hashlib.sha256(data).hexdigest()


def process_audio(record, dataset, vad):
    start = time.perf_counter()
    audio = dataset.read(record["path"])
    try:
        values = extract_features(audio, vad)
        status = "ok" if values["n_segments"] else "no_speech"
    except AudioRejected as error:
        values = {"duration_s": error.duration}
        status = error.reason
    return {
        **record, **values, "status": status, "fingerprint": pcm_fingerprint(audio),
        "extract_ms": round((time.perf_counter() - start) * 1_000, 3),
    }


_worker_dataset = None
_worker_vad = None


def start_worker(source, vad_path):
    global _worker_dataset, _worker_vad
    _worker_dataset = Dataset(source)
    _worker_vad = SileroVAD(vad_path)


def process_record(record):
    return process_audio(record, _worker_dataset, _worker_vad)


def write_csv(frame, target, **kwargs):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    frame.to_csv(temporary, index=False, **kwargs)
    temporary.replace(target)


def finalize_features(raw, destination):
    raw = raw.sort_values(["split", "audio_id"]).reset_index(drop=True)
    assert_disjoint(raw)
    # Takie samo PCM pomiędzy częściami lub z różną etykietą wykluczamy w całości.
    hashes = raw.groupby("fingerprint")
    cross_split = hashes["split"].transform("nunique").gt(1)
    conflict = hashes["label"].transform("nunique").gt(1)
    ok = raw["status"].eq("ok")
    raw.loc[ok & (cross_split | conflict), "status"] = "duplicate_cross_split_or_label"
    eligible = raw["status"].eq("ok")
    duplicates = raw.loc[eligible, "fingerprint"].duplicated()
    raw.loc[duplicates.index[duplicates], "status"] = "duplicate_within_split"
    audit = raw[METADATA + QUALITY_COLUMNS + ["status", "fingerprint", "extract_ms"]]
    features = raw.loc[raw["status"].eq("ok"), list(audit.columns) + VARIANTS["E"]]
    if features["fingerprint"].duplicated().any():
        raise ValueError("Tabela końcowa zawiera zduplikowane PCM.")
    write_csv(audit, destination / "kontrola_danych.csv.gz", compression="gzip")
    write_csv(features, destination / "cechy.csv.gz", compression="gzip")
    return features


def extract_dataset(source, vad_path, destination, workers):
    records = Dataset(source).manifest().to_dict("records")
    rows = []
    with ProcessPoolExecutor(workers, initializer=start_worker, initargs=(source, vad_path)) as pool:
        for count, row in enumerate(pool.map(process_record, records, chunksize=16), 1):
            rows.append(row)
            if count % 1_000 == 0:
                print(f"Cechy: {count:,}/{len(records):,}", flush=True)
    return finalize_features(pd.DataFrame(rows), destination)
