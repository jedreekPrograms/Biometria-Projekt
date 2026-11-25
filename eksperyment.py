"""Trening, dobór progu, niezależny test i analiza błędów."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from sklearn.pipeline import make_pipeline

from cechy import VARIANTS
from zbior import METADATA, REFERENCE, UNSEEN, assert_disjoint, write_csv


MAIN_SEED = 42
SEEDS = (MAIN_SEED, 23, 101)
N_TREES = 120


def select_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(np.isfinite(thresholds))
    best = eligible[np.argmax(np.round(tpr[eligible] - fpr[eligible], 12))]
    return float(thresholds[best])


def equal_error_rate(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])
    left = right - 1
    weight = -difference[left] / (difference[right] - difference[left])
    return float(fpr[left] + weight * (fpr[right] - fpr[left]))


def metrics(labels, scores, threshold):
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "n": len(labels), "n_real": int(tn + fp), "n_spoof": int(fn + tp),
        "threshold": threshold, "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision, "recall": recall, "f1": f1,
        "eer": equal_error_rate(labels, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_alarm_rate": fp / (tn + fp), "miss_rate": fn / (fn + tp),
    }


def paired_bootstrap(test, first, second):
    # Losowanie z powtórzeniami odbywa się parami, na poziomie mówców.
    labels = test["label"].to_numpy()
    counts = pd.DataFrame({
        "speaker_id": test["speaker_id"].to_numpy(),
        "n_real": labels == 0, "n_spoof": labels == 1,
        "delta_real": (labels == 0) * (second == 0).astype(int)
                      - (labels == 0) * (first == 0).astype(int),
        "delta_spoof": (labels == 1) * (second == 1).astype(int)
                       - (labels == 1) * (first == 1).astype(int),
    }).groupby("speaker_id").sum().to_numpy()
    rng = np.random.default_rng(MAIN_SEED)
    draws = rng.integers(0, len(counts), size=(2_000, len(counts)))
    samples = counts[draws].sum(axis=1)
    samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0)]
    deltas = 0.5 * (samples[:, 2] / samples[:, 0] + samples[:, 3] / samples[:, 1])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "balanced_accuracy_delta": float(
            balanced_accuracy_score(labels, second) - balanced_accuracy_score(labels, first)
        ),
        "ci_low": float(low), "ci_high": float(high), "speakers": len(counts),
        "valid_draws": len(samples),
    }


def new_classifier(seed, workers):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            max_features="sqrt", random_state=seed, n_jobs=workers,
        ),
    )


def save_bundle(bundle, model_directory, variant):
    target = model_directory / f"{variant}.joblib"
    temporary = target.with_suffix(".joblib.part")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(target)


def run_experiment(features, result_path, model_directory, workers):
    assert_disjoint(features)
    if features["fingerprint"].duplicated().any():
        raise ValueError("Ekstrakcja zawiera powtórzone próbki PCM.")
    if set(features.loc[features["split"].eq("eval") & features["label"].eq(1),
                        "generator_id"]) != UNSEEN | REFERENCE:
        raise ValueError("Brak pełnego zestawu algorytmów w danych testowych.")
    train, dev, test = (features.loc[features["split"].eq(split)].reset_index(drop=True)
                        for split in ("train", "dev", "eval"))
    real = test["label"].eq(0).to_numpy()
    conditions = {
        "eval_all": np.ones(len(test), dtype=bool),
        "eval_unseen": real | test["generator_id"].isin(UNSEEN).to_numpy(),
        "eval_reference": real | test["generator_id"].isin(REFERENCE).to_numpy(),
    }
    model_directory.mkdir(parents=True, exist_ok=True)
    results = []
    detail = []
    predictions = []
    main_labels = {}
    for seed in SEEDS:
        for variant, columns in VARIANTS.items():
            print(f"Trening {variant}, seed {seed}", flush=True)
            classifier = new_classifier(seed, workers)
            classifier.fit(train[columns], train["label"])
            # Stała kolejność sumowania głosów drzew również przy odtwarzaniu predykcji.
            classifier.set_params(randomforestclassifier__n_jobs=1)
            dev_scores = classifier.predict_proba(dev[columns])[:, 1]
            threshold = select_threshold(dev["label"], dev_scores)
            test_scores = classifier.predict_proba(test[columns])[:, 1]
            identifiers = {"variant": variant, "algorithm": "RandomForest", "seed": seed}
            results.append({**identifiers, "subset": "dev", **metrics(dev["label"], dev_scores, threshold)})
            for condition, mask in conditions.items():
                results.append({
                    **identifiers, "subset": condition,
                    **metrics(test.loc[mask, "label"], test_scores[mask], threshold),
                })
            for split_name, scored, scores in (("dev", dev, dev_scores), ("eval", test, test_scores)):
                predictions.append(scored[METADATA + ["duration_s", "extract_ms"]].assign(
                    **identifiers, score=scores, threshold=threshold,
                    prediction=(scores >= threshold).astype(int),
                ))
            if seed != MAIN_SEED:
                continue
            save_bundle({
                "model": classifier, "columns": columns, "variant": variant,
                "threshold": threshold, "seed": seed,
            }, model_directory, variant)
            main_labels[variant] = (test_scores >= threshold).astype(int)
            for generator in sorted(UNSEEN | REFERENCE):
                mask = real | test["generator_id"].eq(generator).to_numpy()
                detail.append({
                    **identifiers, "generator_id": generator, "reference": generator in REFERENCE,
                    **metrics(test.loc[mask, "label"], test_scores[mask], threshold),
                })

    intervals = {
        condition: paired_bootstrap(test.loc[mask], main_labels["A"][mask], main_labels["E"][mask])
        for condition, mask in conditions.items()
    }
    (result_path / "przedzialy.json").write_text(
        json.dumps(intervals, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(pd.DataFrame(results), result_path / "metryki.csv")
    write_csv(pd.DataFrame(detail), result_path / "generatory.csv")
    write_csv(pd.concat(predictions, ignore_index=True),
              result_path / "predykcje.csv.gz", compression="gzip")


def default_workers():
    return min(8, os.cpu_count() or 1)
