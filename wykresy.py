"""Wykresy i tabele diagnostyczne z zapisanych wyników eksperymentu."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from eksperyment import MAIN_SEED, metrics
from zbior import UNSEEN, write_csv


ROOT = Path(__file__).resolve().parent
INK = "#19293D"
GREY = "#66778A"
BLUE = "#2658A5"
TEAL = "#11786F"
ORANGE = "#AC502A"
PALE = "#E9EEF5"


def prepare_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": PALE,
        "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "savefig.dpi": 200,
    })


def save(fig, target):
    fig.savefig(target, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def duration_plot(audit, target):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.25), sharex=True, sharey=True,
                             layout="constrained")
    bins = np.arange(0, 10.25, .25)
    for ax, (split, title) in zip(axes, [
        ("train", "Trening"), ("dev", "Walidacja"), ("eval", "Test"),
    ], strict=True):
        for label, color, name in [(0, BLUE, "Naturalna"), (1, TEAL, "Syntetyczna")]:
            values = audit.loc[audit["split"].eq(split) & audit["label"].eq(label), "duration_s"]
            counts, _ = np.histogram(values, bins)
            ax.stairs(counts / len(values), bins, color=color, label=name, linewidth=1.6)
        ax.axvline(2, color=ORANGE, lw=1.25, linestyle="--")
        ax.set_title(title, color=INK, fontweight="bold", loc="left", fontsize=9)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color=PALE)
        ax.set_xlabel("Długość [s]")
    axes[0].set_ylabel("Udział w klasie")
    axes[-1].legend(fontsize=8, frameon=False)
    save(fig, target)


def metrics_plot(main, target):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), layout="constrained")
    for ax, metric, axis_name in zip(
        axes, ("balanced_accuracy", "eer"), ("Balanced accuracy", "EER"), strict=True
    ):
        for name, mark, color, label in [
            ("dev", "d", GREY, "Walidacja"),
            ("eval_all", "o", BLUE, "Pełny test"),
            ("eval_unseen", "s", TEAL, "Nowe algorytmy"),
        ]:
            values = main.loc[main["subset"].eq(name)].sort_values("variant")
            ax.plot(values["variant"], values[metric], marker=mark, color=color,
                    linewidth=1.35, markersize=4.5, label=label)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Wariant cech")
        ax.set_ylabel(axis_name)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color=PALE)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, target)


def generator_plot(details, target):
    rates = details.pivot(index="generator_id", columns="variant", values="miss_rate")
    rates = rates.sort_values("E", ascending=False)
    fig, ax = plt.subplots(figsize=(8.25, 4.3), layout="constrained")
    ys = np.arange(len(rates))
    ax.barh(ys - .17, rates["A"], .32, color="#8091AA", label="A · czasowe")
    ax.barh(ys + .17, rates["E"], .32, color=TEAL, label="E · pełny")
    ax.set_yticks(ys, [item + ("*" if item not in UNSEEN else "") for item in rates.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Udział syntez błędnie uznanych za naturalne")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color=PALE)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, target)


def confusion_plot(main, target):
    data = main[main["subset"].eq("eval_unseen")].set_index("variant")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.65), layout="constrained")
    for ax, variant in zip(axes, ["A", "E"], strict=True):
        m = data.loc[variant]
        counts = np.array([[m.tn, m.fp], [m.fn, m.tp]], dtype=int)
        rates = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row in range(2):
            for column in range(2):
                value = rates[row, column]
                n = f"{counts[row, column]:,}".replace(",", " ")
                pct = f"{value:.1%}".replace(".", ",")
                ax.text(column, row, f"{n}\n{pct}", ha="center", va="center",
                        color="white" if value > .62 else INK, linespacing=1.4)
        ax.set_title(f"Wariant {variant}", loc="left", color=INK, fontweight="bold", fontsize=9)
        ax.set_xticks([0, 1], ["Naturalna", "Syntetyczna"])
        ax.set_yticks([0, 1], ["Naturalna", "Syntetyczna"])
        ax.set_xlabel("Predykcja")
        ax.set_ylabel("Klasa rzeczywista")
        ax.tick_params(length=0)
    save(fig, target)


def error_analysis(features, predictions, target):
    real_or_unseen = features["label"].eq(0) | features["generator_id"].isin(UNSEEN)
    test = features[features["split"].eq("eval") & real_or_unseen]
    errors = []
    for variant in ("A", "E"):
        scores = predictions[predictions["variant"].eq(variant)].set_index("audio_id")
        for segment, mask in {
            "all": np.ones(len(test), dtype=bool),
            "short_2_to_4": test["duration_s"].lt(4),
            "long_4_plus": test["duration_s"].ge(4),
            "no_internal_pause": test[["n_micro", "n_macro"]].sum(axis=1).eq(0),
            "at_least_one_pause": test[["n_micro", "n_macro"]].sum(axis=1).gt(0),
        }.items():
            group = test.loc[mask]
            values = scores.loc[group["audio_id"]]
            errors.append({
                "variant": variant, "condition": segment,
                **metrics(group["label"], values["score"], float(values["threshold"].iloc[0])),
            })
    write_csv(pd.DataFrame(errors), target)


def create_plots(result_path):
    prepare_style()
    plots = result_path / "wykresy"
    plots.mkdir(exist_ok=True)
    features = pd.read_csv(result_path / "cechy.csv.gz")
    audit = pd.read_csv(result_path / "kontrola_danych.csv.gz")
    metrics_data = pd.read_csv(result_path / "metryki.csv")
    details = pd.read_csv(result_path / "generatory.csv")
    predictions = pd.read_csv(
        result_path / "predykcje.csv.gz",
        usecols=["audio_id", "split", "variant", "seed", "score", "threshold"],
    ).query("seed == @MAIN_SEED and split == 'eval'")
    write_csv(audit.groupby(["split", "label", "status"]).size().rename("n").reset_index(),
              result_path / "liczebnosci.csv")
    duration_plot(audit, plots / "dlugosci.png")
    metrics_plot(metrics_data[metrics_data["seed"].eq(MAIN_SEED)], plots / "metryki.png")
    generator_plot(details, plots / "generatory.png")
    confusion_plot(metrics_data[metrics_data["seed"].eq(MAIN_SEED)], plots / "macierze.png")
    error_analysis(features, predictions, result_path / "bledy.csv")


if __name__ == "__main__":
    create_plots(ROOT / "wyniki")
