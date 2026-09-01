#!/usr/bin/env python3
"""Render the paired context-robustness summary used in the manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "publication_data" / "context_robustness_summary.csv"
PROVENANCE = ROOT / "publication_data" / "context_robustness_provenance.json"
OUTPUT = ROOT / "generated" / "figures" / "context_robustness.png"

DECODERS = ("Logistic", "CatBoost", "TabPFN-3")
CONTEXTS = ("Filtered uniform", "Structural")
COLORS = {
    "Filtered uniform": "#0072B2",
    "Structural": "#D98B00",
}

ALPHA = 0.55
MARKERS = {"Filtered uniform": "o", "Structural": "s"}
PANELS = (
    ("local_minus_random_ids", "A  Local coordinates\nminus random IDs"),
    ("score_given_local", "B  Score added to\nlocal coordinates"),
    ("local_given_score", "C  Local coordinates\nadded to score"),
)


def load_rows() -> list[dict[str, object]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("effect", "ci_low", "ci_high"):
            row[key] = float(row[key])
    return rows


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if provenance["analysis_status"] != "complete":
        print(
            "WARNING: rendering layout placeholders for "
            + ", ".join(provenance["missing_tabpfn_datasets"])
        )

    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.15), constrained_layout=True)
    offsets = {"Filtered uniform": 0.12, "Structural": -0.12}

    for axis, (contrast, title) in zip(axes, PANELS, strict=True):
        selected = [row for row in rows if row["contrast"] == contrast]
        lows = [float(row["ci_low"]) for row in selected]
        highs = [float(row["ci_high"]) for row in selected]
        span_low = min(0.0, min(lows))
        span_high = max(0.0, max(highs))
        padding = max(0.015, 0.08 * (span_high - span_low))

        axis.axvline(0, color="#8C8C8C", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        for decoder_index, decoder in enumerate(DECODERS):
            y_base = len(DECODERS) - 1 - decoder_index
            decoder_rows = {row["context"]: row for row in selected if row["decoder"] == decoder}
            x_pair = [float(decoder_rows[context]["effect"]) for context in CONTEXTS]
            y_pair = [y_base + offsets[context] for context in CONTEXTS]
            axis.plot(x_pair, y_pair, color=to_rgba("#C7C7C7", ALPHA), linewidth=0.8, zorder=1)
            for context in CONTEXTS:
                row = decoder_rows[context]
                effect = float(row["effect"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])
                y = y_base + offsets[context]
                color = to_rgba(COLORS[context], ALPHA)
                axis.errorbar(
                    effect,
                    y,
                    xerr=[[effect - low], [high - effect]],
                    fmt=MARKERS[context],
                    color=color,
                    markerfacecolor="white" if context == "Structural" else color,
                    markeredgewidth=0.9,
                    markersize=4.6,
                    elinewidth=1.1,
                    capsize=2,
                    zorder=3,
                )

        axis.set_title(title, loc="left", fontweight="semibold", pad=5)
        axis.set_xlim(span_low - padding, span_high + padding)
        axis.set_ylim(-0.55, 2.55)
        axis.set_yticks((2, 1, 0), DECODERS)
        axis.set_xlabel(r"$\Delta$MRR")
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[context],
            color=to_rgba(COLORS[context], ALPHA),
            markerfacecolor=(
                "white"
                if context == "Structural"
                else to_rgba(COLORS[context], ALPHA)
            ),
            markeredgecolor=to_rgba(COLORS[context], ALPHA),
            linewidth=0,
            markersize=5,
            label=context,
        )
        for context in CONTEXTS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.savefig(
        OUTPUT,
        dpi=600,
        bbox_inches="tight",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
