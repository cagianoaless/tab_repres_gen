#!/usr/bin/env python3
"""Render absolute interface performance by query-evidence stratum."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "publication_data" / "query_evidence_absolute.csv"
OUTPUT_PDF = ROOT / "generated" / "figures" / "query_evidence.pdf"
OUTPUT_PNG = ROOT / "generated" / "figures" / "query_evidence.png"

DECODERS = ("Logistic", "CatBoost", "TabPFN-3")
INTERFACES = ("dyadic", "paths", "local")
INTERFACE_LABELS = {
    "dyadic": "Dyadic",
    "paths": "Paths",
    "local": "Full local",
}
COLORS = {
    "dyadic": "#0072B2",
    "paths": "#D98B00",
    "local": "#4D4D4D",
}
ALPHA = 0.55
MARKERS = {"dyadic": "o", "paths": "^", "local": "s"}
STRATA = ("both", "reverse_only", "direct_only", "none")
STRATUM_LABELS = {
    "both": "Direct + reverse",
    "reverse_only": "Reverse only",
    "direct_only": "Direct only",
    "none": "No observed pair edges",
}
OFFSETS = {"dyadic": 0.18, "paths": 0.0, "local": -0.18}


def load_rows() -> list[dict[str, object]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "families",
            "equal_family_mean_rr",
            "family_bootstrap_low",
            "family_bootstrap_high",
            "query_count",
        ):
            row[key] = float(row[key])
    return rows


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.55),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for panel_index, (axis, decoder) in enumerate(zip(axes, DECODERS, strict=True)):
        decoder_rows = [row for row in rows if row["decoder"] == decoder]
        lookup = {
            (str(row["stratum"]), str(row["interface"])): row
            for row in decoder_rows
        }
        for stratum_index, stratum in enumerate(STRATA):
            y_base = len(STRATA) - 1 - stratum_index
            for interface in INTERFACES:
                row = lookup[(stratum, interface)]
                mean = float(row["equal_family_mean_rr"])
                low = float(row["family_bootstrap_low"])
                high = float(row["family_bootstrap_high"])
                color = to_rgba(COLORS[interface], ALPHA)
                axis.errorbar(
                    mean,
                    y_base + OFFSETS[interface],
                    xerr=[[mean - low], [high - mean]],
                    fmt=MARKERS[interface],
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.8,
                    markersize=4.5,
                    elinewidth=1.05,
                    capsize=2,
                    zorder=3,
                )

        letter = chr(ord("A") + panel_index)
        axis.set_title(f"{decoder}", loc="left", fontweight="semibold", pad=5)
        axis.set_xlim(-0.015, 0.83)
        axis.set_ylim(-0.55, 3.55)
        axis.set_yticks(
            (3, 2, 1, 0),
            [STRATUM_LABELS[stratum] for stratum in STRATA],
        )
        axis.set_xlabel("Equal-family mean reciprocal rank")
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[interface],
            color=to_rgba(COLORS[interface], ALPHA),
            markerfacecolor=to_rgba(COLORS[interface], ALPHA),
            markeredgecolor=to_rgba(COLORS[interface], ALPHA),
            linewidth=0,
            markersize=5,
            label=INTERFACE_LABELS[interface],
        )
        for interface in INTERFACES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    print(OUTPUT_PDF)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
