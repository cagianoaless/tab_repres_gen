#!/usr/bin/env python3
"""Plot paired 57-KG interface effects across filtered context constructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DECODER_LABELS = {
    "logistic_nominal": "Logistic",
    "catboost_nominal": "CatBoost",
    "tabpfn": "TabPFN-3",
}
CONTEXT_LABELS = {
    "relation_matched_filtered_uniform": "Filtered uniform",
    "relation_matched_role_degree": "Structurally plausible",
}
COLORS = {
    "relation_matched_filtered_uniform": "#2F6B9A",
    "relation_matched_role_degree": "#C35A3A",
}
PANELS = (
    ("local_minus_tokens", "Local coordinates minus entity tokens"),
    ("score_given_local", "Frozen score added to local coordinates"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.summary)
    data = data.loc[
        (data["direction"] == "combined")
        & data["contrast"].isin([item[0] for item in PANELS])
    ].copy()
    expected = len(PANELS) * len(DECODER_LABELS) * len(CONTEXT_LABELS)
    if len(data) != expected:
        raise ValueError(f"expected {expected} plotted rows, found {len(data)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True)
    decoder_order = list(DECODER_LABELS)
    y = np.arange(len(decoder_order), dtype=float)
    offsets = {
        "relation_matched_filtered_uniform": -0.11,
        "relation_matched_role_degree": 0.11,
    }

    for panel_index, ((contrast, title), axis) in enumerate(zip(PANELS, axes, strict=False)):
        subset = data.loc[data["contrast"] == contrast]
        for context in CONTEXT_LABELS:
            rows = subset.set_index(["decoder", "context_mode"])
            values = np.asarray(
                [rows.loc[(decoder, context), "equal_family_effect"] for decoder in decoder_order]
            )
            low = np.asarray(
                [rows.loc[(decoder, context), "family_bootstrap_low"] for decoder in decoder_order]
            )
            high = np.asarray(
                [rows.loc[(decoder, context), "family_bootstrap_high"] for decoder in decoder_order]
            )
            axis.errorbar(
                values,
                y + offsets[context],
                xerr=np.vstack((values - low, high - values)),
                fmt="o",
                markersize=4.2,
                capsize=2.2,
                capthick=0.8,
                elinewidth=1.1,
                color=COLORS[context],
                label=CONTEXT_LABELS[context],
                zorder=3,
            )
        axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
        axis.set_title(f"{chr(65 + panel_index)}  {title}", loc="left", pad=7)
        axis.set_xlabel("Paired MRR effect (equal family)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [DECODER_LABELS[item] for item in decoder_order])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=7.5)
    figure.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.84, wspace=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
