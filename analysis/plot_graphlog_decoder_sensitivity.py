from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

COLORS = {
    "logistic": "#0072B2",
    "catboost": "#D55E00",
    "catboost_tuned": "#A33B14",
    "tabpfn": "#009E73",
}
LABELS = {
    "logistic": "Logistic",
    "catboost": "CatBoost",
    "catboost_tuned": "CatBoost tuned",
    "tabpfn": "TabPFN",
}
CONTRASTS = {
    "relation multiset - absolute positions": (
        "relation_multiset",
        "ordered_typed_path",
    ),
    "permuted - combined": (
        "aligned_multiset_with_ordered_block_permutation",
        "ordered_plus_multiset",
    ),
}


def _aggregate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby(
            ["decoder", "encoder", "context_per_relation"],
            as_index=False,
        )["equal_length_macro_balanced_accuracy"]
        .mean()
        .rename(
            columns={
                "equal_length_macro_balanced_accuracy": ("equal_world_seed_mean_balanced_accuracy")
            }
        )
    )


def _contrast_curves(cells: pd.DataFrame) -> pd.DataFrame:
    index = [
        "decoder",
        "world",
        "context_per_relation",
        "seed",
    ]
    wide = cells.pivot(
        index=index,
        columns="encoder",
        values="equal_length_macro_balanced_accuracy",
    ).reset_index()
    rows = []
    for label, (left, right) in CONTRASTS.items():
        selected = wide.dropna(subset=[left, right]).copy()
        selected["contrast"] = label
        selected["delta"] = selected[left] - selected[right]
        rows.append(selected)
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["decoder", "contrast", "context_per_relation"], as_index=False)["delta"]
        .mean()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = pd.read_csv(args.input_dir / "cells.csv")
    scores = _aggregate_cells(cells)
    contrasts = _contrast_curves(cells)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    contexts = [5, 10, 20, 40]

    for decoder in ("logistic", "catboost", "tabpfn"):
        for encoder, linestyle, suffix in (
            ("relation_multiset", "-", "relation multiset"),
            ("ordered_typed_path", "--", "absolute positions"),
        ):
            selected = scores[
                (scores["decoder"] == decoder) & (scores["encoder"] == encoder)
            ].sort_values("context_per_relation")
            axes[0].plot(
                selected["context_per_relation"],
                selected["equal_world_seed_mean_balanced_accuracy"],
                color=COLORS[decoder],
                linestyle=linestyle,
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=f"{LABELS[decoder]}, {suffix}",
            )
    axes[0].set_title("(a) Scores: solid relation multiset, dashed absolute positions")
    axes[0].set_xlabel("Context rows per relation")
    axes[0].set_ylabel("Equal-world balanced accuracy")
    axes[0].set_xticks(contexts)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)

    for decoder in ("logistic", "catboost", "tabpfn"):
        for contrast, linestyle in (
            ("relation multiset - absolute positions", "-"),
            ("permuted - combined", "--"),
        ):
            selected = contrasts[
                (contrasts["decoder"] == decoder) & (contrasts["contrast"] == contrast)
            ].sort_values("context_per_relation")
            axes[1].plot(
                selected["context_per_relation"],
                selected["delta"],
                color=COLORS[decoder],
                linestyle=linestyle,
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=f"{LABELS[decoder]}, {contrast}",
            )
    tuned = contrasts[contrasts["decoder"] == "catboost_tuned"]
    for contrast, marker in (
        ("relation multiset - absolute positions", "X"),
        ("permuted - combined", "P"),
    ):
        selected = tuned[tuned["contrast"] == contrast]
        axes[1].scatter(
            selected["context_per_relation"],
            selected["delta"],
            color=COLORS["catboost_tuned"],
            marker=marker,
            s=28,
            zorder=4,
            label=f"{LABELS['catboost_tuned']}, {contrast}",
        )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_title("(b) Fixed-decoder contrasts")
    axes[1].set_xlabel("Context rows per relation")
    axes[1].set_ylabel("Balanced-accuracy delta")
    axes[1].set_xticks(contexts)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    decoder_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[decoder],
            linewidth=1.6,
            marker="o" if decoder != "catboost_tuned" else "X",
            markersize=3.5,
            label=LABELS[decoder],
        )
        for decoder in ("logistic", "catboost", "tabpfn", "catboost_tuned")
    ]
    figure.legend(
        handles=decoder_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        ncol=4,
        columnspacing=1.2,
        handlelength=1.6,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
