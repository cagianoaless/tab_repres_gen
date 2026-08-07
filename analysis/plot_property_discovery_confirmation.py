#!/usr/bin/env python3
"""Create paper-ready plots for the property-discovery confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROPERTY_LABELS = {
    "discovery_reciprocity": "Reciprocity",
    "discovery_typed_composition": "Typed composition",
    "discovery_structural_roles": "Structural roles",
    "discovery_block_affinity": "Block affinity",
}
PROPERTIES = tuple(PROPERTY_LABELS)
CONTEXTS = (20, 50, 100, 200, 400)
LADDER_COLUMNS = (
    ("query_relation_only", "Query only"),
    ("categorical_ordinal", "Random IDs"),
    ("onehot", "One-hot IDs"),
    ("embedding", "Random embed."),
    ("query_dyadic_invariants", "Dyadic"),
    ("query_local_roles", "Local roles"),
    ("query_graph_reachability", "Reachability"),
    ("query_typed_path_k2", "Typed path 2"),
    ("query_typed_path_k3", "Typed path 3"),
    ("aligned", "Complete evidence"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty group")
    return float(np.mean(values))


def plot_context_curves(core: list[dict[str, str]], output: Path) -> None:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in core:
        grouped[
            (
                row["decoder"],
                row["representation_kind"],
                int(row["context_per_class"]),
            )
        ].append(float(row["roc_auc"]))
    styles = (
        ("tabpfn", "aligned", "TabPFN aligned", "#0072B2", "-"),
        ("catboost", "aligned", "CatBoost aligned", "#D55E00", "-"),
        ("tabpfn", "permuted", "TabPFN row permutation", "#56B4E9", "--"),
        ("catboost", "permuted", "CatBoost row permutation", "#E69F00", "--"),
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.1))
    for decoder, representation, label, color, line_style in styles:
        values = [
            mean(grouped[(decoder, representation, context)])
            for context in CONTEXTS
        ]
        axis.plot(
            CONTEXTS,
            values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            linestyle=line_style,
            label=label,
        )
    axis.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    axis.set_xlabel("Labeled context per class")
    axis.set_ylabel("Equal-property macro ROC AUC")
    axis.set_xticks(CONTEXTS)
    axis.set_ylim(0.46, 0.88)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_primary_interaction(core: list[dict[str, str]], output: Path) -> None:
    values = {
        (
            row["decoder"],
            row["property"],
            int(row["context_per_class"]),
            int(row["seed"]),
            row["representation_kind"],
        ): float(row["roc_auc"])
        for row in core
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    rng = np.random.default_rng(20260725)
    for index, property_name in enumerate(PROPERTIES):
        interactions = []
        for seed in range(12001, 12011):
            tabpfn = (
                values[("tabpfn", property_name, 20, seed, "aligned")]
                - values[("tabpfn", property_name, 20, seed, "permuted")]
            )
            catboost = (
                values[("catboost", property_name, 20, seed, "aligned")]
                - values[("catboost", property_name, 20, seed, "permuted")]
            )
            interactions.append(tabpfn - catboost)
        jitter = rng.uniform(-0.10, 0.10, size=len(interactions))
        axis.scatter(
            index + jitter,
            interactions,
            s=24,
            color="#0072B2",
            alpha=0.65,
            edgecolors="none",
        )
        axis.plot(
            [index - 0.20, index + 0.20],
            [mean(interactions), mean(interactions)],
            color="#111111",
            linewidth=2.5,
        )
    axis.axhline(0.0, color="#777777", linewidth=1, linestyle=":")
    axis.set_xticks(range(len(PROPERTIES)), [PROPERTY_LABELS[name] for name in PROPERTIES])
    axis.set_ylabel("TabPFN-CatBoost alignment interaction (ROC AUC)")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def ladder_encoder_key(encoder: str) -> str:
    if encoder in {"role_onehot", "unordered_onehot"}:
        return "onehot"
    if encoder in {"shared_embedding_directed", "shared_embedding_symmetric"}:
        return "embedding"
    return encoder


def plot_encoder_ladder(
    core: list[dict[str, str]],
    ladder: list[dict[str, str]],
    output: Path,
) -> None:
    values: dict[tuple[str, int, str], float] = {}
    for row in ladder:
        if row["decoder"] != "tabpfn":
            continue
        key = ladder_encoder_key(row["encoder"])
        values[(row["property"], int(row["context_per_class"]), key)] = float(
            row["mean_roc_auc"]
        )
    aligned: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in core:
        if row["decoder"] == "tabpfn" and row["representation_kind"] == "aligned":
            aligned[(row["property"], int(row["context_per_class"]))].append(
                float(row["roc_auc"])
            )
    for property_name in PROPERTIES:
        for context in (20, 400):
            values[(property_name, context, "aligned")] = mean(
                aligned[(property_name, context)]
            )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.2),
        sharey=True,
        layout="constrained",
    )
    image = None
    for axis, context in zip(axes, (20, 400), strict=True):
        matrix = np.full((len(PROPERTIES), len(LADDER_COLUMNS)), np.nan)
        for row_index, property_name in enumerate(PROPERTIES):
            for column_index, (encoder, _) in enumerate(LADDER_COLUMNS):
                matrix[row_index, column_index] = values.get(
                    (property_name, context, encoder), np.nan
                )
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, vmin=0.48, vmax=0.91, cmap="viridis", aspect="auto")
        axis.set_title(f"{context}+{context} context")
        axis.set_xticks(
            range(len(LADDER_COLUMNS)),
            [label for _, label in LADDER_COLUMNS],
            rotation=50,
            ha="right",
        )
        axis.tick_params(axis="x", labelsize=8)
        axis.set_yticks(
            range(len(PROPERTIES)),
            [PROPERTY_LABELS[name] for name in PROPERTIES],
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    color = "white" if value < 0.62 or value > 0.82 else "black"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
    if image is None:
        raise ValueError("encoder ladder is empty")
    figure.colorbar(
        image,
        ax=axes,
        label="Mean ROC AUC",
        fraction=0.025,
        pad=0.02,
        shrink=0.82,
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    core = read_csv(args.analysis_dir / "core_cell_metrics.csv")
    ladder = read_csv(args.analysis_dir / "encoder_ladder_summary.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_context_curves(core, args.output_dir / "context_curves")
    plot_primary_interaction(core, args.output_dir / "primary_interaction_by_property")
    plot_encoder_ladder(core, ladder, args.output_dir / "encoder_ladder")
    manifest = {
        "schema_version": 1,
        "figures": {
            "context_curves": (
                "Absolute aligned and row-permutation ROC AUC curves for TabPFN and CatBoost."
            ),
            "primary_interaction_by_property": (
                "All ten graph-seed interactions at the registered 20+20 endpoint."
            ),
            "encoder_ladder": (
                "TabPFN ROC AUC by property, encoder family, and context endpoint."
            ),
        },
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
