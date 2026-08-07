#!/usr/bin/env python3
"""Compute exact accuracy and ROC-AUC ceilings without fitting a decoder."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.synthetic import (
    discovery_latent_labels,
    generate_synthetic_task,
)
from relational_tfm.encoders.synthetic import encode_synthetic_rows
from relational_tfm.evaluation.classification import collision_conditioned_ceiling
from relational_tfm.plan import build_plan


def _select(values: np.ndarray, population_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    positions = {int(query_id): index for index, query_id in enumerate(population_ids)}
    return values[[positions[int(query_id)] for query_id in selected_ids]]


def _unique_cells(configs: list[Path]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in configs:
        plan = build_plan(load_study_config(path))
        maximum = max(int(cell["context_per_class"]) for cell in plan["cells"])
        for cell in plan["cells"]:
            if int(cell["context_per_class"]) != maximum:
                continue
            key = (str(cell["property"]), str(cell["encoder"]), int(cell["seed"]))
            selected.setdefault(key, cell)
    return [selected[key] for key in sorted(selected)]


def _task(cell: dict[str, Any]):
    return generate_synthetic_task(
        property_name=str(cell["property"]),
        graph_size=int(cell["graph_size"]),
        context_per_class=int(cell["context_per_class"]),
        noise_regime=str(cell["noise_regime"]),
        seed=int(cell["seed"]),
        max_context_per_class=int(cell["max_context_per_class"]),
        evaluation_per_class=int(cell["evaluation_per_class"]),
        graph_seed=int(cell["graph_seed"]),
        relabel_seed=int(cell["relabel_seed"]),
        split_seed=int(cell["split_seed"]),
        noise_seed=int(cell["noise_seed"]),
        context_negative_ratio=int(cell["context_negative_ratio"]),
        evaluation_negative_ratio=int(cell["evaluation_negative_ratio"]),
    )


def _row(cell: dict[str, Any], task) -> dict[str, Any]:
    encoded = encode_synthetic_rows(
        task.support_graph,
        task.population,
        str(cell["encoder"]),
        int(cell["encoder_seed"]),
        property_name=str(cell["property"]),
        control_population=task.population,
    )
    evaluation_values = _select(
        encoded.values,
        task.population["query_id"].to_numpy(dtype=np.int64),
        task.evaluation["query_id"].to_numpy(dtype=np.int64),
    )
    summary = collision_conditioned_ceiling(
        evaluation_values,
        task.evaluation["label"].to_numpy(dtype=np.int64),
    )
    latent_summary = collision_conditioned_ceiling(
        evaluation_values,
        discovery_latent_labels(
            task.evaluation["query_id"].to_numpy(dtype=np.int64),
            int(cell["max_context_per_class"]) + int(cell["evaluation_per_class"]),
        ),
    )
    noise_singleton_prediction = (
        float(summary["singleton_fraction"])
        + (1.0 - float(summary["singleton_fraction"])) * 0.9
    )
    return {
        "property": cell["property"],
        "encoder": cell["encoder"],
        "graph_seed": int(cell["graph_seed"]),
        "rows": int(summary["rows"]),
        "majority_prior": float(summary["majority_prior"]),
        "accuracy_ceiling": float(summary["ceiling"]),
        "roc_auc_ceiling": float(summary["roc_auc_ceiling"]),
        "unique_representations": int(summary["unique_representations"]),
        "singleton_classes": int(summary["singleton_classes"]),
        "singleton_rows": int(summary["singleton_rows"]),
        "singleton_fraction": float(summary["singleton_fraction"]),
        "mixed_classes": int(summary["mixed_classes"]),
        "collision_rows": int(summary["collision_rows"]),
        "collision_coverage": float(summary["collision_coverage"]),
        "conditional_label_entropy_bits": float(
            summary["conditional_label_entropy_bits"]
        ),
        "label_entropy_bits": float(summary["label_entropy_bits"]),
        "latent_accuracy_ceiling": float(latent_summary["ceiling"]),
        "latent_roc_auc_ceiling": float(latent_summary["roc_auc_ceiling"]),
        "latent_mixed_classes": int(latent_summary["mixed_classes"]),
        "latent_collision_rows": int(latent_summary["collision_rows"]),
        "latent_collision_coverage": float(latent_summary["collision_coverage"]),
        "latent_conditional_label_entropy_bits": float(
            latent_summary["conditional_label_entropy_bits"]
        ),
        "noise_singleton_prediction": noise_singleton_prediction,
        "noise_singleton_residual": float(summary["ceiling"])
        - noise_singleton_prediction,
        "equivalence_class_histogram": summary["equivalence_class_histogram"],
        "latent_equivalence_class_histogram": latent_summary[
            "equivalence_class_histogram"
        ],
    }


def _group_rows(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task = _task(group[0])
    return [_row(cell, task) for cell in group]


def _cross_graph_row(cell: dict[str, Any]) -> dict[str, Any]:
    ceiling_seeds = tuple(int(seed) for seed in cell["ceiling_seeds"])
    feature_blocks = []
    label_blocks = []
    latent_label_blocks = []
    feature_names: tuple[str, ...] | None = None
    for seed in ceiling_seeds:
        task = generate_synthetic_task(
            property_name=str(cell["property"]),
            graph_size=int(cell["graph_size"]),
            context_per_class=int(cell["max_context_per_class"]),
            noise_regime=str(cell["noise_regime"]),
            seed=seed,
            max_context_per_class=int(cell["max_context_per_class"]),
            evaluation_per_class=int(cell["evaluation_per_class"]),
            graph_seed=seed,
            relabel_seed=seed + 100_000,
            split_seed=seed + 200_000,
            noise_seed=seed + 300_000,
            context_negative_ratio=int(cell["context_negative_ratio"]),
            evaluation_negative_ratio=int(cell["evaluation_negative_ratio"]),
        )
        encoded = encode_synthetic_rows(
            task.support_graph,
            task.population,
            str(cell["encoder"]),
            seed + 400_000,
            property_name=str(cell["property"]),
            control_population=task.population,
        )
        if feature_names is not None and encoded.feature_names != feature_names:
            raise ValueError("cross-graph ceiling feature schemas differ")
        feature_names = encoded.feature_names
        feature_blocks.append(encoded.values)
        label_blocks.append(task.population["label"].to_numpy(dtype=np.int64))
        latent_label_blocks.append(
            discovery_latent_labels(
                task.population["query_id"].to_numpy(dtype=np.int64),
                int(cell["max_context_per_class"]) + int(cell["evaluation_per_class"]),
            )
        )
    summary = collision_conditioned_ceiling(
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(label_blocks),
    )
    latent_summary = collision_conditioned_ceiling(
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(latent_label_blocks),
    )
    return {
        "property": cell["property"],
        "encoder": cell["encoder"],
        "graph_count": len(ceiling_seeds),
        "ceiling_seeds": list(ceiling_seeds),
        **summary,
        "latent_accuracy_ceiling": float(latent_summary["ceiling"]),
        "latent_roc_auc_ceiling": float(latent_summary["roc_auc_ceiling"]),
        "latent_mixed_classes": int(latent_summary["mixed_classes"]),
        "latent_collision_rows": int(latent_summary["collision_rows"]),
        "latent_collision_coverage": float(latent_summary["collision_coverage"]),
        "latent_conditional_label_entropy_bits": float(
            latent_summary["conditional_label_entropy_bits"]
        ),
        "latent_equivalence_class_histogram": latent_summary[
            "equivalence_class_histogram"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    cells = _unique_cells(args.config)
    cross_graph_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        cross_graph_cells.setdefault(
            (str(cell["property"]), str(cell["encoder"])),
            cell,
        )
    grouped_cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped_cells[(str(cell["property"]), int(cell["seed"]))].append(cell)
    groups = [grouped_cells[key] for key in sorted(grouped_cells)]
    if args.workers == 1:
        grouped_rows = map(_group_rows, groups)
        rows = [row for group_rows in grouped_rows for row in group_rows]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [
                row
                for group_rows in pool.map(_group_rows, groups)
                for row in group_rows
            ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_graph.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_rows = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "equivalence_class_histogram",
                    "latent_equivalence_class_histogram",
                }
            }
            for row in rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (args.output_dir / "equivalence_class_histograms.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "property": row["property"],
                        "encoder": row["encoder"],
                        "graph_seed": row["graph_seed"],
                        "observed_label_histogram": row[
                            "equivalence_class_histogram"
                        ],
                        "latent_label_histogram": row[
                            "latent_equivalence_class_histogram"
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    if args.workers == 1:
        cross_graph_rows = [
            _cross_graph_row(cell) for cell in cross_graph_cells.values()
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cross_graph_rows = list(
                pool.map(_cross_graph_row, cross_graph_cells.values())
            )
    (args.output_dir / "cross_graph_ceilings.json").write_text(
        json.dumps(
            {
                "scope": (
                    "exact pooled ceiling over independently generated and relabelled "
                    "graphs; repeated rows must share semantic coordinates across graphs"
                ),
                "groups": cross_graph_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["property"]), str(row["encoder"]))].append(row)
    aggregates = []
    for (property_name, encoder), group in sorted(grouped.items()):
        aggregates.append(
            {
                "property": property_name,
                "encoder": encoder,
                "graphs": len(group),
                "majority_prior": float(np.mean([row["majority_prior"] for row in group])),
                "accuracy_ceiling": float(
                    np.mean([row["accuracy_ceiling"] for row in group])
                ),
                "roc_auc_ceiling": float(
                    np.mean([row["roc_auc_ceiling"] for row in group])
                ),
                "collision_coverage": float(
                    np.mean([row["collision_coverage"] for row in group])
                ),
                "singleton_fraction": float(
                    np.mean([row["singleton_fraction"] for row in group])
                ),
                "latent_accuracy_ceiling": float(
                    np.mean([row["latent_accuracy_ceiling"] for row in group])
                ),
                "latent_roc_auc_ceiling": float(
                    np.mean([row["latent_roc_auc_ceiling"] for row in group])
                ),
                "latent_collision_coverage": float(
                    np.mean([row["latent_collision_coverage"] for row in group])
                ),
                "latent_conditional_label_entropy_bits": float(
                    np.mean(
                        [
                            row["latent_conditional_label_entropy_bits"]
                            for row in group
                        ]
                    )
                ),
                "noise_singleton_prediction": float(
                    np.mean([row["noise_singleton_prediction"] for row in group])
                ),
                "noise_singleton_residual": float(
                    np.mean([row["noise_singleton_residual"] for row in group])
                ),
                "conditional_label_entropy_bits": float(
                    np.mean(
                        [row["conditional_label_entropy_bits"] for row in group]
                    )
                ),
                "mixed_classes": float(
                    np.mean([row["mixed_classes"] for row in group])
                ),
            }
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "scope": "exact finite evaluation-population ceilings averaged over graphs",
                "normalization": "none",
                "rows": len(rows),
                "groups": aggregates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS rows={len(rows)} groups={len(aggregates)}")


if __name__ == "__main__":
    main()
