#!/usr/bin/env python3
"""Measure label-free coordinate support shared by CLUTRR context and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.clutrr import load_clutrr_task
from relational_tfm.encoders.clutrr import encode_clutrr_rows
from relational_tfm.plan import atomic_write_json, build_plan

DEFAULT_ENCODERS = (
    "path_length_only",
    "relation_multiset",
    "ordered_typed_path",
    "relation_transition_counts",
    "ordered_plus_multiset",
)


def coordinate_support(
    context_values: np.ndarray,
    evaluation_values: np.ndarray,
) -> dict[str, float | int]:
    context_active = np.any(np.abs(context_values) > 1e-12, axis=0)
    evaluation_cells = np.abs(evaluation_values) > 1e-12
    evaluation_active = np.any(evaluation_cells, axis=0)
    active_columns = int(evaluation_active.sum())
    shared_columns = int(np.logical_and(context_active, evaluation_active).sum())
    active_cells = int(evaluation_cells.sum())
    covered_cells = int(evaluation_cells[:, context_active].sum())
    unsupported_columns = np.logical_and(evaluation_active, ~context_active)
    fully_supported_rows = int((~np.any(evaluation_cells[:, unsupported_columns], axis=1)).sum())
    return {
        "context_active_columns": int(context_active.sum()),
        "evaluation_active_columns": active_columns,
        "shared_active_columns": shared_columns,
        "active_coordinate_recall": (
            float(shared_columns / active_columns) if active_columns else 1.0
        ),
        "evaluation_active_cells": active_cells,
        "covered_evaluation_active_cells": covered_cells,
        "active_cell_coverage": float(covered_cells / active_cells) if active_cells else 1.0,
        "fully_supported_evaluation_rows": fully_supported_rows,
        "fully_supported_row_fraction": float(
            fully_supported_rows / len(evaluation_values)
        ),
    }


def audit(config_path: Path, repo_root: Path, encoders: tuple[str, ...]) -> dict[str, Any]:
    plan = build_plan(load_study_config(config_path))
    representatives: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in plan["cells"]:
        representatives.setdefault(
            (int(cell["context_per_relation"]), int(cell["seed"])),
            cell,
        )
    records = []
    for (budget, seed), cell in sorted(representatives.items()):
        task = load_clutrr_task(repo_root, cell)
        chain_lengths = task.evaluation["chain_length"].to_numpy(dtype=np.int64)
        for encoder in encoders:
            context = encode_clutrr_rows(
                task.context,
                encoder,
                int(cell["encoder_seed"]),
            )
            evaluation = encode_clutrr_rows(
                task.evaluation,
                encoder,
                int(cell["encoder_seed"]) + 1,
            )
            by_length = {}
            for chain_length in sorted(map(int, np.unique(chain_lengths))):
                selected = chain_lengths == chain_length
                by_length[str(chain_length)] = coordinate_support(
                    context.values,
                    evaluation.values[selected],
                )
            records.append(
                {
                    "encoder": encoder,
                    "feature_count": len(context.feature_names),
                    "context_per_relation": budget,
                    "seed": seed,
                    "overall": coordinate_support(
                        context.values,
                        evaluation.values,
                    ),
                    "by_chain_length": by_length,
                }
            )
    return {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": "label-free context-to-evaluation coordinate support audit",
        "config": str(config_path.relative_to(repo_root)),
        "plan_sha256": plan["plan_sha256"],
        "encoders": list(encoders),
        "records": records,
        "interpretation_boundary": (
            "Coverage measures whether active evaluation coordinates were active in context. "
            "It does not measure label sufficiency, learnability, or decoder performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/studies/clutrr_symbolic_cpu_shared_coordinates_v1.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--encoder", action="append", dest="encoders")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else repo_root / args.config
    )
    result = audit(
        config_path,
        repo_root,
        tuple(args.encoders or DEFAULT_ENCODERS),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    endpoint = [
        record
        for record in result["records"]
        if record["context_per_relation"] == 80
    ]
    print(
        json.dumps(
            {
                record["encoder"]: {
                    "active_cell_coverage": record["overall"]["active_cell_coverage"],
                    "fully_supported_row_fraction": record["overall"][
                        "fully_supported_row_fraction"
                    ],
                }
                for record in endpoint
                if record["seed"] == 101
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
