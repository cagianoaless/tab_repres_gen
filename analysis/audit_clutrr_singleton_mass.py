#!/usr/bin/env python3
"""Report exact CLUTRR collision and singleton diagnostics by chain length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.clutrr import load_clutrr_task
from relational_tfm.encoders.clutrr import encode_clutrr_rows
from relational_tfm.evaluation.classification import collision_conditioned_ceiling
from relational_tfm.plan import atomic_write_json, build_plan

DEFAULT_ENCODERS = (
    "path_length_only",
    "relation_multiset",
    "relation_transition_counts",
    "ordered_typed_path",
    "ordered_plus_multiset",
    "aligned_multiset_with_supported_prefix",
)


def _summary(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    ceiling = collision_conditioned_ceiling(values, labels)
    return {
        "rows": int(ceiling["rows"]),
        "accuracy_ceiling": float(ceiling["ceiling"]),
        "majority_prior": float(ceiling["majority_prior"]),
        "unique_representations": int(ceiling["unique_representations"]),
        "singleton_rows": int(ceiling["singleton_rows"]),
        "singleton_fraction": float(ceiling["singleton_fraction"]),
        "mixed_classes": int(ceiling["mixed_classes"]),
        "collision_rows": int(ceiling["collision_rows"]),
        "collision_coverage": float(ceiling["collision_coverage"]),
        "conditional_label_entropy_bits": float(
            ceiling["conditional_label_entropy_bits"]
        ),
    }


def audit(
    config_path: Path,
    repo_root: Path,
    encoders: tuple[str, ...],
) -> dict[str, Any]:
    plan = build_plan(load_study_config(config_path))
    maximum_context = max(
        int(cell["context_per_relation"]) for cell in plan["cells"]
    )
    cell = next(
        cell
        for cell in plan["cells"]
        if int(cell["context_per_relation"]) == maximum_context
        and int(cell["seed"]) == 101
    )
    task = load_clutrr_task(repo_root, cell)
    labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    lengths = task.evaluation["chain_length"].to_numpy(dtype=np.int64)
    records = []
    for encoder in encoders:
        encoded = encode_clutrr_rows(
            task.evaluation,
            encoder,
            int(cell["encoder_seed"]) + 1,
        )
        by_length = {}
        for length in sorted(map(int, np.unique(lengths))):
            selected = lengths == length
            by_length[str(length)] = _summary(
                encoded.values[selected],
                labels[selected],
            )
        records.append(
            {
                "encoder": encoder,
                "feature_count": len(encoded.feature_names),
                "overall": _summary(encoded.values, labels),
                "by_chain_length": by_length,
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "finite-population collision and singleton audit on the fixed "
            "official CLUTRR evaluation population"
        ),
        "config": str(config_path.relative_to(repo_root)),
        "plan_sha256": plan["plan_sha256"],
        "representative_cell_id": cell["cell_id"],
        "evaluation_rows": len(task.evaluation),
        "encoders": records,
        "interpretation_boundary": (
            "A high finite-population ceiling is not evidence of transferable "
            "semantics when singleton mass is high."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/studies/clutrr_symbolic_cpu_supported_prefix_v1.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--encoder", action="append", dest="encoders")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo_root / args.config).resolve()
    )
    result = audit(
        config_path,
        repo_root,
        tuple(args.encoders or DEFAULT_ENCODERS),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                row["encoder"]: {
                    "accuracy_ceiling": row["overall"]["accuracy_ceiling"],
                    "singleton_fraction": row["overall"]["singleton_fraction"],
                }
                for row in result["encoders"]
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
