from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.graphlog import load_graphlog_task
from relational_tfm.datasets.graphlog_audit import audit_graphlog_task
from relational_tfm.encoders.graphlog import (
    GRAPHLOG_MAX_PATH_LENGTH,
    GRAPHLOG_RELATIONS,
    encode_graphlog_rows,
)
from relational_tfm.evaluation.classification import collision_conditioned_ceiling
from relational_tfm.plan import atomic_write_json, build_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_CONFIG = REPO_ROOT / "configs/studies/graphlog_symbolic_cpu_confirmation_v1.yaml"
GPU_CONFIG = (
    REPO_ROOT / "configs/studies/graphlog_symbolic_gpu_tabpfn_confirmation_v1.yaml"
)
ASSET_AUDIT = REPO_ROOT / "results/graphlog/protocol/asset-audit.json"


def coordinate_support(
    context: np.ndarray, evaluation: np.ndarray
) -> dict[str, float | int]:
    context_active = np.any(context != 0.0, axis=0)
    evaluation_active = evaluation != 0.0
    active_cells = int(evaluation_active.sum())
    supported_cells = int((evaluation_active & context_active[None, :]).sum())
    fully_supported = np.all(~evaluation_active | context_active[None, :], axis=1)
    return {
        "context_active_columns": int(context_active.sum()),
        "evaluation_active_columns": int(np.any(evaluation_active, axis=0).sum()),
        "active_cell_coverage": float(supported_cells / active_cells),
        "fully_supported_row_fraction": float(fully_supported.mean()),
    }


def compact_ceiling(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "ceiling",
            "balanced_accuracy_ceiling",
            "majority_prior",
            "rows",
            "unique_representations",
            "singleton_fraction",
            "collision_coverage",
            "conditional_label_entropy_bits",
        )
    }


def _paired_scientific_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell["dataset"],
        cell["asset"]["sha256"],
        cell["world"],
        tuple(cell["train_chain_lengths"]),
        tuple(cell["evaluation_chain_lengths"]),
        cell["max_path_length"],
        cell["max_context_per_relation"],
        cell["encoder"],
        cell["context_per_relation"],
        cell["seed"],
        cell["split_seed"],
        cell["encoder_seed"],
    )


def _cell_for(
    plan: dict[str, Any],
    *,
    world: str,
    encoder: str,
    context: int,
    seed: int,
) -> dict[str, Any]:
    matches = [
        cell
        for cell in plan["cells"]
        if cell["world"] == world
        and cell["decoder"] == "logistic"
        and cell["encoder"] == encoder
        and cell["context_per_relation"] == context
        and cell["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("GraphLog audit cell is not unique")
    return matches[0]


def build_audit() -> dict[str, Any]:
    if not ASSET_AUDIT.is_file():
        raise FileNotFoundError(
            "run scripts/audit_graphlog_asset.py before the schedule audit"
        )
    asset_audit = json.loads(ASSET_AUDIT.read_text())
    if asset_audit.get("status") != "PASS":
        raise ValueError("GraphLog asset audit did not pass")

    cpu_plan = build_plan(load_study_config(CPU_CONFIG))
    gpu_plan = build_plan(load_study_config(GPU_CONFIG))
    cpu_keys = {_paired_scientific_key(cell) for cell in cpu_plan["cells"]}
    gpu_keys = {_paired_scientific_key(cell) for cell in gpu_plan["cells"]}
    if cpu_keys != gpu_keys:
        raise ValueError("CPU and GPU GraphLog populations are not paired")

    encoders = sorted({cell["encoder"] for cell in cpu_plan["cells"]})
    worlds = [f"rule_{index}" for index in range(57)]
    world_audits: list[dict[str, Any]] = []
    for world in worlds:
        seed_audits: list[dict[str, Any]] = []
        for seed in (101, 102, 103):
            contexts: dict[int, set[str]] = {}
            for budget in (5, 10, 20, 40):
                cell = _cell_for(
                    cpu_plan,
                    world=world,
                    encoder="ordered_plus_multiset",
                    context=budget,
                    seed=seed,
                )
                task = load_graphlog_task(REPO_ROOT, cell)
                audit_graphlog_task(task)
                contexts[budget] = set(task.context["row_id"])
            if not (
                contexts[5] < contexts[10] < contexts[20] < contexts[40]
            ):
                raise ValueError(f"GraphLog contexts are not strictly nested: {world}/{seed}")

            maximum_cell = _cell_for(
                cpu_plan,
                world=world,
                encoder="ordered_plus_multiset",
                context=40,
                seed=seed,
            )
            task = load_graphlog_task(REPO_ROOT, maximum_cell)
            task_audit = audit_graphlog_task(task)
            encoder_audits: dict[str, Any] = {}
            for encoder in encoders:
                context_encoded = encode_graphlog_rows(
                    task.context,
                    encoder,
                    int(maximum_cell["encoder_seed"]),
                )
                evaluation_encoded = encode_graphlog_rows(
                    task.evaluation,
                    encoder,
                    int(maximum_cell["encoder_seed"]) + 1,
                )
                encoder_audits[encoder] = {
                    "feature_count": len(context_encoded.feature_names),
                    "ceiling": compact_ceiling(
                        collision_conditioned_ceiling(
                            evaluation_encoded.values,
                            task.evaluation["label"].to_numpy(dtype=np.int64),
                        )
                    ),
                    "support": coordinate_support(
                        context_encoded.values,
                        evaluation_encoded.values,
                    ),
                }

            aligned = encode_graphlog_rows(
                task.evaluation,
                "ordered_plus_multiset",
                int(maximum_cell["encoder_seed"]) + 1,
            )
            block_null = encode_graphlog_rows(
                task.evaluation,
                "aligned_multiset_with_ordered_block_permutation",
                int(maximum_cell["encoder_seed"]) + 1,
            )
            ordered_width = 1 + GRAPHLOG_MAX_PATH_LENGTH * len(GRAPHLOG_RELATIONS)
            if aligned.values.shape != block_null.values.shape:
                raise ValueError("GraphLog blockwise pair changed dimensionality")
            if not np.array_equal(
                aligned.values[:, ordered_width:],
                block_null.values[:, ordered_width:],
            ):
                raise ValueError("GraphLog blockwise null changed the multiset block")
            if np.array_equal(
                aligned.values[:, 1:ordered_width],
                block_null.values[:, 1:ordered_width],
            ):
                raise ValueError("GraphLog blockwise null did not change the ordered block")

            seed_audits.append(
                {
                    "seed": seed,
                    "task": task_audit,
                    "context_row_ids_sha256": task.metadata[
                        "context_row_ids_sha256"
                    ],
                    "evaluation_row_ids_sha256": task.metadata[
                        "evaluation_row_ids_sha256"
                    ],
                    "nested_contexts": True,
                    "encoders": encoder_audits,
                    "blockwise_pair": {
                        "same_shape": True,
                        "shared_block_byte_identical": True,
                        "ordered_block_changed": True,
                    },
                }
            )
        world_audits.append({"world": world, "seeds": seed_audits})

    return {
        "schema_version": 1,
        "status": "PASS",
        "asset_sha256": asset_audit["asset"]["sha256"],
        "asset_worlds": asset_audit["world_count"],
        "cpu_plan_sha256": cpu_plan["plan_sha256"],
        "gpu_plan_sha256": gpu_plan["plan_sha256"],
        "cpu_cells": len(cpu_plan["cells"]),
        "gpu_cells": len(gpu_plan["cells"]),
        "paired_population_keys": len(cpu_keys),
        "world_count": len(world_audits),
        "context_seeds_per_world": 3,
        "world_audits": world_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit()
    atomic_write_json(args.output, audit)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in audit
                if key not in {"world_audits"}
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
