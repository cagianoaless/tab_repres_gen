from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.compositional_shift import (
    COMPOSITION_CLASS_COUNT,
    generate_compositional_shift_task,
)
from relational_tfm.datasets.compositional_shift_audit import (
    audit_compositional_shift_task,
)
from relational_tfm.encoders.compositional_shift import (
    encode_compositional_shift_rows,
)
from relational_tfm.evaluation.classification import collision_conditioned_ceiling
from relational_tfm.plan import atomic_write_json, build_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_CONFIG = REPO_ROOT / "configs/studies/dihedral16_cpu_confirmation_v1.yaml"
GPU_CONFIG = (
    REPO_ROOT
    / "configs/studies/dihedral16_gpu_tabpfn_confirmation_v1.yaml"
)


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


def _paired_scientific_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell["algebra"],
        tuple(cell["train_chain_lengths"]),
        tuple(cell["evaluation_chain_lengths"]),
        cell["max_path_length"],
        cell["evaluation_per_class_per_length"],
        cell["max_context_per_class"],
        cell["encoder"],
        cell["context_per_class"],
        cell["seed"],
        cell["graph_seed"],
        cell["relabel_seed"],
        cell["split_seed"],
        cell["encoder_seed"],
    )


def build_audit() -> dict[str, Any]:
    cpu_plan = build_plan(load_study_config(CPU_CONFIG))
    gpu_plan = build_plan(load_study_config(GPU_CONFIG))
    cpu_keys = {_paired_scientific_key(cell) for cell in cpu_plan["cells"]}
    gpu_keys = {_paired_scientific_key(cell) for cell in gpu_plan["cells"]}
    if cpu_keys != gpu_keys:
        raise ValueError("CPU and GPU compositional populations are not paired")

    seed_audits: list[dict[str, Any]] = []
    maximum_cells = [
        cell
        for cell in cpu_plan["cells"]
        if cell["decoder"] == "logistic"
        and cell["encoder"] == "ordered_plus_coarse_multiset"
        and cell["context_per_class"] == cell["max_context_per_class"]
    ]
    for cell in maximum_cells:
        task = generate_compositional_shift_task(cell)
        task_audit = audit_compositional_shift_task(task)
        encoder_audits: dict[str, Any] = {}
        encoders = sorted(
            {
                plan_cell["encoder"]
                for plan_cell in cpu_plan["cells"]
                if plan_cell["seed"] == cell["seed"]
            }
        )
        for encoder in encoders:
            context = encode_compositional_shift_rows(
                task.context, encoder, int(cell["encoder_seed"])
            )
            evaluation = encode_compositional_shift_rows(
                task.evaluation, encoder, int(cell["encoder_seed"]) + 1
            )
            encoder_audits[encoder] = {
                "feature_count": len(context.feature_names),
                "ceiling": collision_conditioned_ceiling(
                    evaluation.values,
                    task.evaluation["label"].to_numpy(dtype=np.int64),
                ),
                "support": coordinate_support(
                    context.values,
                    evaluation.values,
                ),
                "provenance": context.provenance,
            }

        aligned = encode_compositional_shift_rows(
            task.evaluation,
            "ordered_plus_coarse_multiset",
            int(cell["encoder_seed"]) + 1,
        )
        block_null = encode_compositional_shift_rows(
            task.evaluation,
            "aligned_coarse_multiset_with_ordered_block_permutation",
            int(cell["encoder_seed"]) + 1,
        )
        ordered_width = 1 + 10 * COMPOSITION_CLASS_COUNT
        if aligned.values.shape != block_null.values.shape:
            raise ValueError("D8 blockwise pair changed dimensionality")
        if not np.array_equal(
            aligned.values[:, ordered_width:],
            block_null.values[:, ordered_width:],
        ):
            raise ValueError("D8 blockwise null changed the shared multiset block")
        if np.array_equal(
            aligned.values[:, 1:ordered_width],
            block_null.values[:, 1:ordered_width],
        ):
            raise ValueError("D8 blockwise null did not change the ordered block")

        seed_audits.append(
            {
                "seed": int(cell["seed"]),
                "task": task_audit,
                "context_row_ids_sha256": task.metadata[
                    "context_row_ids_sha256"
                ],
                "evaluation_row_ids_sha256": task.metadata[
                    "evaluation_row_ids_sha256"
                ],
                "encoders": encoder_audits,
                "blockwise_pair": {
                    "same_shape": True,
                    "shared_block_byte_identical": True,
                    "ordered_block_changed": True,
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "algebra": "dihedral_d8",
        "cpu_plan_sha256": cpu_plan["plan_sha256"],
        "gpu_plan_sha256": gpu_plan["plan_sha256"],
        "cpu_cells": len(cpu_plan["cells"]),
        "gpu_cells": len(gpu_plan["cells"]),
        "paired_population_keys": len(cpu_keys),
        "replication_units": len(seed_audits),
        "seed_audits": seed_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit()
    atomic_write_json(args.output, audit)
    print(json.dumps({key: audit[key] for key in audit if key != "seed_audits"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
