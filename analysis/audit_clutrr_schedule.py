from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.clutrr import (
    CLUTRR_EDGE_RELATIONS,
    CLUTRR_RELATIONS,
    load_clutrr_task,
)
from relational_tfm.datasets.clutrr_audit import audit_clutrr_task
from relational_tfm.encoders.clutrr import encode_clutrr_rows
from relational_tfm.hashing import object_sha256
from relational_tfm.plan import atomic_write_json, build_plan

DEFAULT_CONFIGS = (
    "configs/studies/clutrr_symbolic_cpu_confirmation_v2.yaml",
    "configs/studies/clutrr_symbolic_gpu_tabpfn_confirmation_v2.yaml",
    "configs/studies/clutrr_symbolic_cpu_train234_sensitivity_v2.yaml",
    "configs/studies/clutrr_symbolic_gpu_tabpfn_train234_sensitivity_v2.yaml",
)


def _episode_key(cell: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(cell["dataset_config"]),
        int(cell["context_per_relation"]),
        int(cell["seed"]),
    )


def _representative_cells(
    plans: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    representatives: dict[tuple[str, int, int], dict[str, Any]] = {}
    for plan in plans:
        for cell in plan["cells"]:
            representatives.setdefault(_episode_key(cell), cell)
    return representatives


def _row_multiset(values: np.ndarray) -> Counter[bytes]:
    contiguous = np.ascontiguousarray(values)
    return Counter(row.tobytes() for row in contiguous)


def _audit_v2_controls(
    rows: Any,
    *,
    encoder_seed: int,
    registered_encoders: set[str],
) -> int:
    aligned = encode_clutrr_rows(rows, "ordered_typed_path", encoder_seed).values
    controls = {
        encoder: encode_clutrr_rows(rows, encoder, encoder_seed).values
        for encoder in (
            "within_length_joint_permutation",
            "within_length_independent_permutation",
        )
    }
    chain_lengths = rows["chain_length"].to_numpy(dtype=np.int64)
    checks = 0
    for encoder, values in controls.items():
        if not np.array_equal(values[:, 0], aligned[:, 0]):
            raise ValueError(f"{encoder} changed aligned chain lengths")
        for chain_length in np.unique(chain_lengths):
            selected = chain_lengths == chain_length
            expected = aligned[selected, 1:]
            actual = values[selected, 1:]
            if not np.array_equal(
                np.sort(actual, axis=0),
                np.sort(expected, axis=0),
            ):
                raise ValueError(f"{encoder} crossed a chain-length stratum")
            if (
                encoder == "within_length_joint_permutation"
                and _row_multiset(actual) != _row_multiset(expected)
            ):
                raise ValueError("joint CLUTRR null did not preserve relation rows")
            checks += 1
    for aligned_name, control_name in (
        (
            "relation_transition_counts",
            "within_length_transition_joint_permutation",
        ),
        (
            "ordered_plus_multiset",
            "within_length_ordered_plus_multiset_joint_permutation",
        ),
    ):
        if control_name not in registered_encoders:
            continue
        aligned_values = encode_clutrr_rows(
            rows, aligned_name, encoder_seed
        ).values
        control_values = encode_clutrr_rows(
            rows, control_name, encoder_seed
        ).values
        if not np.array_equal(aligned_values[:, 0], control_values[:, 0]):
            raise ValueError(f"{control_name} changed aligned chain lengths")
        for chain_length in np.unique(chain_lengths):
            selected = chain_lengths == chain_length
            expected = aligned_values[selected, 1:]
            actual = control_values[selected, 1:]
            if _row_multiset(actual) != _row_multiset(expected):
                raise ValueError(f"{control_name} did not preserve complete rows")
            checks += 1
    interference_control = "aligned_multiset_with_ordered_block_permutation"
    if interference_control in registered_encoders:
        aligned_values = encode_clutrr_rows(
            rows, "ordered_plus_multiset", encoder_seed
        ).values
        multiset_values = encode_clutrr_rows(
            rows, "relation_multiset", encoder_seed
        ).values
        control_values = encode_clutrr_rows(
            rows, interference_control, encoder_seed
        ).values
        ordered_width = aligned_values.shape[1] - (multiset_values.shape[1] - 1)
        if not np.array_equal(
            control_values[:, ordered_width:],
            multiset_values[:, 1:],
        ):
            raise ValueError("interference control changed the aligned multiset block")
        if np.array_equal(
            control_values[:, 1:ordered_width],
            aligned_values[:, 1:ordered_width],
        ):
            raise ValueError("interference control did not permute the ordered block")
        for chain_length in np.unique(chain_lengths):
            selected = chain_lengths == chain_length
            if _row_multiset(
                control_values[selected, 1:ordered_width]
            ) != _row_multiset(aligned_values[selected, 1:ordered_width]):
                raise ValueError(
                    "interference control crossed an ordered-block length stratum"
                )
            checks += 1
    supported_prefix_control = "aligned_multiset_with_supported_prefix"
    if supported_prefix_control in registered_encoders:
        aligned_context = encode_clutrr_rows(
            rows, "ordered_plus_multiset", encoder_seed
        ).values
        prefix_context = encode_clutrr_rows(
            rows, supported_prefix_control, encoder_seed
        ).values
        relation_count = len(CLUTRR_EDGE_RELATIONS)
        ordered_width = 1 + 10 * relation_count
        first_unsupported = 1 + 3 * relation_count
        if not np.array_equal(
            aligned_context[:, :first_unsupported],
            prefix_context[:, :first_unsupported],
        ):
            raise ValueError("supported-prefix control changed the aligned prefix")
        if not np.array_equal(
            aligned_context[:, ordered_width:],
            prefix_context[:, ordered_width:],
        ):
            raise ValueError("supported-prefix control changed the aligned multiset")
        if np.any(prefix_context[:, first_unsupported:ordered_width] != 0.0):
            raise ValueError("supported-prefix control retained unsupported suffix cells")
        checks += 1
    return checks


def audit(config_paths: list[Path], repo_root: Path) -> dict[str, Any]:
    plans = [build_plan(load_study_config(path)) for path in config_paths]
    representatives = _representative_cells(plans)
    contexts: dict[tuple[str, int], dict[int, set[str]]] = defaultdict(dict)
    evaluation_ids: dict[str, set[str]] = {}
    encoder_schemas: dict[str, tuple[str, ...]] = {}
    registered_encoders = {
        cell["encoder"] for plan in plans for cell in plan["cells"]
    }
    audited_rows = 0
    control_invariant_checks = 0

    for (dataset_config, budget, seed), cell in sorted(representatives.items()):
        task = load_clutrr_task(repo_root, cell)
        audit_clutrr_task(task)
        context_ids = set(map(str, task.context["row_id"]))
        contexts[(dataset_config, seed)][budget] = context_ids
        current_evaluation_ids = set(map(str, task.evaluation["row_id"]))
        previous_evaluation_ids = evaluation_ids.setdefault(
            dataset_config, current_evaluation_ids
        )
        if current_evaluation_ids != previous_evaluation_ids:
            raise ValueError("CLUTRR evaluation population changed across context episodes")
        audited_rows += len(task.context) + len(task.evaluation)
        control_invariant_checks += _audit_v2_controls(
            task.context,
            encoder_seed=int(cell["encoder_seed"]),
            registered_encoders=registered_encoders,
        )
        control_invariant_checks += _audit_v2_controls(
            task.evaluation,
            encoder_seed=int(cell["encoder_seed"]) + 1,
            registered_encoders=registered_encoders,
        )

        for encoder in sorted(registered_encoders):
            encoded = encode_clutrr_rows(
                task.evaluation,
                encoder,
                int(cell["encoder_seed"]),
            )
            if encoded.values.shape[0] != len(task.evaluation):
                raise ValueError("CLUTRR encoder changed the evaluation population")
            if not np.isfinite(encoded.values).all():
                raise ValueError("CLUTRR encoder produced non-finite values")
            previous_schema = encoder_schemas.setdefault(encoder, encoded.feature_names)
            if encoded.feature_names != previous_schema:
                raise ValueError(f"CLUTRR encoder schema drifted: {encoder}")
            if (
                encoded.provenance["raw_entity_ids_exposed"]
                or encoded.provenance["story_text_exposed"]
                or encoded.provenance["target_relation_exposed"]
            ):
                raise ValueError(f"CLUTRR encoder violates leakage boundary: {encoder}")

    nesting_checks = 0
    for episode_contexts in contexts.values():
        ordered = sorted(episode_contexts)
        for smaller, larger in zip(ordered, ordered[1:], strict=False):
            if not episode_contexts[smaller] < episode_contexts[larger]:
                raise ValueError("CLUTRR contexts are not strict nested prefixes")
            nesting_checks += 1

    unique_plan_hashes = {plan["plan_sha256"] for plan in plans}
    if len(unique_plan_hashes) != len(plans):
        raise ValueError("CLUTRR registered plans are not distinct")
    scientific_body = {
        "schema_version": 1,
        "status": "PASS",
        "configs": [
            {
                "path": str(path.relative_to(repo_root)),
                "study_id": plan["study_id"],
                "cells": len(plan["cells"]),
                "plan_sha256": plan["plan_sha256"],
            }
            for path, plan in zip(config_paths, plans, strict=True)
        ],
        "total_registered_cells": sum(len(plan["cells"]) for plan in plans),
        "unique_episodes": len(representatives),
        "nested_context_checks": nesting_checks,
        "evaluation_populations": {
            dataset_config: len(row_ids)
            for dataset_config, row_ids in sorted(evaluation_ids.items())
        },
        "target_relations": len(CLUTRR_RELATIONS),
        "encoder_feature_counts": {
            encoder: len(feature_names)
            for encoder, feature_names in sorted(encoder_schemas.items())
        },
        "audited_context_and_evaluation_rows": audited_rows,
        "length_stratified_control_invariant_checks": control_invariant_checks,
        "leakage_checks": {
            "raw_entity_ids_exposed": False,
            "story_text_exposed": False,
            "target_relation_exposed": False,
        },
    }
    return {
        **scientific_body,
        "audit_sha256": object_sha256(scientific_body),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="repeat to override the four registered CLUTRR configurations",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_paths = [
        Path(path) if Path(path).is_absolute() else repo_root / path
        for path in (args.configs or DEFAULT_CONFIGS)
    ]
    result = audit(config_paths, repo_root)
    if args.output:
        atomic_write_json(args.output, result)
    print(
        f"PASS cells={result['total_registered_cells']} "
        f"episodes={result['unique_episodes']} "
        f"audit_sha256={result['audit_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
