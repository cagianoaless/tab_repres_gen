from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json, read_plan
from relational_tfm.runner import verify_shard

EXPECTED_PLAN_SHA256 = (
    "dc6872565a8726a715a9437cf5029db6c733c469b08e15fc097b5fd9dc8e7006"
)
EXPECTED_CELLS = 8_208
BOOTSTRAP_SEED = 20_260_728
BOOTSTRAP_REPLICATES = 20_000
PRIMARY_CONTEXT = 40
PRIMARY_CONTRASTS = {
    "shared_multiset_minus_absolute": (
        "relation_multiset",
        "ordered_typed_path",
    ),
    "block_permuted_minus_aligned_combined": (
        "aligned_multiset_with_ordered_block_permutation",
        "ordered_plus_multiset",
    ),
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hierarchical_interval(
    values: dict[str, list[float]],
) -> tuple[float, float]:
    worlds = sorted(values)
    matrix = np.asarray([values[world] for world in worlds], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("GraphLog bootstrap expects three seeds per world")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    world_indices = rng.integers(
        0,
        len(worlds),
        size=(BOOTSTRAP_REPLICATES, len(worlds)),
    )
    seed_indices = rng.integers(
        0,
        matrix.shape[1],
        size=(BOOTSTRAP_REPLICATES, len(worlds), matrix.shape[1]),
    )
    sampled_worlds = matrix[world_indices]
    sampled = np.take_along_axis(sampled_worlds, seed_indices, axis=2)
    draws = sampled.mean(axis=(1, 2))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def _load_cells(
    plan: dict[str, Any],
    result_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    metric_paths = sorted(result_root.glob("shard_*/*.metrics.json"))
    if len(metric_paths) != EXPECTED_CELLS:
        raise ValueError(
            f"GraphLog result is incomplete: {len(metric_paths)}/{EXPECTED_CELLS} metrics"
        )
    shard_manifests = [
        verify_shard(shard_dir)
        for shard_dir in sorted(result_root.glob("shard_*"))
        if shard_dir.is_dir()
    ]
    if not shard_manifests:
        raise ValueError("no GraphLog shard manifests found")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in metric_paths:
        payload = json.loads(path.read_text())
        cell = payload["cell"]
        cell_id = str(cell["cell_id"])
        if cell_id in seen or cell_id not in plan_by_id:
            raise ValueError(f"unexpected or duplicate GraphLog cell: {cell_id}")
        if cell != plan_by_id[cell_id] or payload.get("status") != "complete":
            raise ValueError(f"GraphLog cell does not match the frozen plan: {cell_id}")
        if payload.get("audit", {}).get("status") != "PASS":
            raise ValueError(f"GraphLog cell audit failed: {cell_id}")
        seen.add(cell_id)
        metrics = payload["metrics"]
        metadata = payload["task_metadata"]
        for length, length_metrics in payload["metrics_by_chain_length"].items():
            rows.append(
                {
                    "cell_id": cell_id,
                    "world": cell["world"],
                    "outer_split": metadata["outer_split"],
                    "decoder": cell["decoder"],
                    "encoder": cell["encoder"],
                    "context_per_relation": int(cell["context_per_relation"]),
                    "seed": int(cell["seed"]),
                    "chain_length": int(length),
                    "rows": int(length_metrics["rows"]),
                    "accuracy": float(length_metrics["accuracy"]),
                    "balanced_accuracy": float(
                        length_metrics["balanced_accuracy"]
                    ),
                    "macro_f1": float(length_metrics["macro_f1"]),
                    "equal_length_macro_balanced_accuracy": float(
                        metrics["equal_length_macro_balanced_accuracy"]
                    ),
                    "collision_conditioned_ceiling": float(
                        length_metrics["collision_conditioned_ceiling"]
                    ),
                    "descriptor_graph_path_exact_match_fraction": float(
                        metadata[
                            "evaluation_descriptor_graph_path_exact_match_fraction"
                        ]
                    ),
                }
            )
    if seen != set(plan_by_id):
        raise ValueError("GraphLog result cell population is incomplete")
    return rows, shard_manifests


def _cell_value_map(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    cell_values: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = (
            row["world"],
            row["decoder"],
            row["encoder"],
            row["context_per_relation"],
            row["seed"],
        )
        cell_values[key] = float(row["equal_length_macro_balanced_accuracy"])
    return cell_values


def _aggregate_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cell_values = _cell_value_map(rows)
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (world, decoder, encoder, context, _seed), value in cell_values.items():
        grouped[(decoder, encoder, context)][world].append(value)

    output = []
    for (decoder, encoder, context), by_world in sorted(grouped.items()):
        world_means = [float(np.mean(by_world[world])) for world in sorted(by_world)]
        lower, upper = _hierarchical_interval(by_world)
        output.append(
            {
                "decoder": decoder,
                "encoder": encoder,
                "context_per_relation": context,
                "worlds": len(by_world),
                "seeds_per_world": 3,
                "equal_world_mean_balanced_accuracy": float(np.mean(world_means)),
                "hierarchical_bootstrap_ci95_low": lower,
                "hierarchical_bootstrap_ci95_high": upper,
            }
        )
    return output


def _primary_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_values = _cell_value_map(rows)
    cell_values = {
        (world, decoder, encoder, seed): value
        for (world, decoder, encoder, context, seed), value in all_values.items()
        if context == PRIMARY_CONTEXT
    }
    output = []
    for decoder in ("logistic", "catboost"):
        for contrast, (left, right) in PRIMARY_CONTRASTS.items():
            by_world: dict[str, list[float]] = defaultdict(list)
            for world in (f"rule_{index}" for index in range(57)):
                for seed in (101, 102, 103):
                    by_world[world].append(
                        cell_values[(world, decoder, left, seed)]
                        - cell_values[(world, decoder, right, seed)]
                    )
            world_means = {
                world: float(np.mean(seed_values))
                for world, seed_values in by_world.items()
            }
            panel_mean = float(np.mean(list(world_means.values())))
            lower, upper = _hierarchical_interval(by_world)
            wins = sum(value > 0.0 for value in world_means.values())
            output.append(
                {
                    "decoder": decoder,
                    "contrast": contrast,
                    "left_encoder": left,
                    "right_encoder": right,
                    "context_per_relation": PRIMARY_CONTEXT,
                    "equal_world_mean_delta": panel_mean,
                    "hierarchical_bootstrap_ci95_low": lower,
                    "hierarchical_bootstrap_ci95_high": upper,
                    "world_wins": wins,
                    "world_ties": sum(value == 0.0 for value in world_means.values()),
                    "world_losses": sum(value < 0.0 for value in world_means.values()),
                    "registered_positive_mean": panel_mean > 0.0,
                    "registered_majority_worlds": wins >= 29,
                }
            )
    return output


def _contrast_context_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = _cell_value_map(rows)
    output = []
    for decoder in ("logistic", "catboost"):
        for context in (5, 10, 20, 40):
            for contrast, (left, right) in PRIMARY_CONTRASTS.items():
                by_world: dict[str, list[float]] = defaultdict(list)
                for world in (f"rule_{index}" for index in range(57)):
                    for seed in (101, 102, 103):
                        by_world[world].append(
                            values[(world, decoder, left, context, seed)]
                            - values[(world, decoder, right, context, seed)]
                        )
                world_means = [
                    float(np.mean(seed_values))
                    for seed_values in by_world.values()
                ]
                lower, upper = _hierarchical_interval(by_world)
                output.append(
                    {
                        "decoder": decoder,
                        "contrast": contrast,
                        "context_per_relation": context,
                        "equal_world_mean_delta": float(np.mean(world_means)),
                        "hierarchical_bootstrap_ci95_low": lower,
                        "hierarchical_bootstrap_ci95_high": upper,
                        "world_wins": sum(value > 0.0 for value in world_means),
                        "world_ties": sum(value == 0.0 for value in world_means),
                        "world_losses": sum(value < 0.0 for value in world_means),
                        "analysis_role": (
                            "registered_primary"
                            if context == PRIMARY_CONTEXT
                            else "registered_secondary_context_curve"
                        ),
                    }
                )
    return output


def _contrast_by_length(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = {
        (
            row["world"],
            row["decoder"],
            row["encoder"],
            row["context_per_relation"],
            row["seed"],
            row["chain_length"],
        ): float(row["balanced_accuracy"])
        for row in rows
    }
    output = []
    for decoder in ("logistic", "catboost"):
        for length in (4, 5):
            for contrast, (left, right) in PRIMARY_CONTRASTS.items():
                by_world: dict[str, list[float]] = defaultdict(list)
                for world in (f"rule_{index}" for index in range(57)):
                    for seed in (101, 102, 103):
                        by_world[world].append(
                            values[
                                (
                                    world,
                                    decoder,
                                    left,
                                    PRIMARY_CONTEXT,
                                    seed,
                                    length,
                                )
                            ]
                            - values[
                                (
                                    world,
                                    decoder,
                                    right,
                                    PRIMARY_CONTEXT,
                                    seed,
                                    length,
                                )
                            ]
                        )
                world_means = [
                    float(np.mean(seed_values))
                    for seed_values in by_world.values()
                ]
                lower, upper = _hierarchical_interval(by_world)
                output.append(
                    {
                        "decoder": decoder,
                        "contrast": contrast,
                        "chain_length": length,
                        "context_per_relation": PRIMARY_CONTEXT,
                        "equal_world_mean_delta": float(np.mean(world_means)),
                        "hierarchical_bootstrap_ci95_low": lower,
                        "hierarchical_bootstrap_ci95_high": upper,
                        "world_wins": sum(value > 0.0 for value in world_means),
                        "world_ties": sum(value == 0.0 for value in world_means),
                        "world_losses": sum(value < 0.0 for value in world_means),
                        "analysis_role": "registered_secondary_length_diagnostic",
                    }
                )
    return output


def _primary_by_world(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = _cell_value_map(rows)
    outer_split = {
        str(row["world"]): str(row["outer_split"])
        for row in rows
    }
    output = []
    for decoder in ("logistic", "catboost"):
        for contrast, (left, right) in PRIMARY_CONTRASTS.items():
            for world in (f"rule_{index}" for index in range(57)):
                seed_deltas = [
                    values[(world, decoder, left, PRIMARY_CONTEXT, seed)]
                    - values[(world, decoder, right, PRIMARY_CONTEXT, seed)]
                    for seed in (101, 102, 103)
                ]
                output.append(
                    {
                        "world": world,
                        "outer_split": outer_split[world],
                        "decoder": decoder,
                        "contrast": contrast,
                        "context_per_relation": PRIMARY_CONTEXT,
                        "seed_mean_delta": float(np.mean(seed_deltas)),
                        "seed_min_delta": float(np.min(seed_deltas)),
                        "seed_max_delta": float(np.max(seed_deltas)),
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = read_plan(args.plan.resolve())
    if (
        plan["plan_sha256"] != EXPECTED_PLAN_SHA256
        or len(plan["cells"]) != EXPECTED_CELLS
    ):
        raise ValueError("GraphLog summary received a non-registered plan")
    rows, manifests = _load_cells(plan, args.result_root.resolve())
    curves = _aggregate_curves(rows)
    contrasts = _primary_contrasts(rows)
    contrast_context_curves = _contrast_context_curves(rows)
    contrast_by_length = _contrast_by_length(rows)
    primary_by_world = _primary_by_world(rows)
    decoder_passes = {}
    for decoder in ("logistic", "catboost"):
        selected = [row for row in contrasts if row["decoder"] == decoder]
        decoder_passes[decoder] = all(
            row["registered_positive_mean"]
            and row["registered_majority_worlds"]
            for row in selected
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cell_metrics_by_length.csv", rows)
    _write_csv(output_dir / "aggregate_context_curves.csv", curves)
    _write_csv(output_dir / "primary_contrasts.csv", contrasts)
    _write_csv(output_dir / "contrast_context_curves.csv", contrast_context_curves)
    _write_csv(output_dir / "contrast_by_length.csv", contrast_by_length)
    _write_csv(output_dir / "primary_contrasts_by_world.csv", primary_by_world)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_id": plan["study_id"],
        "plan_sha256": plan["plan_sha256"],
        "cells": EXPECTED_CELLS,
        "shards": len(manifests),
        "worlds": 57,
        "context_seeds": [101, 102, 103],
        "primary_context_per_relation": PRIMARY_CONTEXT,
        "primary_endpoint": "equal_length_macro_balanced_accuracy",
        "bootstrap": {
            "scope": "fixed_panel_sensitivity",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "resampling": "worlds_then_context_seeds_with_replacement",
        },
        "primary_contrasts": contrasts,
        "cpu_decoder_registered_criterion": decoder_passes,
        "cpu_two_decoder_criterion": all(decoder_passes.values()),
        "tabpfn_required_for_cross_decoder_final": True,
        "files": {
            name: {
                "sha256": file_sha256(output_dir / name),
                "rows": sum(1 for _ in (output_dir / name).open()) - 1,
            }
            for name in (
                "cell_metrics_by_length.csv",
                "aggregate_context_curves.csv",
                "primary_contrasts.csv",
                "contrast_context_curves.csv",
                "contrast_by_length.csv",
                "primary_contrasts_by_world.csv",
            )
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cells": summary["cells"],
                "shards": summary["shards"],
                "cpu_two_decoder_criterion": summary[
                    "cpu_two_decoder_criterion"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
