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

PRIMARY_CONTEXT = 40
BOOTSTRAP_SEED = 20_260_729
BOOTSTRAP_REPLICATES = 20_000
REGISTERED_STREAMS = {
    "fixed_cpu": {
        "plan_sha256": (
            "dc6872565a8726a715a9437cf5029db6c733c469b08e15fc097b5fd9dc8e7006"
        ),
        "cells": 8_208,
        "decoders": {"logistic", "catboost"},
    },
    "tuned_catboost": {
        "plan_sha256": (
            "4a84be6406befdf810dd174a8abfa1d307d04b6c65e1c1b1d65c565ec0ac8b2b"
        ),
        "cells": 684,
        "decoders": {"catboost_tuned"},
    },
    "depth1_catboost": {
        "plan_sha256": (
            "37230cb4f827567de0c14428661efe39dfddee212fbec044c32d3c7c85761487"
        ),
        "cells": 684,
        "decoders": {"catboost_depth1"},
    },
    "tabpfn": {
        "plan_sha256": (
            "decac0f768513abfd742bbd3e84bf3934aefa22043043e35a940ebf14406b75d"
        ),
        "cells": 4_104,
        "decoders": {"tabpfn"},
    },
}
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
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _metric_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("shard-*/*.metrics.json"))
    paths.extend(sorted(root.glob("shard_*/*.metrics.json")))
    paths.extend(sorted(root.glob("*.metrics.json")))
    return paths


def _load_stream(
    *,
    stream: str,
    plan_path: Path,
    result_root: Path,
) -> list[dict[str, Any]]:
    specification = REGISTERED_STREAMS[stream]
    plan = read_plan(plan_path.resolve())
    if (
        plan["plan_sha256"] != specification["plan_sha256"]
        or len(plan["cells"]) != specification["cells"]
    ):
        raise ValueError(f"{stream} does not match its frozen plan")

    expected = {cell["cell_id"]: cell for cell in plan["cells"]}
    paths = _metric_paths(result_root.resolve())
    if len(paths) != specification["cells"]:
        raise ValueError(
            f"{stream} is incomplete: {len(paths)}/{specification['cells']} metrics"
        )

    rows = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        cell = payload["cell"]
        cell_id = str(cell["cell_id"])
        if cell_id in seen or cell_id not in expected:
            raise ValueError(f"{stream} has an unexpected cell: {cell_id}")
        if cell != expected[cell_id] or payload.get("status") != "complete":
            raise ValueError(f"{stream} cell does not match its frozen plan: {cell_id}")
        if payload.get("audit", {}).get("status") != "PASS":
            raise ValueError(f"{stream} cell audit failed: {cell_id}")
        if cell["decoder"] not in specification["decoders"]:
            raise ValueError(f"{stream} contains an unexpected decoder")
        seen.add(cell_id)
        rows.append(
            {
                "stream": stream,
                "cell_id": cell_id,
                "world": str(cell["world"]),
                "decoder": str(cell["decoder"]),
                "encoder": str(cell["encoder"]),
                "context_per_relation": int(cell["context_per_relation"]),
                "seed": int(cell["seed"]),
                "equal_length_macro_balanced_accuracy": float(
                    payload["metrics"]["equal_length_macro_balanced_accuracy"]
                ),
                "decoder_package": payload["decoder_provenance"].get("package", ""),
                "decoder_package_version": payload["decoder_provenance"].get(
                    "package_version", ""
                ),
                "decoder_parameters": json.dumps(
                    payload["decoder_provenance"].get("parameters", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    if seen != set(expected):
        raise ValueError(f"{stream} cell population is incomplete")
    return rows


def _fixed_panel_interval(by_world: dict[str, list[float]]) -> tuple[float, float]:
    worlds = sorted(by_world)
    matrix = np.asarray([by_world[world] for world in worlds], dtype=np.float64)
    if matrix.shape != (57, 3):
        raise ValueError(f"expected a 57x3 world/seed matrix, received {matrix.shape}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    world_indices = rng.integers(
        0, matrix.shape[0], size=(BOOTSTRAP_REPLICATES, matrix.shape[0])
    )
    seed_indices = rng.integers(
        0,
        matrix.shape[1],
        size=(BOOTSTRAP_REPLICATES, matrix.shape[0], matrix.shape[1]),
    )
    sampled_worlds = matrix[world_indices]
    sampled = np.take_along_axis(sampled_worlds, seed_indices, axis=2)
    lower, upper = np.quantile(sampled.mean(axis=(1, 2)), [0.025, 0.975])
    return float(lower), float(upper)


def _value_map(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    values = {}
    for row in rows:
        key = (
            row["decoder"],
            row["world"],
            row["encoder"],
            row["context_per_relation"],
            row["seed"],
        )
        if key in values:
            raise ValueError(f"duplicate decoder/world/encoder/context/seed key: {key}")
        values[key] = float(row["equal_length_macro_balanced_accuracy"])
    return values


def _endpoint_encoder_scores(
    values: dict[tuple[Any, ...], float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (decoder, _world, encoder, context, _seed), value in values.items():
        if context == PRIMARY_CONTEXT:
            grouped[(decoder, encoder)].append(value)
    return [
        {
            "decoder": decoder,
            "encoder": encoder,
            "context_per_relation": PRIMARY_CONTEXT,
            "world_seed_cells": len(scores),
            "equal_world_seed_mean_balanced_accuracy": float(np.mean(scores)),
        }
        for (decoder, encoder), scores in sorted(grouped.items())
    ]


def _contrast_rows(
    values: dict[tuple[Any, ...], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decoders = sorted({key[0] for key in values})
    summary = []
    by_world_rows = []
    for decoder in decoders:
        for contrast, (left, right) in PRIMARY_CONTRASTS.items():
            by_world: dict[str, list[float]] = defaultdict(list)
            for world_index in range(57):
                world = f"rule_{world_index}"
                for seed in (101, 102, 103):
                    by_world[world].append(
                        values[(decoder, world, left, PRIMARY_CONTEXT, seed)]
                        - values[(decoder, world, right, PRIMARY_CONTEXT, seed)]
                    )
            world_means = {
                world: float(np.mean(seed_values))
                for world, seed_values in by_world.items()
            }
            lower, upper = _fixed_panel_interval(by_world)
            summary.append(
                {
                    "decoder": decoder,
                    "contrast": contrast,
                    "context_per_relation": PRIMARY_CONTEXT,
                    "equal_world_mean_delta": float(np.mean(list(world_means.values()))),
                    "fixed_panel_sensitivity_low": lower,
                    "fixed_panel_sensitivity_high": upper,
                    "world_wins": sum(value > 0.0 for value in world_means.values()),
                    "world_ties": sum(value == 0.0 for value in world_means.values()),
                    "world_losses": sum(value < 0.0 for value in world_means.values()),
                }
            )
            for world, value in sorted(world_means.items()):
                by_world_rows.append(
                    {
                        "decoder": decoder,
                        "contrast": contrast,
                        "world": world,
                        "seed_mean_delta": value,
                        "seed_min_delta": float(np.min(by_world[world])),
                        "seed_max_delta": float(np.max(by_world[world])),
                    }
                )
    return summary, by_world_rows


def _tuning_deltas(
    contrasts: list[dict[str, Any]],
    by_world_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del contrasts
    lookup = {
        (row["decoder"], row["contrast"], row["world"]): row["seed_mean_delta"]
        for row in by_world_rows
    }
    output = []
    for contrast in PRIMARY_CONTRASTS:
        paired = np.asarray(
            [
                lookup[("catboost_tuned", contrast, f"rule_{index}")]
                - lookup[("catboost", contrast, f"rule_{index}")]
                for index in range(57)
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "contrast": contrast,
                "tuned_minus_fixed_equal_world_delta": float(paired.mean()),
                "world_wins": int(np.sum(paired > 0.0)),
                "world_ties": int(np.sum(paired == 0.0)),
                "world_losses": int(np.sum(paired < 0.0)),
                "world_minimum": float(paired.min()),
                "world_maximum": float(paired.max()),
            }
        )
    return output


def _capacity_summary(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in contrasts
        if row["decoder"] in {"catboost_depth1", "catboost", "catboost_tuned"}
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    for stream in REGISTERED_STREAMS:
        parser.add_argument(f"--{stream.replace('_', '-')}-plan", type=Path, required=True)
        parser.add_argument(
            f"--{stream.replace('_', '-')}-results", type=Path, required=True
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for stream in REGISTERED_STREAMS:
        rows.extend(
            _load_stream(
                stream=stream,
                plan_path=getattr(args, f"{stream}_plan"),
                result_root=getattr(args, f"{stream}_results"),
            )
        )
    values = _value_map(rows)
    encoder_scores = _endpoint_encoder_scores(values)
    contrasts, by_world_rows = _contrast_rows(values)
    tuning = _tuning_deltas(contrasts, by_world_rows)
    capacity = _capacity_summary(contrasts)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cells.csv": rows,
        "endpoint_encoder_scores.csv": encoder_scores,
        "primary_contrasts.csv": contrasts,
        "primary_contrasts_by_world.csv": by_world_rows,
        "catboost_tuning_sensitivity.csv": tuning,
        "catboost_capacity_sensitivity.csv": capacity,
    }
    for filename, table in tables.items():
        _write_csv(output_dir / filename, table)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "complete GraphLog decoder sensitivity on the fixed 57-world panel"
        ),
        "primary_context_per_relation": PRIMARY_CONTEXT,
        "primary_endpoint": "equal_length_macro_balanced_accuracy",
        "depth1_source_commit": "artifact-source-revision",
        "streams": {
            stream: {
                "plan_sha256": specification["plan_sha256"],
                "cells": specification["cells"],
                "decoders": sorted(specification["decoders"]),
            }
            for stream, specification in REGISTERED_STREAMS.items()
        },
        "contrasts": contrasts,
        "catboost_tuning_sensitivity": tuning,
        "catboost_capacity_sensitivity": capacity,
        "fixed_panel_sensitivity": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "resampling": "worlds_then_context_seeds_with_replacement",
            "interpretation": (
                "descriptive sensitivity for this fixed 57-world panel, not a "
                "confidence interval for a population of logical worlds"
            ),
        },
        "files": {
            filename: {
                "sha256": file_sha256(output_dir / filename),
                "rows": len(table),
            }
            for filename, table in tables.items()
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
