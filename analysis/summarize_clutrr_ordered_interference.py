#!/usr/bin/env python3
"""Audit the prospective CLUTRR ordered-coordinate interference control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

CPU_STUDY = "clutrr_symbolic_cpu_ordered_interference_v1"
GPU_STUDY = "clutrr_symbolic_gpu_tabpfn_ordered_interference_v1"
CPU_PLAN = "473f626ccd779dce6333db30ae318878b95b098036ead7e7b7038ae6b3b93a0d"
GPU_PLAN = "145101a0cb89d06fdaac16e55d0fd28af30dee4afb5973aa8209f3770b07f2bd"
ENCODER = "aligned_multiset_with_ordered_block_permutation"
CONTEXTS = (5, 10, 20, 40, 80)
SEEDS = (101, 102, 103)
OOD_LENGTHS = tuple(range(4, 11))
PRIMARY_DECODERS = ("logistic", "catboost", "tabpfn")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_root(
    root: Path,
    *,
    study_id: str,
    plan_sha256: str,
    cells: int,
    shards: int,
) -> list[dict[str, Any]]:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("cells") != cells
        or status.get("shards") != shards
        or status.get("failures") != 0
    ):
        raise ValueError(f"incomplete execution root: {root}")
    registration = json.loads(
        (root / "registration.json").read_text(encoding="utf-8")
    )
    if (
        registration.get("cells") != cells
        or registration.get("shards") != shards
        or registration.get("plan_sha256") != plan_sha256
    ):
        raise ValueError(f"registration drift: {root}")
    paths = sorted((root / "results").glob("shard-*/*.metrics.json"))
    manifests = sorted((root / "results").glob("shard-*/shard_manifest.json"))
    if len(paths) != cells or len(manifests) != shards:
        raise ValueError(f"raw population mismatch: {root}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({payload["cell"]["cell_id"] for payload in payloads}) != cells:
        raise ValueError(f"duplicate cell IDs: {root}")
    for payload in payloads:
        cell = payload["cell"]
        if (
            payload.get("status") != "complete"
            or payload.get("prediction_policy") != "decoder"
            or cell.get("study_id") != study_id
            or cell.get("encoder") != ENCODER
        ):
            raise ValueError(f"invalid cell provenance: {cell.get('cell_id')}")
    return payloads


def _load_reference_metrics(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in root.glob("**/*.metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload["cell"]
        if (
            cell.get("encoder") != "relation_multiset"
            or cell.get("decoder") not in PRIMARY_DECODERS
        ):
            continue
        key = (
            str(cell["decoder"]),
            int(cell["context_per_relation"]),
            int(cell["seed"]),
        )
        rows[key] = {
            "task_sha256": payload["task_metadata"]["task_sha256"],
            "ood_equal_length_accuracy": mean(
                float(payload["metrics_by_chain_length"][str(length)]["accuracy"])
                for length in OOD_LENGTHS
            ),
        }
    return rows


def _load_shared_cells(
    cpu_path: Path, gpu_path: Path
) -> dict[tuple[str, int, int], float]:
    rows: dict[tuple[str, int, int], float] = {}
    for path, forced_decoder in ((cpu_path, None), (gpu_path, "tabpfn")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["encoder"] != "ordered_plus_multiset":
                    continue
                decoder = forced_decoder or row["decoder"]
                key = (
                    str(decoder),
                    int(row["context_per_relation"]),
                    int(row["seed"]),
                )
                rows[key] = float(row["ood_equal_length_accuracy"])
    return rows


def _summarize_new(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        cell = payload["cell"]
        rows.append(
            {
                "decoder": str(cell["decoder"]),
                "context_per_relation": int(cell["context_per_relation"]),
                "seed": int(cell["seed"]),
                "task_sha256": payload["task_metadata"]["task_sha256"],
                "ood_equal_length_accuracy": mean(
                    float(payload["metrics_by_chain_length"][str(length)]["accuracy"])
                    for length in OOD_LENGTHS
                ),
            }
        )
    expected = {
        (decoder, context, seed)
        for decoder in (
            "logistic",
            "logistic_tuned",
            "catboost",
            "catboost_tuned",
            "mlp",
            "tabpfn",
        )
        for context in CONTEXTS
        for seed in SEEDS
    }
    observed = {
        (row["decoder"], row["context_per_relation"], row["seed"]) for row in rows
    }
    if observed != expected:
        raise ValueError("interference-control factorial population changed")
    return sorted(
        rows,
        key=lambda row: (
            str(row["decoder"]),
            int(row["context_per_relation"]),
            int(row["seed"]),
        ),
    )


def _comparisons(
    new_rows: list[dict[str, Any]],
    multiset: dict[tuple[str, int, int], dict[str, Any]],
    ordered_plus_multiset: dict[tuple[str, int, int], float],
) -> list[dict[str, Any]]:
    indexed = {
        (str(row["decoder"]), int(row["context_per_relation"]), int(row["seed"])): row
        for row in new_rows
    }
    output = []
    for decoder in PRIMARY_DECODERS:
        for context in CONTEXTS:
            for seed in SEEDS:
                key = (decoder, context, seed)
                current = indexed[key]
                reference = multiset[key]
                if current["task_sha256"] != reference["task_sha256"]:
                    raise ValueError(f"task population drift for {key}")
                for name, value in (
                    (
                        "relation_multiset",
                        float(reference["ood_equal_length_accuracy"]),
                    ),
                    ("ordered_plus_multiset", ordered_plus_multiset[key]),
                ):
                    output.append(
                        {
                            "decoder": decoder,
                            "context_per_relation": context,
                            "seed": seed,
                            "reference": name,
                            "interference_control_accuracy": current[
                                "ood_equal_length_accuracy"
                            ],
                            "reference_accuracy": value,
                            "difference": current["ood_equal_length_accuracy"] - value,
                        }
                    )
    return output


def _aggregate(
    new_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_new: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in new_rows:
        grouped_new[(row["decoder"], row["context_per_relation"])].append(
            float(row["ood_equal_length_accuracy"])
        )
    aggregate = [
        {
            "decoder": decoder,
            "context_per_relation": context,
            "seed_mean_ood_accuracy": mean(values),
            "seed_min_ood_accuracy": min(values),
            "seed_max_ood_accuracy": max(values),
        }
        for (decoder, context), values in sorted(grouped_new.items())
    ]
    grouped_comparisons: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in comparisons:
        grouped_comparisons[
            (row["decoder"], row["context_per_relation"], row["reference"])
        ].append(float(row["difference"]))
    comparison_aggregate = [
        {
            "decoder": decoder,
            "context_per_relation": context,
            "reference": reference,
            "seed_mean_difference": mean(values),
            "seed_min_difference": min(values),
            "seed_max_difference": max(values),
        }
        for (decoder, context, reference), values in sorted(
            grouped_comparisons.items()
        )
    ]
    return aggregate, comparison_aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--primary-cpu-root", type=Path, required=True)
    parser.add_argument("--primary-gpu-root", type=Path, required=True)
    parser.add_argument("--shared-cpu-cells", type=Path, required=True)
    parser.add_argument("--shared-gpu-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payloads = _validate_root(
        args.new_root / "cpu",
        study_id=CPU_STUDY,
        plan_sha256=CPU_PLAN,
        cells=75,
        shards=15,
    )
    payloads.extend(
        _validate_root(
            args.new_root / "gpu",
            study_id=GPU_STUDY,
            plan_sha256=GPU_PLAN,
            cells=15,
            shards=5,
        )
    )
    new_rows = _summarize_new(payloads)
    multiset = _load_reference_metrics(args.primary_cpu_root)
    multiset.update(_load_reference_metrics(args.primary_gpu_root))
    shared = _load_shared_cells(args.shared_cpu_cells, args.shared_gpu_cells)
    expected_reference = {
        (decoder, context, seed)
        for decoder in PRIMARY_DECODERS
        for context in CONTEXTS
        for seed in SEEDS
    }
    if set(multiset) != expected_reference or not expected_reference <= set(shared):
        raise ValueError("reference populations are incomplete")

    comparisons = _comparisons(new_rows, multiset, shared)
    aggregate, comparison_aggregate = _aggregate(new_rows, comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "cells.csv", new_rows)
    _write_csv(args.output_dir / "aggregate.csv", aggregate)
    _write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    _write_csv(
        args.output_dir / "paired_comparison_aggregate.csv", comparison_aggregate
    )

    largest = [
        row
        for row in aggregate
        if row["context_per_relation"] == 80
        and row["decoder"] in PRIMARY_DECODERS
    ]
    largest_comparisons = [
        row
        for row in comparison_aggregate
        if row["context_per_relation"] == 80
    ]
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_ids": [CPU_STUDY, GPU_STUDY],
        "cells": 90,
        "shards": 20,
        "failures": 0,
        "plan_sha256": {"cpu": CPU_PLAN, "gpu": GPU_PLAN},
        "largest_context": largest,
        "largest_context_comparisons": largest_comparisons,
        "interpretation": (
            "Permuting only the absolute-position block while preserving the aligned "
            "multiset recovers much of the performance lost by ordered-plus-multiset "
            "at high context. The remaining gap to multiset-only at smaller contexts "
            "shows a separate dimensional or nuisance-feature burden."
        ),
        "uncertainty_scope": (
            "Descriptive paired variation over three context seeds on one fixed "
            "official CLUTRR population; no sampled-task inference."
        ),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(f"PASS cells=90 shards=20 output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
