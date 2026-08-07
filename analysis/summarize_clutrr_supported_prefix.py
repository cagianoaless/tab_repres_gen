#!/usr/bin/env python3
"""Summarize the paired CLUTRR supported-prefix intervention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json, read_plan

EXPECTED = {
    "cpu": {
        "plan_sha256": (
            "b79695cce1226249e4e76b6173603744d05d40ddc9ad77f9544181d63eff219a"
        ),
        "cells": 30,
        "decoders": {"logistic", "catboost"},
    },
    "gpu": {
        "plan_sha256": (
            "d79d377854011288e0a54e03d631391d886b0e421fbe9a7cc3a1608d27ef4c75"
        ),
        "cells": 15,
        "decoders": {"tabpfn"},
    },
}
OOD_LENGTHS = tuple(range(4, 11))


def _metric_paths(root: Path) -> list[Path]:
    return sorted(
        {
            *root.glob("results/shard-*/*.metrics.json"),
            *root.glob("results/shard_*/*.metrics.json"),
        }
    )


def _load(root: Path, stream: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = EXPECTED[stream]
    plan = read_plan(root / "plan.json")
    if (
        plan["plan_sha256"] != expected["plan_sha256"]
        or len(plan["cells"]) != expected["cells"]
    ):
        raise ValueError(f"{stream} plan does not match the frozen intervention")
    plan_by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    paths = _metric_paths(root)
    if len(paths) != expected["cells"]:
        raise ValueError(f"{stream} results are incomplete: {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    seen = {payload["cell"]["cell_id"] for payload in payloads}
    if seen != set(plan_by_id):
        raise ValueError(f"{stream} result cells differ from the frozen plan")
    if any(
        payload.get("status") != "complete"
        or payload["cell"] != plan_by_id[payload["cell"]["cell_id"]]
        or payload.get("audit", {}).get("status") != "PASS"
        for payload in payloads
    ):
        raise ValueError(f"{stream} contains an invalid cell")
    if {payload["cell"]["decoder"] for payload in payloads} != expected["decoders"]:
        raise ValueError(f"{stream} decoder population changed")
    return payloads, plan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _rows(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = []
    by_length = []
    for payload in payloads:
        cell = payload["cell"]
        common = {
            "decoder": cell["decoder"],
            "context_per_relation": int(cell["context_per_relation"]),
            "seed": int(cell["seed"]),
        }
        ood = [
            payload["metrics_by_chain_length"][str(length)]
            for length in OOD_LENGTHS
        ]
        cells.append(
            {
                **common,
                "ood_equal_length_accuracy": float(
                    np.mean([row["accuracy"] for row in ood])
                ),
                "ood_equal_length_balanced_accuracy": float(
                    np.mean([row["balanced_accuracy"] for row in ood])
                ),
                "ood_equal_length_macro_f1": float(
                    np.mean([row["macro_f1"] for row in ood])
                ),
                "ood_equal_length_top3_accuracy": float(
                    np.mean([row["top3_accuracy"] for row in ood])
                ),
                "ood_equal_length_log_loss": float(
                    np.mean([row["log_loss"] for row in ood])
                ),
            }
        )
        for length in OOD_LENGTHS:
            metrics = payload["metrics_by_chain_length"][str(length)]
            by_length.append(
                {
                    **common,
                    "chain_length": length,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "rows": int(metrics["rows"]),
                }
            )
    return cells, by_length


def _baseline_rows(cpu_path: Path, gpu_path: Path) -> dict[tuple[str, int], float]:
    output = {}
    for path, default_decoder in (
        (cpu_path, None),
        (gpu_path, "tabpfn"),
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["encoder"] != "ordered_plus_multiset":
                    continue
                decoder = row.get("decoder") or default_decoder
                if decoder is None:
                    raise ValueError(f"baseline decoder is missing: {path}")
                output[(decoder, int(row["context_per_relation"]))] = float(
                    row["seed_mean_ood_accuracy"]
                )
    return output


def summarize(
    cpu_root: Path,
    gpu_root: Path,
    baseline_cpu: Path,
    baseline_gpu: Path,
    output: Path,
) -> dict[str, Any]:
    cpu_payloads, cpu_plan = _load(cpu_root, "cpu")
    gpu_payloads, gpu_plan = _load(gpu_root, "gpu")
    cells, by_length = _rows(cpu_payloads + gpu_payloads)
    baselines = _baseline_rows(baseline_cpu, baseline_gpu)
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in cells:
        key = (str(row["decoder"]), int(row["context_per_relation"]))
        grouped.setdefault(key, []).append(float(row["ood_equal_length_accuracy"]))
    comparisons = []
    for key, values in sorted(grouped.items()):
        decoder, context = key
        supported = float(np.mean(values))
        ordered = baselines[key]
        comparisons.append(
            {
                "decoder": decoder,
                "context_per_relation": context,
                "supported_prefix_ood_accuracy": supported,
                "ordered_plus_multiset_ood_accuracy": ordered,
                "supported_prefix_minus_ordered_plus_multiset": supported - ordered,
                "seed_min_supported_prefix_accuracy": float(np.min(values)),
                "seed_max_supported_prefix_accuracy": float(np.max(values)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "cells.csv", cells)
    _write_csv(output / "by_length.csv", by_length)
    _write_csv(output / "comparison.csv", comparisons)
    maximum = max(
        abs(row["supported_prefix_minus_ordered_plus_multiset"])
        for row in comparisons
    )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "targeted intervention that retains the aligned multiset and absolute "
            "positions 1--3 while zeroing only positions 4--10"
        ),
        "cells": len(cells),
        "cpu_plan_sha256": cpu_plan["plan_sha256"],
        "gpu_plan_sha256": gpu_plan["plan_sha256"],
        "decoders": sorted({row["decoder"] for row in cells}),
        "contexts_per_relation": sorted(
            {row["context_per_relation"] for row in cells}
        ),
        "maximum_absolute_accuracy_change": maximum,
        "interpretation": (
            "Removing evaluation-only suffix coordinates did not change the OOD "
            "accuracy curve. The earlier block-permutation gain therefore cannot be "
            "attributed to unsupported suffix coordinates alone."
        ),
        "files": {
            name: file_sha256(output / name)
            for name in ("cells.csv", "by_length.csv", "comparison.csv")
        },
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--baseline-cpu", type=Path, required=True)
    parser.add_argument("--baseline-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.cpu_root.resolve(),
        args.gpu_root.resolve(),
        args.baseline_cpu.resolve(),
        args.baseline_gpu.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
