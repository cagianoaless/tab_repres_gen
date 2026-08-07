#!/usr/bin/env python3
"""Audit and summarize the registered CLUTRR shared-coordinate TabPFN cells."""

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

STUDY_ID = "clutrr_symbolic_gpu_tabpfn_shared_coordinates_v1"
EXPECTED_CELLS = 60
EXPECTED_SHARDS = 15
EXPECTED_PLAN = "f911307adcd680acaeb25db68429fd8357c1b190544f6f318fa06b59f92229d5"
CONTEXTS = (5, 10, 20, 40, 80)
SEEDS = (101, 102, 103)
OOD_LENGTHS = tuple(range(4, 11))
ENCODERS = (
    "relation_transition_counts",
    "within_length_transition_joint_permutation",
    "ordered_plus_multiset",
    "within_length_ordered_plus_multiset_joint_permutation",
)
CONTRASTS = (
    (
        "transition_aligned_minus_null",
        "relation_transition_counts",
        "within_length_transition_joint_permutation",
    ),
    (
        "ordered_plus_multiset_aligned_minus_null",
        "ordered_plus_multiset",
        "within_length_ordered_plus_multiset_joint_permutation",
    ),
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically without requiring the package to be installed."""
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate(root: Path) -> list[dict[str, Any]]:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status != {
        "cells": EXPECTED_CELLS,
        "complete": True,
        "completed_at_utc": status.get("completed_at_utc"),
        "failures": 0,
        "shards": EXPECTED_SHARDS,
    }:
        raise ValueError("terminal status does not describe the registered population")

    registration = json.loads(
        (root / "registration.json").read_text(encoding="utf-8")
    )
    if (
        registration.get("cells") != EXPECTED_CELLS
        or registration.get("shards") != EXPECTED_SHARDS
        or registration.get("plan_sha256") != EXPECTED_PLAN
    ):
        raise ValueError("registration does not match the frozen plan")

    metric_paths = sorted((root / "results").glob("shard-*/*.metrics.json"))
    manifests = sorted((root / "results").glob("shard-*/shard_manifest.json"))
    if len(metric_paths) != EXPECTED_CELLS or len(manifests) != EXPECTED_SHARDS:
        raise ValueError("raw output population is incomplete")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    cell_ids = [payload["cell"]["cell_id"] for payload in payloads]
    if len(set(cell_ids)) != EXPECTED_CELLS:
        raise ValueError("duplicate TabPFN cell IDs")

    observed = set()
    for payload in payloads:
        cell = payload["cell"]
        identity = cell["model_identity"]
        key = (
            cell["encoder"],
            int(cell["context_per_relation"]),
            int(cell["seed"]),
        )
        observed.add(key)
        if (
            payload.get("status") != "complete"
            or payload.get("prediction_policy") != "decoder"
            or cell.get("study_id") != STUDY_ID
            or cell.get("decoder") != "tabpfn"
            or identity.get("package_version") != "8.1.0"
            or identity.get("model_version") != "V3"
            or identity.get("n_estimators") != 4
            or payload["decoder_provenance"].get("resolved_devices") != ["cuda:0"]
        ):
            raise ValueError(f"invalid cell provenance: {cell['cell_id']}")

    expected = {
        (encoder, context, seed)
        for encoder in ENCODERS
        for context in CONTEXTS
        for seed in SEEDS
    }
    if observed != expected:
        raise ValueError("TabPFN factorial population changed")
    return payloads


def summarize(payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cells = []
    pivot: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for payload in payloads:
        cell = payload["cell"]
        context = int(cell["context_per_relation"])
        seed = int(cell["seed"])
        encoder = str(cell["encoder"])
        accuracy = mean(
            float(payload["metrics_by_chain_length"][str(length)]["accuracy"])
            for length in OOD_LENGTHS
        )
        cells.append(
            {
                "encoder": encoder,
                "context_per_relation": context,
                "seed": seed,
                "ood_equal_length_accuracy": accuracy,
            }
        )
        pivot[(context, seed)][encoder] = accuracy
        grouped[(encoder, context)].append(accuracy)

    aggregate = [
        {
            "encoder": encoder,
            "context_per_relation": context,
            "seed_mean_ood_accuracy": mean(values),
            "seed_min_ood_accuracy": min(values),
            "seed_max_ood_accuracy": max(values),
        }
        for (encoder, context), values in sorted(grouped.items())
    ]

    contrasts = []
    for (context, seed), values in sorted(pivot.items()):
        if set(values) != set(ENCODERS):
            raise ValueError("encoder pairing drift")
        for name, aligned, null in CONTRASTS:
            contrasts.append(
                {
                    "context_per_relation": context,
                    "seed": seed,
                    "contrast": name,
                    "aligned_accuracy": values[aligned],
                    "null_accuracy": values[null],
                    "difference": values[aligned] - values[null],
                }
            )
    return {"cells": cells, "aggregate": aggregate, "contrasts": contrasts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tables = summarize(load_and_validate(args.results_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)

    largest = [
        row for row in tables["aggregate"] if row["context_per_relation"] == 80
    ]
    largest_contrasts = [
        row for row in tables["contrasts"] if row["context_per_relation"] == 80
    ]
    contrast_means = {
        name: mean(
            float(row["difference"])
            for row in largest_contrasts
            if row["contrast"] == name
        )
        for name, _, _ in CONTRASTS
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_id": STUDY_ID,
        "cells": EXPECTED_CELLS,
        "shards": EXPECTED_SHARDS,
        "plan_sha256": EXPECTED_PLAN,
        "largest_context_aggregate": largest,
        "largest_context_contrasts": contrast_means,
        "uncertainty_scope": (
            "descriptive variation over three context seeds on one fixed official "
            "CLUTRR population"
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(
        f"PASS cells={EXPECTED_CELLS} shards={EXPECTED_SHARDS} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
