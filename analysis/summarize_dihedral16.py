#!/usr/bin/env python3
"""Summarize the registered D8 compositional length-shift replication."""

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

import numpy as np

STUDY_ID = "compositional_d8_cpu_confirmation_v1"
PLAN_SHA256 = "91558079d61c7006ced18d1885d074fd2de31ac221bfdcd2c96c1591137c0d55"
SOURCE_COMMIT = "artifact-source-revision"
EXPECTED_CELLS = 600
EXPECTED_SHARDS = 6
CONTEXTS = (20, 50, 100, 200, 400)
SEEDS = tuple(range(701, 711))
OOD_LENGTHS = tuple(range(4, 11))
DECODERS = ("logistic", "catboost")
ENCODERS = (
    "path_length_only",
    "coarse_relation_multiset",
    "relation_transition_counts",
    "ordered_typed_path",
    "ordered_plus_coarse_multiset",
    "aligned_coarse_multiset_with_ordered_block_permutation",
)
CONTRASTS = (
    (
        "shared_lossy_minus_absolute",
        "coarse_relation_multiset",
        "ordered_typed_path",
    ),
    (
        "block_permuted_minus_aligned_combined",
        "aligned_coarse_multiset_with_ordered_block_permutation",
        "ordered_plus_coarse_multiset",
    ),
)
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_DRAWS = 20_000


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
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    estimates = array[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _load_and_validate(root: Path) -> list[dict[str, Any]]:
    manifests = sorted(root.glob("shard-*/shard_manifest.json"))
    metrics = sorted(root.glob("shard-*/*.metrics.json"))
    if len(manifests) != EXPECTED_SHARDS or len(metrics) != EXPECTED_CELLS:
        raise ValueError("D8 CPU output population is incomplete")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("study_id") != STUDY_ID
            or manifest.get("plan_sha256") != PLAN_SHA256
            or len(manifest.get("cell_ids", [])) != 100
        ):
            raise ValueError(f"invalid shard manifest: {manifest_path}")
        provenance = json.loads(
            (manifest_path.parent / "runtime_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        if provenance["runtime"]["git"]["commit"] != SOURCE_COMMIT:
            raise ValueError(f"execution source drift: {manifest_path.parent}")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in metrics]
    if len({payload["cell"]["cell_id"] for payload in payloads}) != EXPECTED_CELLS:
        raise ValueError("D8 CPU output contains duplicate cell IDs")
    observed = set()
    for payload in payloads:
        cell = payload["cell"]
        key = (
            str(cell["encoder"]),
            str(cell["decoder"]),
            int(cell["context_per_class"]),
            int(cell["seed"]),
        )
        observed.add(key)
        if (
            payload.get("status") != "complete"
            or payload.get("audit", {}).get("status") != "PASS"
            or cell.get("study_id") != STUDY_ID
            or cell.get("algebra") != "dihedral_d8"
            or payload.get("prediction_policy") != "decoder"
        ):
            raise ValueError(f"invalid D8 cell: {cell.get('cell_id')}")
    expected = {
        (encoder, decoder, context, seed)
        for encoder in ENCODERS
        for decoder in DECODERS
        for context in CONTEXTS
        for seed in SEEDS
    }
    if observed != expected:
        raise ValueError("D8 CPU factorial population changed")
    return payloads


def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str, int, int], float] = {}
    length_values: dict[tuple[str, str, int, int, int], float] = {}
    ceilings: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        cell = payload["cell"]
        encoder = str(cell["encoder"])
        decoder = str(cell["decoder"])
        context = int(cell["context_per_class"])
        seed = int(cell["seed"])
        values[(encoder, decoder, context, seed)] = mean(
            float(payload["metrics_by_chain_length"][str(length)]["accuracy"])
            for length in OOD_LENGTHS
        )
        for length in OOD_LENGTHS:
            length_values[(encoder, decoder, context, seed, length)] = float(
                payload["metrics_by_chain_length"][str(length)]["accuracy"]
            )
        ceilings[(encoder, seed)].append(payload["collision_conditioned_ceiling"])

    aggregate = []
    for encoder in ENCODERS:
        for decoder in DECODERS:
            for context in CONTEXTS:
                cell_values = [
                    values[(encoder, decoder, context, seed)] for seed in SEEDS
                ]
                lower, upper = _bootstrap_mean_interval(cell_values)
                aggregate.append(
                    {
                        "encoder": encoder,
                        "decoder": decoder,
                        "context_per_class": context,
                        "seed_mean_ood_accuracy": mean(cell_values),
                        "seed_min_ood_accuracy": min(cell_values),
                        "seed_max_ood_accuracy": max(cell_values),
                        "fixed_panel_bootstrap_lower": lower,
                        "fixed_panel_bootstrap_upper": upper,
                    }
                )

    primary_seed_rows = []
    primary_summary = []
    primary_length_rows = []
    for contrast_name, positive_encoder, negative_encoder in CONTRASTS:
        for decoder in DECODERS:
            differences = []
            for seed in SEEDS:
                difference = (
                    values[(positive_encoder, decoder, 400, seed)]
                    - values[(negative_encoder, decoder, 400, seed)]
                )
                differences.append(difference)
                primary_seed_rows.append(
                    {
                        "contrast": contrast_name,
                        "decoder": decoder,
                        "seed": seed,
                        "positive_encoder_accuracy": values[
                            (positive_encoder, decoder, 400, seed)
                        ],
                        "negative_encoder_accuracy": values[
                            (negative_encoder, decoder, 400, seed)
                        ],
                        "difference": difference,
                    }
                )
            lower, upper = _bootstrap_mean_interval(differences)
            length_differences = []
            for length in OOD_LENGTHS:
                per_seed = [
                    length_values[(positive_encoder, decoder, 400, seed, length)]
                    - length_values[(negative_encoder, decoder, 400, seed, length)]
                    for seed in SEEDS
                ]
                length_difference = mean(per_seed)
                length_differences.append(length_difference)
                primary_length_rows.append(
                    {
                        "contrast": contrast_name,
                        "decoder": decoder,
                        "chain_length": length,
                        "seed_mean_difference": length_difference,
                        "positive_seeds": sum(value > 0.0 for value in per_seed),
                        "zero_seeds": sum(value == 0.0 for value in per_seed),
                        "negative_seeds": sum(value < 0.0 for value in per_seed),
                    }
                )
            primary_summary.append(
                {
                    "contrast": contrast_name,
                    "decoder": decoder,
                    "context_per_class": 400,
                    "seed_mean_difference": mean(differences),
                    "fixed_panel_bootstrap_lower": lower,
                    "fixed_panel_bootstrap_upper": upper,
                    "positive_seeds": sum(value > 0.0 for value in differences),
                    "positive_ood_lengths": sum(
                        value > 0.0 for value in length_differences
                    ),
                }
            )

    ceiling_summary = []
    for encoder in ENCODERS:
        by_seed = []
        for seed in SEEDS:
            repeated = ceilings[(encoder, seed)]
            canonical = repeated[0]
            if any(item != canonical for item in repeated[1:]):
                raise ValueError(
                    f"representation ceiling drift across cells: {encoder}, {seed}"
                )
            by_seed.append(canonical)
        ceiling_summary.append(
            {
                "encoder": encoder,
                "seed_mean_ceiling": mean(
                    float(item["ceiling"]) for item in by_seed
                ),
                "seed_mean_singleton_fraction": mean(
                    float(item["singleton_fraction"]) for item in by_seed
                ),
                "seed_mean_collision_coverage": mean(
                    float(item["collision_coverage"]) for item in by_seed
                ),
            }
        )

    cpu_replication_supported = all(
        row["seed_mean_difference"] > 0.0
        and row["positive_ood_lengths"] >= 5
        for row in primary_summary
    )
    return {
        "aggregate": aggregate,
        "primary_seed_rows": primary_seed_rows,
        "primary_summary": primary_summary,
        "primary_length_rows": primary_length_rows,
        "ceiling_summary": ceiling_summary,
        "cpu_replication_supported": cpu_replication_supported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = _load_and_validate(args.results_root)
    tables = summarize(payloads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "aggregate.csv", tables["aggregate"])
    _write_csv(
        args.output_dir / "primary_seed_contrasts.csv",
        tables["primary_seed_rows"],
    )
    _write_csv(
        args.output_dir / "primary_length_contrasts.csv",
        tables["primary_length_rows"],
    )
    _write_csv(
        args.output_dir / "ceiling_summary.csv",
        tables["ceiling_summary"],
    )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_id": STUDY_ID,
        "source_commit": SOURCE_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "cells": EXPECTED_CELLS,
        "shards": EXPECTED_SHARDS,
        "failures": 0,
        "contexts_per_class": list(CONTEXTS),
        "seeds": list(SEEDS),
        "ood_lengths": list(OOD_LENGTHS),
        "primary_context_per_class": 400,
        "primary_contrasts": tables["primary_summary"],
        "cpu_replication_supported": tables["cpu_replication_supported"],
        "final_replication_status": (
            "awaiting_registered_tabpfn_cells"
        ),
        "bootstrap": {
            "unit": "generated_population_seed",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "scope": "fixed D8 panel sensitivity, not graph-population inference",
        },
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(
        f"PASS cells={EXPECTED_CELLS} shards={EXPECTED_SHARDS} "
        f"cpu_replication_supported={tables['cpu_replication_supported']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
