#!/usr/bin/env python3
"""Validate and summarize the registered D8 TabPFN confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

STUDY_ID = "compositional_d8_gpu_tabpfn_confirmation_v1"
PLAN_SHA256 = "cec127de036bca92f4f710e53b429d72431c1c1c83f3509edfb436d9547085d1"
SOURCE_COMMIT = "artifact-source-revision"
EXPECTED_CELLS = 300
EXPECTED_SHARDS = 19
CONTEXTS = (20, 50, 100, 200, 400)
SEEDS = tuple(range(701, 711))
OOD_LENGTHS = tuple(range(4, 11))
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
BOOTSTRAP_SEED = 20_260_730
BOOTSTRAP_DRAWS = 20_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("D8 TabPFN output population is incomplete")

    manifest_cell_ids: list[str] = []
    registered_metrics: dict[Path, str] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("study_id") != STUDY_ID
            or manifest.get("plan_sha256") != PLAN_SHA256
        ):
            raise ValueError(f"invalid shard manifest: {manifest_path}")
        manifest_cell_ids.extend(manifest.get("cell_ids", []))
        for item in manifest.get("files", []):
            if item["path"].endswith(".metrics.json"):
                registered_metrics[manifest_path.parent / item["path"]] = item[
                    "sha256"
                ]
        provenance = json.loads(
            (manifest_path.parent / "runtime_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        if provenance["runtime"]["git"]["commit"] != SOURCE_COMMIT:
            raise ValueError(f"execution source drift: {manifest_path.parent}")

    if (
        len(manifest_cell_ids) != EXPECTED_CELLS
        or len(set(manifest_cell_ids)) != EXPECTED_CELLS
        or set(registered_metrics) != set(metrics)
    ):
        raise ValueError("shard manifests do not cover the metric population")
    for path, expected in registered_metrics.items():
        if file_sha256(path) != expected:
            raise ValueError(f"metric checksum mismatch: {path}")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in metrics]
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
            or cell.get("decoder") != "tabpfn"
            or payload.get("prediction_policy") != "decoder"
        ):
            raise ValueError(f"invalid D8 TabPFN cell: {cell.get('cell_id')}")
    expected = {
        (encoder, "tabpfn", context, seed)
        for encoder in ENCODERS
        for context in CONTEXTS
        for seed in SEEDS
    }
    if observed != expected:
        raise ValueError("D8 TabPFN factorial population changed")
    return payloads


def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, int, int], float] = {}
    length_values: dict[tuple[str, int, int, int], float] = {}
    ceilings: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        cell = payload["cell"]
        encoder = str(cell["encoder"])
        context = int(cell["context_per_class"])
        seed = int(cell["seed"])
        values[(encoder, context, seed)] = mean(
            float(payload["metrics_by_chain_length"][str(length)]["accuracy"])
            for length in OOD_LENGTHS
        )
        for length in OOD_LENGTHS:
            length_values[(encoder, context, seed, length)] = float(
                payload["metrics_by_chain_length"][str(length)]["accuracy"]
            )
        ceilings[(encoder, seed)].append(payload["collision_conditioned_ceiling"])

    aggregate = []
    for encoder in ENCODERS:
        for context in CONTEXTS:
            cell_values = [
                values[(encoder, context, seed)] for seed in SEEDS
            ]
            lower, upper = _bootstrap_mean_interval(cell_values)
            aggregate.append(
                {
                    "encoder": encoder,
                    "decoder": "tabpfn",
                    "context_per_class": context,
                    "seed_mean_ood_accuracy": mean(cell_values),
                    "seed_min_ood_accuracy": min(cell_values),
                    "seed_max_ood_accuracy": max(cell_values),
                    "fixed_panel_bootstrap_lower": lower,
                    "fixed_panel_bootstrap_upper": upper,
                }
            )

    primary_seed_rows = []
    primary_length_rows = []
    primary_summary = []
    for contrast_name, positive_encoder, negative_encoder in CONTRASTS:
        differences = []
        for seed in SEEDS:
            difference = (
                values[(positive_encoder, 400, seed)]
                - values[(negative_encoder, 400, seed)]
            )
            differences.append(difference)
            primary_seed_rows.append(
                {
                    "contrast": contrast_name,
                    "decoder": "tabpfn",
                    "seed": seed,
                    "positive_encoder_accuracy": values[
                        (positive_encoder, 400, seed)
                    ],
                    "negative_encoder_accuracy": values[
                        (negative_encoder, 400, seed)
                    ],
                    "difference": difference,
                }
            )
        lower, upper = _bootstrap_mean_interval(differences)
        length_differences = []
        for length in OOD_LENGTHS:
            per_seed = [
                length_values[(positive_encoder, 400, seed, length)]
                - length_values[(negative_encoder, 400, seed, length)]
                for seed in SEEDS
            ]
            length_difference = mean(per_seed)
            length_differences.append(length_difference)
            primary_length_rows.append(
                {
                    "contrast": contrast_name,
                    "decoder": "tabpfn",
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
                "decoder": "tabpfn",
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
                    f"representation ceiling drift: encoder={encoder}, seed={seed}"
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
    return {
        "aggregate": aggregate,
        "primary_seed_rows": primary_seed_rows,
        "primary_length_rows": primary_length_rows,
        "primary_summary": primary_summary,
        "ceiling_summary": ceiling_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    tables = summarize(_load_and_validate(args.results_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "aggregate.csv": tables["aggregate"],
        "primary_seed_contrasts.csv": tables["primary_seed_rows"],
        "primary_length_contrasts.csv": tables["primary_length_rows"],
        "ceiling_summary.csv": tables["ceiling_summary"],
    }
    for name, rows in outputs.items():
        _write_csv(args.output_dir / name, rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_id": STUDY_ID,
        "source_commit": SOURCE_COMMIT,
        "analysis_commit": args.analysis_commit,
        "plan_sha256": PLAN_SHA256,
        "cells": EXPECTED_CELLS,
        "shards": EXPECTED_SHARDS,
        "failures": 0,
        "contexts_per_class": list(CONTEXTS),
        "seeds": list(SEEDS),
        "ood_lengths": list(OOD_LENGTHS),
        "primary_context_per_class": 400,
        "primary_contrasts": tables["primary_summary"],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "generated_population_seed",
            "scope": "fixed D8 panel sensitivity",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_names = list(outputs) + ["summary.json"]
    manifest = {
        "analysis": "compositional_d8_gpu_tabpfn_confirmation_v1",
        "analysis_commit": args.analysis_commit,
        "execution_commit": SOURCE_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "status": "PASS",
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": file_sha256(args.output_dir / name),
            }
            for name in output_names
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
