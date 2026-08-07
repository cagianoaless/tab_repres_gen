#!/usr/bin/env python3
"""Audit whether the primary CLUTRR encoder ordering is metric-robust."""

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

CPU_STUDY = "clutrr_symbolic_cpu_confirmation_v2"
GPU_STUDY = "clutrr_symbolic_gpu_tabpfn_confirmation_v2"
CPU_PLAN = "cca4114229dbfac4938d707252c4257817149817becabb4b8e8b43a7bf86df98"
GPU_PLAN = "d118695450af93907a347aab5760dfb6bb3067ed30ffe3255771722bfd348a4b"
CPU_RESULT_COMMIT = "artifact-source-revision"
GPU_RESULT_COMMIT = "artifact-source-revision"
EXPECTED_CPU_CELLS = 150
EXPECTED_GPU_CELLS = 75
PRIMARY_CONTEXT = 80
OOD_LENGTHS = tuple(range(4, 11))
PRIMARY_ENCODERS = ("ordered_typed_path", "relation_multiset")
PRIMARY_DECODERS = ("logistic", "catboost", "tabpfn")
HIGHER_IS_BETTER = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "top3_accuracy",
)
LOWER_IS_BETTER = ("log_loss", "multiclass_brier")
METRICS = (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_paths(root: Path) -> list[Path]:
    return sorted(root.glob("shard-*/*.metrics.json"))


def _load_population(
    root: Path,
    *,
    study_id: str,
    expected_cells: int,
    expected_decoders: set[str],
) -> list[dict[str, Any]]:
    paths = _metric_paths(root)
    if len(paths) != expected_cells:
        raise ValueError(
            f"{study_id}: expected {expected_cells} metric files, found {len(paths)}"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({payload["cell"]["cell_id"] for payload in payloads}) != expected_cells:
        raise ValueError(f"{study_id}: duplicate cell IDs")
    for path, payload in zip(paths, payloads, strict=True):
        cell = payload["cell"]
        if payload.get("status") != "complete" or cell.get("study_id") != study_id:
            raise ValueError(f"{study_id}: invalid result cell {cell.get('cell_id')}")
        if cell.get("decoder") not in expected_decoders:
            raise ValueError(f"{study_id}: unexpected decoder {cell.get('decoder')}")
        prediction_path = path.with_name(payload["predictions"]["path"])
        if not prediction_path.is_file():
            raise ValueError(f"{study_id}: missing prediction file {prediction_path}")
        if _sha256(prediction_path) != payload["predictions"]["sha256"]:
            raise ValueError(f"{study_id}: prediction checksum mismatch")
        if set(payload["metrics_by_chain_length"]) != {
            str(length) for length in range(2, 11)
        }:
            raise ValueError(f"{study_id}: incomplete chain-length metrics")
    return payloads


def _mean_metric(payload: dict[str, Any], metric: str) -> float:
    return mean(
        float(payload["metrics_by_chain_length"][str(length)][metric])
        for length in OOD_LENGTHS
    )


def _primary_payloads(
    cpu_payloads: list[dict[str, Any]],
    gpu_payloads: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for payload in (*cpu_payloads, *gpu_payloads):
        cell = payload["cell"]
        if (
            int(cell["context_per_relation"]) != PRIMARY_CONTEXT
            or cell["encoder"] not in PRIMARY_ENCODERS
            or cell["decoder"] not in PRIMARY_DECODERS
        ):
            continue
        key = (str(cell["decoder"]), str(cell["encoder"]), int(cell["seed"]))
        if key in selected:
            raise ValueError(f"duplicate primary CLUTRR cell: {key}")
        selected[key] = payload
    expected = {
        (decoder, encoder, seed)
        for decoder in PRIMARY_DECODERS
        for encoder in PRIMARY_ENCODERS
        for seed in (101, 102, 103)
    }
    if set(selected) != expected:
        raise ValueError("primary CLUTRR metric population is incomplete")
    return selected


def _metric_rows(
    selected: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for decoder in PRIMARY_DECODERS:
        for metric in METRICS:
            values: dict[str, float] = {}
            for encoder in PRIMARY_ENCODERS:
                values[encoder] = mean(
                    _mean_metric(selected[(decoder, encoder, seed)], metric)
                    for seed in (101, 102, 103)
                )
            if metric in HIGHER_IS_BETTER:
                advantage = (
                    values["relation_multiset"] - values["ordered_typed_path"]
                )
            else:
                advantage = (
                    values["ordered_typed_path"] - values["relation_multiset"]
                )
            summary_rows.append(
                {
                    "decoder": decoder,
                    "metric": metric,
                    "ordered_typed_path": values["ordered_typed_path"],
                    "relation_multiset": values["relation_multiset"],
                    "multiset_advantage": advantage,
                    "preferred_direction": (
                        "higher" if metric in HIGHER_IS_BETTER else "lower"
                    ),
                    "multiset_better": advantage > 0.0,
                }
            )

        for length in OOD_LENGTHS:
            row: dict[str, Any] = {"decoder": decoder, "chain_length": length}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                ordered = mean(
                    float(
                        selected[(decoder, "ordered_typed_path", seed)][
                            "metrics_by_chain_length"
                        ][str(length)][metric]
                    )
                    for seed in (101, 102, 103)
                )
                multiset = mean(
                    float(
                        selected[(decoder, "relation_multiset", seed)][
                            "metrics_by_chain_length"
                        ][str(length)][metric]
                    )
                    for seed in (101, 102, 103)
                )
                row[f"ordered_{metric}"] = ordered
                row[f"multiset_{metric}"] = multiset
                row[f"multiset_minus_ordered_{metric}"] = multiset - ordered
            length_rows.append(row)
    return summary_rows, length_rows


def _target_rows(
    selected: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for (decoder, encoder, _seed), payload in selected.items():
        metric_path = next(
            path
            for path in _metric_paths(Path(payload["_result_root"]))
            if path.stem.removesuffix(".metrics") == payload["cell"]["cell_id"]
        )
        prediction_path = metric_path.with_name(payload["predictions"]["path"])
        arrays = np.load(prediction_path, allow_pickle=False)
        labels = arrays["labels"]
        classes = arrays["classes"]
        predictions = classes[np.argmax(arrays["probabilities"], axis=1)]
        mask = np.isin(arrays["chain_lengths"], OOD_LENGTHS)
        class_names = tuple(payload["class_names"])
        for class_index, class_name in enumerate(class_names):
            current = mask & (labels == class_index)
            count = int(current.sum())
            if count == 0:
                continue
            accuracy = float(np.mean(predictions[current] == labels[current]))
            values[(decoder, encoder, class_name)].append((accuracy, count))

    rows: list[dict[str, Any]] = []
    for decoder in PRIMARY_DECODERS:
        names = sorted(
            {
                class_name
                for current_decoder, _encoder, class_name in values
                if current_decoder == decoder
            }
        )
        for class_name in names:
            ordered = values[(decoder, "ordered_typed_path", class_name)]
            multiset = values[(decoder, "relation_multiset", class_name)]
            if len(ordered) != 3 or len(multiset) != 3:
                raise ValueError("target-relation seed population is incomplete")
            rows.append(
                {
                    "decoder": decoder,
                    "target_relation": class_name,
                    "evaluation_rows_per_seed": ordered[0][1],
                    "ordered_accuracy": mean(value for value, _count in ordered),
                    "multiset_accuracy": mean(value for value, _count in multiset),
                    "multiset_minus_ordered": (
                        mean(value for value, _count in multiset)
                        - mean(value for value, _count in ordered)
                    ),
                }
            )
    return rows


def audit(cpu_root: Path, gpu_root: Path, output_dir: Path) -> dict[str, Any]:
    cpu_payloads = _load_population(
        cpu_root,
        study_id=CPU_STUDY,
        expected_cells=EXPECTED_CPU_CELLS,
        expected_decoders={"logistic", "catboost"},
    )
    gpu_payloads = _load_population(
        gpu_root,
        study_id=GPU_STUDY,
        expected_cells=EXPECTED_GPU_CELLS,
        expected_decoders={"tabpfn"},
    )
    for payload in cpu_payloads:
        payload["_result_root"] = str(cpu_root)
    for payload in gpu_payloads:
        payload["_result_root"] = str(gpu_root)

    selected = _primary_payloads(cpu_payloads, gpu_payloads)
    metric_rows, length_rows = _metric_rows(selected)
    target_rows = _target_rows(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "primary_metric_robustness.csv", metric_rows)
    _write_csv(output_dir / "primary_per_length.csv", length_rows)
    _write_csv(output_dir / "primary_per_target_relation.csv", target_rows)

    per_length_metrics = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        deltas = [
            float(row[f"multiset_minus_ordered_{metric}"]) for row in length_rows
        ]
        per_length_metrics[metric] = {
            "comparisons": len(deltas),
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
            "minimum": min(deltas),
            "maximum": max(deltas),
        }
    target_counts = {}
    for decoder in PRIMARY_DECODERS:
        deltas = [
            float(row["multiset_minus_ordered"])
            for row in target_rows
            if row["decoder"] == decoder
        ]
        target_counts[decoder] = {
            "positive": sum(delta > 0 for delta in deltas),
            "zero": sum(delta == 0 for delta in deltas),
            "negative": sum(delta < 0 for delta in deltas),
        }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "estimand": (
            "equal-length mean over fixed CLUTRR OOD lengths 4--10 at 80 "
            "context examples per target relation, then mean over seeds 101--103"
        ),
        "cpu_study": CPU_STUDY,
        "cpu_plan_sha256": CPU_PLAN,
        "cpu_result_commit": CPU_RESULT_COMMIT,
        "cpu_cells": len(cpu_payloads),
        "gpu_study": GPU_STUDY,
        "gpu_plan_sha256": GPU_PLAN,
        "gpu_result_commit": GPU_RESULT_COMMIT,
        "gpu_cells": len(gpu_payloads),
        "primary_cells": len(selected),
        "all_six_metric_decoder_comparisons_favor_multiset": all(
            bool(row["multiset_better"]) for row in metric_rows
        ),
        "per_length_metrics": per_length_metrics,
        "per_target_relation_accuracy": target_counts,
        "scope": (
            "fixed official CLUTRR test population and three context seeds; "
            "not uncertainty over datasets or relation ontologies"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-results-root", type=Path, required=True)
    parser.add_argument("--gpu-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.cpu_results_root, args.gpu_results_root, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
