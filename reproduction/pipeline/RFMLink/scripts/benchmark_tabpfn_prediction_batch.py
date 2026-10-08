#!/usr/bin/env python3
"""Select a TabPFN prediction chunk size using throughput, never test quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ultra4g_ood_reassessment as suite  # noqa: E402


DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "results" / "analysis"
    / "tabpfn_prediction_batch_benchmark.json"
)
BASELINE_BATCH = 50_000
CANDIDATE_BATCH = 500_000
TRIAL_ORDER = (BASELINE_BATCH, CANDIDATE_BATCH, CANDIDATE_BATCH, BASELINE_BATCH)
MIN_SPEEDUP = 1.03
MAX_ABSOLUTE_DIFFERENCE = 1e-5
MAX_AGGREGATE_RESERVED_FRACTION = 0.75
GPU_WORKERS = 4


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hetionet")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import tabpfn
    import torch
    from tabpfn import TabPFNClassifier

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prediction-batch benchmark")
    if suite.file_sha256(suite.TABPFN_CHECKPOINT) != suite.EXPECTED_TABPFN_SHA256:
        raise RuntimeError("TabPFN checkpoint hash mismatch")

    context = suite.context_payload(
        args.dataset, args.seed, args.budget, "correct_labels"
    )
    valid = suite.compact_asset(
        suite.engine.load_split_asset(args.dataset, "valid"), queries=args.queries
    )
    train_X = suite.select_columns(context["X"], "full").astype(
        np.float32, copy=False
    )
    eval_X = suite.select_columns(valid["X"], "full").astype(
        np.float32, copy=False
    )
    _, categorical = suite.representation_indices("full")
    max_cardinality = max(
        int(np.unique(train_X[:, index]).size) for index in categorical
    )
    model = TabPFNClassifier(
        model_path=suite.TABPFN_CHECKPOINT,
        device="cuda",
        n_estimators=suite.N_ESTIMATORS,
        random_state=args.seed,
        fit_mode="fit_with_cache",
        categorical_features_indices=categorical,
        inference_config=suite.build_tabpfn_inference_config(
            max(max_cardinality, 2)
        ),
    )
    fit_started = time.perf_counter()
    model.fit(train_X, context["y"])
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_started

    # Exclude one-time CUDA initialization from both measured conditions.
    suite.positive_class_probabilities(model, eval_X[:10_000])
    torch.cuda.synchronize()

    trials = []
    predictions_by_batch: dict[int, np.ndarray] = {}
    ranks_by_batch: dict[int, np.ndarray] = {}
    batch_errors: dict[int, str] = {}
    for batch_rows in TRIAL_ORDER:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            chunks = [
                suite.positive_class_probabilities(
                    model, eval_X[start:start + batch_rows]
                )
                for start in range(0, len(eval_X), batch_rows)
            ]
            prediction = np.concatenate(chunks).astype(np.float32, copy=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            metrics, frame = suite.engine.metrics_from_scores(valid, prediction)
            trials.append({
                "batch_rows": batch_rows,
                "seconds": seconds,
                "rows_per_second": len(eval_X) / seconds,
                "chunks": len(chunks),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "prediction_sha256": array_sha256(prediction),
                "metrics": json_ready(metrics),
            })
            predictions_by_batch.setdefault(batch_rows, prediction)
            ranks_by_batch.setdefault(
                batch_rows, frame["rank"].to_numpy(dtype=np.int64, copy=True)
            )
        except torch.cuda.OutOfMemoryError as error:
            batch_errors[batch_rows] = f"{type(error).__name__}: {error}"
            torch.cuda.empty_cache()
            if batch_rows == BASELINE_BATCH:
                raise

    baseline_trials = [
        trial for trial in trials if trial["batch_rows"] == BASELINE_BATCH
    ]
    candidate_trials = [
        trial for trial in trials if trial["batch_rows"] == CANDIDATE_BATCH
    ]
    candidate_completed = len(candidate_trials) == 2
    if candidate_completed:
        baseline_rate = float(np.median([
            trial["rows_per_second"] for trial in baseline_trials
        ]))
        candidate_rate = float(np.median([
            trial["rows_per_second"] for trial in candidate_trials
        ]))
        speedup = candidate_rate / baseline_rate
        difference = np.abs(
            predictions_by_batch[BASELINE_BATCH]
            - predictions_by_batch[CANDIDATE_BATCH]
        )
        max_difference = float(difference.max(initial=0.0))
        mean_difference = float(difference.mean())
        rank_equal = bool(np.array_equal(
            ranks_by_batch[BASELINE_BATCH], ranks_by_batch[CANDIDATE_BATCH]
        ))
        candidate_peak_reserved = max(
            trial["peak_reserved_bytes"] for trial in candidate_trials
        )
        aggregate_reserved_fraction = (
            GPU_WORKERS * candidate_peak_reserved
            / torch.cuda.get_device_properties(0).total_memory
        )
    else:
        baseline_rate = float(np.median([
            trial["rows_per_second"] for trial in baseline_trials
        ]))
        candidate_rate = None
        speedup = None
        max_difference = None
        mean_difference = None
        rank_equal = False
        aggregate_reserved_fraction = None

    accepted = bool(
        candidate_completed
        and speedup >= MIN_SPEEDUP
        and max_difference <= MAX_ABSOLUTE_DIFFERENCE
        and rank_equal
        and aggregate_reserved_fraction <= MAX_AGGREGATE_RESERVED_FRACTION
    )
    payload = {
        "status": "PASS_500K" if accepted else "KEEP_50K",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_basis": "runtime_and_prediction_equivalence_only",
        "recommended_batch_rows": CANDIDATE_BATCH if accepted else BASELINE_BATCH,
        "dataset": args.dataset,
        "seed": args.seed,
        "context_rows": int(len(train_X)),
        "positive_context_rows": int(np.sum(context["y"] == 1)),
        "negative_context_rows": int(np.sum(context["y"] == 0)),
        "evaluation_queries": int(len(valid["filtered"])),
        "evaluation_rows": int(len(eval_X)),
        "num_features": int(eval_X.shape[1]),
        "n_estimators": suite.N_ESTIMATORS,
        "fit_seconds": fit_seconds,
        "trial_order": list(TRIAL_ORDER),
        "trials": trials,
        "errors": batch_errors,
        "comparison": {
            "baseline_rows_per_second": baseline_rate,
            "candidate_rows_per_second": candidate_rate,
            "candidate_speedup": speedup,
            "max_absolute_prediction_difference": max_difference,
            "mean_absolute_prediction_difference": mean_difference,
            "all_query_ranks_equal": rank_equal,
            "four_worker_reserved_memory_fraction": aggregate_reserved_fraction,
        },
        "preregistered_acceptance_rule": {
            "minimum_speedup": MIN_SPEEDUP,
            "maximum_absolute_prediction_difference": MAX_ABSOLUTE_DIFFERENCE,
            "all_query_ranks_equal": True,
            "maximum_four_worker_reserved_memory_fraction": (
                MAX_AGGREGATE_RESERVED_FRACTION
            ),
            "quality_metrics_used_for_batch_selection": False,
        },
        "software": {
            "python": platform.python_version(),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "checkpoint": str(suite.TABPFN_CHECKPOINT),
            "checkpoint_sha256": suite.EXPECTED_TABPFN_SHA256,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "gpu_workers_assumed": GPU_WORKERS,
            "hostname": platform.node(),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "recommended_batch_rows": payload["recommended_batch_rows"],
        "candidate_speedup": speedup,
        "all_query_ranks_equal": rank_equal,
    }, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
