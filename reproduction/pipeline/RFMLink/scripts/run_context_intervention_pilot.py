#!/usr/bin/env python3
"""Paired UMLS intervention: fixed context rows with correct vs permuted labels.

This is deliberately a diagnostic runner, not a new RFMLink architecture. It
reuses the frozen primary feature, ULTRA, TabPFN, and evaluator paths and stops
after applying the pre-registered pilot rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path.insert(0, str(RFMLINK_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_primary_experiment import (  # noqa: E402
    CATEGORICAL,
    INVARIANT_COLUMNS,
    PrimaryAssets,
    side_metrics,
)
from rfmlink.contexts import load_context  # noqa: E402
from rfmlink.decoders import fit_predict_tabpfn3  # noqa: E402
from rfmlink.paths import MODELS_CKPT  # noqa: E402
from rfmlink.sbm_import import tied_rank_positions  # noqa: E402


DATASET = "umls"
SEEDS = (101, 102, 103, 104, 105)
SIZE = "200pos_200neg"
CONDITIONS = (
    "correct_labels",
    "permuted_labels",
    "random_rows_correct_balance",
)
CHECKPOINT = MODELS_CKPT / "tabpfn-v3-classifier-v3_default.ckpt"
RESULTS_ROOT = WORKSPACE_ROOT / "results"
SUITE_ROOT = RESULTS_ROOT / "context_intervention_suite"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
RULE_PATH = ANALYSIS_ROOT / "context_intervention_decision_rule.json"
PLAN_PATH = ANALYSIS_ROOT / "context_intervention_overnight_plan.json"
SUMMARY_JSON = ANALYSIS_ROOT / "context_intervention_overnight_summary.json"
SUMMARY_MD = ANALYSIS_ROOT / "context_intervention_overnight_summary.md"
FAILURE_MD = ANALYSIS_ROOT / "context_intervention_failure_report.md"
PRIMARY_RESULTS = RFMLINK_ROOT / "results" / "primary_runs"
PRIMARY_REPRODUCTION_TOLERANCE = 0.002
REQUIRED_RUN_FILES = (
    "config.json",
    "metrics.json",
    "timings.json",
    "predictions.parquet",
    "context_manifest.json",
    "run.log",
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return array_hash(values)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", f"--git-dir={WORKSPACE_ROOT / '.git'}", f"--work-tree={WORKSPACE_ROOT}",
             "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware_info() -> dict:
    import torch

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def validate_preregistration() -> tuple[dict, dict]:
    if not RULE_PATH.exists() or not PLAN_PATH.exists():
        raise FileNotFoundError("pre-registered decision rule or suite plan is missing")
    rule = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["design_review"]["decision"] != "GO_WITH_CORRECTIONS":
        raise RuntimeError("pilot is not authorized by the Stage 0 design review")
    if rule["seeds"] != list(SEEDS):
        raise RuntimeError("decision-rule seeds do not match the runner")
    return rule, plan


def run_complete(run_dir: Path) -> bool:
    if not all((run_dir / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    try:
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
        json.loads((run_dir / "context_manifest.json").read_text(encoding="utf-8"))
        predictions = pd.read_parquet(run_dir / "predictions.parquet", columns=["split", "score"])
        return len(predictions) > 0 and np.isfinite(predictions["score"]).all()
    except Exception:
        return False


def stratified_label_permutation(
    labels: np.ndarray,
    relations: np.ndarray,
    orientations: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute within relation and orientation, preserving all stratum marginals."""
    labels = np.asarray(labels, dtype=np.int64)
    permutation = np.arange(len(labels), dtype=np.int64)
    strata = np.asarray([f"{r}:{o}" for r, o in zip(relations, orientations)])
    for attempt in range(100):
        rng = np.random.default_rng(seed + 910_001 + attempt)
        candidate = np.arange(len(labels), dtype=np.int64)
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            candidate[indices] = indices[rng.permutation(len(indices))]
        changed = int(np.sum(labels[candidate] != labels))
        if changed >= len(labels) // 4:
            permutation = candidate
            break
    else:
        raise RuntimeError("could not construct a nontrivial stratified label permutation")
    permuted = labels[permutation]
    if int(permuted.sum()) != int(labels.sum()):
        raise AssertionError("label permutation changed global class balance")
    for stratum in np.unique(strata):
        mask = strata == stratum
        if int(permuted[mask].sum()) != int(labels[mask].sum()):
            raise AssertionError("label permutation changed a relation/orientation marginal")
    return permuted, permutation


def _sample_relation_negatives(
    train_triples: np.ndarray,
    relation: int,
    count: int,
    num_entities: int,
    forbidden: set[tuple[int, int, int]],
    excluded: set[tuple[int, int, int]],
    used: set[tuple[int, int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    sources = train_triples[train_triples[:, 1] == relation]
    rows = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > max(10_000, count * 2_000):
            raise RuntimeError(f"negative sampling failed for relation {relation}")
        h, r, t = sources[rng.integers(0, len(sources))]
        if rng.random() < 0.5:
            h = int(rng.integers(0, num_entities))
        else:
            t = int(rng.integers(0, num_entities))
        candidate = (int(h), int(r), int(t))
        if candidate in forbidden or candidate in excluded or candidate in used:
            continue
        used.add(candidate)
        rows.append(candidate)
    return rows


def relation_matched_random_rows(
    pa: PrimaryAssets,
    triples: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Replace rows in place while preserving relation, class, and orientation slots."""
    train = pa.train_frame[["h", "r", "t"]].to_numpy(dtype=np.int64)
    train_set = {tuple(row) for row in train.tolist()}
    source_set = {tuple(row) for row in triples.tolist()}
    rng = np.random.default_rng(seed + 720_003)
    replacement = np.empty_like(triples)
    used: set[tuple[int, int, int]] = set()
    positive_overlap = 0

    for relation in np.unique(triples[:, 1]):
        slots = np.flatnonzero((triples[:, 1] == relation) & (labels == 1))
        if not len(slots):
            continue
        pool = train[train[:, 1] == relation]
        preferred = np.asarray([row for row in pool.tolist() if tuple(row) not in source_set],
                               dtype=np.int64)
        if len(preferred) >= len(slots):
            chosen = preferred[rng.choice(len(preferred), len(slots), replace=False)]
        else:
            chosen = pool[rng.choice(len(pool), len(slots), replace=False)]
        for slot, row in zip(slots, chosen):
            value = tuple(int(x) for x in row)
            replacement[slot] = row
            used.add(value)
            positive_overlap += int(value in source_set)

    negative_source = {tuple(row) for row in triples[labels == 0].tolist()}
    for relation in np.unique(triples[:, 1]):
        slots = np.flatnonzero((triples[:, 1] == relation) & (labels == 0))
        if not len(slots):
            continue
        rows = _sample_relation_negatives(
            train, int(relation), len(slots), pa.num_entities, train_set,
            negative_source, used, rng)
        replacement[slots] = np.asarray(rows, dtype=np.int64)

    if not np.array_equal(replacement[:, 1], triples[:, 1]):
        raise AssertionError("random rows changed relation slots")
    if len({tuple(row) for row in replacement.tolist()}) != len(replacement):
        raise AssertionError("random context contains duplicate rows")
    return replacement, labels.copy(), {
        "sampling_seed": seed + 720_003,
        "matched_marginals": ["relation", "label", "orientation_slot"],
        "positive_overlap_with_correct_context": positive_overlap,
        "negative_overlap_with_correct_context": 0,
        "negative_filtering": "train_only",
        "uses_valid_or_test_truths": False,
    }


def context_rows_payload(
    source_triples: np.ndarray,
    source_indices: np.ndarray,
    orientations: np.ndarray,
    original_labels: np.ndarray,
    supplied_labels: np.ndarray,
) -> list[dict]:
    rows = []
    for out_index, (source_index, orientation, original, supplied) in enumerate(
        zip(source_indices, orientations, original_labels, supplied_labels)
    ):
        triple = source_triples[int(source_index)]
        rows.append({
            "output_index": out_index,
            "source_index": int(source_index),
            "source_triple": [int(x) for x in triple],
            "orientation": str(orientation),
            "original_label": int(original),
            "supplied_label": int(supplied),
        })
    return rows


def prepare_seed_contexts(pa: PrimaryAssets, seed: int) -> tuple[dict, dict]:
    t0 = time.time()
    triples, labels, source_config = load_context(DATASET, seed, SIZE)
    X, y, indices, orientations = pa.context_matrix_from_rows(
        triples, labels, seed, balanced=True, equal_budget=True, return_manifest=True)
    relations = triples[indices, 1]
    permuted, permutation = stratified_label_permutation(y, relations, orientations, seed)
    random_triples, random_labels, random_info = relation_matched_random_rows(
        pa, triples, labels, seed)
    random_X, random_y, random_indices, random_orientations = pa.context_matrix_from_rows(
        random_triples, random_labels, seed, balanced=True, equal_budget=True,
        return_manifest=True)

    if not np.array_equal(y, labels[indices]):
        raise AssertionError("balanced-equal labels are not aligned to source rows")
    if not np.array_equal(random_y, random_labels[random_indices]):
        raise AssertionError("random-row labels are not aligned")
    shared = {
        "correct_labels": {
            "X": X,
            "y": y,
            "triples": triples,
            "indices": indices,
            "orientations": orientations,
            "original_y": y,
            "permutation": np.arange(len(y), dtype=np.int64),
        },
        "permuted_labels": {
            "X": X,
            "y": permuted,
            "triples": triples,
            "indices": indices,
            "orientations": orientations,
            "original_y": y,
            "permutation": permutation,
        },
        "random_rows_correct_balance": {
            "X": random_X,
            "y": random_y,
            "triples": random_triples,
            "indices": random_indices,
            "orientations": random_orientations,
            "original_y": random_y,
            "permutation": np.arange(len(random_y), dtype=np.int64),
        },
    }
    if array_hash(shared["correct_labels"]["X"]) != array_hash(shared["permuted_labels"]["X"]):
        raise AssertionError("correct and permuted feature matrices differ")
    metadata = {
        "build_seconds": time.time() - t0,
        "source_context_config": source_config,
        "random_rows": random_info,
    }
    return shared, metadata


def candidate_payload(pa: PrimaryAssets) -> tuple[dict, dict]:
    tables = {split: pa.eval_tables[split] for split in ("valid", "test")}
    matrices = {split: pa.eval_X[split] for split in ("valid", "test")}
    metadata = {}
    for split in ("valid", "test"):
        table = tables[split]
        metadata[split] = {
            "rows": len(table),
            "queries": int(table["query_index"].nunique()),
            "candidate_hash": frame_hash(
                table,
                ["query_index", "side", "h", "r", "t", "is_target", "is_filtered"],
            ),
            "feature_hash": array_hash(matrices[split]),
            "ultra_score_hash": array_hash(matrices[split][:, -1]),
        }
    return {"tables": tables, "matrices": matrices}, metadata


def predictions_frame(pa: PrimaryAssets, scores_by_split: dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for split in ("valid", "test"):
        table = pa.eval_tables[split]
        frame = table[["query_index", "side", "h", "r", "t", "is_target", "is_filtered"]].copy()
        frame.insert(0, "split", split)
        frame["score"] = np.asarray(scores_by_split[split], dtype=np.float64)
        frame["ultra_score"] = pa.eval_X[split][:, -1].astype(np.float64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def primary_reference_mrr(seed: int) -> float:
    path = PRIMARY_RESULTS / f"umls__primary_balanced_equal__tabpfn_v3__s{seed}" / "metrics.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["test"]["combined"]["mrr"])


def run_condition(
    pa: PrimaryAssets,
    seed: int,
    condition: str,
    payload: dict,
    build_metadata: dict,
    candidate_metadata: dict,
    smoke: bool = False,
) -> dict:
    suffix = "_smoke" if smoke else ""
    run_dir = SUITE_ROOT / DATASET / f"{condition}{suffix}" / "tabpfn_v3" / str(seed)
    if not smoke and run_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.time()
    try:
        X = payload["X"]
        y = payload["y"]
        eval_Xs = [pa.eval_X["valid"], pa.eval_X["test"]]
        eval_tables = [pa.eval_tables["valid"], pa.eval_tables["test"]]
        n_estimators = 1 if smoke else 8
        if smoke:
            selected_Xs, selected_tables = [], []
            for matrix, table in zip(eval_Xs, eval_tables):
                qids = table["query_index"].drop_duplicates().iloc[:4]
                mask = table["query_index"].isin(qids).to_numpy()
                selected_Xs.append(matrix[mask])
                selected_tables.append(table.loc[mask].reset_index(drop=True))
            eval_Xs, eval_tables = selected_Xs, selected_tables

        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        decode_start = time.time()
        predictions, decoder_info = fit_predict_tabpfn3(
            X,
            y,
            eval_Xs,
            seed,
            categorical_indices=[INVARIANT_COLUMNS.index(c) for c in CATEGORICAL],
            fit_mode="fit_with_cache",
            checkpoint_override=str(CHECKPOINT),
            n_estimators=n_estimators,
        )
        decode_seconds = time.time() - decode_start
        peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        metrics = {}
        for split, table, scores in zip(("valid", "test"), eval_tables, predictions):
            metrics[split] = side_metrics(table, np.asarray(scores))

        if smoke:
            atomic_json(run_dir / "smoke_metrics.json", metrics)
            log_path.write_text(f"SMOKE OK {condition} seed={seed}\n", encoding="utf-8")
            return metrics

        scores_by_split = {split: np.asarray(scores) for split, scores in zip(("valid", "test"), predictions)}
        prediction_table = predictions_frame(pa, scores_by_split)
        rows = context_rows_payload(
            payload["triples"], payload["indices"], payload["orientations"],
            payload["original_y"], payload["y"])
        manifest = {
            "dataset": DATASET,
            "seed": seed,
            "condition": condition,
            "rows": rows,
            "num_rows": len(rows),
            "num_positive_labels": int(np.sum(payload["y"] == 1)),
            "num_negative_labels": int(np.sum(payload["y"] == 0)),
            "context_row_identity_hash": hashlib.sha256(
                json.dumps([(row["source_triple"], row["orientation"]) for row in rows],
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "feature_matrix_hash": array_hash(X),
            "original_label_hash": array_hash(payload["original_y"]),
            "supplied_label_hash": array_hash(payload["y"]),
            "label_permutation_hash": array_hash(payload["permutation"]),
            "labels_changed": int(np.sum(payload["original_y"] != payload["y"])),
            "candidate_metadata": candidate_metadata,
            "source_context_config": build_metadata["source_context_config"],
            "random_rows_info": build_metadata.get("random_rows") if condition == "random_rows_correct_balance" else None,
        }
        config = {
            "experiment_family": "context_intervention_suite",
            "dataset": DATASET,
            "seed": seed,
            "condition": condition,
            "representation": "primary_balanced_equal invariant explicit + frozen ULTRA repr + score",
            "num_features": int(X.shape[1]),
            "context_size": SIZE,
            "context_rows": int(len(X)),
            "decoder": "tabpfn_v3",
            "checkpoint": str(CHECKPOINT),
            "n_estimators": n_estimators,
            "fit_mode": "fit_with_cache",
            "evaluation": "full all-entity filtered ranking, combined head+tail with separate sides",
            "filter_set": "train+valid+test inside evaluator only",
            "tie_policy": "expected uniform tie breaking",
            "score_normalization": "none; direct TabPFN positive-class probability",
            "git_commit": git_commit(),
            "packages": {
                "tabpfn": package_version("tabpfn"),
                "torch": package_version("torch"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scikit_learn": package_version("scikit-learn"),
            },
            "hardware": hardware_info(),
        }
        timings = {
            "context_build_shared_seconds": build_metadata["build_seconds"],
            "fit_seconds": decoder_info["fit_time"],
            "prediction_seconds": decoder_info["predict_time"],
            "decoder_wall_seconds": decode_seconds,
            "total_run_seconds_excluding_shared_context_build": time.time() - started,
            "peak_gpu_memory_bytes": peak_gpu,
        }
        if condition == "correct_labels":
            reference = primary_reference_mrr(seed)
            observed = float(metrics["test"]["combined"]["mrr"])
            config["primary_reproduction"] = {
                "reference_mrr": reference,
                "observed_mrr": observed,
                "absolute_difference": abs(reference - observed),
                "tolerance": PRIMARY_REPRODUCTION_TOLERANCE,
                "passed": abs(reference - observed) <= PRIMARY_REPRODUCTION_TOLERANCE,
            }
            if not config["primary_reproduction"]["passed"]:
                raise RuntimeError(
                    f"identical-condition check failed: {observed} != {reference}")

        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "metrics.json", metrics)
        atomic_json(run_dir / "timings.json", timings)
        atomic_json(run_dir / "context_manifest.json", manifest)
        atomic_parquet(run_dir / "predictions.parquet", prediction_table)
        log_path.write_text(
            f"OK {condition} seed={seed} test_mrr={metrics['test']['combined']['mrr']:.9f}\n",
            encoding="utf-8",
        )
        (run_dir / "failure.json").unlink(missing_ok=True)
        return metrics
    except Exception as exc:
        failure = {
            "condition": condition,
            "seed": seed,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(run_dir / "failure.json", failure)
        log_path.write_text(failure["traceback"], encoding="utf-8")
        raise


def paired_summary(values_a: list[float], values_b: list[float]) -> dict:
    differences = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    bootstrap = np.asarray([
        differences[np.asarray(indices, dtype=np.int64)].mean()
        for indices in itertools.product(range(len(differences)), repeat=len(differences))
    ])
    return {
        "per_seed_differences": {str(seed): float(value) for seed, value in zip(SEEDS, differences)},
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "sample_standard_deviation": float(np.std(differences, ddof=1)),
        "paired_bootstrap_90pct_ci": [
            float(np.quantile(bootstrap, 0.05)), float(np.quantile(bootstrap, 0.95))],
        "positive_seeds": int(np.sum(differences > 0)),
        "negative_seeds": int(np.sum(differences < 0)),
        "zero_seeds": int(np.sum(differences == 0)),
    }


def metric_values(all_metrics: dict, condition: str, side: str, metric: str) -> list[float]:
    return [float(all_metrics[condition][seed]["test"][side][metric]) for seed in SEEDS]


def query_ranks(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for (side, query_index), group in frame.groupby(["side", "query_index"], sort=True):
        target_score = float(group.loc[group["is_target"] == 1, score_column].iloc[0])
        valid_scores = group.loc[group["is_filtered"] == 0, score_column].to_numpy(dtype=np.float64)
        positions = tied_rank_positions(valid_scores, target_score)
        expected_rank = float(np.mean(positions))
        rr = float(np.mean(1.0 / positions))
        rows.append({"side": side, "query_index": int(query_index), "rank": expected_rank, "rr": rr})
    return pd.DataFrame(rows)


def rank_group(rank: float) -> str:
    if rank <= 1.0:
        return "rank_1"
    if rank <= 10.0:
        return "rank_2_10"
    if rank <= 100.0:
        return "rank_11_100"
    return "rank_gt_100"


def rank_stratified_analysis() -> dict:
    records = []
    for seed in SEEDS:
        condition_frames = {}
        for condition in CONDITIONS:
            path = SUITE_ROOT / DATASET / condition / "tabpfn_v3" / str(seed) / "predictions.parquet"
            frame = pd.read_parquet(path)
            condition_frames[condition] = frame[frame["split"] == "test"].reset_index(drop=True)
        base = query_ranks(condition_frames["correct_labels"], "ultra_score")
        base = base.rename(columns={"rank": "ultra_rank", "rr": "ultra_rr"})
        merged = base
        for condition, frame in condition_frames.items():
            values = query_ranks(frame, "score").rename(
                columns={"rank": f"{condition}_rank", "rr": f"{condition}_rr"})
            merged = merged.merge(values, on=["side", "query_index"], validate="one_to_one")
        merged["seed"] = seed
        merged["rank_group"] = merged["ultra_rank"].map(rank_group)
        records.append(merged)
    data = pd.concat(records, ignore_index=True)
    output = {}
    for group_name in ("rank_1", "rank_2_10", "rank_11_100", "rank_gt_100"):
        group = data[data["rank_group"] == group_name]
        entry = {
            "query_seed_pairs": int(len(group)),
            "unique_queries_per_seed": int(group.groupby("seed").size().median()) if len(group) else 0,
            "ultra_mean_rr": float(group["ultra_rr"].mean()) if len(group) else None,
            "conditions": {},
        }
        for condition in CONDITIONS:
            delta = group[f"{condition}_rr"] - group["ultra_rr"]
            entry["conditions"][condition] = {
                "mean_rr": float(group[f"{condition}_rr"].mean()) if len(group) else None,
                "mean_change_from_ultra": float(delta.mean()) if len(group) else None,
                "improved": int(np.sum(delta > 1e-12)),
                "worsened": int(np.sum(delta < -1e-12)),
                "unchanged": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        output[group_name] = entry
    return output


def decide(rule: dict, all_metrics: dict) -> tuple[str, dict]:
    comparison = paired_summary(
        metric_values(all_metrics, "correct_labels", "combined", "mrr"),
        metric_values(all_metrics, "permuted_labels", "combined", "mrr"),
    )
    threshold = float(rule["practical_threshold"]["value"])
    small = float(rule["practical_threshold"]["small_effect_boundary"])
    head_mean = paired_summary(
        metric_values(all_metrics, "correct_labels", "head", "mrr"),
        metric_values(all_metrics, "permuted_labels", "head", "mrr"),
    )["mean"]
    tail_mean = paired_summary(
        metric_values(all_metrics, "correct_labels", "tail", "mrr"),
        metric_values(all_metrics, "permuted_labels", "tail", "mrr"),
    )["mean"]
    hits_means = {
        metric: paired_summary(
            metric_values(all_metrics, "correct_labels", "combined", metric),
            metric_values(all_metrics, "permuted_labels", "combined", metric),
        )["mean"]
        for metric in ("hits_at_1", "hits_at_3", "hits_at_10")
    }
    lower, upper = comparison["paired_bootstrap_90pct_ci"]
    pass_checks = {
        "practical_mean": comparison["mean"] >= threshold,
        "direction_consistency": comparison["positive_seeds"] >= 4,
        "bootstrap_lower_positive": lower > 0,
        "head_tail_not_materially_negative": min(head_mean, tail_mean) >= -0.005,
        "hits_coherent": sum(value >= 0 for value in hits_means.values()) >= 2,
    }
    fail_checks = {
        "nonpositive_mean": comparison["mean"] <= 0,
        "at_most_two_positive_seeds": comparison["positive_seeds"] <= 2,
        "small_effect_with_upper_below_practical": comparison["mean"] <= small and upper < threshold,
    }
    if all(pass_checks.values()):
        status = "PASS"
    elif any(fail_checks.values()):
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    details = {
        "primary_comparison": comparison,
        "head_mean_difference": head_mean,
        "tail_mean_difference": tail_mean,
        "combined_hits_mean_differences": hits_means,
        "pass_checks": pass_checks,
        "fail_checks": fail_checks,
    }
    return status, details


def analyze_pilot(rule: dict) -> dict:
    all_metrics = {condition: {} for condition in CONDITIONS}
    per_seed = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = SUITE_ROOT / DATASET / condition / "tabpfn_v3" / str(seed)
            if not run_complete(run_dir):
                raise RuntimeError(f"incomplete pilot run: {run_dir}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            all_metrics[condition][seed] = metrics
            per_seed.append({
                "condition": condition,
                "seed": seed,
                "test": metrics["test"],
                "timings": json.loads((run_dir / "timings.json").read_text(encoding="utf-8")),
            })
    status, decision_details = decide(rule, all_metrics)
    correct_vs_permuted = {}
    correct_vs_random = {}
    permuted_vs_random = {}
    for side in ("combined", "head", "tail"):
        for metric in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10", "mean_rank"):
            key = f"{side}.{metric}"
            correct = metric_values(all_metrics, "correct_labels", side, metric)
            permuted = metric_values(all_metrics, "permuted_labels", side, metric)
            random_rows = metric_values(all_metrics, "random_rows_correct_balance", side, metric)
            correct_vs_permuted[key] = paired_summary(correct, permuted)
            correct_vs_random[key] = paired_summary(correct, random_rows)
            permuted_vs_random[key] = paired_summary(permuted, random_rows)
    rank_results = rank_stratified_analysis()
    if status == "PASS":
        supported = ["Correct target-context labels have a pre-registered, practically meaningful effect on UMLS."]
        unsupported = ["The pilot alone does not establish a TabPFN-specific residual mechanism or selectivity."]
        alternative = "Ordinary nonlinear reranking remains viable until matched classical residual controls are run."
        next_step = "Run the pre-specified UMLS residual controls under identical inputs."
    elif status == "FAIL":
        supported = []
        unsupported = ["Semantic in-context adaptation is not supported under the frozen UMLS protocol."]
        alternative = "Feature/ULTRA-score-driven ordinary nonlinear reranking or row-distribution shortcuts."
        next_step = "Stop residual/gate development and inspect which frozen feature families retain performance under permuted labels."
    else:
        supported = []
        unsupported = ["Semantic in-context adaptation remains unproven."]
        alternative = "Both a modest label effect and ordinary nonlinear reranking remain compatible with the data."
        next_step = "Add paired UMLS seeds 106-110 only, stopping as soon as the frozen rule becomes decisive."
    summary = {
        "design_review": json.loads(PLAN_PATH.read_text(encoding="utf-8"))["design_review"],
        "decision_rule": rule,
        "pilot": {
            "status": status,
            "decision_details": decision_details,
            "per_seed_results": per_seed,
            "correct_vs_permuted": correct_vs_permuted,
            "correct_vs_random_rows": correct_vs_random,
            "permuted_vs_random_rows": permuted_vs_random,
            "rank_stratified_results": rank_results,
        },
        "residual_suite_executed": False,
        "baseline_results": {},
        "context_ablations": {},
        "dataset_transfer_results": {},
        "supported_conclusions": supported,
        "unsupported_conclusions": unsupported,
        "main_alternative_explanation": alternative,
        "recommended_next_step": next_step,
    }
    atomic_json(SUMMARY_JSON, summary)
    primary = decision_details["primary_comparison"]
    markdown = f"""# Context Intervention Pilot

## Decision

**{status}**

- Correct minus permuted combined MRR: {primary['mean']:.6f}
- Median paired difference: {primary['median']:.6f}
- 90% paired-bootstrap interval: [{primary['paired_bootstrap_90pct_ci'][0]:.6f}, {primary['paired_bootstrap_90pct_ci'][1]:.6f}]
- Positive/negative seeds: {primary['positive_seeds']}/{primary['negative_seeds']}
- Practical threshold: {rule['practical_threshold']['value']:.6f}

## Interpretation

{supported[0] if supported else unsupported[0]}

Main alternative explanation: {alternative}

Next step: {next_step}
"""
    tmp = SUMMARY_MD.with_name(f".{SUMMARY_MD.name}.{os.getpid()}.tmp")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(markdown, encoding="utf-8")
    os.replace(tmp, SUMMARY_MD)
    if status == "FAIL":
        FAILURE_MD.write_text(
            "# Context Intervention Failure\n\n"
            "**The current evidence does not support semantic in-context adaptation.**\n\n"
            f"Correct minus permuted combined MRR was {primary['mean']:.6f}, "
            f"with 90% interval {primary['paired_bootstrap_90pct_ci']}.\n\n"
            f"Leading explanation: {alternative}\n",
            encoding="utf-8",
        )
    return summary


def run_smokes(pa: PrimaryAssets, contexts: dict, metadata: dict, candidate_metadata: dict) -> None:
    for condition in CONDITIONS:
        run_condition(pa, 101, condition, contexts[condition], metadata,
                      candidate_metadata, smoke=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    rule, plan = validate_preregistration()
    print(
        f"PLAN decision={plan['design_review']['decision']} "
        f"smoke_jobs={plan['pilot']['smoke_job_count']} "
        f"pilot_jobs={plan['pilot']['full_job_count']} "
        f"estimated_minutes={plan['pilot']['estimated_runtime_minutes']}",
        flush=True,
    )
    pa = PrimaryAssets(DATASET)
    _, candidate_metadata = candidate_payload(pa)
    contexts_101, metadata_101 = prepare_seed_contexts(pa, 101)
    if not args.skip_smoke:
        run_smokes(pa, contexts_101, metadata_101, candidate_metadata)
        print("SMOKE TESTS PASSED", flush=True)
    if args.smoke_only:
        return 0

    for seed in SEEDS:
        if seed == 101:
            contexts, metadata = contexts_101, metadata_101
        else:
            contexts, metadata = prepare_seed_contexts(pa, seed)
        for condition in CONDITIONS:
            metrics = run_condition(
                pa, seed, condition, contexts[condition], metadata, candidate_metadata)
            print(
                f"OK seed={seed} condition={condition} "
                f"mrr={metrics['test']['combined']['mrr']:.6f}",
                flush=True,
            )

    summary = analyze_pilot(rule)
    status = summary["pilot"]["status"]
    primary = summary["pilot"]["decision_details"]["primary_comparison"]
    if status == "FAIL":
        print("=" * 72, flush=True)
        print("WARNING: The current evidence does not support semantic in-context adaptation.", flush=True)
        print("THE RESIDUAL SUITE AND GATE WERE NOT RUN.", flush=True)
        print("=" * 72, flush=True)
        return 2
    if status == "INCONCLUSIVE":
        print("INCONCLUSIVE: residual suite not authorized; add only paired seeds 106-110.", flush=True)
        return 3
    print(
        f"PASS: correct labels mattered (mean paired MRR={primary['mean']:.6f}); "
        "UMLS residual controls are authorized.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
