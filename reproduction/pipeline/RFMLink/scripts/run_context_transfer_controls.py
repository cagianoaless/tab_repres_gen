#!/usr/bin/env python3
"""Minimal weak/strong transfer after the UMLS residual controls."""

from __future__ import annotations

import argparse
import json
import os
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

import run_context_intervention_pilot as pilot  # noqa: E402
import run_context_residual_controls as residual  # noqa: E402
from run_primary_experiment import CATEGORICAL, INVARIANT_COLUMNS, side_metrics  # noqa: E402
from rfmlink.contexts import load_context  # noqa: E402
from rfmlink.decoders import fit_predict_hgb, fit_predict_tabpfn3  # noqa: E402


DATASET_CONDITIONS = {
    "kinships": ("correct_labels", "permuted_labels"),
    "codex-small": ("correct_labels",),
}
SEEDS = pilot.SEEDS
REFERENCE_TOLERANCE = 0.003


def prepare_contexts(pa, dataset: str, seed: int):
    triples, labels, source_config = load_context(dataset, seed, pilot.SIZE)
    started = time.time()
    X, y, indices, orientations = pa.context_matrix_from_rows(
        triples, labels, seed, balanced=True, equal_budget=True, return_manifest=True)
    relations = triples[indices, 1]
    permuted, permutation = pilot.stratified_label_permutation(
        y, relations, orientations, seed)
    common = {
        "X": X,
        "triples": triples,
        "indices": indices,
        "orientations": orientations,
        "original_y": y,
    }
    return {
        "correct_labels": {
            **common, "y": y, "permutation": np.arange(len(y), dtype=np.int64)},
        "permuted_labels": {
            **common, "y": permuted, "permutation": permutation},
    }, {
        "context_build_seconds": time.time() - started,
        "source_context_config": source_config,
    }


def candidate_metadata(pa):
    output = {}
    for split in ("valid", "test"):
        table = pa.eval_tables[split]
        output[split] = {
            "rows": len(table),
            "queries": int(table["query_index"].nunique()),
            "candidate_hash": pilot.frame_hash(
                table,
                ["query_index", "side", "h", "r", "t", "is_target", "is_filtered"],
            ),
            "feature_hash": pilot.array_hash(pa.eval_X[split]),
            "ultra_score_hash": pilot.array_hash(pa.eval_X[split][:, -1]),
        }
    return output


def prediction_frame(pa, predictions):
    frames = []
    for split, scores in zip(("valid", "test"), predictions):
        table = pa.eval_tables[split]
        frame = table[["query_index", "side", "h", "r", "t", "is_target", "is_filtered"]].copy()
        frame.insert(0, "split", split)
        frame["score"] = np.asarray(scores, dtype=np.float64)
        frame["ultra_score"] = pa.eval_X[split][:, -1].astype(np.float64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def base_dir(dataset, condition, decoder, seed, smoke=False):
    name = f"{decoder}_transfer" + ("_smoke" if smoke else "")
    return pilot.SUITE_ROOT / dataset / condition / name / str(seed)


def primary_reference(dataset, seed):
    path = (pilot.PRIMARY_RESULTS /
            f"{dataset}__primary_balanced_equal__tabpfn_v3__s{seed}" / "metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))["test"]["combined"]["mrr"]


def execute_base(
    pa,
    dataset,
    seed,
    condition,
    decoder,
    payload,
    metadata,
    smoke=False,
    enforce_historical_reference=True,
):
    run_dir = base_dir(dataset, condition, decoder, seed, smoke)
    if not smoke and pilot.run_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    try:
        eval_Xs = [pa.eval_X[split] for split in ("valid", "test")]
        eval_tables = [pa.eval_tables[split] for split in ("valid", "test")]
        if smoke:
            reduced_Xs, reduced_tables = [], []
            for X, table in zip(eval_Xs, eval_tables):
                qids = table["query_index"].drop_duplicates().iloc[:2]
                mask = table["query_index"].isin(qids).to_numpy()
                reduced_Xs.append(X[mask])
                reduced_tables.append(table.loc[mask].reset_index(drop=True))
            eval_Xs, eval_tables = reduced_Xs, reduced_tables
        cat = [INVARIANT_COLUMNS.index(column) for column in CATEGORICAL]
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        if decoder == "tabpfn_v3":
            predictions, info = fit_predict_tabpfn3(
                payload["X"], payload["y"], eval_Xs, seed,
                categorical_indices=cat,
                fit_mode="fit_with_cache",
                checkpoint_override=str(pilot.CHECKPOINT),
                n_estimators=1 if smoke else 8,
            )
        elif decoder == "hgb":
            predictions, info = fit_predict_hgb(
                payload["X"], payload["y"], eval_Xs, seed,
                categorical_indices=cat)
        else:
            raise ValueError(decoder)
        peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        metrics = {
            split: side_metrics(table, np.asarray(scores))
            for split, table, scores in zip(("valid", "test"), eval_tables, predictions)
        }
        if smoke:
            pilot.atomic_json(run_dir / "smoke_metrics.json", metrics)
            log_path.write_text(f"SMOKE OK {dataset} {condition} {decoder}\n", encoding="utf-8")
            return metrics
        frame = prediction_frame(pa, predictions)
        rows = pilot.context_rows_payload(
            payload["triples"], payload["indices"], payload["orientations"],
            payload["original_y"], payload["y"])
        manifest = {
            "rows": rows,
            "condition": condition,
            "context_row_identity_hash": pilot.hashlib.sha256(
                json.dumps([(row["source_triple"], row["orientation"]) for row in rows],
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "feature_matrix_hash": pilot.array_hash(payload["X"]),
            "original_label_hash": pilot.array_hash(payload["original_y"]),
            "supplied_label_hash": pilot.array_hash(payload["y"]),
            "label_permutation_hash": pilot.array_hash(payload["permutation"]),
            "candidate_metadata": candidate_metadata(pa),
            "source_context_config": metadata["source_context_config"],
        }
        config = {
            "experiment_family": metadata.get(
                "experiment_family", "context_intervention_suite"),
            "stage": "minimal_dataset_transfer",
            "dataset": dataset,
            "seed": seed,
            "condition": condition,
            "decoder": decoder,
            "representation": "primary_balanced_equal, 147 columns",
            "context_rows": len(payload["y"]),
            "evaluation": "full candidate ranking on the frozen primary query set; combined head+tail",
            "query_subset": "fixed 500-triple subset" if dataset == "codex-small" else None,
            "checkpoint": str(pilot.CHECKPOINT) if decoder == "tabpfn_v3" else None,
            "git_commit": pilot.git_commit(),
            "hardware": pilot.hardware_info(),
        }
        if metadata.get("protocol_revision"):
            config["protocol_revision"] = metadata["protocol_revision"]
        if metadata.get("canonical_context_artifact"):
            config["canonical_context_artifact"] = metadata["canonical_context_artifact"]
            config["canonical_context_sha256"] = metadata["canonical_context_sha256"]
        if metadata.get("package_versions"):
            config["package_versions"] = metadata["package_versions"]
        if metadata.get("checkpoint_sha256"):
            config["checkpoint_sha256"] = metadata["checkpoint_sha256"]
        if (enforce_historical_reference
                and condition == "correct_labels" and decoder == "tabpfn_v3"):
            reference = float(primary_reference(dataset, seed))
            observed = float(metrics["test"]["combined"]["mrr"])
            config["primary_reproduction"] = {
                "reference_mrr": reference,
                "observed_mrr": observed,
                "absolute_difference": abs(reference - observed),
                "tolerance": REFERENCE_TOLERANCE,
                "passed": abs(reference - observed) <= REFERENCE_TOLERANCE,
            }
            if not config["primary_reproduction"]["passed"]:
                raise RuntimeError(
                    f"{dataset} seed {seed} primary reproduction drift exceeded tolerance: "
                    f"observed={observed:.12f} reference={reference:.12f} "
                    f"difference={abs(reference - observed):.12f}")
        timings = {
            "context_build_shared_seconds": metadata["context_build_seconds"],
            "fit_seconds": info["fit_time"],
            "prediction_seconds": info["predict_time"],
            "total_decoder_seconds": time.time() - started,
            "peak_gpu_memory_bytes": peak_gpu,
        }
        pilot.atomic_json(run_dir / "config.json", config)
        pilot.atomic_json(run_dir / "metrics.json", metrics)
        pilot.atomic_json(run_dir / "timings.json", timings)
        pilot.atomic_json(run_dir / "context_manifest.json", manifest)
        pilot.atomic_parquet(run_dir / "predictions.parquet", frame)
        log_path.write_text(
            f"OK {dataset} {condition} {decoder} seed={seed} "
            f"mrr={metrics['test']['combined']['mrr']:.9f}\n", encoding="utf-8")
        (run_dir / "failure.json").unlink(missing_ok=True)
        return metrics
    except Exception as exc:
        failure = {
            "dataset": dataset, "condition": condition, "decoder": decoder,
            "seed": seed, "exception_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if "metrics" in locals():
            failure["attempted_metrics"] = metrics
        pilot.atomic_json(run_dir / "failure.json", failure)
        log_path.write_text(failure["traceback"], encoding="utf-8")
        raise


def fast_query_standardize(frame, values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(len(values), dtype=np.float64)
    for split in ("valid", "test"):
        split_indices = np.flatnonzero(frame["split"].to_numpy() == split)
        part = frame.iloc[split_indices]
        counts = part.groupby("query_index", sort=False).size().to_numpy()
        if len(np.unique(counts)) != 1:
            raise RuntimeError("fast normalization requires fixed candidate count per query")
        width = int(counts[0])
        matrix = values[split_indices].reshape(-1, width)
        keep = (part["is_filtered"].to_numpy() == 0).reshape(-1, width)
        denominator = keep.sum(axis=1)
        mean = np.sum(matrix * keep, axis=1) / denominator
        centered = matrix - mean[:, None]
        scale = np.sqrt(np.sum(centered * centered * keep, axis=1) / denominator)
        scale[scale < 1e-8] = 1.0
        output[split_indices] = (centered / scale[:, None]).reshape(-1)
    return output


def fast_mrr(table, scores):
    counts = table.groupby("query_index", sort=False).size().to_numpy()
    if len(np.unique(counts)) != 1:
        raise RuntimeError("fast MRR requires fixed candidate count per query")
    width = int(counts[0])
    score_matrix = np.asarray(scores, dtype=np.float64).reshape(-1, width)
    target_mask = table["is_target"].to_numpy(dtype=bool).reshape(-1, width)
    keep = (table["is_filtered"].to_numpy() == 0).reshape(-1, width)
    if not np.all(target_mask.sum(axis=1) == 1):
        raise RuntimeError("each query must have exactly one target")
    target = score_matrix[target_mask]
    greater = np.sum((score_matrix > target[:, None]) & keep, axis=1)
    tied = np.sum((score_matrix == target[:, None]) & keep, axis=1)
    harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, width + 1))])
    rr = (harmonic[greater + tied] - harmonic[greater]) / tied
    return float(rr.mean())


def derived_dir(dataset, condition, method, seed):
    return pilot.SUITE_ROOT / dataset / condition / f"{method}_transfer" / str(seed)


def load_base(dataset, condition, decoder, seed):
    return pd.read_parquet(base_dir(dataset, condition, decoder, seed) / "predictions.parquet")


def save_derived(dataset, condition, method, source_decoder, seed):
    run_dir = derived_dir(dataset, condition, method, seed)
    if pilot.run_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    frame = load_base(dataset, condition, source_decoder or "tabpfn_v3", seed)
    source_run_dir = base_dir(dataset, condition, source_decoder or "tabpfn_v3", seed)
    source_config = json.loads((source_run_dir / "config.json").read_text(encoding="utf-8"))
    ultra = fast_query_standardize(frame, frame["ultra_score"].to_numpy())
    if method == "ultra_only":
        final, selected, valid_mrr = ultra, 0.0, None
        formula = "z_query(ULTRA)"
        grid = [0.0]
    else:
        probability = np.clip(frame["score"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
        model = fast_query_standardize(frame, np.log(probability / (1 - probability)))
        grid = residual.ALPHA_GRID if method == "fixed_blend" else residual.BETA_GRID
        valid_mask = frame["split"].to_numpy() == "valid"
        valid_table = frame.loc[valid_mask].reset_index(drop=True)
        candidates = []
        for weight in grid:
            score = ((1 - weight) * ultra + weight * model
                     if method == "fixed_blend" else ultra + weight * model)
            candidates.append((fast_mrr(valid_table, score[valid_mask]), float(weight)))
        valid_mrr = max(value for value, _ in candidates)
        selected = min(weight for value, weight in candidates if abs(value - valid_mrr) <= 1e-12)
        final = ((1 - selected) * ultra + selected * model
                 if method == "fixed_blend" else ultra + selected * model)
        formula = ("(1-alpha)*z_query(ULTRA)+alpha*z_query(logit(model))"
                   if method == "fixed_blend" else
                   "z_query(ULTRA)+beta*z_query(logit(model))")
    metrics = residual.metrics_for_frame(frame, final)
    predictions = frame.copy()
    predictions["base_model_score"] = predictions["score"]
    predictions["score"] = final
    predictions["normalized_ultra_score"] = ultra
    config = {
        "experiment_family": source_config.get(
            "experiment_family", "context_intervention_suite"),
        "stage": "minimal_dataset_transfer_derived",
        "dataset": dataset, "condition": condition, "method": method,
        "source_decoder": source_decoder, "seed": seed,
        "normalization": "per-query z-score over unfiltered candidates",
        "formula": formula, "weight_grid": list(map(float, grid)),
        "selected_weight": selected, "selected_validation_mrr": valid_mrr,
        "selection_split": "validation only", "top_k": None,
        "git_commit": pilot.git_commit(), "hardware": pilot.hardware_info(),
    }
    for key in (
        "protocol_revision", "canonical_context_artifact",
        "canonical_context_sha256", "package_versions", "checkpoint_sha256",
    ):
        if key in source_config:
            config[key] = source_config[key]
    manifest = {
        "source_context_manifest": str(base_dir(
            dataset, condition, source_decoder or "tabpfn_v3", seed) / "context_manifest.json"),
        "candidate_hashes_in_source_manifest": True,
    }
    pilot.atomic_json(run_dir / "config.json", config)
    pilot.atomic_json(run_dir / "metrics.json", metrics)
    pilot.atomic_json(run_dir / "timings.json", {
        "derivation_only": True, "peak_gpu_memory_bytes": 0})
    pilot.atomic_json(run_dir / "context_manifest.json", manifest)
    pilot.atomic_parquet(run_dir / "predictions.parquet", predictions)
    (run_dir / "run.log").write_text(
        f"OK {dataset} {condition} {method} seed={seed} "
        f"weight={selected:.3f} mrr={metrics['test']['combined']['mrr']:.9f}\n",
        encoding="utf-8")
    return metrics


def aggregate(dataset, condition, method):
    values = []
    for seed in SEEDS:
        path = derived_dir(dataset, condition, method, seed)
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        values.append({
            "seed": seed,
            "mrr": metrics["test"]["combined"]["mrr"],
            "head_mrr": metrics["test"]["head"]["mrr"],
            "tail_mrr": metrics["test"]["tail"]["mrr"],
            "selected_weight": config["selected_weight"],
        })
    mrr = np.asarray([row["mrr"] for row in values])
    return {
        "per_seed": values,
        "mean_mrr": float(mrr.mean()),
        "sample_standard_deviation_mrr": float(mrr.std(ddof=1)),
    }


def update_summary():
    summary = json.loads(pilot.SUMMARY_JSON.read_text(encoding="utf-8"))
    transfer = {}
    for dataset, conditions in DATASET_CONDITIONS.items():
        transfer[dataset] = {}
        for condition in conditions:
            methods = ("ultra_only", "fixed_blend", "tabpfn_residual", "hgb_residual")
            transfer[dataset][condition] = {
                method: aggregate(dataset, condition, method) for method in methods}
            direct = []
            for seed in SEEDS:
                metrics = json.loads((base_dir(dataset, condition, "tabpfn_v3", seed) /
                                      "metrics.json").read_text(encoding="utf-8"))
                direct.append(metrics["test"]["combined"]["mrr"])
            transfer[dataset][condition]["tabpfn_direct"] = {
                "per_seed_mrr": dict(zip(map(str, SEEDS), direct)),
                "mean_mrr": float(np.mean(direct)),
                "sample_standard_deviation_mrr": float(np.std(direct, ddof=1)),
            }
    summary["dataset_transfer_results"] = transfer
    codex = transfer["codex-small"]["correct_labels"]
    ultra = codex["ultra_only"]["mean_mrr"]
    blend = codex["fixed_blend"]["mean_mrr"]
    tab_residual = codex["tabpfn_residual"]["mean_mrr"]
    if max(blend, tab_residual) >= ultra:
        summary["supported_conclusions"].append(
            "Validation-selected combination preserves or improves the strong CoDEx-Small ULTRA baseline on the frozen primary subset.")
    else:
        summary["unsupported_conclusions"].append(
            "Neither the fixed blend nor TabPFN residual preserves the strong CoDEx-Small ULTRA baseline.")
    summary["recommended_next_step"] = (
        "Do not build a gate. Replicate the best validation-selected fixed blend on one additional strong-ULTRA dataset only if a broader claim is needed.")
    pilot.atomic_json(pilot.SUMMARY_JSON, summary)
    markdown = pilot.SUMMARY_MD.read_text(encoding="utf-8")
    markdown += "\n## Dataset Transfer\n\n"
    for dataset in DATASET_CONDITIONS:
        correct = transfer[dataset]["correct_labels"]
        markdown += (
            f"- {dataset}: ULTRA {correct['ultra_only']['mean_mrr']:.6f}; "
            f"direct {correct['tabpfn_direct']['mean_mrr']:.6f}; "
            f"blend {correct['fixed_blend']['mean_mrr']:.6f}; "
            f"TabPFN residual {correct['tabpfn_residual']['mean_mrr']:.6f}; "
            f"HGB residual {correct['hgb_residual']['mean_mrr']:.6f}.\n")
    tmp = pilot.SUMMARY_MD.with_name(f".{pilot.SUMMARY_MD.name}.{os.getpid()}.tmp")
    tmp.write_text(markdown, encoding="utf-8")
    os.replace(tmp, pilot.SUMMARY_MD)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    summary = json.loads(pilot.SUMMARY_JSON.read_text(encoding="utf-8"))
    if not summary.get("residual_suite_executed"):
        raise RuntimeError("UMLS residual controls must complete before transfer")
    print("PLAN transfer_tabpfn_jobs=15 transfer_hgb_jobs=15 derived_jobs=60 estimated_hours=[3,4]", flush=True)
    assets = {}
    context_cache = {}
    if not args.skip_smoke:
        for dataset in DATASET_CONDITIONS:
            pa = pilot.PrimaryAssets(dataset)
            contexts, metadata = prepare_contexts(pa, dataset, 101)
            execute_base(pa, dataset, 101, "correct_labels", "tabpfn_v3",
                         contexts["correct_labels"], metadata, smoke=True)
            execute_base(pa, dataset, 101, "correct_labels", "hgb",
                         contexts["correct_labels"], metadata, smoke=True)
        print("TRANSFER SMOKES PASSED", flush=True)
    if args.smoke_only:
        return 0

    for dataset, conditions in DATASET_CONDITIONS.items():
        pa = pilot.PrimaryAssets(dataset)
        assets[dataset] = pa
        for seed in SEEDS:
            contexts, metadata = prepare_contexts(pa, dataset, seed)
            context_cache[(dataset, seed)] = contexts
            for condition in conditions:
                for decoder in ("tabpfn_v3", "hgb"):
                    metrics = execute_base(
                        pa, dataset, seed, condition, decoder, contexts[condition], metadata)
                    print(
                        f"OK transfer base {dataset} seed={seed} {condition} {decoder} "
                        f"mrr={metrics['test']['combined']['mrr']:.6f}", flush=True)
        for seed in SEEDS:
            for condition in conditions:
                for method, decoder in (
                    ("ultra_only", None), ("fixed_blend", "tabpfn_v3"),
                    ("tabpfn_residual", "tabpfn_v3"), ("hgb_residual", "hgb"),
                ):
                    metrics = save_derived(dataset, condition, method, decoder, seed)
                    print(
                        f"OK transfer derived {dataset} seed={seed} {condition} {method} "
                        f"mrr={metrics['test']['combined']['mrr']:.6f}", flush=True)
    summary = update_summary()
    codex = summary["dataset_transfer_results"]["codex-small"]["correct_labels"]
    print(
        "TRANSFER COMPLETE: "
        f"CoDEx ULTRA={codex['ultra_only']['mean_mrr']:.6f} "
        f"blend={codex['fixed_blend']['mean_mrr']:.6f} "
        f"TabPFN_residual={codex['tabpfn_residual']['mean_mrr']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
