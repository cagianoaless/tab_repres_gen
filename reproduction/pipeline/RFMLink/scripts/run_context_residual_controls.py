#!/usr/bin/env python3
"""Matched residual controls authorized by the UMLS context pilot PASS."""

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
from run_primary_experiment import CATEGORICAL, INVARIANT_COLUMNS, logistic_onehot, side_metrics  # noqa: E402
from rfmlink.decoders import fit_predict_hgb, fit_predict_tabpfn3  # noqa: E402


FEATURE_VARIANTS = {
    "full": {
        "description": "18 invariant explicit + 128 frozen ULTRA representation + ULTRA score",
        "indices": None,
    },
    "structural_without_ultra_score": {
        "description": "18 invariant explicit + 128 frozen ULTRA representation; ULTRA score removed",
        "indices": np.arange(146, dtype=np.int64),
    },
    "ultra_score_only": {
        "description": "frozen ULTRA scalar score only",
        "indices": np.array([146], dtype=np.int64),
    },
}
MLP_HIDDEN = (64,)
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
BETA_GRID = np.linspace(0.0, 2.0, 41)


def select_features(X: np.ndarray, variant: str) -> np.ndarray:
    indices = FEATURE_VARIANTS[variant]["indices"]
    return X if indices is None else X[:, indices]


def categorical_indices(variant: str) -> list[int]:
    if variant == "ultra_score_only":
        return []
    return [INVARIANT_COLUMNS.index(column) for column in CATEGORICAL]


def feature_names(pa, variant: str) -> list[str]:
    if variant == "full":
        return pa.feature_names
    if variant == "structural_without_ultra_score":
        return pa.feature_names[:-1]
    return ["ultra_score"]


def fit_predict_mlp(train_X, train_y, eval_Xs, seed, names, categorical):
    from sklearn.compose import ColumnTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    train = pd.DataFrame(train_X, columns=names)
    evaluations = [pd.DataFrame(X, columns=names) for X in eval_Xs]
    cat_names = [names[index] for index in categorical]
    numeric = [name for name in names if name not in cat_names]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_names),
        ("num", StandardScaler(), numeric),
    ])
    model = Pipeline([
        ("pre", preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            activation="relu",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=seed,
        )),
    ])
    start = time.time()
    model.fit(train, train_y)
    fit_seconds = time.time() - start
    start = time.time()
    predictions = [model.predict_proba(frame)[:, 1] for frame in evaluations]
    return predictions, {
        "model": "mlp",
        "fit_time": fit_seconds,
        "predict_time": time.time() - start,
        "hidden_layers": list(MLP_HIDDEN),
        "alpha": 1e-3,
        "early_stopping": True,
    }


def base_run_dir(condition: str, decoder: str, variant: str, seed: int, smoke=False) -> Path:
    name = f"{decoder}_direct_{variant}" + ("_smoke" if smoke else "")
    return pilot.SUITE_ROOT / pilot.DATASET / condition / name / str(seed)


def base_run_complete(path: Path) -> bool:
    return pilot.run_complete(path)


def build_manifest(payload: dict, condition: str, variant: str, candidate_metadata: dict) -> dict:
    rows = pilot.context_rows_payload(
        payload["triples"], payload["indices"], payload["orientations"],
        payload["original_y"], payload["y"])
    X = select_features(payload["X"], variant)
    return {
        "condition": condition,
        "feature_variant": variant,
        "rows": rows,
        "num_rows": len(rows),
        "context_row_identity_hash": pilot.hashlib.sha256(
            json.dumps([(row["source_triple"], row["orientation"]) for row in rows],
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "feature_matrix_hash": pilot.array_hash(X),
        "original_label_hash": pilot.array_hash(payload["original_y"]),
        "supplied_label_hash": pilot.array_hash(payload["y"]),
        "label_permutation_hash": pilot.array_hash(payload["permutation"]),
        "labels_changed": int(np.sum(payload["original_y"] != payload["y"])),
        "candidate_metadata": candidate_metadata,
    }


def execute_base(
    pa,
    seed: int,
    condition: str,
    decoder: str,
    variant: str,
    payload: dict,
    candidate_metadata: dict,
    smoke=False,
) -> dict:
    run_dir = base_run_dir(condition, decoder, variant, seed, smoke)
    if not smoke and base_run_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.time()
    try:
        X = select_features(payload["X"], variant)
        eval_Xs = [select_features(pa.eval_X[split], variant) for split in ("valid", "test")]
        eval_tables = [pa.eval_tables[split] for split in ("valid", "test")]
        if smoke:
            compact_Xs, compact_tables = [], []
            for matrix, table in zip(eval_Xs, eval_tables):
                qids = table["query_index"].drop_duplicates().iloc[:4]
                mask = table["query_index"].isin(qids).to_numpy()
                compact_Xs.append(matrix[mask])
                compact_tables.append(table.loc[mask].reset_index(drop=True))
            eval_Xs, eval_tables = compact_Xs, compact_tables
        cat = categorical_indices(variant)

        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if decoder == "tabpfn_v3":
            predictions, info = fit_predict_tabpfn3(
                X, payload["y"], eval_Xs, seed,
                categorical_indices=cat,
                fit_mode="fit_with_cache",
                checkpoint_override=str(pilot.CHECKPOINT),
                n_estimators=1 if smoke else 8,
            )
        elif decoder == "hgb":
            predictions, info = fit_predict_hgb(
                X, payload["y"], eval_Xs, seed, categorical_indices=cat)
        elif decoder == "logistic":
            names = feature_names(pa, variant)
            predictions, info = logistic_onehot(
                pd.DataFrame(X, columns=names), payload["y"],
                [pd.DataFrame(matrix, columns=names) for matrix in eval_Xs], seed)
        elif decoder == "mlp":
            predictions, info = fit_predict_mlp(
                X, payload["y"], eval_Xs, seed, feature_names(pa, variant), cat)
        else:
            raise ValueError(decoder)
        peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        metrics = {
            split: side_metrics(table, np.asarray(scores))
            for split, table, scores in zip(("valid", "test"), eval_tables, predictions)
        }
        if smoke:
            pilot.atomic_json(run_dir / "smoke_metrics.json", metrics)
            log_path.write_text(f"SMOKE OK {condition} {decoder} {variant}\n", encoding="utf-8")
            return metrics

        scores_by_split = dict(zip(("valid", "test"), predictions))
        prediction_frame = pilot.predictions_frame(pa, scores_by_split)
        config = {
            "experiment_family": "context_intervention_suite",
            "stage": "matched_base_decoder_for_residual_controls",
            "dataset": pilot.DATASET,
            "seed": seed,
            "condition": condition,
            "decoder": decoder,
            "feature_variant": variant,
            "feature_description": FEATURE_VARIANTS[variant]["description"],
            "num_features": int(X.shape[1]),
            "context_rows": int(len(X)),
            "evaluation": "full all-entity filtered ranking, both sides",
            "checkpoint": str(pilot.CHECKPOINT) if decoder == "tabpfn_v3" else None,
            "git_commit": pilot.git_commit(),
            "packages": {
                "tabpfn": pilot.package_version("tabpfn"),
                "torch": pilot.package_version("torch"),
                "scikit_learn": pilot.package_version("scikit-learn"),
            },
            "hardware": pilot.hardware_info(),
        }
        timings = {
            "fit_seconds": info.get("fit_time"),
            "prediction_seconds": info.get("predict_time"),
            "total_seconds": time.time() - started,
            "peak_gpu_memory_bytes": peak_gpu,
        }
        pilot.atomic_json(run_dir / "config.json", config)
        pilot.atomic_json(run_dir / "metrics.json", metrics)
        pilot.atomic_json(run_dir / "timings.json", timings)
        pilot.atomic_json(run_dir / "context_manifest.json",
                          build_manifest(payload, condition, variant, candidate_metadata))
        pilot.atomic_parquet(run_dir / "predictions.parquet", prediction_frame)
        log_path.write_text(
            f"OK {condition} {decoder} {variant} seed={seed} "
            f"mrr={metrics['test']['combined']['mrr']:.9f}\n", encoding="utf-8")
        (run_dir / "failure.json").unlink(missing_ok=True)
        return metrics
    except Exception as exc:
        failure = {
            "condition": condition,
            "decoder": decoder,
            "variant": variant,
            "seed": seed,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        pilot.atomic_json(run_dir / "failure.json", failure)
        log_path.write_text(failure["traceback"], encoding="utf-8")
        raise


def load_predictions(condition: str, decoder: str, variant: str, seed: int) -> pd.DataFrame:
    if decoder == "tabpfn_v3" and variant == "full":
        path = (pilot.SUITE_ROOT / pilot.DATASET / condition / "tabpfn_v3" /
                str(seed) / "predictions.parquet")
    else:
        path = base_run_dir(condition, decoder, variant, seed) / "predictions.parquet"
    return pd.read_parquet(path)


def query_standardize(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.empty(len(frame), dtype=np.float64)
    work = frame[["split", "side", "query_index", "is_filtered"]].copy()
    work["value"] = np.asarray(values, dtype=np.float64)
    for _, group in work.groupby(["split", "side", "query_index"], sort=False):
        reference = group.loc[group["is_filtered"] == 0, "value"].to_numpy()
        mean = float(reference.mean())
        scale = float(reference.std())
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        result[group.index.to_numpy()] = (group["value"].to_numpy() - mean) / scale
    return result


def normalized_model_score(frame: pd.DataFrame) -> np.ndarray:
    probability = np.clip(frame["score"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    return query_standardize(frame, logit)


def normalized_ultra_score(frame: pd.DataFrame) -> np.ndarray:
    return query_standardize(frame, frame["ultra_score"].to_numpy(dtype=np.float64))


def metrics_for_frame(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    output = {}
    for split in ("valid", "test"):
        mask = frame["split"].to_numpy() == split
        table = frame.loc[mask].reset_index(drop=True)
        output[split] = side_metrics(table, np.asarray(scores)[mask])
    return output


def select_validation_weight(frame: pd.DataFrame, model_score: np.ndarray, mode: str):
    ultra = normalized_ultra_score(frame)
    valid_mask = frame["split"].to_numpy() == "valid"
    valid_frame = frame.loc[valid_mask].reset_index(drop=True)
    grid = ALPHA_GRID if mode == "fixed_blend" else BETA_GRID
    candidates = []
    for weight in grid:
        if mode == "fixed_blend":
            scores = (1.0 - weight) * ultra + weight * model_score
        else:
            scores = ultra + weight * model_score
        metric = side_metrics(valid_frame, scores[valid_mask])["combined"]["mrr"]
        candidates.append((float(metric), float(weight)))
    best_metric = max(metric for metric, _ in candidates)
    best_weight = min(weight for metric, weight in candidates if abs(metric - best_metric) <= 1e-12)
    if mode == "fixed_blend":
        final = (1.0 - best_weight) * ultra + best_weight * model_score
    else:
        final = ultra + best_weight * model_score
    return final, best_weight, best_metric


def derived_run_dir(condition: str, method: str, variant: str, seed: int) -> Path:
    return pilot.SUITE_ROOT / pilot.DATASET / condition / f"{method}_{variant}" / str(seed)


def save_derived(
    condition: str,
    method: str,
    variant: str,
    seed: int,
    source_decoder: str | None,
) -> dict:
    run_dir = derived_run_dir(condition, method, variant, seed)
    if pilot.run_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    if source_decoder is None:
        frame = load_predictions("correct_labels", "tabpfn_v3", "full", seed)
        final = normalized_ultra_score(frame)
        weight = 0.0
        valid_mrr = metrics_for_frame(frame, final)["valid"]["combined"]["mrr"]
        formula = "per-query zscore(ULTRA raw score)"
    else:
        frame = load_predictions(condition, source_decoder, variant, seed)
        model_score = normalized_model_score(frame)
        mode = "fixed_blend" if method == "fixed_blend" else "residual"
        final, weight, valid_mrr = select_validation_weight(frame, model_score, mode)
        formula = (
            "(1-alpha)*z_query(ULTRA)+alpha*z_query(logit(model_probability))"
            if mode == "fixed_blend"
            else "z_query(ULTRA)+beta*z_query(logit(model_probability))"
        )
    metrics = metrics_for_frame(frame, final)
    predictions = frame.copy()
    predictions["base_model_score"] = predictions["score"]
    predictions["score"] = final
    predictions["normalized_ultra_score"] = normalized_ultra_score(frame)
    config = {
        "experiment_family": "context_intervention_suite",
        "stage": "matched_residual_control",
        "dataset": pilot.DATASET,
        "condition": condition,
        "method": method,
        "source_decoder": source_decoder,
        "feature_variant": variant,
        "residual_target": "binary context label; residual is defined in ranking-score space, not as an observed regression target",
        "normalization": "per-query z-score over unfiltered candidates; model probability converted to clipped logit first",
        "formula": formula,
        "validation_selection": "combined filtered validation MRR only; ties choose smallest weight",
        "weight_grid": (ALPHA_GRID.tolist() if method == "fixed_blend" else
                        (BETA_GRID.tolist() if source_decoder is not None else [0.0])),
        "selected_weight": weight,
        "selected_validation_mrr": valid_mrr,
        "outside_top_k": "not applicable; full candidate set scored",
        "seed": seed,
        "git_commit": pilot.git_commit(),
        "hardware": pilot.hardware_info(),
    }
    timings = {"derivation_seconds": 0.0, "peak_gpu_memory_bytes": 0}
    source_manifest = (
        pilot.SUITE_ROOT / pilot.DATASET / condition /
        (("tabpfn_v3" if source_decoder == "tabpfn_v3" and variant == "full"
          else f"{source_decoder}_direct_{variant}")) / str(seed) / "context_manifest.json"
        if source_decoder is not None else None
    )
    manifest = {
        "source_context_manifest": str(source_manifest) if source_manifest else None,
        "condition": condition,
        "feature_variant": variant,
        "candidate_hashes_in_source_manifest": True,
    }
    pilot.atomic_json(run_dir / "config.json", config)
    pilot.atomic_json(run_dir / "metrics.json", metrics)
    pilot.atomic_json(run_dir / "timings.json", timings)
    pilot.atomic_json(run_dir / "context_manifest.json", manifest)
    pilot.atomic_parquet(run_dir / "predictions.parquet", predictions)
    (run_dir / "run.log").write_text(
        f"OK {method} {condition} {variant} seed={seed} "
        f"weight={weight:.3f} mrr={metrics['test']['combined']['mrr']:.9f}\n",
        encoding="utf-8")
    return metrics


def smoke_tests(pa, contexts, candidate_metadata):
    jobs = [
        ("correct_labels", "logistic", "full"),
        ("correct_labels", "hgb", "full"),
        ("permuted_labels", "hgb", "full"),
        ("random_rows_correct_balance", "hgb", "full"),
        ("correct_labels", "mlp", "full"),
        ("correct_labels", "tabpfn_v3", "structural_without_ultra_score"),
        ("correct_labels", "tabpfn_v3", "ultra_score_only"),
    ]
    for condition, decoder, variant in jobs:
        execute_base(pa, 101, condition, decoder, variant, contexts[condition],
                     candidate_metadata, smoke=True)


def full_base_jobs():
    for seed in pilot.SEEDS:
        for decoder in ("logistic", "hgb", "mlp"):
            yield seed, "correct_labels", decoder, "full"
        for condition in ("permuted_labels", "random_rows_correct_balance"):
            yield seed, condition, "hgb", "full"
        for variant in ("structural_without_ultra_score", "ultra_score_only"):
            yield seed, "correct_labels", "hgb", variant
            yield seed, "correct_labels", "tabpfn_v3", variant


def derived_jobs():
    for seed in pilot.SEEDS:
        yield seed, "correct_labels", "ultra_only", "full", None
        yield seed, "correct_labels", "fixed_blend", "full", "tabpfn_v3"
        yield seed, "correct_labels", "logistic_residual", "full", "logistic"
        yield seed, "correct_labels", "hgb_residual", "full", "hgb"
        yield seed, "correct_labels", "mlp_residual", "full", "mlp"
        yield seed, "correct_labels", "tabpfn_residual", "full", "tabpfn_v3"
        for condition in ("permuted_labels", "random_rows_correct_balance"):
            yield seed, condition, "tabpfn_residual", "full", "tabpfn_v3"
            yield seed, condition, "hgb_residual", "full", "hgb"
        for variant in ("structural_without_ultra_score", "ultra_score_only"):
            yield seed, "correct_labels", "tabpfn_residual", variant, "tabpfn_v3"
            yield seed, "correct_labels", "hgb_residual", variant, "hgb"


def method_summary(condition: str, method: str, variant: str) -> dict:
    values = []
    for seed in pilot.SEEDS:
        metrics = json.loads((derived_run_dir(condition, method, variant, seed) /
                              "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((derived_run_dir(condition, method, variant, seed) /
                             "config.json").read_text(encoding="utf-8"))
        values.append({
            "seed": seed,
            "mrr": metrics["test"]["combined"]["mrr"],
            "head_mrr": metrics["test"]["head"]["mrr"],
            "tail_mrr": metrics["test"]["tail"]["mrr"],
            "hits_at_1": metrics["test"]["combined"]["hits_at_1"],
            "hits_at_3": metrics["test"]["combined"]["hits_at_3"],
            "hits_at_10": metrics["test"]["combined"]["hits_at_10"],
            "selected_weight": config["selected_weight"],
        })
    mrr = np.asarray([row["mrr"] for row in values])
    return {
        "condition": condition,
        "method": method,
        "feature_variant": variant,
        "per_seed": values,
        "mean_mrr": float(mrr.mean()),
        "sample_standard_deviation_mrr": float(mrr.std(ddof=1)),
    }


def update_summary():
    summary = json.loads(pilot.SUMMARY_JSON.read_text(encoding="utf-8"))
    main_methods = {
        method: method_summary("correct_labels", method, "full")
        for method in ("ultra_only", "fixed_blend", "logistic_residual",
                       "hgb_residual", "mlp_residual", "tabpfn_residual")
    }
    direct_values = []
    for seed in pilot.SEEDS:
        metrics = json.loads((pilot.SUITE_ROOT / pilot.DATASET / "correct_labels" /
                              "tabpfn_v3" / str(seed) / "metrics.json").read_text())
        direct_values.append(metrics["test"]["combined"]["mrr"])
    main_methods["tabpfn_direct"] = {
        "mean_mrr": float(np.mean(direct_values)),
        "sample_standard_deviation_mrr": float(np.std(direct_values, ddof=1)),
        "per_seed_mrr": dict(zip(map(str, pilot.SEEDS), direct_values)),
    }
    context_ablations = {}
    for model in ("tabpfn_residual", "hgb_residual"):
        context_ablations[model] = {
            condition: method_summary(condition, model, "full")
            for condition in pilot.CONDITIONS
        }
    feature_ablations = {}
    for model in ("tabpfn_residual", "hgb_residual"):
        feature_ablations[model] = {
            variant: method_summary("correct_labels", model, variant)
            for variant in FEATURE_VARIANTS
        }
    best_classical = max(
        ("logistic_residual", "hgb_residual", "mlp_residual"),
        key=lambda method: main_methods[method]["mean_mrr"],
    )
    tabpfn_mean = main_methods["tabpfn_residual"]["mean_mrr"]
    classical_mean = main_methods[best_classical]["mean_mrr"]
    summary["residual_suite_executed"] = True
    summary["baseline_results"] = main_methods
    summary["context_ablations"] = {
        "label_and_row_conditions": context_ablations,
        "feature_information": feature_ablations,
    }
    summary["supported_conclusions"].append(
        "Correct target labels matter under a relation/orientation-marginal-preserving intervention.")
    if tabpfn_mean > classical_mean:
        summary["supported_conclusions"].append(
            f"TabPFN residual exceeds the strongest tested classical residual ({best_classical}) on mean UMLS MRR under matched inputs.")
    else:
        summary["unsupported_conclusions"].append(
            f"TabPFN-specific residual superiority is not supported: {best_classical} matches or exceeds it on UMLS.")
    summary["unsupported_conclusions"].append(
        "Selectivity remains unproven until a strong-ULTRA preservation dataset is evaluated.")
    summary["main_alternative_explanation"] = (
        "Supervised nonlinear reranking using the same target labels and frozen features; "
        "the matched residual controls determine whether this remains sufficient.")
    summary["recommended_next_step"] = (
        "Run Kinships and one protocol-aligned strong-ULTRA preservation dataset only if the UMLS residual advantage over classical controls is material.")
    pilot.atomic_json(pilot.SUMMARY_JSON, summary)
    lines = [
        "# Context Intervention Overnight Summary",
        "",
        "## Pilot",
        "",
        f"PASS: correct minus permuted MRR = {summary['pilot']['decision_details']['primary_comparison']['mean']:.6f}.",
        "",
        "## Matched Residual Controls",
        "",
        "| Method | Mean combined MRR | Seed SD |",
        "|---|---:|---:|",
    ]
    for method, values in main_methods.items():
        lines.append(
            f"| {method} | {values['mean_mrr']:.6f} | "
            f"{values['sample_standard_deviation_mrr']:.6f} |")
    lines.extend([
        "",
        "## Scientific Limit",
        "",
        "The pilot proves that supplied context labels matter under this protocol. "
        "It does not by itself prove selective adaptation or a TabPFN-specific mechanism.",
        "",
    ])
    tmp = pilot.SUMMARY_MD.with_name(f".{pilot.SUMMARY_MD.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, pilot.SUMMARY_MD)
    return summary, best_classical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    existing = json.loads(pilot.SUMMARY_JSON.read_text(encoding="utf-8"))
    if existing["pilot"]["status"] != "PASS":
        raise RuntimeError("residual controls are forbidden because the pilot did not PASS")
    base_jobs = list(full_base_jobs())
    residual_jobs = list(derived_jobs())
    print(
        f"PLAN smoke_jobs=7 base_jobs={len(base_jobs)} derived_jobs={len(residual_jobs)} "
        "estimated_minutes=[35, 55]", flush=True)
    pa = pilot.PrimaryAssets(pilot.DATASET)
    _, candidate_metadata = pilot.candidate_payload(pa)
    contexts_101, metadata_101 = pilot.prepare_seed_contexts(pa, 101)
    if not args.skip_smoke:
        smoke_tests(pa, contexts_101, candidate_metadata)
        print("RESIDUAL CONTROL SMOKES PASSED", flush=True)
    if args.smoke_only:
        return 0

    contexts_by_seed = {101: contexts_101}
    for seed, condition, decoder, variant in base_jobs:
        if seed not in contexts_by_seed:
            contexts_by_seed[seed], _ = pilot.prepare_seed_contexts(pa, seed)
        metrics = execute_base(
            pa, seed, condition, decoder, variant, contexts_by_seed[seed][condition],
            candidate_metadata)
        print(
            f"OK base seed={seed} {condition} {decoder} {variant} "
            f"mrr={metrics['test']['combined']['mrr']:.6f}", flush=True)
    for seed, condition, method, variant, decoder in residual_jobs:
        metrics = save_derived(condition, method, variant, seed, decoder)
        print(
            f"OK derived seed={seed} {condition} {method} {variant} "
            f"mrr={metrics['test']['combined']['mrr']:.6f}", flush=True)
    summary, best_classical = update_summary()
    tabpfn = summary["baseline_results"]["tabpfn_residual"]["mean_mrr"]
    classical = summary["baseline_results"][best_classical]["mean_mrr"]
    print(
        f"RESIDUAL CONTROLS COMPLETE: TabPFN={tabpfn:.6f}, "
        f"best_classical={best_classical}:{classical:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
