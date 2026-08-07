#!/usr/bin/env python3
"""Audit and integrate the corrected Ultra57 logistic ranking matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from analyze_ultra4g_decoder_compatibility import (
    best_interfaces,
    decoder_interactions,
    encoder_contrasts,
    equal_family_summary,
)
from extract_query_conditioned_compatibility import (
    collect_seed_averaged_outcomes,
    ladder_effects,
    reused_query_properties,
)

from relational_tfm.hashing import file_sha256
from relational_tfm.runner import verify_shard

EXPECTED_CELLS = 6_156
EXPECTED_DATASET_ROWS = 57 * 9 * 4
EXPECTED_SCORE_ONLY_CELLS = 57 * 4 * 3
EXPECTED_QUERY_ROWS = 57 * 128
DECODER_LABELS = {
    "logistic": "Logistic",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
}
DYADIC_STRATUM_LABELS = {
    "none": "None",
    "direct_only": "Direct only",
    "reverse_only": "Reverse only",
    "both": "Both",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fixed_panel_bootstrap(
    values: np.ndarray,
    seed: int = 260803,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    random = np.random.default_rng(seed)
    samples = random.choice(
        values,
        size=(20_000, len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_family_blocked(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    dataset_means = (
        frame.groupby(
            ["provenance_family", "dataset", *group_columns],
            observed=True,
        )["delta_reciprocal_rank"]
        .mean()
        .reset_index()
    )
    family_means = (
        dataset_means.groupby(
            ["provenance_family", *group_columns],
            observed=True,
        )["delta_reciprocal_rank"]
        .mean()
        .reset_index()
    )
    rows = []
    for keys, group in family_means.groupby(group_columns, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["delta_reciprocal_rank"].to_numpy(dtype=float)
        low, high = fixed_panel_bootstrap(values)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "families": len(values),
                "equal_family_mean_delta_rr": float(values.mean()),
                "family_bootstrap_low": low,
                "family_bootstrap_high": high,
            }
        )
    return pd.DataFrame(rows)


def build_query_evidence_strata(
    effects: pd.DataFrame,
    properties: pd.DataFrame,
) -> pd.DataFrame:
    query_strata = properties[
        [
            "dataset",
            "query_id",
            "pair_other_relation_count",
            "reverse_pair_relation_count",
        ]
    ].copy()
    direct = query_strata["pair_other_relation_count"].gt(0)
    reverse = query_strata["reverse_pair_relation_count"].gt(0)
    query_strata["dyadic_stratum"] = np.select(
        [direct & reverse, direct, reverse],
        ["both", "direct_only", "reverse_only"],
        default="none",
    )
    contrasts = ["local_given_paths", "paths_minus_dyadic"]
    selected = effects.loc[
        (effects["effect_family"] == "encoder")
        & effects["contrast"].isin(contrasts)
        & (effects["context_mode"] == "original_uniform")
        & effects["decoder"].isin(DECODER_LABELS)
    ].copy()
    selected = selected.merge(
        query_strata[["dataset", "query_id", "dyadic_stratum"]],
        on=["dataset", "query_id"],
        validate="many_to_one",
    )
    summary = summarize_family_blocked(
        selected,
        ["contrast", "decoder", "dyadic_stratum"],
    )
    pooled = (
        selected.groupby(
            ["contrast", "decoder", "dyadic_stratum"],
            observed=True,
        )["delta_reciprocal_rank"]
        .agg(query_count="size", pooled_query_median="median")
        .reset_index()
    )
    summary = summary.merge(
        pooled,
        on=["contrast", "decoder", "dyadic_stratum"],
        validate="one_to_one",
    )
    unique_counts = query_strata.groupby("dyadic_stratum", observed=True).size()
    summary["stratum_label"] = summary["dyadic_stratum"].map(
        lambda value: (
            f"{DYADIC_STRATUM_LABELS[value]}\n(n={int(unique_counts[value]):,})"
        )
    )
    summary["contrast_label"] = summary["contrast"].map(
        {
            "local_given_paths": "Add remaining local block to paths",
            "paths_minus_dyadic": "Replace dyadic evidence with paths",
        }
    )
    summary["decoder"] = summary["decoder"].map(DECODER_LABELS)
    if len(selected) != EXPECTED_QUERY_ROWS * 3 * 2 or len(summary) != 24:
        raise ValueError("query evidence summary has unexpected dimensions")
    return summary


def provenance_maps(
    dataset_metrics: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    identity = dataset_metrics[
        ["dataset", "provenance_family", "transfer_regime"]
    ].drop_duplicates()
    if len(identity) != 57 or identity["dataset"].duplicated().any():
        raise ValueError("existing dataset metadata is not a 57-dataset bijection")
    provenance = dict(
        zip(identity["dataset"], identity["provenance_family"], strict=True)
    )
    regimes = dict(zip(identity["dataset"], identity["transfer_regime"], strict=True))
    return provenance, regimes


def collect_corrected_dataset_metrics(
    run_root: Path,
    provenance: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    registration = read_json(run_root / "registration.json")
    if registration.get("cells") != EXPECTED_CELLS:
        raise ValueError("correction registration has the wrong cell count")
    metric_paths: list[Path] = []
    shard_count = int(registration["shards"])
    for shard_index in range(shard_count):
        shard = run_root / "results" / f"shard-{shard_index:04d}"
        verify_shard(shard)
        metric_paths.extend(sorted(shard.glob("*.metrics.json")))
    if len(metric_paths) != EXPECTED_CELLS:
        raise ValueError("corrected metric matrix is incomplete")

    rows: list[dict[str, Any]] = []
    query_frames: list[pd.DataFrame] = []
    score_only_cells = 0
    exact_native_rank_cells = 0
    maximum_native_rank_difference = 0.0
    policies: set[str] = set()
    for metric_path in metric_paths:
        payload = read_json(metric_path)
        cell = payload["cell"]
        if cell["decoder"] != "logistic" or payload.get("status") != "complete":
            raise ValueError("correction output contains a non-logistic or incomplete cell")
        policies.add(str(payload["prediction_policy"]))
        row = {
            "dataset": cell["dataset"],
            "transfer_regime": cell["transfer_regime"],
            "interface": cell["interface"],
            "decoder": "logistic",
            "context_per_class": int(cell["context_per_class"]),
            "seed": int(cell["seed"]),
            "provenance_family": provenance[cell["dataset"]],
        }
        for direction in ("combined", "head", "tail"):
            row[f"{direction}_mrr"] = float(
                payload["direction_metrics"][direction]["mrr"]
            )
            row[f"native_{direction}_mrr"] = float(
                payload["native_ultra_direction_metrics"][direction]["mrr"]
            )
        rows.append(row)

        prediction_path = metric_path.parent / payload["predictions"]["path"]
        if file_sha256(prediction_path) != payload["predictions"]["sha256"]:
            raise ValueError("corrected prediction checksum changed")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            prediction_arrays = {name: np.asarray(arrays[name]) for name in arrays.files}
        query_frames.append(
            pd.DataFrame(
                {
                    "dataset": cell["dataset"],
                    "provenance_family": provenance[cell["dataset"]],
                    "transfer_regime": cell["transfer_regime"],
                    "interface": cell["interface"],
                    "decoder": "logistic",
                    "context_per_class": int(cell["context_per_class"]),
                    "seed": int(cell["seed"]),
                    "query_id": prediction_arrays["query_ids"],
                    "side_code": prediction_arrays["side_codes"],
                    "target_entity": prediction_arrays["target_entities"],
                    "target_score": prediction_arrays["target_scores"],
                    "rank": prediction_arrays["ranks"],
                    "reciprocal_rank": prediction_arrays["reciprocal_ranks"],
                    "hit_at_1": prediction_arrays["hits_at_1"],
                    "hit_at_3": prediction_arrays["hits_at_3"],
                    "hit_at_10": prediction_arrays["hits_at_10"],
                    "hit_at_50": prediction_arrays["hits_at_50"],
                    "native_ultra_target_score": prediction_arrays[
                        "native_ultra_target_scores"
                    ],
                    "native_ultra_rank": prediction_arrays["native_ultra_ranks"],
                    "native_ultra_reciprocal_rank": prediction_arrays[
                        "native_ultra_reciprocal_ranks"
                    ],
                    "native_ultra_hit_at_1": prediction_arrays[
                        "native_ultra_hits_at_1"
                    ],
                    "native_ultra_hit_at_3": prediction_arrays[
                        "native_ultra_hits_at_3"
                    ],
                    "native_ultra_hit_at_10": prediction_arrays[
                        "native_ultra_hits_at_10"
                    ],
                    "native_ultra_hit_at_50": prediction_arrays[
                        "native_ultra_hits_at_50"
                    ],
                }
            )
        )

        if cell["interface"] == "ultra_score_only":
            score_only_cells += 1
            ranks = np.asarray(prediction_arrays["ranks"], dtype=np.float64)
            native = np.asarray(
                prediction_arrays["native_ultra_ranks"],
                dtype=np.float64,
            )
            difference = float(np.max(np.abs(ranks - native)))
            maximum_native_rank_difference = max(
                maximum_native_rank_difference,
                difference,
            )
            exact_native_rank_cells += int(np.array_equal(ranks, native))

    if score_only_cells != EXPECTED_SCORE_ONLY_CELLS:
        raise ValueError("score-only monotonicity audit is incomplete")
    if exact_native_rank_cells != score_only_cells:
        raise ValueError("score-only logistic does not preserve every native ranking")

    per_seed = pd.DataFrame(rows)
    values = [
        f"{prefix}{direction}_mrr"
        for direction in ("combined", "head", "tail")
        for prefix in ("", "native_")
    ]
    keys = [
        "dataset",
        "transfer_regime",
        "interface",
        "decoder",
        "context_per_class",
        "provenance_family",
    ]
    seed_counts = per_seed.groupby(keys, observed=True)["seed"].nunique()
    if not seed_counts.eq(3).all():
        raise ValueError("at least one corrected dataset cell lacks three seeds")
    dataset = per_seed.groupby(keys, as_index=False, observed=True)[values].mean()
    if len(dataset) != EXPECTED_DATASET_ROWS:
        raise ValueError("corrected dataset matrix has the wrong dimensions")
    query_predictions = pd.concat(query_frames, ignore_index=True)
    expected_query_predictions = EXPECTED_CELLS * 128
    if len(query_predictions) != expected_query_predictions:
        raise ValueError("corrected query-prediction matrix is incomplete")
    return dataset, query_predictions, {
        "registered_cells": EXPECTED_CELLS,
        "verified_shards": shard_count,
        "query_prediction_rows": expected_query_predictions,
        "score_only_cells": score_only_cells,
        "score_only_exact_native_rank_cells": exact_native_rank_cells,
        "maximum_score_only_rank_difference": maximum_native_rank_difference,
        "prediction_policies": sorted(policies),
        "source_commit": registration["source_commit"],
        "plan_sha256": registration["plan_sha256"],
        "registration_sha256": file_sha256(run_root / "registration.json"),
    }


def corrected_query_effects(
    run_root: Path,
    properties_path: Path,
    existing_metrics: pd.DataFrame,
    analysis_plan: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    provenance, regimes = provenance_maps(existing_metrics)
    properties, identity, property_audit = reused_query_properties(
        properties_path,
        provenance,
        regimes,
    )
    outcomes, outcome_audit = collect_seed_averaged_outcomes(
        [run_root],
        EXPECTED_CELLS,
        identity,
        "encoder_ladder",
    )
    # This correction reruns only logistic regression.  Recompute its encoder and
    # context effects here; cross-decoder effects are retained from the complete
    # three-decoder query table below.
    logistic_plan = {**analysis_plan, "decoder_interfaces": []}
    effects = ladder_effects(outcomes, logistic_plan)
    effects = effects.merge(
        properties[
            [
                "dataset",
                "query_id",
                "source_triple_id",
                "side",
                "provenance_family",
                "transfer_regime",
            ]
        ],
        on=["dataset", "query_id"],
        validate="many_to_one",
    )
    if effects["decoder"].ne("logistic").any():
        raise ValueError("corrected query effects include another decoder")
    return effects, {"properties": property_audit, "outcomes": outcome_audit}, properties


def integrate_decoder_matrix(
    existing: pd.DataFrame,
    corrected: pd.DataFrame,
    output: Path,
    correction_audit: dict[str, Any],
    existing_summary: dict[str, Any],
) -> None:
    combined = pd.concat(
        [existing.loc[existing["decoder"] != "logistic"], corrected],
        ignore_index=True,
    )
    keys = ["dataset", "decoder", "interface", "context_per_class"]
    if len(combined) != 57 * 3 * 9 * 4 or combined.duplicated(keys).any():
        raise ValueError("integrated decoder matrix is incomplete")
    equal_family = equal_family_summary(combined)
    family_absolute = (
        combined.groupby(
            ["provenance_family", "decoder", "interface", "context_per_class"],
            as_index=False,
            observed=True,
        )[["combined_mrr", "head_mrr", "tail_mrr"]]
        .mean()
    )
    equal_family_overall = (
        family_absolute.groupby(
            ["decoder", "interface", "context_per_class"],
            as_index=False,
            observed=True,
        )[["combined_mrr", "head_mrr", "tail_mrr"]]
        .mean()
    )
    equal_family_overall["family_count"] = family_absolute["provenance_family"].nunique()
    effects, contrast_summary = encoder_contrasts(combined)
    interactions = decoder_interactions(effects)
    winners = best_interfaces(equal_family)
    artifacts = {
        "dataset_metrics.csv": combined.sort_values(keys, ignore_index=True),
        "equal_family_absolute_mrr.csv": equal_family,
        "equal_family_overall_mrr.csv": equal_family_overall,
        "encoder_contrast_by_dataset.csv": effects,
        "encoder_contrast_summary.csv": contrast_summary,
        "decoder_interaction_summary.csv": interactions,
        "best_interface_by_regime.csv": winners,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False)
    summary = dict(existing_summary)
    summary.update(
        {
            "schema_version": 2,
            "status": "COMPLETE_AND_AUDITED",
            "scientific_role": (
                "post-hoc fixed-panel encoder-by-decoder compatibility "
                "characterization"
            ),
            "datasets": 57,
            "decoders": ["tabpfn", "logistic", "catboost"],
            "contexts_per_class": [20, 50, 100, 200],
            "logistic_ranking_correction": correction_audit,
        }
    )
    write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 2,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--existing-dataset-metrics", type=Path, required=True)
    parser.add_argument("--existing-query-effects", type=Path, required=True)
    parser.add_argument("--query-properties", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decoder-output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    existing_summary_path = args.existing_dataset_metrics.resolve().parent / "summary.json"
    existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
    existing_metrics = pd.read_csv(args.existing_dataset_metrics)
    provenance, _ = provenance_maps(existing_metrics)
    corrected_metrics, query_predictions, correction_audit = collect_corrected_dataset_metrics(
        run_root,
        provenance,
    )
    plan = yaml.safe_load(args.analysis_plan.read_text(encoding="utf-8"))
    effects, query_audit, properties = corrected_query_effects(
        run_root,
        args.query_properties.resolve(),
        existing_metrics,
        plan,
    )
    old_effects = pd.read_csv(args.existing_query_effects, low_memory=False)
    retained = old_effects.loc[
        ~(
            (old_effects["stream"] == "encoder_ladder")
            & (old_effects["effect_family"] == "encoder")
            & (old_effects["decoder"] == "logistic")
        )
    ]
    corrected_encoder = effects.loc[effects["effect_family"] == "encoder"]
    query_evidence = build_query_evidence_strata(
        pd.concat([retained, corrected_encoder], ignore_index=True),
        properties,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corrected_metrics.to_csv(output / "corrected_logistic_dataset_metrics.csv", index=False)
    query_predictions.to_csv(
        output / "corrected_logistic_query_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    effects.to_csv(
        output / "corrected_logistic_query_effects.csv.gz",
        index=False,
        compression="gzip",
    )
    query_evidence.to_csv(output / "query_evidence_strata.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": "implementation correction of binary ranking scores",
        "ranking_audit": correction_audit,
        "query_audit": query_audit,
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "audit.json":
            audit["outputs"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    write_json(output / "audit.json", audit)
    integrate_decoder_matrix(
        existing_metrics,
        corrected_metrics,
        args.decoder_output.resolve(),
        correction_audit,
        existing_summary,
    )
    print(json.dumps(correction_audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
