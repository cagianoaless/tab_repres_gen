#!/usr/bin/env python3
"""Analyze relation coverage, relation-ID use, and decoder compatibility on Ultra57."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.runner import verify_shard

try:
    from scripts.analyze_rfmlink_ultra57_q64_complete57 import provenance_lookup
except ModuleNotFoundError:
    from analyze_rfmlink_ultra57_q64_complete57 import provenance_lookup

ASSET_ID = "rfmlink_ultra57_relation_context_v1"
SOURCE_COMMIT = "a48dd52dfdedcf38428c489909b1f31be27ca3d9"
STUDY_IDS = {
    "cpu": "rfmlink_ultra57_relation_context_cpu_decoders_v1",
    "tabpfn": "rfmlink_ultra57_relation_context_gpu_tabpfn_v1",
}
EXPECTED_CELLS = {"cpu": 2_736, "tabpfn": 1_368}
DECODERS = ("catboost", "logistic", "tabpfn")
INTERFACES = (
    "b0_local_only",
    "b0_local_no_relation_id",
    "b0_local_ultra",
    "b0_local_ultra_no_relation_id",
)
CONTEXT_MODES = (
    "frequency_random_filtered_uniform",
    "relation_matched_filtered_uniform",
)
SEEDS = (101, 102, 103)
DIRECTIONS = ("combined", "head", "tail")
TIE_BAND = 0.005
EXPECTED_DATASETS = 57
EXPECTED_FAMILIES = 16


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_metrics(
    context_relations: np.ndarray,
    evaluation_relations: np.ndarray,
) -> dict[str, float | int]:
    context = np.asarray(context_relations, dtype=np.int64).reshape(-1)
    evaluation = np.asarray(evaluation_relations, dtype=np.int64).reshape(-1)
    if len(context) == 0 or len(evaluation) == 0:
        raise ValueError("relation distributions must be non-empty")
    support = np.union1d(context, evaluation)
    context_counts = np.asarray([(context == value).sum() for value in support])
    evaluation_counts = np.asarray(
        [(evaluation == value).sum() for value in support]
    )
    context_probability = context_counts / context_counts.sum()
    evaluation_probability = evaluation_counts / evaluation_counts.sum()
    evaluation_support = evaluation_counts > 0
    covered = context_counts[evaluation_support] > 0
    positive_context_probability = context_probability[context_probability > 0]
    normalized_entropy = (
        -float(np.sum(positive_context_probability * np.log(positive_context_probability)))
        / np.log(len(positive_context_probability))
        if len(positive_context_probability) > 1
        else 0.0
    )
    return {
        "context_unique_relations": int(np.count_nonzero(context_counts)),
        "evaluation_unique_relations": int(np.count_nonzero(evaluation_counts)),
        "evaluation_relation_type_coverage": float(covered.mean()),
        "evaluation_relation_mass_coverage": float(
            evaluation_probability[evaluation_support][covered].sum()
        ),
        "relation_distribution_tv": float(
            0.5 * np.abs(context_probability - evaluation_probability).sum()
        ),
        "context_relation_gini_impurity": float(
            1.0 - np.square(context_probability).sum()
        ),
        "context_relation_normalized_entropy": normalized_entropy,
        "off_target_directional_rows": int(
            np.count_nonzero(~np.isin(context, evaluation))
        ),
    }


def collect_context_audits(manifest_path: Path) -> pd.DataFrame:
    manifest = read_json(manifest_path)
    if manifest.get("asset_id") != ASSET_ID:
        raise ValueError("relation-context manifest has the wrong asset ID")
    datasets = manifest.get("datasets", {})
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError("relation-context manifest is not the 57-dataset panel")
    lookup = provenance_lookup()
    rows: list[dict[str, Any]] = []
    for dataset, metadata in sorted(datasets.items()):
        archive_path = manifest_path.parent / f"{dataset}.npz"
        if file_sha256(archive_path) != metadata["archive_sha256"]:
            raise ValueError(f"archive hash mismatch for {dataset}")
        with np.load(archive_path, allow_pickle=False) as archive:
            evaluation_relations = archive["direction_relations"]
            for mode in CONTEXT_MODES:
                audit_key = (
                    "frequency_random_audits"
                    if mode == CONTEXT_MODES[0]
                    else "relation_matched_audits"
                )
                audits = metadata[audit_key]
                if set(map(int, audits)) != set(SEEDS):
                    raise ValueError(f"{dataset}/{mode} has the wrong seeds")
                for seed in SEEDS:
                    prefix = f"context_{mode}_{seed}"
                    labels = archive[f"{prefix}_y"]
                    features = archive[f"{prefix}_X"]
                    positive_relations = features[labels == 1, 0].astype(np.int64)
                    if len(positive_relations) != 200:
                        raise ValueError(
                            f"{dataset}/{mode}/{seed} does not have 200 positives"
                        )
                    metrics = _distribution_metrics(
                        positive_relations, evaluation_relations
                    )
                    recorded = audits[str(seed)]
                    if int(recorded["q64_overlap"]) != 0:
                        raise ValueError(f"{dataset}/{mode}/{seed} overlaps q64")
                    if int(recorded["positive_directional_rows"]) != 200:
                        raise ValueError("context-positive row count drift")
                    if mode == CONTEXT_MODES[1]:
                        for key in (
                            "evaluation_relation_type_coverage",
                            "evaluation_relation_mass_coverage",
                            "relation_distribution_tv",
                        ):
                            if not np.isclose(
                                float(metrics[key]), float(recorded[key]), atol=1e-12
                            ):
                                raise ValueError(
                                    f"{dataset}/{seed} manifest drift at {key}"
                                )
                        fallback = metrics["off_target_directional_rows"] // 2
                        if fallback != int(
                            recorded.get("off_target_relation_fallback_triples", 0)
                        ):
                            raise ValueError(
                                f"{dataset}/{seed} fallback-count drift"
                            )
                    rows.append(
                        {
                            "dataset": dataset,
                            "provenance_family": lookup[dataset],
                            "transfer_regime": metadata["transfer_regime"],
                            "context_mode": mode,
                            "seed": seed,
                            "context_source_split": metadata[
                                "context_source_split"
                            ],
                            "q64_overlap": int(recorded["q64_overlap"]),
                            **metrics,
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "context_mode", "seed"], ignore_index=True
    )
    expected = EXPECTED_DATASETS * len(CONTEXT_MODES) * len(SEEDS)
    if len(frame) != expected or frame.duplicated(
        ["dataset", "context_mode", "seed"]
    ).any():
        raise ValueError("context audit is incomplete or duplicated")
    return frame


def collect_cells(
    run_root: Path,
    manifest_path: Path,
    *,
    profile: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registration = read_json(run_root / "registration.json")
    plan = read_json(run_root / "plan.json")
    expected_cells = EXPECTED_CELLS[profile]
    expected_decoders = {"tabpfn"} if profile == "tabpfn" else {
        "catboost",
        "logistic",
    }
    if registration["source_commit"] != SOURCE_COMMIT:
        raise ValueError(f"{profile} source commit differs from the frozen plan")
    if registration["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError(f"{profile} registration and plan hashes differ")
    if registration["asset_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError(f"{profile} asset manifest hash differs")
    if int(registration["cells"]) != expected_cells:
        raise ValueError(f"{profile} registration cell count differs")
    planned = {cell["cell_id"]: cell for cell in plan["cells"]}
    if len(planned) != expected_cells:
        raise ValueError(f"{profile} plan cells are missing or duplicated")
    if {cell["study_id"] for cell in planned.values()} != {STUDY_IDS[profile]}:
        raise ValueError(f"{profile} plan has the wrong study ID")
    if {cell["decoder"] for cell in planned.values()} != expected_decoders:
        raise ValueError(f"{profile} decoder panel differs")

    shard_dirs = sorted((run_root / "results").glob("shard-*"))
    if len(shard_dirs) != int(registration["shards"]):
        raise ValueError(
            f"{profile} shards incomplete: {len(shard_dirs)}/"
            f"{registration['shards']}"
        )
    observed: set[str] = set()
    shard_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        verified = verify_shard(shard_dir)
        if verified.get("plan_sha256") != registration["plan_sha256"]:
            raise ValueError(f"{shard_dir.name} has the wrong plan hash")
        shard_hashes[shard_dir.name] = file_sha256(
            shard_dir / "shard_manifest.json"
        )
        for cell_id in verified["cell_ids"]:
            if cell_id in observed:
                raise ValueError(f"duplicate result cell {cell_id}")
            metric = read_json(shard_dir / f"{cell_id}.metrics.json")
            cell = metric["cell"]
            if cell != planned.get(cell_id) or metric.get("status") != "complete":
                raise ValueError(f"result payload differs from plan for {cell_id}")
            observed.add(cell_id)
            row: dict[str, Any] = {
                "cell_id": cell_id,
                "dataset": cell["dataset"],
                "transfer_regime": cell["transfer_regime"],
                "decoder": cell["decoder"],
                "interface": cell["interface"],
                "context_mode": cell["context_mode"],
                "seed": int(cell["seed"]),
                "context_per_class": int(cell["context_per_class"]),
                "feature_count": int(metric["feature_count"]),
                "duration_seconds": float(metric["duration_seconds"]),
                "decoder_fit_status": metric["decoder_fit_status"],
                "decoder_prediction_status": metric["decoder_prediction_status"],
            }
            for direction in DIRECTIONS:
                row[f"{direction}_mrr"] = float(
                    metric["direction_metrics"][direction]["mrr"]
                )
                row[f"native_{direction}_mrr"] = float(
                    metric["native_ultra_direction_metrics"][direction]["mrr"]
                )
            rows.append(row)
    if observed != set(planned):
        raise ValueError(f"{profile} result-plan cell mismatch")
    frame = pd.DataFrame(rows)
    lookup = provenance_lookup()
    frame["provenance_family"] = frame["dataset"].map(lookup)
    if frame["provenance_family"].isna().any():
        raise ValueError("at least one dataset lacks a provenance family")
    audit = {
        "profile": profile,
        "study_id": STUDY_IDS[profile],
        "source_commit": registration["source_commit"],
        "plan_sha256": registration["plan_sha256"],
        "cells": len(frame),
        "shards": len(shard_dirs),
        "shard_manifest_sha256": shard_hashes,
        "status": "PASS",
    }
    return frame, audit


def validate_joint_cells(frame: pd.DataFrame) -> None:
    expected = sum(EXPECTED_CELLS.values())
    keys = ["dataset", "decoder", "interface", "context_mode", "seed"]
    if len(frame) != expected or frame.duplicated(keys).any():
        raise ValueError("joint result panel is incomplete or duplicated")
    if set(frame["dataset"]) != set(provenance_lookup()):
        raise ValueError("joint result panel is not the frozen 57-dataset panel")
    if set(frame["decoder"]) != set(DECODERS):
        raise ValueError("joint decoder panel differs")
    if set(frame["interface"]) != set(INTERFACES):
        raise ValueError("joint interface panel differs")
    if set(frame["context_mode"]) != set(CONTEXT_MODES):
        raise ValueError("joint context-mode panel differs")
    if set(frame["seed"]) != set(SEEDS):
        raise ValueError("joint seed panel differs")
    if (frame["decoder_fit_status"] != "succeeded").any() or (
        frame["decoder_prediction_status"] != "succeeded"
    ).any():
        raise ValueError("at least one decoder cell did not succeed")
    metrics = [
        f"{prefix}{direction}_mrr"
        for prefix in ("", "native_")
        for direction in DIRECTIONS
    ]
    if not np.isfinite(frame[metrics].to_numpy(dtype=float)).all():
        raise ValueError("joint result panel contains non-finite metrics")
    native_range = frame.groupby("dataset", observed=True)[
        [f"native_{direction}_mrr" for direction in DIRECTIONS]
    ].agg(lambda values: float(values.max() - values.min()))
    if (native_range.to_numpy(dtype=float) > 1e-12).any():
        raise ValueError("native ULTRA metrics differ across paired cells")


def dataset_means(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        f"{prefix}{direction}_mrr"
        for prefix in ("", "native_")
        for direction in DIRECTIONS
    ]
    keys = [
        "dataset",
        "provenance_family",
        "transfer_regime",
        "decoder",
        "interface",
        "context_mode",
    ]
    return (
        frame.groupby(keys, as_index=False, observed=True)[metrics]
        .mean()
        .sort_values(keys, ignore_index=True)
    )


def _add_effect(
    rows: list[dict[str, Any]],
    index: tuple[str, str, str],
    *,
    effect: float,
    contrast_family: str,
    contrast_id: str,
    direction: str,
    decoder: str,
    interface: str,
    treatment: str,
    reference: str,
) -> None:
    dataset, family, regime = index
    rows.append(
        {
            "dataset": dataset,
            "provenance_family": family,
            "transfer_regime": regime,
            "contrast_family": contrast_family,
            "contrast_id": contrast_id,
            "direction": direction,
            "decoder": decoder,
            "interface": interface,
            "treatment": treatment,
            "reference": reference,
            "effect": float(effect),
        }
    )


def build_effects(dataset: pd.DataFrame) -> pd.DataFrame:
    index_keys = ["dataset", "provenance_family", "transfer_regime"]
    interface_pairs = {
        "without_ultra": ("b0_local_only", "b0_local_no_relation_id"),
        "with_ultra": ("b0_local_ultra", "b0_local_ultra_no_relation_id"),
    }
    score_pairs = {
        "with_relation_id": ("b0_local_ultra", "b0_local_only"),
        "without_relation_id": (
            "b0_local_ultra_no_relation_id",
            "b0_local_no_relation_id",
        ),
    }
    decoder_pairs = (
        ("tabpfn", "catboost"),
        ("tabpfn", "logistic"),
        ("catboost", "logistic"),
    )
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        pivot = dataset.pivot(
            index=index_keys,
            columns=["decoder", "interface", "context_mode"],
            values=f"{direction}_mrr",
        )
        if pivot.shape[0] != EXPECTED_DATASETS or pivot.isna().any().any():
            raise ValueError(f"configuration matrix is incomplete for {direction}")
        for index, values in pivot.iterrows():
            context_effect: dict[tuple[str, str], float] = {}
            for decoder in DECODERS:
                for interface in INTERFACES:
                    effect = (
                        values[(decoder, interface, CONTEXT_MODES[1])]
                        - values[(decoder, interface, CONTEXT_MODES[0])]
                    )
                    context_effect[(decoder, interface)] = float(effect)
                    _add_effect(
                        rows,
                        index,
                        effect=effect,
                        contrast_family="context_sampling",
                        contrast_id=f"sampling|{decoder}|{interface}",
                        direction=direction,
                        decoder=decoder,
                        interface=interface,
                        treatment=CONTEXT_MODES[1],
                        reference=CONTEXT_MODES[0],
                    )

                for score_status, (with_id, without_id) in interface_pairs.items():
                    for mode in CONTEXT_MODES:
                        effect = values[(decoder, with_id, mode)] - values[
                            (decoder, without_id, mode)
                        ]
                        _add_effect(
                            rows,
                            index,
                            effect=effect,
                            contrast_family="relation_id",
                            contrast_id=(
                                f"relation_id|{decoder}|{score_status}|{mode}"
                            ),
                            direction=direction,
                            decoder=decoder,
                            interface=score_status,
                            treatment=with_id,
                            reference=without_id,
                        )
                    interaction = (
                        context_effect[(decoder, with_id)]
                        - context_effect[(decoder, without_id)]
                    )
                    _add_effect(
                        rows,
                        index,
                        effect=interaction,
                        contrast_family="sampling_x_relation_id",
                        contrast_id=f"sampling_x_relation_id|{decoder}|{score_status}",
                        direction=direction,
                        decoder=decoder,
                        interface=score_status,
                        treatment="sampling_effect_with_relation_id",
                        reference="sampling_effect_without_relation_id",
                    )

                for id_status, (with_score, without_score) in score_pairs.items():
                    for mode in CONTEXT_MODES:
                        effect = values[(decoder, with_score, mode)] - values[
                            (decoder, without_score, mode)
                        ]
                        _add_effect(
                            rows,
                            index,
                            effect=effect,
                            contrast_family="ultra_score",
                            contrast_id=f"ultra_score|{decoder}|{id_status}|{mode}",
                            direction=direction,
                            decoder=decoder,
                            interface=id_status,
                            treatment=with_score,
                            reference=without_score,
                        )
                    interaction = (
                        context_effect[(decoder, with_score)]
                        - context_effect[(decoder, without_score)]
                    )
                    _add_effect(
                        rows,
                        index,
                        effect=interaction,
                        contrast_family="sampling_x_ultra_score",
                        contrast_id=f"sampling_x_ultra_score|{decoder}|{id_status}",
                        direction=direction,
                        decoder=decoder,
                        interface=id_status,
                        treatment="sampling_effect_with_ultra_score",
                        reference="sampling_effect_without_ultra_score",
                    )

            for interface in INTERFACES:
                for left, right in decoder_pairs:
                    effect = context_effect[(left, interface)] - context_effect[
                        (right, interface)
                    ]
                    _add_effect(
                        rows,
                        index,
                        effect=effect,
                        contrast_family="decoder_x_sampling",
                        contrast_id=f"decoder_x_sampling|{left}-{right}|{interface}",
                        direction=direction,
                        decoder=f"{left}-{right}",
                        interface=interface,
                        treatment=f"sampling_effect_{left}",
                        reference=f"sampling_effect_{right}",
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["contrast_id", "direction", "dataset"], ignore_index=True
    )
    expected_contrasts = 60
    expected_rows = expected_contrasts * len(DIRECTIONS) * EXPECTED_DATASETS
    if len(frame) != expected_rows or frame["contrast_id"].nunique() != expected_contrasts:
        raise ValueError("effect family is incomplete")
    return frame


def summarize_effects(
    effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = (
        effects.groupby(
            [
                "contrast_family",
                "contrast_id",
                "direction",
                "decoder",
                "interface",
                "treatment",
                "reference",
                "provenance_family",
            ],
            as_index=False,
            observed=True,
        )["effect"]
        .mean()
        .sort_values(["contrast_id", "direction", "provenance_family"])
    )
    rows: list[dict[str, Any]] = []
    group_keys = [
        "contrast_family",
        "contrast_id",
        "direction",
        "decoder",
        "interface",
        "treatment",
        "reference",
    ]
    for key, group in effects.groupby(group_keys, observed=True):
        family_values = family.loc[
            (family[group_keys] == pd.Series(key, index=group_keys)).all(axis=1),
            "effect",
        ].to_numpy(dtype=float)
        dataset_values = group["effect"].to_numpy(dtype=float)
        leave_one_out = np.asarray(
            [np.delete(family_values, i).mean() for i in range(len(family_values))]
        )
        row = dict(zip(group_keys, key, strict=True))
        row.update(
            {
                "equal_family_effect": float(family_values.mean()),
                "equal_dataset_effect": float(dataset_values.mean()),
                "families": len(family_values),
                "datasets": len(dataset_values),
                "dataset_wins": int((dataset_values > TIE_BAND).sum()),
                "dataset_ties": int((np.abs(dataset_values) <= TIE_BAND).sum()),
                "dataset_losses": int((dataset_values < -TIE_BAND).sum()),
                "leave_one_family_out_min": float(leave_one_out.min()),
                "leave_one_family_out_max": float(leave_one_out.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["contrast_id", "direction"], ignore_index=True
    ), family


def _t_statistic(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    statistic = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0.0,
    )
    constant = (standard_errors == 0.0) & (means != 0.0)
    statistic[constant] = np.sign(means[constant]) * np.inf
    return statistic


def exact_joint_max_t(family: pd.DataFrame) -> pd.DataFrame:
    selected = family.loc[family["direction"] == "combined"]
    matrix = selected.pivot(
        index="provenance_family", columns="contrast_id", values="effect"
    ).sort_index(axis=1)
    if matrix.shape != (EXPECTED_FAMILIES, 60) or matrix.isna().any().any():
        raise ValueError("combined family-effect matrix is incomplete")
    base = matrix.to_numpy(dtype=float)
    observed = _t_statistic(base)
    raw_counts = np.zeros(base.shape[1], dtype=np.int64)
    adjusted_counts = np.zeros(base.shape[1], dtype=np.int64)
    patterns = 0
    for signs in itertools.product((-1.0, 1.0), repeat=base.shape[0]):
        statistic = np.abs(
            _t_statistic(base * np.asarray(signs, dtype=float)[:, None])
        )
        maximum = statistic.max()
        raw_counts += statistic >= np.abs(observed) - 1e-12
        adjusted_counts += maximum >= np.abs(observed) - 1e-12
        patterns += 1
    return pd.DataFrame(
        {
            "contrast_id": matrix.columns,
            "observed_t": observed,
            "exact_two_sided_p": raw_counts / patterns,
            "exact_max_t_adjusted_p": adjusted_counts / patterns,
            "sign_patterns": patterns,
            "joint_contrasts": matrix.shape[1],
        }
    )


def configuration_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    keys = ["decoder", "interface", "context_mode"]
    family = dataset.groupby(
        keys + ["provenance_family"], as_index=False, observed=True
    )[[f"{direction}_mrr" for direction in DIRECTIONS]].mean()
    rows: list[dict[str, Any]] = []
    for key, group in dataset.groupby(keys, observed=True):
        family_group = family.loc[
            (family[keys] == pd.Series(key, index=keys)).all(axis=1)
        ]
        row = dict(zip(keys, key, strict=True))
        for direction in DIRECTIONS:
            row[f"equal_dataset_{direction}_mrr"] = float(
                group[f"{direction}_mrr"].mean()
            )
            row[f"equal_family_{direction}_mrr"] = float(
                family_group[f"{direction}_mrr"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def distribution_changes(context: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "relation_distribution_tv",
        "evaluation_relation_type_coverage",
        "evaluation_relation_mass_coverage",
        "context_relation_gini_impurity",
        "context_relation_normalized_entropy",
        "off_target_directional_rows",
    )
    averaged = context.groupby(
        ["dataset", "provenance_family", "transfer_regime", "context_mode"],
        as_index=False,
        observed=True,
    )[list(metrics)].mean()
    rows: list[dict[str, Any]] = []
    for index, group in averaged.groupby(
        ["dataset", "provenance_family", "transfer_regime"], observed=True
    ):
        values = group.set_index("context_mode")
        if set(values.index) != set(CONTEXT_MODES):
            raise ValueError("context-distribution pair is incomplete")
        row = dict(
            zip(
                ["dataset", "provenance_family", "transfer_regime"],
                index,
                strict=True,
            )
        )
        for metric in metrics:
            reference = float(values.loc[CONTEXT_MODES[0], metric])
            matched = float(values.loc[CONTEXT_MODES[1], metric])
            row[f"reference_{metric}"] = reference
            row[f"matched_{metric}"] = matched
            row[f"matched_minus_reference_{metric}"] = matched - reference
        row["tv_reduction"] = (
            row["reference_relation_distribution_tv"]
            - row["matched_relation_distribution_tv"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset", ignore_index=True)


def sampling_moderators(
    effects: pd.DataFrame, changes: pd.DataFrame
) -> pd.DataFrame:
    selected = effects.loc[
        (effects["contrast_family"] == "context_sampling")
        & (effects["direction"] == "combined")
    ].merge(
        changes,
        on=["dataset", "provenance_family", "transfer_regime"],
        validate="many_to_one",
    )
    moderators = (
        "tv_reduction",
        "matched_minus_reference_evaluation_relation_type_coverage",
        "matched_minus_reference_evaluation_relation_mass_coverage",
        "matched_minus_reference_context_relation_gini_impurity",
        "matched_minus_reference_context_relation_normalized_entropy",
        "matched_minus_reference_off_target_directional_rows",
    )
    rows: list[dict[str, Any]] = []
    for (decoder, interface), group in selected.groupby(
        ["decoder", "interface"], observed=True
    ):
        for moderator in moderators:
            dataset_rho = group[moderator].rank().corr(group["effect"].rank())
            family = group.groupby("provenance_family", observed=True)[
                [moderator, "effect"]
            ].mean()
            family_rho = family[moderator].rank().corr(family["effect"].rank())
            rows.append(
                {
                    "decoder": decoder,
                    "interface": interface,
                    "moderator": moderator,
                    "dataset_spearman_rho": float(dataset_rho),
                    "family_spearman_rho": float(family_rho),
                    "datasets": len(group),
                    "families": len(family),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["decoder", "interface", "moderator"], ignore_index=True
    )


def write_readme(
    path: Path,
    summary: pd.DataFrame,
    exact: pd.DataFrame,
) -> None:
    primary = summary.loc[
        (summary["contrast_family"] == "context_sampling")
        & (summary["direction"] == "combined")
    ].copy()
    if "exact_max_t_adjusted_p" not in primary:
        primary = primary.merge(exact, on="contrast_id", validate="one_to_one")
    rows = [
        "# Relation-context compatibility",
        "",
        "Post-hoc fixed-panel factorial on 57 knowledge graphs. Evaluation rows, "
        "q64 candidates, filtered masks, feature matrices, context size, and "
        "negative sampling are fixed; only positive relation coverage changes.",
        "",
        "The simultaneous sensitivity uses exact family-level sign flips and "
        "max-|t| adjustment over all 60 combined-direction contrasts.",
        "",
        "## Relation-matched minus frequency-random context",
        "",
        "| Decoder | Interface | Equal-family MRR | W/T/L | max-|t| p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary.sort_values(["decoder", "interface"]).itertuples():
        rows.append(
            f"| {row.decoder} | {row.interface} | "
            f"{row.equal_family_effect:+.4f} | "
            f"{row.dataset_wins}/{row.dataset_ties}/{row.dataset_losses} | "
            f"{row.exact_max_t_adjusted_p:.5f} |"
        )
    rows.extend(
        [
            "",
            "Full cell, dataset, family, distribution, interaction, and "
            "multiplicity tables are stored as CSV files in this directory.",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn-run-root", type=Path, required=True)
    parser.add_argument("--cpu-run-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    context = collect_context_audits(args.asset_manifest)
    cpu, cpu_audit = collect_cells(
        args.cpu_run_root, args.asset_manifest, profile="cpu"
    )
    tabpfn, tabpfn_audit = collect_cells(
        args.tabpfn_run_root, args.asset_manifest, profile="tabpfn"
    )
    cells = pd.concat([cpu, tabpfn], ignore_index=True).sort_values(
        ["dataset", "decoder", "interface", "context_mode", "seed"],
        ignore_index=True,
    )
    validate_joint_cells(cells)
    dataset = dataset_means(cells)
    effects = build_effects(dataset)
    summary, family = summarize_effects(effects)
    exact = exact_joint_max_t(family)
    summary = summary.merge(
        exact,
        on="contrast_id",
        how="left",
        validate="many_to_one",
    )
    configurations = configuration_summary(dataset)
    changes = distribution_changes(context)
    moderators = sampling_moderators(effects, changes)

    outputs = {
        "cell_metrics.csv": cells,
        "dataset_configuration_metrics.csv": dataset,
        "configuration_summary.csv": configurations,
        "context_distribution_audit.csv": context,
        "context_distribution_changes.csv": changes,
        "contrast_by_dataset.csv": effects,
        "contrast_by_family.csv": family,
        "contrast_summary.csv": summary,
        "combined_exact_max_t.csv": exact,
        "sampling_effect_moderators.csv": moderators,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)
    write_readme(args.output / "README.md", summary, exact)

    audit = {
        "analysis": "rfmlink_ultra57_relation_context_compatibility_v1",
        "analysis_commit": args.analysis_commit,
        "scientific_status": "post-hoc fixed-panel factorial",
        "asset_id": ASSET_ID,
        "asset_manifest_sha256": file_sha256(args.asset_manifest),
        "cells": len(cells),
        "datasets": cells["dataset"].nunique(),
        "provenance_families": cells["provenance_family"].nunique(),
        "decoders": sorted(cells["decoder"].unique()),
        "interfaces": sorted(cells["interface"].unique()),
        "context_modes": sorted(cells["context_mode"].unique()),
        "combined_contrasts": exact["contrast_id"].nunique(),
        "multiplicity": (
            "exact joint family sign-flip max-|t| over all 60 "
            "combined-direction contrasts"
        ),
        "tie_band": TIE_BAND,
        "cpu_run": cpu_audit,
        "tabpfn_run": tabpfn_audit,
        "status": "PASS",
    }
    write_json(args.output / "completion_audit.json", audit)
    files = sorted(
        path
        for path in args.output.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    write_json(
        args.output / "analysis_manifest.json",
        {
            "analysis": audit["analysis"],
            "analysis_commit": args.analysis_commit,
            "scientific_status": audit["scientific_status"],
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in files
            },
        },
    )


if __name__ == "__main__":
    main()
