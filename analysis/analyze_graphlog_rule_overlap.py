#!/usr/bin/env python3
"""Dependence-aware fixed-panel analysis of the 57 GraphLog worlds."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json

CLUSTER_COUNTS = (6, 8, 10, 12, 15, 20)
PRIMARY_CONTEXT = 40
PRIMARY_CONTRAST = "shared_multiset_minus_absolute"
ENCODERS = ("relation_multiset", "ordered_typed_path")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(comb(trials, value) for value in range(min(wins, losses) + 1))
    return float(min(1.0, 2.0 * tail / (2**trials)))


def load_rule_sets(asset: Path) -> dict[str, frozenset[str]]:
    output: dict[str, frozenset[str]] = {}
    with zipfile.ZipFile(asset) as archive:
        for member in archive.namelist():
            parts = member.split("/")
            if (
                len(parts) == 4
                and parts[0] == "graphlog_v1.1"
                and parts[2].startswith("rule_")
                and parts[3] == f"rules_{parts[2][5:]}.json"
            ):
                world = parts[2]
                if world in output:
                    raise ValueError(f"duplicate rule file for {world}")
                mapping = json.loads(archive.read(member))
                output[world] = frozenset(
                    f"{body}=>{head}" for body, head in mapping.items()
                )
    expected = {f"rule_{index}" for index in range(57)}
    if set(output) != expected:
        raise ValueError("GraphLog archive does not contain exactly 57 world rule sets")
    if {len(rules) for rules in output.values()} != {20}:
        raise ValueError("GraphLog worlds do not contain exactly 20 rules")
    return output


def rule_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def similarity_matrix(
    rule_sets: dict[str, frozenset[str]],
) -> tuple[list[str], np.ndarray]:
    worlds = sorted(rule_sets, key=lambda item: int(item.split("_")[1]))
    matrix = np.eye(len(worlds), dtype=np.float64)
    for row, left in enumerate(worlds):
        for column in range(row + 1, len(worlds)):
            value = rule_jaccard(rule_sets[left], rule_sets[worlds[column]])
            matrix[row, column] = value
            matrix[column, row] = value
    return worlds, matrix


def cluster_assignments(
    worlds: list[str], similarity: np.ndarray
) -> dict[int, dict[str, int]]:
    distances = np.clip(1.0 - similarity, 0.0, 1.0)
    tree = linkage(squareform(distances, checks=True), method="average")
    return {
        count: {
            world: int(label)
            for world, label in zip(
                worlds,
                fcluster(tree, t=count, criterion="maxclust"),
                strict=True,
            )
        }
        for count in CLUSTER_COUNTS
    }


def _world_covariates(
    *,
    worlds: list[str],
    similarity: np.ndarray,
    representation_path: Path,
) -> pd.DataFrame:
    representation = pd.read_csv(representation_path)
    representation = representation.loc[
        representation["encoder"].isin(ENCODERS)
    ].copy()
    pivot = representation.pivot(
        index="world",
        columns="encoder",
        values=[
            "evaluation_rows",
            "evaluation_classes",
            "balanced_accuracy_chance",
            "accuracy_ceiling",
            "singleton_fraction",
            "collision_coverage",
        ],
    )
    rows = []
    for index, world in enumerate(worlds):
        other = np.delete(similarity[index], index)
        row: dict[str, Any] = {
            "world": world,
            "mean_rule_jaccard": float(other.mean()),
            "max_rule_jaccard": float(other.max()),
            "min_rule_jaccard": float(other.min()),
        }
        for encoder, prefix in (
            ("relation_multiset", "multiset"),
            ("ordered_typed_path", "absolute"),
        ):
            for column in (
                "evaluation_rows",
                "evaluation_classes",
                "balanced_accuracy_chance",
                "accuracy_ceiling",
                "singleton_fraction",
                "collision_coverage",
            ):
                row[f"{prefix}_{column}"] = float(pivot.loc[world, (column, encoder)])
        row["ceiling_difference_absolute_minus_multiset"] = (
            row["absolute_accuracy_ceiling"] - row["multiset_accuracy_ceiling"]
        )
        row["singleton_difference_absolute_minus_multiset"] = (
            row["absolute_singleton_fraction"] - row["multiset_singleton_fraction"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _absolute_endpoint_rows(
    cells_path: Path,
    covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(cells_path)
    subset = cells.loc[
        (cells["context_per_relation"] == PRIMARY_CONTEXT)
        & cells["encoder"].isin(ENCODERS)
    ].copy()
    scores = (
        subset.groupby(["decoder", "world", "encoder"], as_index=False)[
            "equal_length_macro_balanced_accuracy"
        ]
        .mean()
        .pivot(
            index=["decoder", "world"],
            columns="encoder",
            values="equal_length_macro_balanced_accuracy",
        )
        .reset_index()
        .rename(
            columns={
                "relation_multiset": "multiset_balanced_accuracy",
                "ordered_typed_path": "absolute_balanced_accuracy",
            }
        )
    )
    scores["shared_minus_absolute"] = (
        scores["multiset_balanced_accuracy"] - scores["absolute_balanced_accuracy"]
    )
    scores = scores.merge(
        covariates[
            ["world", "multiset_balanced_accuracy_chance"]
        ].rename(
            columns={
                "multiset_balanced_accuracy_chance": "balanced_accuracy_chance"
            }
        ),
        on="world",
        validate="many_to_one",
    )
    denominator = 1.0 - scores["balanced_accuracy_chance"]
    scores["multiset_chance_normalized"] = (
        scores["multiset_balanced_accuracy"] - scores["balanced_accuracy_chance"]
    ) / denominator
    scores["absolute_chance_normalized"] = (
        scores["absolute_balanced_accuracy"] - scores["balanced_accuracy_chance"]
    ) / denominator
    level_summary = (
        scores.groupby("decoder", as_index=False)
        .agg(
            worlds=("world", "size"),
            equal_world_multiset_ba=("multiset_balanced_accuracy", "mean"),
            equal_world_absolute_ba=("absolute_balanced_accuracy", "mean"),
            equal_world_chance=("balanced_accuracy_chance", "mean"),
            equal_world_multiset_chance_normalized=(
                "multiset_chance_normalized",
                "mean",
            ),
            equal_world_absolute_chance_normalized=(
                "absolute_chance_normalized",
                "mean",
            ),
            multiset_worlds_at_or_below_chance=(
                "multiset_chance_normalized",
                lambda values: int((values <= 0.0).sum()),
            ),
            absolute_worlds_at_or_below_chance=(
                "absolute_chance_normalized",
                lambda values: int((values <= 0.0).sum()),
            ),
        )
    )
    return scores, level_summary


def _cluster_sensitivities(
    outcomes: pd.DataFrame,
    assignments: dict[int, dict[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for decoder, decoder_frame in outcomes.groupby("decoder", observed=True):
        for requested_count, mapping in assignments.items():
            frame = decoder_frame.copy()
            frame["cluster"] = frame["world"].map(mapping)
            cluster_means = frame.groupby("cluster", observed=True)[
                "shared_minus_absolute"
            ].mean()
            wins = int((cluster_means > 0.0).sum())
            ties = int((cluster_means == 0.0).sum())
            losses = int((cluster_means < 0.0).sum())
            summary_rows.append(
                {
                    "decoder": decoder,
                    "requested_clusters": requested_count,
                    "realized_clusters": len(cluster_means),
                    "cluster_wins": wins,
                    "cluster_ties": ties,
                    "cluster_losses": losses,
                    "cluster_sign_sensitivity_p": exact_two_sided_sign_pvalue(
                        wins, losses
                    ),
                    "unweighted_cluster_mean_delta": float(cluster_means.mean()),
                    "equal_world_mean_delta": float(
                        frame["shared_minus_absolute"].mean()
                    ),
                    "interpretation": "dependence sensitivity, not population inference",
                }
            )
            for cluster, value in cluster_means.items():
                members = sorted(frame.loc[frame["cluster"] == cluster, "world"])
                cluster_rows.append(
                    {
                        "decoder": decoder,
                        "requested_clusters": requested_count,
                        "cluster": int(cluster),
                        "worlds": len(members),
                        "members": "|".join(members),
                        "mean_delta": float(value),
                    }
                )
    return summary_rows, cluster_rows


def _moderator_rows(
    outcomes: pd.DataFrame,
    covariates: pd.DataFrame,
) -> list[dict[str, Any]]:
    predictors = [
        "mean_rule_jaccard",
        "max_rule_jaccard",
        "multiset_evaluation_classes",
        "multiset_balanced_accuracy_chance",
        "multiset_accuracy_ceiling",
        "absolute_accuracy_ceiling",
        "multiset_singleton_fraction",
        "absolute_singleton_fraction",
        "multiset_collision_coverage",
        "absolute_collision_coverage",
        "ceiling_difference_absolute_minus_multiset",
        "singleton_difference_absolute_minus_multiset",
    ]
    rows: list[dict[str, Any]] = []
    for decoder, decoder_frame in outcomes.groupby("decoder", observed=True):
        merged = decoder_frame.merge(covariates, on="world", validate="one_to_one")
        target = merged["shared_minus_absolute"].to_numpy(dtype=np.float64)
        for predictor in predictors:
            values = merged[predictor].to_numpy(dtype=np.float64)
            if np.ptp(values) == 0.0 or np.ptp(target) == 0.0:
                rho = None
                pvalue = None
            else:
                statistic = spearmanr(values, target)
                rho = float(statistic.statistic)
                pvalue = float(statistic.pvalue)
            rows.append(
                {
                    "decoder": decoder,
                    "predictor": predictor,
                    "spearman_rho": rho,
                    "unadjusted_p": pvalue,
                    "worlds": len(merged),
                    "analysis_role": "post_hoc_exploratory_no_multiplicity_claim",
                }
            )
    return rows


def _ridge_predictions(
    outcomes: pd.DataFrame,
    covariates: pd.DataFrame,
    assignments: dict[str, int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    predictors = [
        "mean_rule_jaccard",
        "max_rule_jaccard",
        "multiset_evaluation_classes",
        "multiset_accuracy_ceiling",
        "absolute_accuracy_ceiling",
        "multiset_singleton_fraction",
        "absolute_singleton_fraction",
        "multiset_collision_coverage",
        "absolute_collision_coverage",
    ]
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for decoder, decoder_frame in outcomes.groupby("decoder", observed=True):
        merged = decoder_frame.merge(covariates, on="world", validate="one_to_one")
        merged["cluster"] = merged["world"].map(assignments)
        predictions = np.empty(len(merged), dtype=np.float64)
        for cluster in sorted(merged["cluster"].unique()):
            test = merged["cluster"] == cluster
            train = ~test
            scaler = StandardScaler().fit(merged.loc[train, predictors])
            model = Ridge(alpha=1.0).fit(
                scaler.transform(merged.loc[train, predictors]),
                merged.loc[train, "shared_minus_absolute"],
            )
            for predictor, coefficient in zip(
                predictors, model.coef_, strict=True
            ):
                coefficient_rows.append(
                    {
                        "decoder": decoder,
                        "held_out_cluster": int(cluster),
                        "predictor": predictor,
                        "standardized_coefficient": float(coefficient),
                        "intercept": float(model.intercept_),
                    }
                )
            predictions[test] = model.predict(
                scaler.transform(merged.loc[test, predictors])
            )
        observed = merged["shared_minus_absolute"].to_numpy(dtype=np.float64)
        residual = observed - predictions
        baseline = observed - observed.mean()
        r2 = 1.0 - float(np.dot(residual, residual) / np.dot(baseline, baseline))
        correlation = spearmanr(observed, predictions)
        summary_rows.append(
            {
                "decoder": decoder,
                "clusters": len(set(assignments.values())),
                "worlds": len(merged),
                "leave_cluster_out_r2": r2,
                "observed_prediction_spearman": float(correlation.statistic),
                "mean_absolute_error": float(np.abs(residual).mean()),
                "predictors": "|".join(predictors),
                "analysis_role": "post_hoc_descriptive_predictability",
            }
        )
        for row, prediction in zip(
            merged.itertuples(index=False), predictions, strict=True
        ):
            prediction_rows.append(
                {
                    "decoder": decoder,
                    "world": row.world,
                    "cluster": int(row.cluster),
                    "observed_delta": float(row.shared_minus_absolute),
                    "predicted_delta": float(prediction),
                    "residual": float(row.shared_minus_absolute - prediction),
                }
            )
    return summary_rows, prediction_rows, coefficient_rows


def analyze(
    *,
    asset: Path,
    representation: Path,
    cells: Path,
    output: Path,
) -> dict[str, Any]:
    rule_sets = load_rule_sets(asset)
    worlds, similarity = similarity_matrix(rule_sets)
    assignments = cluster_assignments(worlds, similarity)
    covariates = _world_covariates(
        worlds=worlds,
        similarity=similarity,
        representation_path=representation,
    )
    outcomes, level_summary = _absolute_endpoint_rows(cells, covariates)
    cluster_summary, cluster_rows = _cluster_sensitivities(outcomes, assignments)
    moderators = _moderator_rows(outcomes, covariates)
    ridge_summary, ridge_predictions, ridge_coefficients = _ridge_predictions(
        outcomes, covariates, assignments[10]
    )

    pair_rows = []
    matrix_rows = []
    for row, world in enumerate(worlds):
        matrix_row: dict[str, Any] = {"world": world}
        for column, other in enumerate(worlds):
            matrix_row[other] = float(similarity[row, column])
            if column > row:
                pair_rows.append(
                    {
                        "left_world": world,
                        "right_world": other,
                        "shared_rules": len(rule_sets[world] & rule_sets[other]),
                        "rule_jaccard": float(similarity[row, column]),
                    }
                )
        matrix_rows.append(matrix_row)

    output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]]] = {
        "rule_overlap_pairs.csv": pair_rows,
        "rule_overlap_matrix.csv": matrix_rows,
        "world_covariates.csv": covariates.to_dict(orient="records"),
        "endpoint_absolute_by_world.csv": outcomes.to_dict(orient="records"),
        "endpoint_absolute_summary.csv": level_summary.to_dict(orient="records"),
        "cluster_sign_sensitivity.csv": cluster_summary,
        "cluster_members_and_effects.csv": cluster_rows,
        "moderator_spearman.csv": moderators,
        "ridge_leave_cluster_out_summary.csv": ridge_summary,
        "ridge_leave_cluster_out_predictions.csv": ridge_predictions,
        "ridge_leave_cluster_out_coefficients.csv": ridge_coefficients,
    }
    for filename, rows in tables.items():
        _write_csv(output / filename, rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "post-hoc fixed-panel dependence and moderator sensitivity; "
            "not population inference"
        ),
        "worlds": len(worlds),
        "rules_per_world": 20,
        "unique_rules": len(set().union(*rule_sets.values())),
        "pairwise_rule_jaccard": {
            "minimum": float(
                similarity[np.triu_indices(len(worlds), k=1)].min()
            ),
            "mean": float(
                similarity[np.triu_indices(len(worlds), k=1)].mean()
            ),
            "maximum": float(
                similarity[np.triu_indices(len(worlds), k=1)].max()
            ),
        },
        "cluster_grid": list(CLUSTER_COUNTS),
        "primary_contrast": PRIMARY_CONTRAST,
        "primary_context_per_relation": PRIMARY_CONTEXT,
        "inputs": {
            "asset_sha256": file_sha256(asset),
            "representation_sha256": file_sha256(representation),
            "cells_sha256": file_sha256(cells),
        },
        "files": {
            filename: {"rows": len(rows), "sha256": file_sha256(output / filename)}
            for filename, rows in tables.items()
        },
        "interpretation_boundary": (
            "World-overlap clusters are constructed without decoder outcomes. "
            "Cluster sign values are robustness sensitivities and do not restore "
            "independent sampling or a superpopulation estimand."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        asset=args.asset.resolve(),
        representation=args.representation.resolve(),
        cells=args.cells.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
