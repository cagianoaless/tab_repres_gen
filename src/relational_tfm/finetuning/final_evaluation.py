from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.datasets.genealogy import generate_genealogy_task
from relational_tfm.datasets.genealogy_audit import audit_genealogy_task
from relational_tfm.datasets.rule_induction import generate_rule_induction_task
from relational_tfm.datasets.rule_induction_audit import audit_rule_induction_task
from relational_tfm.encoders.genealogy import encode_genealogy_rows
from relational_tfm.encoders.rule_induction import encode_rule_induction_rows
from relational_tfm.evaluation.classification import classification_metrics


@dataclass(frozen=True)
class FinalEvaluationEpisode:
    context_values: np.ndarray
    context_labels: np.ndarray
    evaluation_values: np.ndarray
    evaluation_labels: np.ndarray
    feature_names: tuple[str, ...]
    evaluation_relations: np.ndarray | None
    audit: dict[str, Any]
    metadata: dict[str, Any]


def final_graph_seeds(panel: dict[str, Any]) -> tuple[int, ...]:
    start = int(panel["final_graph_seed_start"])
    count = int(panel["final_graphs"])
    if start < 0 or count <= 0:
        raise ValueError("final graph seed range must be non-negative and non-empty")
    return tuple(range(start, start + count))


def validate_final_seed_isolation(
    final_seeds: tuple[int, ...],
    training_config: dict[str, Any],
) -> None:
    corpus = training_config["corpus"]
    train_seeds = set(
        range(
            int(corpus["train_seed_start"]),
            int(corpus["train_seed_start"]) + int(corpus["train_graphs"]),
        )
    )
    validation_seeds = set(
        range(
            int(corpus["validation_seed_start"]),
            int(corpus["validation_seed_start"])
            + int(corpus["validation_graphs"]),
        )
    )
    final = set(final_seeds)
    if len(final) != len(final_seeds):
        raise ValueError("final graph seeds are not unique")
    if final & train_seeds or final & validation_seeds:
        raise ValueError("final graph seeds overlap training or validation graph seeds")


def build_final_episode(
    *,
    panel_name: str,
    task_specification: dict[str, Any],
    episode_config: dict[str, Any],
    graph_seed: int,
) -> FinalEvaluationEpisode:
    common = {
        "context_per_class": int(episode_config["context_per_class"]),
        "max_context_per_class": int(episode_config["max_context_per_class"]),
        "evaluation_per_class": int(episode_config["evaluation_per_class"]),
        "graph_seed": int(graph_seed),
        "split_seed": int(graph_seed) + 1_000_000,
        "relabel_seed": int(graph_seed) + 2_000_000,
    }
    encoder_seed = int(graph_seed) + 3_000_000
    encoder = str(task_specification["encoder"])

    if panel_name == "atomic":
        task = generate_rule_induction_task(
            rule_name=str(task_specification["rule"]),
            graph_size=int(task_specification["graph_size"]),
            **common,
        )
        audit = audit_rule_induction_task(task)
        context = encode_rule_induction_rows(
            task.support_graph,
            task.context,
            encoder,
            encoder_seed,
        )
        evaluation = encode_rule_induction_rows(
            task.support_graph,
            task.evaluation,
            encoder,
            encoder_seed,
        )
        relations = None
    elif panel_name == "genealogy":
        task = generate_genealogy_task(
            split_name=str(task_specification["split"]),
            observed_fraction=float(task_specification["observed_fraction"]),
            family_count=int(task_specification["family_count"]),
            **common,
        )
        audit = audit_genealogy_task(task)
        context = encode_genealogy_rows(
            task.support_graph,
            task.context,
            encoder,
            encoder_seed,
        )
        evaluation = encode_genealogy_rows(
            task.support_graph,
            task.evaluation,
            encoder,
            encoder_seed,
        )
        relations = task.evaluation["relation"].to_numpy(dtype=np.str_)
    else:
        raise ValueError(f"unsupported final-evaluation panel: {panel_name}")

    if context.feature_names != evaluation.feature_names:
        raise ValueError("context and evaluation feature schemas differ")
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    _validate_balanced_binary(context_labels, "context")
    _validate_balanced_binary(evaluation_labels, "evaluation")
    if not np.isfinite(context.values).all() or not np.isfinite(evaluation.values).all():
        raise ValueError("final-evaluation episode contains non-finite features")

    return FinalEvaluationEpisode(
        context_values=np.asarray(context.values, dtype=np.float32),
        context_labels=context_labels,
        evaluation_values=np.asarray(evaluation.values, dtype=np.float32),
        evaluation_labels=evaluation_labels,
        feature_names=context.feature_names,
        evaluation_relations=relations,
        audit=audit,
        metadata=dict(task.metadata),
    )


def per_relation_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    relations: np.ndarray | None,
) -> dict[str, dict[str, float | int | None]]:
    if relations is None:
        return {}
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "relation": np.asarray(relations, dtype=np.str_),
        }
    )
    result: dict[str, dict[str, float | int | None]] = {}
    for relation, group in frame.groupby("relation", sort=True):
        relation_labels = group["label"].to_numpy(dtype=np.int64)
        relation_probabilities = group["probability"].to_numpy(dtype=np.float64)
        counts = np.bincount(relation_labels, minlength=2)
        metrics: dict[str, float | int | None] = {
            "rows": int(len(group)),
            "negative_rows": int(counts[0]),
            "positive_rows": int(counts[1]),
        }
        if np.count_nonzero(counts) == 2:
            metrics.update(classification_metrics(relation_labels, relation_probabilities))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        result[str(relation)] = metrics
    return result


def _validate_balanced_binary(labels: np.ndarray, name: str) -> None:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if len(counts) != 2 or counts[0] == 0 or counts[0] != counts[1]:
        raise ValueError(f"{name} labels must be balanced and binary")

