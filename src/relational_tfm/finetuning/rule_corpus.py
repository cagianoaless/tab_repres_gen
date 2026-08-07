from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relational_tfm.datasets.rule_induction import generate_rule_induction_task
from relational_tfm.datasets.rule_induction_audit import audit_rule_induction_task
from relational_tfm.encoders.rule_induction import encode_rule_induction_rows


@dataclass(frozen=True)
class RuleCorpusSplit:
    values: np.ndarray
    labels: np.ndarray
    graph_seeds: np.ndarray
    feature_names: tuple[str, ...]
    graph_metadata: tuple[dict[str, object], ...]


def build_rule_corpus_split(
    *,
    rule_name: str,
    encoder: str,
    graph_size: int,
    graph_seeds: tuple[int, ...],
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
) -> RuleCorpusSplit:
    """Build pooled rows from disjoint generated graphs for a fine-tuning split."""

    if not graph_seeds:
        raise ValueError("graph_seeds must not be empty")

    value_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    graph_seed_blocks: list[np.ndarray] = []
    graph_metadata: list[dict[str, object]] = []
    feature_names: tuple[str, ...] | None = None

    for graph_seed in graph_seeds:
        task = generate_rule_induction_task(
            rule_name=rule_name,
            graph_size=graph_size,
            context_per_class=context_per_class,
            max_context_per_class=max_context_per_class,
            evaluation_per_class=evaluation_per_class,
            graph_seed=graph_seed,
            split_seed=graph_seed + 1_000_000,
            relabel_seed=graph_seed + 2_000_000,
        )
        audit = audit_rule_induction_task(task)
        encoded = encode_rule_induction_rows(
            task.support_graph,
            task.population,
            encoder,
            graph_seed + 3_000_000,
        )
        if feature_names is None:
            feature_names = encoded.feature_names
        elif encoded.feature_names != feature_names:
            raise ValueError("feature schema changed across generated graphs")

        labels = task.population["label"].to_numpy(dtype=np.int8)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("each fine-tuning graph must contain both labels")
        if int(labels.sum()) * 2 != len(labels):
            raise ValueError("each fine-tuning graph must be class-balanced")

        value_blocks.append(np.asarray(encoded.values, dtype=np.float32))
        label_blocks.append(labels)
        graph_seed_blocks.append(np.full(len(labels), graph_seed, dtype=np.int64))
        graph_metadata.append(
            {
                **task.metadata,
                "audit": audit,
                "encoder": encoder,
                "feature_count": len(encoded.feature_names),
            }
        )

    assert feature_names is not None
    return RuleCorpusSplit(
        values=np.concatenate(value_blocks, axis=0),
        labels=np.concatenate(label_blocks, axis=0),
        graph_seeds=np.concatenate(graph_seed_blocks, axis=0),
        feature_names=feature_names,
        graph_metadata=tuple(graph_metadata),
    )
