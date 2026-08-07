from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relational_tfm.datasets.genealogy import generate_genealogy_task
from relational_tfm.datasets.genealogy_audit import audit_genealogy_task
from relational_tfm.encoders.genealogy import encode_genealogy_rows


@dataclass(frozen=True)
class GenealogyCorpusSplit:
    values: np.ndarray
    labels: np.ndarray
    graph_seeds: np.ndarray
    feature_names: tuple[str, ...]
    graph_metadata: tuple[dict[str, object], ...]


def build_genealogy_corpus_split(
    *,
    split_name: str,
    observed_fraction: float,
    encoder: str,
    family_count: int,
    graph_seeds: tuple[int, ...],
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
) -> GenealogyCorpusSplit:
    if not graph_seeds:
        raise ValueError("graph_seeds must not be empty")

    value_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    graph_seed_blocks: list[np.ndarray] = []
    graph_metadata: list[dict[str, object]] = []
    feature_names: tuple[str, ...] | None = None
    for graph_seed in graph_seeds:
        task = generate_genealogy_task(
            split_name=split_name,
            observed_fraction=observed_fraction,
            family_count=family_count,
            context_per_class=context_per_class,
            max_context_per_class=max_context_per_class,
            evaluation_per_class=evaluation_per_class,
            graph_seed=graph_seed,
            split_seed=graph_seed + 1_000_000,
            relabel_seed=graph_seed + 2_000_000,
        )
        audit = audit_genealogy_task(task)
        encoded = encode_genealogy_rows(
            task.support_graph,
            task.population,
            encoder,
            graph_seed + 3_000_000,
        )
        if feature_names is None:
            feature_names = encoded.feature_names
        elif encoded.feature_names != feature_names:
            raise ValueError("genealogy feature schema changed across graphs")
        labels = task.population["label"].to_numpy(dtype=np.int8)
        if set(np.unique(labels)) != {0, 1} or int(labels.sum()) * 2 != len(labels):
            raise ValueError("each genealogy graph must be binary and balanced")
        value_blocks.append(encoded.values)
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
    return GenealogyCorpusSplit(
        values=np.concatenate(value_blocks, axis=0),
        labels=np.concatenate(label_blocks, axis=0),
        graph_seeds=np.concatenate(graph_seed_blocks, axis=0),
        feature_names=feature_names,
        graph_metadata=tuple(graph_metadata),
    )
