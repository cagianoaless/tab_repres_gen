from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relational_tfm.datasets.synthetic import generate_synthetic_task
from relational_tfm.encoders.synthetic import encode_synthetic_rows


@dataclass(frozen=True)
class CorpusSplit:
    values: np.ndarray
    labels: np.ndarray
    graph_seeds: np.ndarray
    feature_names: tuple[str, ...]
    graph_metadata: tuple[dict[str, object], ...]


def build_property_corpus_split(
    *,
    property_name: str,
    encoder: str,
    graph_size: int,
    graph_seeds: tuple[int, ...],
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
    noise_regime: str,
) -> CorpusSplit:
    if not graph_seeds:
        raise ValueError("graph_seeds must not be empty")

    value_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    graph_seed_blocks: list[np.ndarray] = []
    graph_metadata: list[dict[str, object]] = []
    feature_names: tuple[str, ...] | None = None

    for graph_seed in graph_seeds:
        task = generate_synthetic_task(
            property_name=property_name,
            graph_size=graph_size,
            context_per_class=context_per_class,
            noise_regime=noise_regime,
            seed=graph_seed,
            max_context_per_class=max_context_per_class,
            evaluation_per_class=evaluation_per_class,
            graph_seed=graph_seed,
            relabel_seed=graph_seed + 1_000_000,
            split_seed=graph_seed + 2_000_000,
            noise_seed=graph_seed + 3_000_000,
        )
        encoded = encode_synthetic_rows(
            task.support_graph,
            task.population,
            encoder,
            graph_seed + 4_000_000,
            property_name=property_name,
            control_population=task.population,
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
        graph_metadata.append(dict(task.metadata))

    assert feature_names is not None
    return CorpusSplit(
        values=np.concatenate(value_blocks, axis=0),
        labels=np.concatenate(label_blocks, axis=0),
        graph_seeds=np.concatenate(graph_seed_blocks, axis=0),
        feature_names=feature_names,
        graph_metadata=tuple(graph_metadata),
    )
