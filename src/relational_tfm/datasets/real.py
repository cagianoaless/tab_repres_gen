from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.encoders.controls import frozen_context_increment_tiers
from relational_tfm.hashing import object_sha256

Graph = nx.MultiDiGraph
Triple = tuple[int, str, int]


@dataclass(frozen=True)
class RealTask:
    dataset_name: str
    support_graph: Graph
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, object]


LoadedDataset = tuple[
    list[Triple],
    list[Triple],
    list[Triple],
    list[int],
    list[int],
    str,
]


@dataclass(frozen=True)
class RealDatasetAdapter:
    """One registered raw-data layout for the shared link-discrimination task."""

    dataset_name: str
    asset_id: str
    macro_graph_family: str
    loader: Callable[[Path], LoadedDataset]


def _frame(rows: list[Triple], label: int, offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": np.arange(offset, offset + len(rows), dtype=np.int64),
            "head": [head for head, _, _ in rows],
            "relation": [relation for _, relation, _ in rows],
            "tail": [tail for _, _, tail in rows],
            "label": np.full(len(rows), label, dtype=np.int8),
        }
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    return object_sha256(
        [
            [int(row.query_id), int(row.head), str(row.relation), int(row.tail), int(row.label)]
            for row in frame.sort_values("query_id").itertuples(index=False)
        ]
    )


def _map_triples(
    raw_splits: tuple[list[tuple[str, str, str]], ...]
) -> tuple[tuple[list[Triple], ...], int]:
    entities = sorted(
        {value for split in raw_splits for head, _, tail in split for value in (head, tail)}
    )
    entity_id = {entity: index for index, entity in enumerate(entities)}
    mapped = tuple(
        [(entity_id[h], relation, entity_id[t]) for h, relation, t in split]
        for split in raw_splits
    )
    return mapped, len(entities)


def _load_kg(repo_root: Path, dataset: str) -> LoadedDataset:
    root = repo_root / "external/auto-kbc/data" / dataset
    raw = []
    for split in ("train", "dev", "test"):
        rows = []
        for line in (root / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
            head, relation, tail = line.split("\t")
            rows.append((head, relation, tail))
        raw.append(rows)
    (train, dev, test), count = _map_triples(tuple(raw))
    nodes = list(range(count))
    return train, dev, test, nodes, nodes, "multi_relational_kg"


def _load_email(repo_root: Path) -> LoadedDataset:
    path = repo_root / "data/raw/snap_email_eu_core/email-Eu-core.txt"
    edges = sorted(
        {
            (int(parts[0]), "email", int(parts[1]))
            for line in path.read_text(encoding="utf-8").splitlines()
            if (parts := line.split())
        }
    )
    order = np.random.default_rng(17_311).permutation(len(edges))
    shuffled = [edges[int(index)] for index in order]
    train_end = int(0.8 * len(shuffled))
    dev_end = int(0.9 * len(shuffled))
    nodes = sorted({value for h, _, t in edges for value in (h, t)})
    return (
        shuffled[:train_end],
        shuffled[train_end:dev_end],
        shuffled[dev_end:],
        nodes,
        nodes,
        "directed_homogeneous",
    )


def _load_movielens(repo_root: Path) -> LoadedDataset:
    path = repo_root / "data/raw/movielens/ml-100k/u.data"
    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        user, item, _rating, timestamp = line.split("\t")
        raw.append((int(timestamp), int(user), int(item)))
    users = sorted({user for _, user, _ in raw})
    items = sorted({item for _, _, item in raw})
    user_id = {value: index for index, value in enumerate(users)}
    item_id = {value: len(users) + index for index, value in enumerate(items)}
    triples = [(user_id[user], "interacts", item_id[item]) for _, user, item in sorted(raw)]
    train_end = int(0.8 * len(triples))
    dev_end = int(0.9 * len(triples))
    return (
        triples[:train_end],
        triples[train_end:dev_end],
        triples[dev_end:],
        list(user_id.values()),
        list(item_id.values()),
        "bipartite_implicit",
    )


def _kg_adapter(dataset_name: str) -> RealDatasetAdapter:
    return RealDatasetAdapter(
        dataset_name=dataset_name,
        asset_id="auto_kbc_pinned",
        macro_graph_family="multi_relational_kg",
        loader=lambda repo_root: _load_kg(repo_root, dataset_name),
    )


REAL_DATASET_ADAPTERS: dict[str, RealDatasetAdapter] = {
    "email_eu_core": RealDatasetAdapter(
        dataset_name="email_eu_core",
        asset_id="snap_email_eu_core_official",
        macro_graph_family="directed_homogeneous",
        loader=_load_email,
    ),
    "movielens_100k": RealDatasetAdapter(
        dataset_name="movielens_100k",
        asset_id="movielens_100k_official",
        macro_graph_family="bipartite_implicit",
        loader=_load_movielens,
    ),
    **{
        dataset_name: _kg_adapter(dataset_name)
        for dataset_name in ("umls", "kinship", "nations")
    },
}


def real_dataset_adapter(dataset_name: str) -> RealDatasetAdapter:
    try:
        return REAL_DATASET_ADAPTERS[dataset_name]
    except KeyError as error:
        raise ValueError(f"unsupported real dataset: {dataset_name}") from error


def _sample_negatives(
    references: list[Triple],
    count: int,
    all_true: set[Triple],
    sources: list[int],
    targets: list[int],
    seed: int,
    forbidden: set[Triple] | None = None,
) -> list[Triple]:
    rng = np.random.default_rng(seed)
    selected: set[Triple] = set()
    forbidden = forbidden or set()
    attempts = 0
    while len(selected) < count and attempts < count * 500:
        head, relation, tail = references[attempts % len(references)]
        if sources is targets and attempts % 2:
            candidate = (int(sources[int(rng.integers(len(sources)))]), relation, tail)
        else:
            candidate = (head, relation, int(targets[int(rng.integers(len(targets)))]))
        attempts += 1
        if candidate in all_true or candidate in forbidden:
            continue
        selected.add(candidate)
    if len(selected) != count:
        raise ValueError(f"could sample only {len(selected)} of {count} real negatives")
    return sorted(selected)


def generate_real_task(
    repo_root: Path,
    dataset_name: str,
    context_positive_count: int,
    context_negative_ratio: int,
    max_context_positive: int,
    max_context_negative: int,
    evaluation_positive_count: int,
    evaluation_negative_ratio: int,
    split_seed: int,
) -> RealTask:
    adapter = real_dataset_adapter(dataset_name)
    loaded = adapter.loader(repo_root)
    train, dev, test, sources, targets, regime = loaded
    if regime != adapter.macro_graph_family:
        raise ValueError(
            f"{dataset_name} loader returned regime {regime!r}; "
            f"registered={adapter.macro_graph_family!r}"
        )
    if len(test) < evaluation_positive_count:
        raise ValueError(f"{dataset_name} has insufficient positive split rows")
    rng = np.random.default_rng(split_seed)
    all_true = set(train) | set(dev) | set(test)
    if regime == "multi_relational_kg":
        if len(train) <= max_context_positive:
            raise ValueError(f"{dataset_name} has insufficient training rows for context holdout")
        train_order = rng.permutation(len(train))
        context_positives = [
            train[int(index)] for index in train_order[:max_context_positive]
        ]
        train = [train[int(index)] for index in train_order[max_context_positive:]]
        context_source = "deterministic holdout from official train; excluded from support"
    else:
        if len(dev) < max_context_positive:
            raise ValueError(f"{dataset_name} has insufficient positive context rows")
        context_positives = [
            dev[int(index)] for index in rng.permutation(len(dev))[:max_context_positive]
        ]
        context_source = "development split"
    evaluation_positives = [
        test[int(index)] for index in rng.permutation(len(test))[:evaluation_positive_count]
    ]
    context_negatives = _sample_negatives(
        context_positives,
        max_context_negative,
        all_true,
        sources,
        targets,
        split_seed + 1,
    )
    evaluation_negatives = _sample_negatives(
        evaluation_positives,
        evaluation_positive_count * evaluation_negative_ratio,
        all_true,
        sources,
        targets,
        split_seed + 2,
        set(context_negatives),
    )
    positive_pool = _frame(context_positives, 1, 0)
    negative_pool = _frame(context_negatives, 0, max_context_positive)
    evaluation_positive = _frame(
        evaluation_positives, 1, max_context_positive + max_context_negative
    )
    evaluation_negative = _frame(
        evaluation_negatives,
        0,
        max_context_positive + max_context_negative + evaluation_positive_count,
    )
    positive_pool["_control_partition"] = "context"
    negative_pool["_control_partition"] = "context"
    evaluation_positive["_control_partition"] = "evaluation"
    evaluation_negative["_control_partition"] = "evaluation"
    positive_pool["_control_tier"] = frozen_context_increment_tiers(
        len(positive_pool)
    )
    negative_pool["_control_tier"] = frozen_context_increment_tiers(
        len(negative_pool),
        negative_ratio=context_negative_ratio,
    )
    evaluation_positive["_control_tier"] = "evaluation"
    evaluation_negative["_control_tier"] = "evaluation"
    context = pd.concat(
        [
            positive_pool.iloc[:context_positive_count],
            negative_pool.iloc[: context_positive_count * context_negative_ratio],
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=split_seed + 3)
    evaluation = pd.concat(
        [evaluation_positive, evaluation_negative], ignore_index=True
    ).sample(frac=1.0, random_state=split_seed + 4)
    population = pd.concat(
        [positive_pool, negative_pool, evaluation_positive, evaluation_negative],
        ignore_index=True,
    ).sort_values("query_id")

    graph = Graph()
    graph.add_nodes_from(sorted(set(sources) | set(targets)))
    for head, relation, tail in train:
        graph.add_edge(head, tail, relation=relation)
    support_triples = set(train)
    context_positive_support_overlap = len(set(context_positives) & support_triples)
    evaluation_positive_support_overlap = len(set(evaluation_positives) & support_triples)
    if context_positive_support_overlap or evaluation_positive_support_overlap:
        raise ValueError("a real positive query is present in the feature support graph")
    metadata = {
        "dataset": dataset_name,
        "macro_graph_family": regime,
        "support_nodes": graph.number_of_nodes(),
        "support_edges": graph.number_of_edges(),
        "context_positive_count": context_positive_count,
        "context_negative_ratio": context_negative_ratio,
        "evaluation_positive_count": evaluation_positive_count,
        "evaluation_negative_ratio": evaluation_negative_ratio,
        "context_source": context_source,
        "feature_graph_policy": (
            "support/training edges only; all context and evaluation positives excluded"
        ),
        "context_positive_support_overlap": context_positive_support_overlap,
        "evaluation_positive_support_overlap": evaluation_positive_support_overlap,
        "negative_filter_policy": (
            "relation-conditioned endpoint-role corruption, false in complete known graph"
        ),
        "context_negative_known_true_overlap": len(set(context_negatives) & all_true),
        "evaluation_negative_known_true_overlap": len(set(evaluation_negatives) & all_true),
        "reverse_support_policy": "retained as observed support evidence",
        "split_seed": split_seed,
        "context_query_sha256": _frame_hash(context),
        "evaluation_query_sha256": _frame_hash(evaluation),
        "population_query_sha256": _frame_hash(population),
    }
    return RealTask(
        dataset_name=dataset_name,
        support_graph=graph,
        context=context.reset_index(drop=True),
        evaluation=evaluation.reset_index(drop=True),
        population=population.reset_index(drop=True),
        metadata=metadata,
    )
