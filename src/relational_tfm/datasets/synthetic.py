from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.encoders.controls import frozen_context_increment_tiers
from relational_tfm.hashing import object_sha256
from relational_tfm.protocol import MIN_GRAPH_SIZE, SUPPORTED_NOISE_REGIMES

Graph = nx.MultiDiGraph
Query = tuple[int, int, str]


@dataclass(frozen=True)
class SyntheticTask:
    property_name: str
    clean_support_graph: Graph
    support_graph: Graph
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, object]


def _query_frame(queries: list[Query], label: int, offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": np.arange(offset, offset + len(queries), dtype=np.int64),
            "head": [head for head, _, _ in queries],
            "tail": [tail for _, tail, _ in queries],
            "relation": [relation for _, _, relation in queries],
            "label": np.full(len(queries), label, dtype=np.int8),
        }
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    records = [
        {
            "query_id": int(row.query_id),
            "head": int(row.head),
            "tail": int(row.tail),
            "relation": str(row.relation),
            "label": int(row.label),
        }
        for row in frame.sort_values("query_id").itertuples(index=False)
    ]
    return object_sha256(records)


def _graph_hash(graph: Graph) -> str:
    nodes = [
        {"node": int(node), **{str(key): value for key, value in sorted(attributes.items())}}
        for node, attributes in sorted(graph.nodes(data=True))
    ]
    edges = [
        {
            "head": int(head),
            "tail": int(tail),
            "key": int(key),
            **{str(name): value for name, value in sorted(attributes.items())},
        }
        for head, tail, key, attributes in sorted(graph.edges(keys=True, data=True))
    ]
    return object_sha256({"nodes": nodes, "edges": edges})


def _sample_unique_pairs(
    left: list[int],
    right: list[int],
    count: int,
    rng: np.random.Generator,
    *,
    allow_equal: bool = False,
    unordered: bool = False,
    predicate: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int]]:
    capacity = len(left) * len(right)
    if left is right and not allow_equal:
        capacity -= len(left)
    if unordered and left is right:
        capacity //= 2
    if count > capacity:
        raise ValueError(f"cannot sample {count} unique pairs from capacity {capacity}")
    selected: set[tuple[int, int]] = set()
    attempts = 0
    limit = max(10_000, count * 200)
    while len(selected) < count and attempts < limit:
        head = int(left[int(rng.integers(len(left)))])
        tail = int(right[int(rng.integers(len(right)))])
        attempts += 1
        if not allow_equal and head == tail:
            continue
        pair = tuple(sorted((head, tail))) if unordered else (head, tail)
        if predicate is not None and not predicate(*pair):
            continue
        selected.add(pair)
    if len(selected) != count:
        raise ValueError(f"could sample only {len(selected)} of {count} required pairs")
    return sorted(selected)


def _balanced_frames(
    positive_queries: list[Query],
    negative_queries: list[Query],
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
    split_seed: int,
    context_negative_ratio: int = 1,
    evaluation_negative_ratio: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive_required = max_context_per_class + evaluation_per_class
    negative_required = (
        max_context_per_class + evaluation_per_class * evaluation_negative_ratio
    )
    if context_per_class > max_context_per_class:
        raise ValueError("context_per_class exceeds the frozen maximum")
    negative_context_size = context_per_class * context_negative_ratio
    if negative_context_size > max_context_per_class:
        raise ValueError("negative context budget exceeds the frozen maximum")
    if len(positive_queries) < positive_required or len(negative_queries) < negative_required:
        raise ValueError(
            f"need {positive_required} positive and {negative_required} negative queries, "
            f"found {len(positive_queries)} positive "
            f"and {len(negative_queries)} negative"
        )
    rng = np.random.default_rng(split_seed)
    positive_order = rng.permutation(len(positive_queries))[:positive_required]
    negative_order = rng.permutation(len(negative_queries))[:negative_required]
    positives = [positive_queries[int(index)] for index in positive_order]
    negatives = [negative_queries[int(index)] for index in negative_order]

    positive_frame = _query_frame(positives, 1, 0)
    negative_frame = _query_frame(negatives, 0, positive_required)
    positive_frame["_control_partition"] = np.where(
        np.arange(len(positive_frame)) < max_context_per_class,
        "context",
        "evaluation",
    )
    negative_frame["_control_partition"] = np.where(
        np.arange(len(negative_frame)) < max_context_per_class,
        "context",
        "evaluation",
    )
    positive_frame["_control_tier"] = np.concatenate(
        [
            frozen_context_increment_tiers(max_context_per_class),
            np.full(evaluation_per_class, "evaluation", dtype=object),
        ]
    )
    negative_frame["_control_tier"] = np.concatenate(
        [
            frozen_context_increment_tiers(
                max_context_per_class,
                negative_ratio=context_negative_ratio,
            ),
            np.full(
                evaluation_per_class * evaluation_negative_ratio,
                "evaluation",
                dtype=object,
            ),
        ]
    )
    context_pool = pd.concat(
        [
            positive_frame.iloc[:max_context_per_class],
            negative_frame.iloc[:max_context_per_class],
        ],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [
            positive_frame.iloc[max_context_per_class:],
            negative_frame.iloc[
                max_context_per_class : max_context_per_class
                + evaluation_per_class * evaluation_negative_ratio
            ],
        ],
        ignore_index=True,
    )
    context_ids = set(positive_frame.iloc[:context_per_class]["query_id"]) | set(
        negative_frame.iloc[:negative_context_size]["query_id"]
    )
    context = context_pool[context_pool["query_id"].isin(context_ids)].copy()
    context = context.iloc[np.random.default_rng(split_seed + 1).permutation(len(context))]
    evaluation = evaluation.iloc[
        np.random.default_rng(split_seed + 2).permutation(len(evaluation))
    ]
    population = pd.concat([context_pool, evaluation], ignore_index=True)
    return (
        context.reset_index(drop=True),
        evaluation.reset_index(drop=True),
        population.sort_values("query_id").reset_index(drop=True),
    )


def _relabel(
    graph: Graph,
    frames: tuple[pd.DataFrame, ...],
    relabel_seed: int,
    additional_nodes: Iterable[int] = (),
) -> tuple[Graph, tuple[pd.DataFrame, ...], dict[int, int]]:
    nodes = np.asarray(
        sorted(
            {int(node) for node in graph.nodes()}
            | {int(node) for node in additional_nodes}
        ),
        dtype=np.int64,
    )
    rng = np.random.default_rng(relabel_seed)
    while True:
        labels = rng.integers(2**40, 2**52, size=len(nodes), dtype=np.int64)
        if len(np.unique(labels)) == len(labels):
            break
    mapping = {int(node): int(label) for node, label in zip(nodes, labels, strict=True)}
    relabeled_graph = nx.relabel_nodes(graph, mapping, copy=True)
    relabeled_frames = []
    for frame in frames:
        result = frame.copy()
        result["head"] = result["head"].map(mapping).astype(np.int64)
        result["tail"] = result["tail"].map(mapping).astype(np.int64)
        relabeled_frames.append(result)
    return relabeled_graph, tuple(relabeled_frames), mapping


def _apply_noise(clean_graph: Graph, graph_size: int, regime: str, seed: int) -> Graph:
    if regime not in SUPPORTED_NOISE_REGIMES:
        raise ValueError(f"unsupported noise regime: {regime}")
    graph = clean_graph.copy()
    if regime == "none":
        return graph
    rng = np.random.default_rng(seed)
    auxiliary_count = max(8, graph_size // 8)
    auxiliary = list(range(graph_size, graph_size + auxiliary_count))
    graph.add_nodes_from((node, {"community": -1, "role": -1}) for node in auxiliary)
    if regime == "disconnected":
        for index, node in enumerate(auxiliary):
            graph.add_edge(
                node,
                auxiliary[(index + 1) % len(auxiliary)],
                relation="noise",
                noise=True,
            )
        return graph
    base_nodes = list(range(graph_size))
    if regime == "endpoint_attached":
        for node in auxiliary:
            endpoint = int(base_nodes[int(rng.integers(len(base_nodes)))])
            graph.add_edge(endpoint, node, relation="noise", noise=True)
            graph.add_edge(node, endpoint, relation="noise", noise=True)
        return graph
    relations = sorted(
        {
            str(attributes.get("relation", "r0"))
            for _, _, attributes in graph.edges(data=True)
            if attributes.get("relation") != "noise"
        }
    ) or ["r0"]
    requested = max(1, graph.number_of_edges() // 20)
    for _ in range(requested):
        head, tail = rng.choice(base_nodes, size=2, replace=False)
        relation = relations[int(rng.integers(len(relations)))]
        graph.add_edge(int(head), int(tail), relation=relation, noise=True)
    return graph


def _reflexivity(
    graph_size: int, positive_required: int, negative_required: int, rng: np.random.Generator
):
    graph = Graph()
    graph.add_nodes_from(range(graph_size))
    for node in range(graph_size):
        graph.add_edge(node, (node + 1) % graph_size, relation="r0", noise=False)
    positives = [(node, node, "q0") for node in range(positive_required)]
    negative_pairs = _sample_unique_pairs(
        list(range(graph_size)), list(range(graph_size)), negative_required, rng
    )
    negatives = [(head, tail, "q0") for head, tail in negative_pairs]
    return graph, positives, negatives, "positive iff head and tail denote the same entity"


def _symmetry(
    graph_size: int, positive_required: int, negative_required: int, rng: np.random.Generator
):
    graph = Graph()
    graph.add_nodes_from(range(graph_size))
    pairs = _sample_unique_pairs(
        list(range(graph_size)),
        list(range(graph_size)),
        positive_required + negative_required,
        rng,
        unordered=True,
    )
    positives = []
    for left, right in pairs[:positive_required]:
        if rng.random() < 0.5:
            left, right = right, left
        graph.add_edge(left, right, relation="r0", noise=False)
        positives.append((right, left, "r0"))
    negatives = [
        (left, right, "r0")
        for left, right in pairs[positive_required : positive_required + negative_required]
    ]
    return graph, positives, negatives, "positive iff the reverse support edge exists"


def _antisymmetry(
    graph_size: int, positive_required: int, negative_required: int, rng: np.random.Generator
):
    graph = Graph()
    split = graph_size // 2
    sources = list(range(split))
    sinks = list(range(split, graph_size))
    graph.add_nodes_from((node, {"role": 0}) for node in sources)
    graph.add_nodes_from((node, {"role": 1}) for node in sinks)
    for index, source in enumerate(sources):
        for offset in range(4):
            sink = sinks[(index + 7 * offset) % len(sinks)]
            graph.add_edge(source, sink, relation="r0", noise=False)
    pairs = _sample_unique_pairs(
        sources,
        sinks,
        max(positive_required, negative_required),
        rng,
        predicate=lambda head, tail: not graph.has_edge(head, tail),
    )
    positives = [(head, tail, "r0") for head, tail in pairs[:positive_required]]
    negatives = [(tail, head, "r0") for head, tail in pairs[:negative_required]]
    return graph, positives, negatives, "positive iff the query follows the support direction"


def _transitivity(
    graph_size: int, positive_required: int, negative_required: int, rng: np.random.Generator
):
    graph = Graph()
    graph.add_nodes_from(range(graph_size))
    for node in range(graph_size - 1):
        graph.add_edge(node, node + 1, relation="r0", noise=False)
    pairs = _sample_unique_pairs(
        list(range(graph_size)),
        list(range(graph_size)),
        max(positive_required, negative_required),
        rng,
        predicate=lambda head, tail: tail >= head + 2,
    )
    positives = [(head, tail, "r0") for head, tail in pairs[:positive_required]]
    negatives = [(tail, head, "r0") for head, tail in pairs[:negative_required]]
    return (
        graph,
        positives,
        negatives,
        "positive iff a forward support path of length at least two exists",
    )


def _typed_composition(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    length: int,
    rng: np.random.Generator,
) -> tuple[Graph, list[Query], list[Query], str]:
    graph = Graph()
    layers = [list(map(int, part)) for part in np.array_split(np.arange(graph_size), length + 1)]
    if any(len(layer) % 2 for layer in layers):
        raise ValueError("typed-composition layers must have even size")
    groups = []
    for layer_index, layer in enumerate(layers):
        shuffled = [int(node) for node in rng.permutation(layer)]
        midpoint = len(shuffled) // 2
        layer_groups = (shuffled[:midpoint], shuffled[midpoint:])
        groups.append(layer_groups)
        for group_index, group in enumerate(layer_groups):
            graph.add_nodes_from(
                (node, {"role": layer_index, "latent_group": group_index}) for node in group
            )
    for layer_index in range(length):
        for group_index in (0, 1):
            for head in groups[layer_index][group_index]:
                for tail in groups[layer_index + 1][group_index]:
                    graph.add_edge(
                        head, tail, relation=f"r{layer_index + 1}", noise=False
                    )
                for tail in groups[layer_index + 1][1 - group_index]:
                    graph.add_edge(
                        head, tail, relation=f"s{layer_index + 1}", noise=False
                    )
    positive_pairs = [
        (head, tail)
        for group_index in (0, 1)
        for head in groups[0][group_index]
        for tail in groups[-1][group_index]
    ]
    negative_pairs = [
        (head, tail)
        for group_index in (0, 1)
        for head in groups[0][group_index]
        for tail in groups[-1][1 - group_index]
    ]
    if len(positive_pairs) < positive_required or len(negative_pairs) < negative_required:
        raise ValueError("typed-composition graph is too small for the requested budget")
    positive_order = rng.permutation(len(positive_pairs))[:positive_required]
    negative_order = rng.permutation(len(negative_pairs))[:negative_required]
    query_relation = f"compose_l{length}"
    positives = [(*positive_pairs[int(index)], query_relation) for index in positive_order]
    negatives = [(*negative_pairs[int(index)], query_relation) for index in negative_order]
    pattern = "->".join(f"r{index}" for index in range(1, length + 1))
    return (
        graph,
        positives,
        negatives,
        f"positive iff the typed support path {pattern} exists; all queries have an untyped path",
    )


def _local_roles(
    graph_size: int, positive_required: int, negative_required: int, rng: np.random.Generator
):
    if graph_size % 4:
        raise ValueError("local-structural-role graphs require a size divisible by four")
    graph = Graph()
    roles = {node: node % 4 for node in range(graph_size)}
    graph.add_nodes_from((node, {"role": role}) for node, role in roles.items())
    for role in range(4):
        sources = [node for node in range(graph_size) if roles[node] == role]
        targets = [node for node in range(graph_size) if roles[node] == (role + 1) % 4]
        for source_index, node in enumerate(sources):
            for offset in range(role + 1):
                graph.add_edge(
                    node,
                    targets[(source_index + offset) % len(targets)],
                    relation=f"r{role}",
                    noise=False,
                )
    nodes = list(range(graph_size))
    positives = _sample_unique_pairs(
        nodes,
        nodes,
        positive_required,
        rng,
        predicate=lambda head, tail: roles[head] == roles[tail],
    )
    negatives = _sample_unique_pairs(
        nodes,
        nodes,
        negative_required,
        rng,
        predicate=lambda head, tail: roles[head] != roles[tail],
    )
    return (
        graph,
        [(head, tail, "role_eq") for head, tail in positives],
        [(head, tail, "role_eq") for head, tail in negatives],
        "positive iff endpoints have the same relation-specific structural role",
    )


def _block_affinity(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    rng: np.random.Generator,
    homophilic: bool,
):
    graph = Graph()
    communities = {node: int(node >= graph_size // 2) for node in range(graph_size)}
    graph.add_nodes_from(
        (node, {"community": community}) for node, community in communities.items()
    )
    for head in range(graph_size):
        for tail in range(head + 1, graph_size):
            same = communities[head] == communities[tail]
            probability = 0.08 if same == homophilic else 0.01
            if rng.random() < probability:
                graph.add_edge(head, tail, relation="r0", noise=False)
                graph.add_edge(tail, head, relation="r0", noise=False)
    nodes = list(range(graph_size))
    positive_pairs = _sample_unique_pairs(
        nodes,
        nodes,
        positive_required,
        rng,
        predicate=lambda head, tail: (communities[head] == communities[tail]) == homophilic
        and not graph.has_edge(head, tail),
    )
    negative_pairs = _sample_unique_pairs(
        nodes,
        nodes,
        negative_required,
        rng,
        predicate=lambda head, tail: (communities[head] == communities[tail]) != homophilic
        and not graph.has_edge(head, tail),
    )
    mode = "same" if homophilic else "different"
    return (
        graph,
        [(head, tail, "block") for head, tail in positive_pairs],
        [(head, tail, "block") for head, tail in negative_pairs],
        f"positive iff endpoint communities are {mode}",
    )


DISCOVERY_QUERY_RELATIONS = ("q0", "q1", "q2", "q3")
DISCOVERY_NOISE_PERIOD = 10


def discovery_latent_labels(
    query_ids: np.ndarray, positive_required: int
) -> np.ndarray:
    """Recover frozen pre-flip labels from the synthetic query-ID partition."""
    query_ids = np.asarray(query_ids, dtype=np.int64)
    return (query_ids < positive_required).astype(np.int8)


def _sample_queries_by_relation(
    left: list[int],
    right: list[int],
    count_per_relation: int,
    rng: np.random.Generator,
    predicate: Callable[[str, int, int], bool],
) -> tuple[list[Query], list[Query]]:
    positives: list[Query] = []
    negatives: list[Query] = []
    for relation in DISCOVERY_QUERY_RELATIONS:
        positive_pairs = _sample_unique_pairs(
            left,
            right,
            count_per_relation,
            rng,
            predicate=lambda head, tail, relation=relation: predicate(relation, head, tail),
        )
        negative_pairs = _sample_unique_pairs(
            left,
            right,
            count_per_relation,
            rng,
            predicate=lambda head, tail, relation=relation: not predicate(relation, head, tail),
        )
        positives.extend((head, tail, relation) for head, tail in positive_pairs)
        negatives.extend((head, tail, relation) for head, tail in negative_pairs)
    return positives, negatives


def _discovery_reciprocity(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    rng: np.random.Generator,
) -> tuple[Graph, list[Query], list[Query], str]:
    graph = Graph()
    graph.add_nodes_from(range(graph_size))
    nodes = list(range(graph_size))
    support_relations = tuple(f"r{index}" for index in range(4))
    for relation in support_relations:
        for head, tail in _sample_unique_pairs(nodes, nodes, max(1_200, positive_required), rng):
            graph.add_edge(head, tail, relation=relation, noise=False)
    mapping = {
        query: str(support)
        for query, support in zip(
            DISCOVERY_QUERY_RELATIONS,
            rng.permutation(support_relations),
            strict=True,
        )
    }
    graph.graph["discovery_mapping"] = mapping
    count = max(positive_required, negative_required)
    positives, negatives = _sample_queries_by_relation(
        nodes,
        nodes,
        count,
        rng,
        lambda relation, head, tail: any(
            str(attributes.get("relation")) == mapping[relation]
            for attributes in graph.get_edge_data(tail, head, default={}).values()
        ),
    )
    return (
        graph,
        positives,
        negatives,
        "query relation selects one hidden support relation; positive iff its reverse edge exists",
    )


def _discovery_typed_composition(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    rng: np.random.Generator,
) -> tuple[Graph, list[Query], list[Query], str]:
    if graph_size % 3:
        raise ValueError("discovery typed-composition size must be divisible by three")
    graph = Graph()
    layers = [[int(node) for node in part] for part in np.array_split(np.arange(graph_size), 3)]
    graph.add_nodes_from(
        (node, {"layer": layer}) for layer, nodes in enumerate(layers) for node in nodes
    )
    templates = tuple((f"r{2 * index}", f"r{2 * index + 1}") for index in range(4))
    edge_count = max(500, positive_required // 2)
    for first_relation, second_relation in templates:
        for head, tail in _sample_unique_pairs(layers[0], layers[1], edge_count, rng):
            graph.add_edge(head, tail, relation=first_relation, noise=False)
        for head, tail in _sample_unique_pairs(layers[1], layers[2], edge_count, rng):
            graph.add_edge(head, tail, relation=second_relation, noise=False)
    mapping = {
        query: tuple(str(relation) for relation in template)
        for query, template in zip(
            DISCOVERY_QUERY_RELATIONS,
            rng.permutation(np.asarray(templates, dtype=object)),
            strict=True,
        )
    }
    graph.graph["discovery_mapping"] = mapping

    typed_targets: dict[tuple[int, tuple[str, str]], set[int]] = {}

    def has_mapped_path(relation: str, head: int, tail: int) -> bool:
        template = tuple(mapping[relation])
        key = (head, template)
        if key not in typed_targets:
            frontier = {head}
            for edge_relation in template:
                frontier = {
                    int(neighbor)
                    for node in frontier
                    for _, neighbor, attributes in graph.out_edges(node, data=True)
                    if str(attributes.get("relation")) == edge_relation
                }
            typed_targets[key] = frontier
        return tail in typed_targets[key]

    count = max(positive_required, negative_required)
    positives, negatives = _sample_queries_by_relation(
        layers[0], layers[2], count, rng, has_mapped_path
    )
    return (
        graph,
        positives,
        negatives,
        "query relation selects one hidden member of a four-template typed-path bank",
    )


def _discovery_structural_roles(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    rng: np.random.Generator,
) -> tuple[Graph, list[Query], list[Query], str]:
    if graph_size % 4:
        raise ValueError("discovery role graph size must be divisible by four")
    graph = Graph()
    roles = {node: node % 4 for node in range(graph_size)}
    graph.add_nodes_from((node, {"role": role}) for node, role in roles.items())
    role_nodes = {
        role: [node for node, value in roles.items() if value == role] for role in range(4)
    }
    for role in range(4):
        targets = role_nodes[(role + 1) % 4]
        for source_index, head in enumerate(role_nodes[role]):
            for offset in range(role + 2):
                tail = targets[(source_index + 7 * offset) % len(targets)]
                graph.add_edge(head, tail, relation=f"r{role}", noise=False)
    tail_roles = [int(value) for value in rng.permutation(4)]
    mapping = {
        query: (head_role, tail_roles[head_role])
        for head_role, query in enumerate(DISCOVERY_QUERY_RELATIONS)
    }
    graph.graph["discovery_mapping"] = mapping
    nodes = list(range(graph_size))
    count = max(positive_required, negative_required)
    positives, negatives = _sample_queries_by_relation(
        nodes,
        nodes,
        count,
        rng,
        lambda relation, head, tail: (
            (
                roles[head],
                roles[tail],
            )
            == tuple(mapping[relation])
        ),
    )
    return (
        graph,
        positives,
        negatives,
        "query relation selects one hidden ordered pair of graph-derived structural roles",
    )


def _discovery_block_affinity(
    graph_size: int,
    positive_required: int,
    negative_required: int,
    rng: np.random.Generator,
) -> tuple[Graph, list[Query], list[Query], str]:
    if graph_size % 4:
        raise ValueError("discovery block graph size must be divisible by four")
    graph = Graph()
    communities = {node: node // (graph_size // 4) for node in range(graph_size)}
    graph.add_nodes_from(
        (node, {"community": community}) for node, community in communities.items()
    )
    for head in range(graph_size):
        for tail in range(head + 1, graph_size):
            same = communities[head] == communities[tail]
            if rng.random() < (0.12 if same else 0.015):
                graph.add_edge(head, tail, relation="r0", noise=False)
                graph.add_edge(tail, head, relation="r0", noise=False)
    modes = [True, True, False, False]
    rng.shuffle(modes)
    mapping = {
        query: bool(mode) for query, mode in zip(DISCOVERY_QUERY_RELATIONS, modes, strict=True)
    }
    graph.graph["discovery_mapping"] = mapping
    nodes = list(range(graph_size))
    count = max(positive_required, negative_required)
    positives, negatives = _sample_queries_by_relation(
        nodes,
        nodes,
        count,
        rng,
        lambda relation, head, tail: (communities[head] == communities[tail]) == mapping[relation],
    )
    return (
        graph,
        positives,
        negatives,
        "query relation selects a hidden homophilic or heterophilic block-affinity rule",
    )


def _apply_discovery_label_noise(
    frames: tuple[pd.DataFrame, ...],
    positive_required: int,
) -> tuple[pd.DataFrame, ...]:
    noisy: list[pd.DataFrame] = []
    for frame in frames:
        result = frame.copy()
        query_ids = result["query_id"].to_numpy(dtype=np.int64)
        within_class = np.where(
            query_ids < positive_required,
            query_ids,
            query_ids - positive_required,
        )
        flip = within_class % DISCOVERY_NOISE_PERIOD == 0
        labels = result["label"].to_numpy(dtype=np.int8)
        result["label"] = np.where(flip, 1 - labels, labels).astype(np.int8)
        noisy.append(result)
    return tuple(noisy)


def generate_synthetic_task(
    property_name: str,
    graph_size: int,
    context_per_class: int,
    noise_regime: str,
    seed: int,
    *,
    max_context_per_class: int | None = None,
    evaluation_per_class: int | None = None,
    graph_seed: int | None = None,
    relabel_seed: int | None = None,
    split_seed: int | None = None,
    noise_seed: int | None = None,
    context_negative_ratio: int = 1,
    evaluation_negative_ratio: int = 1,
) -> SyntheticTask:
    max_context = max_context_per_class or context_per_class
    evaluation_size = evaluation_per_class or context_per_class
    minimum = MIN_GRAPH_SIZE.get(property_name)
    if minimum is None:
        raise ValueError(f"synthetic property is not implemented: {property_name}")
    if max_context >= 400 and evaluation_size >= 400 and graph_size < minimum:
        raise ValueError(
            f"graph_size={graph_size} is below the registered minimum {minimum} for "
            f"{property_name} at the 400-per-class budget"
        )
    graph_rng = np.random.default_rng(graph_seed if graph_seed is not None else seed)
    positive_required = max_context + evaluation_size
    negative_required = max_context + evaluation_size * evaluation_negative_ratio
    generators = {
        "reflexivity": lambda: _reflexivity(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "symmetry": lambda: _symmetry(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "antisymmetry": lambda: _antisymmetry(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "transitivity": lambda: _transitivity(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "typed_composition_l2": lambda: _typed_composition(
            graph_size, positive_required, negative_required, 2, graph_rng
        ),
        "typed_composition_l3": lambda: _typed_composition(
            graph_size, positive_required, negative_required, 3, graph_rng
        ),
        "local_structural_roles": lambda: _local_roles(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "block_affinity_homophilic": lambda: _block_affinity(
            graph_size, positive_required, negative_required, graph_rng, True
        ),
        "block_affinity_heterophilic": lambda: _block_affinity(
            graph_size, positive_required, negative_required, graph_rng, False
        ),
        "discovery_reciprocity": lambda: _discovery_reciprocity(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "discovery_typed_composition": lambda: _discovery_typed_composition(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "discovery_structural_roles": lambda: _discovery_structural_roles(
            graph_size, positive_required, negative_required, graph_rng
        ),
        "discovery_block_affinity": lambda: _discovery_block_affinity(
            graph_size, positive_required, negative_required, graph_rng
        ),
    }
    clean_graph, positives, negatives, definition = generators[property_name]()
    context, evaluation, population = _balanced_frames(
        positives,
        negatives,
        context_per_class,
        max_context,
        evaluation_size,
        split_seed if split_seed is not None else seed + 20_000,
        context_negative_ratio,
        evaluation_negative_ratio,
    )
    observed_graph = _apply_noise(
        clean_graph,
        graph_size,
        noise_regime,
        noise_seed if noise_seed is not None else seed + 30_000,
    )
    relabel_value = relabel_seed if relabel_seed is not None else seed + 10_000
    clean_graph, relabeled, mapping = _relabel(
        clean_graph,
        (context, evaluation, population),
        relabel_value,
        additional_nodes=observed_graph.nodes(),
    )
    context, evaluation, population = relabeled
    observed_graph = nx.relabel_nodes(observed_graph, mapping, copy=True)
    if property_name.startswith("discovery_"):
        if context_negative_ratio != 1 or evaluation_negative_ratio != 1:
            raise ValueError("discovery tasks require balanced context and evaluation")
        context, evaluation, population = _apply_discovery_label_noise(
            (context, evaluation, population), positive_required
        )

    support_queries = {
        (int(head), int(tail), str(attributes.get("relation", "r0")))
        for head, tail, attributes in clean_graph.edges(data=True)
    }
    sampled_queries = {
        (int(row.head), int(row.tail), str(row.relation))
        for row in population.itertuples(index=False)
    }
    if support_queries & sampled_queries:
        raise ValueError("support graph contains a context or evaluation query")

    metadata: dict[str, object] = {
        "definition": definition,
        "graph_size": graph_size,
        "clean_support_edges": clean_graph.number_of_edges(),
        "observed_support_edges": observed_graph.number_of_edges(),
        "noise_edges": observed_graph.number_of_edges() - clean_graph.number_of_edges(),
        "noise_regime": noise_regime,
        "graph_seed": graph_seed if graph_seed is not None else seed,
        "relabel_seed": relabel_value,
        "split_seed": split_seed if split_seed is not None else seed + 20_000,
        "noise_seed": noise_seed if noise_seed is not None else seed + 30_000,
        "max_context_per_class": max_context,
        "evaluation_per_class": evaluation_size,
        "context_positive_count": context_per_class,
        "context_negative_ratio": context_negative_ratio,
        "evaluation_negative_ratio": evaluation_negative_ratio,
        "entity_relabeling": "independent random non-contiguous 52-bit tokens",
        "clean_support_sha256": _graph_hash(clean_graph),
        "observed_support_sha256": _graph_hash(observed_graph),
        "context_query_sha256": _frame_hash(context),
        "evaluation_query_sha256": _frame_hash(evaluation),
        "population_query_sha256": _frame_hash(population),
    }
    if property_name.startswith("discovery_"):
        metadata.update(
            {
                "label_noise": "deterministic symmetric flips within each latent class",
                "label_noise_selection": (
                    "every tenth position after an independent split-seed permutation "
                    "within each latent class; query_id is never encoded"
                ),
                "label_noise_fraction": 1.0 / DISCOVERY_NOISE_PERIOD,
                "label_noise_period": DISCOVERY_NOISE_PERIOD,
                "latent_rule_sha256": object_sha256(clean_graph.graph["discovery_mapping"]),
                "encoder_access_to_latent_rule": False,
            }
        )
    return SyntheticTask(
        property_name=property_name,
        clean_support_graph=clean_graph,
        support_graph=observed_graph,
        context=context,
        evaluation=evaluation,
        population=population,
        metadata=metadata,
    )
