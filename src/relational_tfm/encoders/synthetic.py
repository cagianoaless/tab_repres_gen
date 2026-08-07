from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.encoders.controls import stratified_joint_derangement
from relational_tfm.protocol import CONTROL_ENCODERS, aligned_encoder

Graph = nx.MultiDiGraph
ROLE_RELATIONS = ("r0", "r1", "r2", "r3")
LEGACY_EMBEDDING_DIM = 64


@dataclass(frozen=True)
class EncodedRows:
    values: np.ndarray
    feature_names: tuple[str, ...]
    provenance: dict[str, object]


def _edge_count(graph: Graph, head: int, tail: int, relation: str | None = None) -> int:
    edges = graph.get_edge_data(head, tail, default={})
    return sum(
        relation is None or str(attributes.get("relation", "r0")) == relation
        for attributes in edges.values()
    )


def _degrees(graph: Graph, node: int) -> tuple[float, float]:
    return float(graph.in_degree(node)), float(graph.out_degree(node))


def _dyadic_row(graph: Graph, head: int, tail: int, relation: str) -> list[float]:
    return [
        float(head == tail),
        float(_edge_count(graph, head, tail, relation) > 0),
        float(_edge_count(graph, tail, head, relation) > 0),
    ]


DYADIC_NAMES = (
    "same_entity",
    "direct_support_edge",
    "reverse_support_edge",
)


def _relation_categories(graph: Graph, population: pd.DataFrame | None) -> tuple[str, ...]:
    relations = {
        str(attributes.get("relation", "r0")) for _, _, attributes in graph.edges(data=True)
    }
    if population is not None:
        relations.update(str(value) for value in population["relation"].unique())
    return tuple(sorted(relations))


def _pointwise_rows(
    graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
    population: pd.DataFrame | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    node_ids = np.asarray(sorted(int(node) for node in graph.nodes()), dtype=np.int64)
    entity_rng = np.random.default_rng(seed + 9_187)
    node_order = entity_rng.permutation(node_ids)
    node_position = {int(node): index for index, node in enumerate(node_order)}
    head_positions = rows["head"].map(node_position).to_numpy(dtype=np.int64)
    tail_positions = rows["tail"].map(node_position).to_numpy(dtype=np.int64)
    entity_tokens = entity_rng.uniform(-1.0, 1.0, size=len(node_ids))
    node_token = {
        int(node): float(token)
        for node, token in zip(node_order, entity_tokens, strict=True)
    }

    relations = _relation_categories(graph, population)
    relation_rng = np.random.default_rng(seed + 19_187)
    relation_order = relation_rng.permutation(relations)
    relation_position = {
        str(relation): index for index, relation in enumerate(relation_order)
    }
    relation_positions = rows["relation"].map(relation_position).to_numpy(dtype=np.int64)
    relation_values = np.zeros((len(rows), len(relations)), dtype=np.float64)
    relation_values[np.arange(len(rows)), relation_positions] = 1.0
    relation_names = tuple(f"relation_code_{index:03d}" for index in range(len(relations)))

    if encoder == "categorical_ordinal":
        relation_tokens = relation_rng.uniform(-1.0, 1.0, size=len(relations))
        relation_token = {
            str(relation): float(token)
            for relation, token in zip(relation_order, relation_tokens, strict=True)
        }
        values = np.column_stack(
            [
                rows["head"].map(node_token).to_numpy(dtype=np.float64),
                rows["tail"].map(node_token).to_numpy(dtype=np.float64),
                rows["relation"].map(relation_token).to_numpy(dtype=np.float64),
            ]
        )
        return values, ("head_random_token", "tail_random_token", "relation_random_token")

    if encoder in {"role_onehot", "unordered_onehot"}:
        if encoder == "role_onehot":
            entity_values = np.zeros((len(rows), 2 * len(node_ids)), dtype=np.float64)
            entity_values[np.arange(len(rows)), head_positions] = 1.0
            entity_values[np.arange(len(rows)), len(node_ids) + tail_positions] = 1.0
            entity_names = tuple(
                [f"head_entity_{index:04d}" for index in range(len(node_ids))]
                + [f"tail_entity_{index:04d}" for index in range(len(node_ids))]
            )
        else:
            entity_values = np.zeros((len(rows), len(node_ids)), dtype=np.float64)
            entity_values[np.arange(len(rows)), head_positions] = 1.0
            entity_values[np.arange(len(rows)), tail_positions] = 1.0
            entity_names = tuple(f"unordered_entity_{index:04d}" for index in range(len(node_ids)))
        return np.column_stack([entity_values, relation_values]), entity_names + relation_names

    rng = np.random.default_rng(seed + 29_187)
    embeddings = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(LEGACY_EMBEDDING_DIM)),
        size=(len(node_ids), LEGACY_EMBEDDING_DIM),
    )
    head = embeddings[head_positions]
    tail = embeddings[tail_positions]
    if encoder == "shared_embedding_symmetric":
        entity_values = np.column_stack([head + tail, np.abs(head - tail), head * tail])
        entity_names = tuple(
            f"{block}_{index:03d}"
            for block in ("embedding_sum", "embedding_absdiff", "embedding_product")
            for index in range(LEGACY_EMBEDDING_DIM)
        )
    elif encoder == "shared_embedding_directed":
        entity_values = np.column_stack([head, tail, head - tail, head * tail])
        entity_names = tuple(
            f"{block}_{index:03d}"
            for block in ("head_embedding", "tail_embedding", "embedding_diff", "embedding_product")
            for index in range(LEGACY_EMBEDDING_DIM)
        )
    elif encoder == "embedding_diff":
        entity_values = head - tail
        entity_names = tuple(
            f"embedding_diff_{index:03d}" for index in range(LEGACY_EMBEDDING_DIM)
        )
    else:
        raise ValueError(f"unsupported pointwise encoder: {encoder}")
    return np.column_stack([entity_values, relation_values]), entity_names + relation_names


def _relation_degree(graph: Graph, node: int, relation: str, incoming: bool) -> int:
    edges = graph.in_edges(node, data=True) if incoming else graph.out_edges(node, data=True)
    return sum(str(attributes.get("relation", "r0")) == relation for *_, attributes in edges)


def _role_row(graph: Graph, head: int, tail: int) -> list[float]:
    head_in, head_out = _degrees(graph, head)
    tail_in, tail_out = _degrees(graph, tail)
    head_successors = set(graph.successors(head))
    tail_successors = set(graph.successors(tail))
    head_predecessors = set(graph.predecessors(head))
    tail_predecessors = set(graph.predecessors(tail))
    head_signature = []
    tail_signature = []
    for relation in ROLE_RELATIONS:
        head_signature.extend(
            [
                _relation_degree(graph, head, relation, True),
                _relation_degree(graph, head, relation, False),
            ]
        )
        tail_signature.extend(
            [
                _relation_degree(graph, tail, relation, True),
                _relation_degree(graph, tail, relation, False),
            ]
        )
    return [
        head_in,
        head_out,
        tail_in,
        tail_out,
        head_in - tail_in,
        head_out - tail_out,
        float(len(head_successors & tail_successors)),
        float(len(head_predecessors & tail_predecessors)),
        *[float(value) for value in head_signature + tail_signature],
        float(head_signature == tail_signature),
    ]


ROLE_NAMES = (
    "head_in_degree",
    "head_out_degree",
    "tail_in_degree",
    "tail_out_degree",
    "in_degree_difference",
    "out_degree_difference",
    "common_successors",
    "common_predecessors",
) + tuple(
    f"{endpoint}_{direction}_{relation}_degree"
    for endpoint in ("head", "tail")
    for relation in ROLE_RELATIONS
    for direction in ("in", "out")
) + ("same_relation_role_signature",)


def _path_count(graph: Graph, head: int, tail: int, length: int) -> int:
    frontier: dict[int, int] = {head: 1}
    for _ in range(length):
        next_frontier: dict[int, int] = {}
        for node, count in frontier.items():
            for neighbor in graph.successors(node):
                next_frontier[neighbor] = next_frontier.get(neighbor, 0) + count
        frontier = next_frontier
    return frontier.get(tail, 0)


def _typed_path_count(graph: Graph, head: int, tail: int, relations: tuple[str, ...]) -> int:
    frontier: dict[int, int] = {head: 1}
    for relation in relations:
        next_frontier: dict[int, int] = {}
        for node, count in frontier.items():
            for _, neighbor, attributes in graph.out_edges(node, data=True):
                if str(attributes.get("relation", "r0")) == relation:
                    next_frontier[neighbor] = next_frontier.get(neighbor, 0) + count
        frontier = next_frontier
    return frontier.get(tail, 0)


def _path_row(graph: Graph, head: int, tail: int) -> list[float]:
    simple = nx.DiGraph(graph)
    try:
        forward_distance = nx.shortest_path_length(simple, head, tail)
    except nx.NetworkXNoPath:
        forward_distance = None
    try:
        reverse_distance = nx.shortest_path_length(simple, tail, head)
    except nx.NetworkXNoPath:
        reverse_distance = None
    return [
        float(forward_distance is not None),
        float(reverse_distance is not None),
        0.0 if forward_distance is None else 1.0 / float(forward_distance),
        0.0 if reverse_distance is None else 1.0 / float(reverse_distance),
        float(_path_count(graph, head, tail, 2)),
        float(_path_count(graph, tail, head, 2)),
        float(_path_count(graph, head, tail, 3)),
        float(_path_count(graph, tail, head, 3)),
        float(_typed_path_count(graph, head, tail, ("r1", "r2"))),
        float(_typed_path_count(graph, tail, head, ("r1", "r2"))),
        float(_typed_path_count(graph, head, tail, ("r1", "r2", "r3"))),
        float(_typed_path_count(graph, tail, head, ("r1", "r2", "r3"))),
    ]


def _successor_index(
    graph: Graph,
    relation: str | None,
) -> dict[int, tuple[int, ...]]:
    if relation is None:
        return {
            int(node): tuple(int(neighbor) for neighbor in graph.successors(node))
            for node in graph.nodes()
        }
    indexed: dict[int, list[int]] = {int(node): [] for node in graph.nodes()}
    for head, tail, attributes in graph.edges(data=True):
        if str(attributes.get("relation", "r0")) == relation:
            indexed[int(head)].append(int(tail))
    return {node: tuple(neighbors) for node, neighbors in indexed.items()}


def _batched_path_maps(
    graph: Graph,
    sources: set[int],
    relations: tuple[str | None, ...],
) -> dict[int, tuple[dict[int, int], ...]]:
    indices = tuple(_successor_index(graph, relation) for relation in relations)
    step_caches: list[
        dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    ] = [{} for _ in relations]
    result: dict[int, tuple[dict[int, int], ...]] = {}
    for source in sources:
        frontier: tuple[tuple[int, int], ...] = ((source, 1),)
        steps: list[dict[int, int]] = []
        for step, index in enumerate(indices):
            cached = step_caches[step].get(frontier)
            if cached is None:
                following: dict[int, int] = {}
                for node, count in frontier:
                    for neighbor in index.get(node, ()):
                        following[neighbor] = following.get(neighbor, 0) + count
                cached = tuple(sorted(following.items()))
                step_caches[step][frontier] = cached
            frontier = cached
            steps.append(dict(frontier))
        result[source] = tuple(steps)
    return result


def _path_rows(graph: Graph, rows: pd.DataFrame) -> list[list[float]]:
    sources = set(map(int, rows["head"])) | set(map(int, rows["tail"]))
    simple = nx.DiGraph(graph)
    distances = {
        source: dict(nx.single_source_shortest_path_length(simple, source))
        for source in sources
    }
    untyped = _batched_path_maps(graph, sources, (None, None, None))
    typed = _batched_path_maps(graph, sources, ("r1", "r2", "r3"))
    encoded: list[list[float]] = []
    for row in rows.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        forward_distance = distances[head].get(tail)
        reverse_distance = distances[tail].get(head)
        encoded.append(
            [
                float(forward_distance is not None),
                float(reverse_distance is not None),
                (
                    0.0
                    if forward_distance is None
                    else 1.0 / float(forward_distance)
                ),
                (
                    0.0
                    if reverse_distance is None
                    else 1.0 / float(reverse_distance)
                ),
                float(untyped[head][1].get(tail, 0)),
                float(untyped[tail][1].get(head, 0)),
                float(untyped[head][2].get(tail, 0)),
                float(untyped[tail][2].get(head, 0)),
                float(typed[head][1].get(tail, 0)),
                float(typed[tail][1].get(head, 0)),
                float(typed[head][2].get(tail, 0)),
                float(typed[tail][2].get(head, 0)),
            ]
        )
    return encoded


PATH_NAMES = (
    "forward_reachable",
    "reverse_reachable",
    "forward_inverse_distance",
    "reverse_inverse_distance",
    "forward_path_count_l2",
    "reverse_path_count_l2",
    "forward_path_count_l3",
    "reverse_path_count_l3",
    "forward_typed_path_count_r1_r2",
    "reverse_typed_path_count_r1_r2",
    "forward_typed_path_count_r1_r2_r3",
    "reverse_typed_path_count_r1_r2_r3",
)


def _same_relation_distance_maps(
    graph: Graph, relations: set[str]
) -> dict[str, dict[int, dict[int, int]]]:
    result = {}
    for relation in relations:
        relation_graph = nx.DiGraph()
        relation_graph.add_nodes_from(graph.nodes())
        relation_graph.add_edges_from(
            (left, right)
            for left, right, attributes in graph.edges(data=True)
            if str(attributes.get("relation", "r0")) == relation
        )
        result[relation] = {
            int(source): {int(target): int(distance) for target, distance in distances.items()}
            for source, distances in nx.all_pairs_shortest_path_length(relation_graph)
        }
    return result


def _same_relation_reachability_row(
    graph: Graph,
    head: int,
    tail: int,
    relation: str,
    distance_maps: dict[str, dict[int, dict[int, int]]],
) -> list[float]:
    relation_distances = distance_maps[relation]
    forward_distance = relation_distances.get(head, {}).get(tail)
    reverse_distance = relation_distances.get(tail, {}).get(head)
    forward = forward_distance is not None and forward_distance > 0
    reverse = reverse_distance is not None and reverse_distance > 0
    forward_inverse = 0.0 if not forward else 1.0 / float(forward_distance)
    reverse_inverse = 0.0 if not reverse else 1.0 / float(reverse_distance)
    return [
        float(head == tail),
        float(_edge_count(graph, head, tail, relation) > 0),
        float(_edge_count(graph, tail, head, relation) > 0),
        float(forward),
        float(reverse),
        forward_inverse,
        reverse_inverse,
        float(forward) - float(reverse),
        forward_inverse - reverse_inverse,
    ]


REACHABILITY_NAMES = (
    "graph_same_entity",
    "graph_direct_edge",
    "graph_reverse_edge",
    "graph_forward_reachable",
    "graph_reverse_reachable",
    "graph_forward_inverse_distance",
    "graph_reverse_inverse_distance",
    "graph_reachability_direction",
    "graph_inverse_distance_direction",
)


def _targeted_typed_path_row(graph: Graph, head: int, tail: int, max_length: int) -> list[float]:
    values: list[float] = []
    for relations in (("r1", "r2"), ("r1", "r2", "r3")):
        if len(relations) > max_length:
            continue
        forward = float(_typed_path_count(graph, head, tail, relations))
        reverse = float(_typed_path_count(graph, tail, head, relations))
        values.extend([forward, reverse, float(forward > 0.0), float(reverse > 0.0)])
    return values


def _targeted_typed_path_rows(
    graph: Graph,
    rows: pd.DataFrame,
    max_length: int,
) -> list[list[float]]:
    sources = set(map(int, rows["head"])) | set(map(int, rows["tail"]))
    typed = _batched_path_maps(graph, sources, ("r1", "r2", "r3"))
    encoded: list[list[float]] = []
    for row in rows.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        values: list[float] = []
        for step in range(1, max_length):
            forward = float(typed[head][step].get(tail, 0))
            reverse = float(typed[tail][step].get(head, 0))
            values.extend(
                [forward, reverse, float(forward > 0.0), float(reverse > 0.0)]
            )
        encoded.append(values)
    return encoded


def _targeted_typed_path_names(max_length: int) -> tuple[str, ...]:
    names = []
    for label, length in (("r1_r2", 2), ("r1_r2_r3", 3)):
        if length > max_length:
            continue
        names.extend(
            [
                f"forward_typed_path_count_{label}",
                f"reverse_typed_path_count_{label}",
                f"forward_typed_path_exists_{label}",
                f"reverse_typed_path_exists_{label}",
            ]
        )
    return tuple(names)


def _community_rows(graph: Graph, rows: pd.DataFrame) -> list[list[float]]:
    simple = nx.Graph(graph)
    components = {
        int(node): index
        for index, component in enumerate(nx.connected_components(simple))
        for node in component
    }
    scale = max(graph.number_of_nodes() - 1, 1)
    encoded = []
    for row in rows.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        head_community = int(graph.nodes[head].get("community", -1))
        tail_community = int(graph.nodes[tail].get("community", -1))
        encoded.append(
            [
                float(head_community == tail_community and head_community >= 0),
                float(head_community),
                float(tail_community),
                float(components[head] == components[tail]),
                float(graph.degree(head)) / scale,
                float(graph.degree(tail)) / scale,
            ]
        )
    return encoded


COMMUNITY_NAMES = (
    "same_community",
    "head_community",
    "tail_community",
    "same_connected_component",
    "head_normalized_degree",
    "tail_normalized_degree",
)


DISCOVERY_QUERY_RELATIONS = ("q0", "q1", "q2", "q3")
DISCOVERY_SUPPORT_RELATIONS = ("r0", "r1", "r2", "r3")
DISCOVERY_COMPOSITION_TEMPLATES = (
    ("r0", "r1"),
    ("r2", "r3"),
    ("r4", "r5"),
    ("r6", "r7"),
)


def _query_relation_row(relation: str) -> list[float]:
    if relation not in DISCOVERY_QUERY_RELATIONS:
        raise ValueError(f"unsupported discovery query relation: {relation}")
    return [float(relation == candidate) for candidate in DISCOVERY_QUERY_RELATIONS]


DISCOVERY_QUERY_NAMES = tuple(
    f"query_relation_{index}" for index in range(len(DISCOVERY_QUERY_RELATIONS))
)

DISCOVERY_QUERY_AUGMENTED_ENCODERS = {
    "query_dyadic_invariants": "dyadic_invariants",
    "query_local_roles": "local_roles",
    "query_graph_reachability": "graph_reachability",
    "query_typed_path_k2": "typed_path_k2",
    "query_typed_path_k3": "typed_path_k3",
}


def _reciprocity_evidence_row(
    graph: Graph, head: int, tail: int, query_relation: str
) -> list[float]:
    values = _query_relation_row(query_relation)
    for relation in DISCOVERY_SUPPORT_RELATIONS:
        values.extend(
            [
                float(_edge_count(graph, head, tail, relation) > 0),
                float(_edge_count(graph, tail, head, relation) > 0),
                float(_relation_degree(graph, head, relation, True)),
                float(_relation_degree(graph, head, relation, False)),
                float(_relation_degree(graph, tail, relation, True)),
                float(_relation_degree(graph, tail, relation, False)),
            ]
        )
    return values


RECIPROCITY_EVIDENCE_NAMES = DISCOVERY_QUERY_NAMES + tuple(
    f"{feature}_{relation}"
    for relation in DISCOVERY_SUPPORT_RELATIONS
    for feature in (
        "direct_edge",
        "reverse_edge",
        "head_in_degree",
        "head_out_degree",
        "tail_in_degree",
        "tail_out_degree",
    )
)


def _composition_evidence_row(
    graph: Graph, head: int, tail: int, query_relation: str
) -> list[float]:
    values = _query_relation_row(query_relation)
    for template in DISCOVERY_COMPOSITION_TEMPLATES:
        count = float(_typed_path_count(graph, head, tail, template))
        values.extend([count, float(count > 0.0)])
    return values


COMPOSITION_EVIDENCE_NAMES = DISCOVERY_QUERY_NAMES + tuple(
    f"{feature}_template_{index}"
    for index in range(len(DISCOVERY_COMPOSITION_TEMPLATES))
    for feature in ("path_count", "path_exists")
)


def _role_evidence_row(graph: Graph, head: int, tail: int, query_relation: str) -> list[float]:
    values = _query_relation_row(query_relation)
    for relation in DISCOVERY_SUPPORT_RELATIONS:
        values.extend(
            [
                float(_relation_degree(graph, head, relation, True)),
                float(_relation_degree(graph, head, relation, False)),
                float(_relation_degree(graph, tail, relation, True)),
                float(_relation_degree(graph, tail, relation, False)),
            ]
        )
    return values


ROLE_EVIDENCE_NAMES = DISCOVERY_QUERY_NAMES + tuple(
    f"{endpoint}_{direction}_degree_{relation}"
    for relation in DISCOVERY_SUPPORT_RELATIONS
    for endpoint, direction in (
        ("head", "in"),
        ("head", "out"),
        ("tail", "in"),
        ("tail", "out"),
    )
)


def _topology_evidence_row(
    graph: Graph,
    simple: nx.Graph,
    head: int,
    tail: int,
    query_relation: str,
) -> list[float]:
    head_neighbors = set(simple.neighbors(head))
    tail_neighbors = set(simple.neighbors(tail))
    common = head_neighbors & tail_neighbors
    union = head_neighbors | tail_neighbors
    head_degree = float(len(head_neighbors))
    tail_degree = float(len(tail_neighbors))
    adamic_adar = sum(1.0 / np.log(max(float(simple.degree(node)), 2.0)) for node in common)
    return _query_relation_row(query_relation) + [
        head_degree,
        tail_degree,
        head_degree - tail_degree,
        abs(head_degree - tail_degree),
        float(len(common)),
        0.0 if not union else float(len(common)) / float(len(union)),
        adamic_adar,
        head_degree * tail_degree,
        float(nx.has_path(simple, head, tail)),
    ]


TOPOLOGY_EVIDENCE_NAMES = DISCOVERY_QUERY_NAMES + (
    "head_degree",
    "tail_degree",
    "signed_degree_difference",
    "absolute_degree_difference",
    "common_neighbors",
    "jaccard_similarity",
    "adamic_adar",
    "preferential_attachment",
    "same_connected_component",
)


def _encode_registered(
    graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
    control_population: pd.DataFrame | None = None,
) -> EncodedRows:
    if encoder == "query_relation_only":
        return EncodedRows(
            values=np.asarray(
                [_query_relation_row(str(relation)) for relation in rows["relation"]],
                dtype=np.float64,
            ),
            feature_names=DISCOVERY_QUERY_NAMES,
            provenance={"encoder": encoder, "control": None, "seed": seed},
        )
    if encoder in DISCOVERY_QUERY_AUGMENTED_ENCODERS:
        base_encoder = DISCOVERY_QUERY_AUGMENTED_ENCODERS[encoder]
        base = _encode_registered(
            graph,
            rows,
            base_encoder,
            seed,
            control_population,
        )
        query_values = np.asarray(
            [_query_relation_row(str(relation)) for relation in rows["relation"]],
            dtype=np.float64,
        )
        return EncodedRows(
            values=np.column_stack([query_values, base.values]),
            feature_names=DISCOVERY_QUERY_NAMES + base.feature_names,
            provenance={
                "encoder": encoder,
                "base_encoder": base_encoder,
                "control": None,
                "seed": seed,
            },
        )
    node_ids = sorted(int(node) for node in graph.nodes())
    code_rng = np.random.default_rng(seed + 9_187)
    random_order = code_rng.permutation(node_ids)
    random_tokens = code_rng.uniform(-1.0, 1.0, size=len(node_ids))
    random_code = {
        int(node): float(token)
        for node, token in zip(random_order, random_tokens, strict=True)
    }
    encoded: list[list[float]]
    if encoder in {
        "categorical_ordinal",
        "role_onehot",
        "unordered_onehot",
        "shared_embedding_symmetric",
        "shared_embedding_directed",
        "embedding_diff",
    }:
        values, names = _pointwise_rows(graph, rows, encoder, seed, control_population)
        return EncodedRows(
            values=values,
            feature_names=names,
            provenance={
                "encoder": encoder,
                "control": None,
                "seed": seed,
                "embedding_dim": (
                    LEGACY_EMBEDDING_DIM if "embedding" in encoder else None
                ),
            },
        )
    if encoder in {"artifact_ids", "artifact_random_ids"}:
        encoded = []
        for row in rows.itertuples(index=False):
            head_code = random_code[int(row.head)]
            tail_code = random_code[int(row.tail)]
            encoded.append(
                [head_code, tail_code, head_code - tail_code, abs(head_code - tail_code)]
            )
        names = (
            "head_random_code",
            "tail_random_code",
            "signed_code_gap",
            "absolute_code_gap",
        )
    elif encoder == "dyadic_invariants":
        encoded = [
            _dyadic_row(graph, int(row.head), int(row.tail), str(row.relation))
            for row in rows.itertuples(index=False)
        ]
        names = DYADIC_NAMES
    elif encoder == "local_roles":
        encoded = [
            _dyadic_row(graph, int(row.head), int(row.tail), str(row.relation))
            + _role_row(graph, int(row.head), int(row.tail))
            for row in rows.itertuples(index=False)
        ]
        names = DYADIC_NAMES + ROLE_NAMES
    elif encoder == "typed_paths":
        local_rows = [
            _dyadic_row(graph, int(row.head), int(row.tail), str(row.relation))
            + _role_row(graph, int(row.head), int(row.tail))
            for row in rows.itertuples(index=False)
        ]
        path_rows = _path_rows(graph, rows)
        encoded = [
            local + paths
            for local, paths in zip(local_rows, path_rows, strict=True)
        ]
        names = DYADIC_NAMES + ROLE_NAMES + PATH_NAMES
    elif encoder == "graph_reachability":
        distance_maps = _same_relation_distance_maps(
            graph, {str(value) for value in rows["relation"].unique()}
        )
        encoded = [
            _same_relation_reachability_row(
                graph,
                int(row.head),
                int(row.tail),
                str(row.relation),
                distance_maps,
            )
            for row in rows.itertuples(index=False)
        ]
        names = REACHABILITY_NAMES
    elif encoder in {"typed_path_k2", "typed_path_k3"}:
        max_length = 2 if encoder == "typed_path_k2" else 3
        distance_maps = _same_relation_distance_maps(
            graph, {str(value) for value in rows["relation"].unique()}
        )
        reachability_rows = [
            _same_relation_reachability_row(
                graph,
                int(row.head),
                int(row.tail),
                str(row.relation),
                distance_maps,
            )
            for row in rows.itertuples(index=False)
        ]
        path_rows = _targeted_typed_path_rows(graph, rows, max_length)
        encoded = [
            reachability + paths
            for reachability, paths in zip(
                reachability_rows, path_rows, strict=True
            )
        ]
        names = REACHABILITY_NAMES + _targeted_typed_path_names(max_length)
    elif encoder == "community_global":
        role_rows = [
            _dyadic_row(graph, int(row.head), int(row.tail), str(row.relation))
            + _role_row(graph, int(row.head), int(row.tail))
            for row in rows.itertuples(index=False)
        ]
        global_rows = _community_rows(graph, rows)
        encoded = [left + right for left, right in zip(role_rows, global_rows, strict=True)]
        names = DYADIC_NAMES + ROLE_NAMES + COMMUNITY_NAMES
    elif encoder == "reciprocity_evidence_bank":
        encoded = [
            _reciprocity_evidence_row(graph, int(row.head), int(row.tail), str(row.relation))
            for row in rows.itertuples(index=False)
        ]
        names = RECIPROCITY_EVIDENCE_NAMES
    elif encoder == "composition_evidence_bank":
        encoded = [
            _composition_evidence_row(graph, int(row.head), int(row.tail), str(row.relation))
            for row in rows.itertuples(index=False)
        ]
        names = COMPOSITION_EVIDENCE_NAMES
    elif encoder == "role_evidence_bank":
        encoded = [
            _role_evidence_row(graph, int(row.head), int(row.tail), str(row.relation))
            for row in rows.itertuples(index=False)
        ]
        names = ROLE_EVIDENCE_NAMES
    elif encoder == "topology_evidence_bank":
        simple = nx.Graph(graph)
        encoded = [
            _topology_evidence_row(
                graph,
                simple,
                int(row.head),
                int(row.tail),
                str(row.relation),
            )
            for row in rows.itertuples(index=False)
        ]
        names = TOPOLOGY_EVIDENCE_NAMES
    else:
        raise ValueError(f"synthetic encoder is not implemented: {encoder}")
    values = np.asarray(encoded, dtype=np.float64)
    return EncodedRows(
        values=values,
        feature_names=names,
        provenance={"encoder": encoder, "control": None, "seed": seed},
    )


def _control_rows(
    graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
    property_name: str,
    control_population: pd.DataFrame,
) -> EncodedRows:
    base_encoder = aligned_encoder(property_name)
    base = _encode_registered(graph, control_population, base_encoder, seed, control_population)
    query_ids = control_population["query_id"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed + (71_117 if encoder == "aligned_permutation" else 93_101))
    if encoder == "aligned_permutation":
        controlled = base.values[rng.permutation(len(base.values))]
        control_metadata: dict[str, object] = {}
    elif encoder == "split_relation_permutation":
        if "_control_partition" not in control_population:
            raise ValueError("split-relation control requires frozen partition metadata")
        strata = [
            (str(partition), str(tier), str(relation))
            for partition, tier, relation in zip(
                control_population["_control_partition"],
                control_population["_control_tier"],
                control_population["relation"],
                strict=True,
            )
        ]
        controlled, control_metadata = stratified_joint_derangement(
            base.values,
            strata,
            seed + 81_119,
        )
    else:
        controlled = base.values.copy()
        for column in range(controlled.shape[1]):
            controlled[:, column] = controlled[rng.permutation(len(controlled)), column]
        control_metadata = {}
    by_query = {int(query_id): controlled[index] for index, query_id in enumerate(query_ids)}
    values = np.asarray(
        [by_query[int(query_id)] for query_id in rows["query_id"].to_numpy()],
        dtype=np.float64,
    )
    prefix = {
        "aligned_permutation": "permuted",
        "split_relation_permutation": "split_relation_permuted",
        "dimension_matched_noise": "dimension_matched_noise",
    }[encoder]
    return EncodedRows(
        values=values,
        feature_names=tuple(f"{prefix}_{name}" for name in base.feature_names),
        provenance={
            "encoder": encoder,
            "control": prefix,
            "aligned_source_encoder": base_encoder,
            "reference_population_rows": len(control_population),
            "permutation_strata": (
                ["control_partition", "nested_context_increment", "query_relation"]
                if encoder == "split_relation_permutation"
                else []
            ),
            "uses_target_label": False,
            **control_metadata,
            "seed": seed,
        },
    )


def encode_synthetic_rows(
    graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
    *,
    property_name: str | None = None,
    control_population: pd.DataFrame | None = None,
) -> EncodedRows:
    if encoder in CONTROL_ENCODERS:
        if property_name is None or control_population is None:
            raise ValueError(f"{encoder} requires property_name and control_population")
        result = _control_rows(
            graph, rows, encoder, seed, property_name, control_population
        )
    else:
        result = _encode_registered(graph, rows, encoder, seed, control_population)
    if result.values.ndim != 2 or result.values.shape[0] != len(rows):
        raise ValueError("encoder returned an invalid feature matrix")
    if not np.isfinite(result.values).all():
        raise ValueError("encoder returned non-finite features")
    return result
