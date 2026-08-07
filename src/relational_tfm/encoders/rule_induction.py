from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.encoders.synthetic import EncodedRows

Graph = nx.MultiDiGraph

RULE_INDUCTION_ENCODERS = frozenset(
    {
        "random_entity_tokens",
        "dyadic_premises",
        "bounded_path_evidence",
        "proof_oracle",
    }
)

DYADIC_FEATURE_NAMES = (
    "query_polarity",
    "same_entity",
    "direct_premise",
    "reverse_premise",
    "head_in_degree_normalized",
    "head_out_degree_normalized",
    "tail_in_degree_normalized",
    "tail_out_degree_normalized",
    "common_predecessors_normalized",
    "common_successors_normalized",
)

BOUNDED_PATH_FEATURE_NAMES = (
    "forward_path_count_l2_normalized",
    "reverse_path_count_l2_normalized",
    "forward_path_count_l3_normalized",
    "reverse_path_count_l3_normalized",
)

PROOF_FEATURE_NAMES = (
    "forward_reachable",
    "reverse_reachable",
    "same_weak_component",
    "forward_inverse_distance",
    "reverse_inverse_distance",
)


def _relation_graph(graph: Graph, relation: str) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes())
    result.add_edges_from(
        (int(head), int(tail))
        for head, tail, attributes in graph.edges(data=True)
        if str(attributes.get("relation", "r0")) == relation
    )
    return result


def _path_count(graph: nx.DiGraph, head: int, tail: int, length: int) -> int:
    frontier: dict[int, int] = {head: 1}
    for _ in range(length):
        following: dict[int, int] = {}
        for node, count in frontier.items():
            for neighbor in graph.successors(node):
                following[int(neighbor)] = following.get(int(neighbor), 0) + count
        frontier = following
    return frontier.get(tail, 0)


def _random_token_rows(
    graph: Graph,
    rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    node_ids = np.asarray(sorted(map(int, graph.nodes())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(node_ids)
    tokens = rng.uniform(-1.0, 1.0, size=len(node_ids))
    by_node = {
        int(node): float(token)
        for node, token in zip(random_order, tokens, strict=True)
    }
    values = np.column_stack(
        [
            rows["head"].map(by_node).to_numpy(dtype=np.float64),
            rows["tail"].map(by_node).to_numpy(dtype=np.float64),
            rows["query_polarity"].to_numpy(dtype=np.float64),
        ]
    )
    return values, (
        "head_random_token",
        "tail_random_token",
        "query_polarity",
    )


def _dyadic_row(
    graph: nx.DiGraph,
    head: int,
    tail: int,
    polarity: int,
) -> list[float]:
    scale = float(max(graph.number_of_nodes() - 1, 1))
    predecessors_head = set(graph.predecessors(head))
    predecessors_tail = set(graph.predecessors(tail))
    successors_head = set(graph.successors(head))
    successors_tail = set(graph.successors(tail))
    return [
        float(polarity),
        float(head == tail),
        float(graph.has_edge(head, tail)),
        float(graph.has_edge(tail, head)),
        float(graph.in_degree(head)) / scale,
        float(graph.out_degree(head)) / scale,
        float(graph.in_degree(tail)) / scale,
        float(graph.out_degree(tail)) / scale,
        float(len(predecessors_head & predecessors_tail)) / scale,
        float(len(successors_head & successors_tail)) / scale,
    ]


def encode_rule_induction_rows(
    support_graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
) -> EncodedRows:
    """Encode signed candidate conclusions using only observed positive premises."""

    if encoder not in RULE_INDUCTION_ENCODERS:
        raise ValueError(f"unsupported rule-induction encoder: {encoder}")
    required_columns = {
        "head",
        "tail",
        "relation",
        "query_polarity",
    }
    if not required_columns <= set(rows.columns):
        raise ValueError(
            f"rule-induction rows omit columns: {sorted(required_columns - set(rows.columns))}"
        )
    if set(map(int, rows["query_polarity"].unique())) - {-1, 1}:
        raise ValueError("query_polarity must contain only -1 and 1")

    if encoder == "random_entity_tokens":
        values, feature_names = _random_token_rows(support_graph, rows, seed)
    else:
        relation_graphs = {
            relation: _relation_graph(support_graph, relation)
            for relation in sorted(map(str, rows["relation"].unique()))
        }
        undirected_graphs = {
            relation: graph.to_undirected()
            for relation, graph in relation_graphs.items()
        }
        values_list: list[list[float]] = []
        for row in rows.itertuples(index=False):
            relation = str(row.relation)
            graph = relation_graphs[relation]
            undirected = undirected_graphs[relation]
            head = int(row.head)
            tail = int(row.tail)
            values = _dyadic_row(
                graph,
                head,
                tail,
                int(row.query_polarity),
            )
            if encoder in {"bounded_path_evidence", "proof_oracle"}:
                scale = float(max(graph.number_of_nodes() - 1, 1))
                values.extend(
                    [
                        float(_path_count(graph, head, tail, 2)) / scale,
                        float(_path_count(graph, tail, head, 2)) / scale,
                        float(_path_count(graph, head, tail, 3)) / scale,
                        float(_path_count(graph, tail, head, 3)) / scale,
                    ]
                )
            if encoder == "proof_oracle":
                try:
                    forward_distance = int(nx.shortest_path_length(graph, head, tail))
                except nx.NetworkXNoPath:
                    forward_distance = 0
                try:
                    reverse_distance = int(nx.shortest_path_length(graph, tail, head))
                except nx.NetworkXNoPath:
                    reverse_distance = 0
                values.extend(
                    [
                        float(forward_distance > 0),
                        float(reverse_distance > 0),
                        float(nx.has_path(undirected, head, tail)),
                        0.0 if forward_distance == 0 else 1.0 / forward_distance,
                        0.0 if reverse_distance == 0 else 1.0 / reverse_distance,
                    ]
                )
            values_list.append(values)
        values = np.asarray(values_list, dtype=np.float64)
        feature_names = DYADIC_FEATURE_NAMES
        if encoder in {"bounded_path_evidence", "proof_oracle"}:
            feature_names += BOUNDED_PATH_FEATURE_NAMES
        if encoder == "proof_oracle":
            feature_names += PROOF_FEATURE_NAMES

    if values.ndim != 2 or values.shape[0] != len(rows):
        raise ValueError("rule-induction encoder returned an invalid feature matrix")
    if values.shape[1] != len(feature_names):
        raise ValueError("rule-induction feature names do not match the matrix width")
    if not np.isfinite(values).all():
        raise ValueError("rule-induction encoder returned non-finite values")
    return EncodedRows(
        values=values,
        feature_names=feature_names,
        provenance={
            "encoder": encoder,
            "seed": seed,
            "input": "support_graph_and_signed_candidate_rows",
            "closure_or_relation_matrix_exposed": False,
        },
    )
