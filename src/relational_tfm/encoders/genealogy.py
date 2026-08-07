from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.datasets.genealogy import (
    ALL_RELATIONS,
    CORE_RELATIONS,
    DERIVED_RELATIONS,
    derive_genealogy_facts_from_core,
)
from relational_tfm.encoders.synthetic import EncodedRows

Graph = nx.MultiDiGraph

GENEALOGY_ENCODERS = frozenset(
    {"relation_local", "bounded_core_paths", "typed_rule_evidence"}
)
RELATION_ORDER = tuple(sorted(ALL_RELATIONS))
DERIVED_ORDER = tuple(sorted(DERIVED_RELATIONS))


def _relation_edges(graph: Graph) -> dict[str, set[tuple[int, int]]]:
    edges: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for head, tail, attributes in graph.edges(data=True):
        edges[str(attributes["relation"])].add((int(head), int(tail)))
    return edges


def _core_graph(graph: Graph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes())
    result.add_edges_from(
        (int(head), int(tail))
        for head, tail, attributes in graph.edges(data=True)
        if str(attributes["relation"]) in CORE_RELATIONS
    )
    return result


def _path_count(graph: nx.DiGraph, head: int, tail: int, length: int) -> int:
    frontier: dict[int, int] = {head: 1}
    for _ in range(length):
        following: dict[int, int] = defaultdict(int)
        for node, count in frontier.items():
            for neighbor in graph.successors(node):
                following[int(neighbor)] += count
        frontier = dict(following)
    return frontier.get(tail, 0)


def encode_genealogy_rows(
    support_graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
) -> EncodedRows:
    if encoder not in GENEALOGY_ENCODERS:
        raise ValueError(f"unsupported genealogy encoder: {encoder}")
    required = {"head", "tail", "relation"}
    if not required <= set(rows.columns):
        raise ValueError(
            f"genealogy rows omit columns: {sorted(required - set(rows.columns))}"
        )
    if set(map(str, rows["relation"].unique())) - ALL_RELATIONS:
        raise ValueError("genealogy rows contain an unregistered target relation")

    relation_edges = _relation_edges(support_graph)
    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(support_graph.nodes())
    collapsed.add_edges_from(
        (int(head), int(tail)) for head, tail in support_graph.edges()
    )
    core = _core_graph(support_graph)
    weak = core.to_undirected()
    proof_by_relation: dict[str, set[tuple[int, int]]] = defaultdict(set)
    if encoder == "typed_rule_evidence":
        for head, relation, tail in derive_genealogy_facts_from_core(support_graph):
            if relation in DERIVED_RELATIONS:
                proof_by_relation[relation].add((head, tail))

    scale = float(max(support_graph.number_of_nodes() - 1, 1))
    values: list[list[float]] = []
    for row in rows.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        relation = str(row.relation)
        row_values = [float(relation == item) for item in RELATION_ORDER]
        row_values.extend(
            [
                float((head, tail) in relation_edges[relation]),
                float((tail, head) in relation_edges[relation]),
                float(collapsed.out_degree(head)) / scale,
                float(collapsed.in_degree(head)) / scale,
                float(collapsed.out_degree(tail)) / scale,
                float(collapsed.in_degree(tail)) / scale,
            ]
        )
        if encoder in {"bounded_core_paths", "typed_rule_evidence"}:
            row_values.extend(
                [
                    float(_path_count(core, head, tail, 2)) / scale,
                    float(_path_count(core, tail, head, 2)) / scale,
                    float(_path_count(core, head, tail, 3)) / scale,
                    float(_path_count(core, tail, head, 3)) / scale,
                    float(nx.has_path(weak, head, tail)),
                ]
            )
        if encoder == "typed_rule_evidence":
            row_values.extend(
                [
                    float((head, tail) in proof_by_relation[item])
                    for item in DERIVED_ORDER
                ]
            )
        values.append(row_values)

    feature_names = tuple(f"query_relation__{item}" for item in RELATION_ORDER) + (
        "direct_target_relation_observed",
        "reverse_target_relation_observed",
        "head_out_degree_normalized",
        "head_in_degree_normalized",
        "tail_out_degree_normalized",
        "tail_in_degree_normalized",
    )
    if encoder in {"bounded_core_paths", "typed_rule_evidence"}:
        feature_names += (
            "forward_core_path_count_l2_normalized",
            "reverse_core_path_count_l2_normalized",
            "forward_core_path_count_l3_normalized",
            "reverse_core_path_count_l3_normalized",
            "same_core_weak_component",
        )
    if encoder == "typed_rule_evidence":
        feature_names += tuple(f"proof__{item}" for item in DERIVED_ORDER)

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (len(rows), len(feature_names)):
        raise ValueError("genealogy feature matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError("genealogy encoder returned non-finite values")
    return EncodedRows(
        values=matrix,
        feature_names=feature_names,
        provenance={
            "encoder": encoder,
            "seed": seed,
            "input": "observed_genealogy_and_candidate_rows",
            "closure_or_relation_matrix_exposed": False,
            "raw_entity_ids_exposed": False,
            "family_generation_or_sex_exposed": False,
        },
    )
