from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.encoders.controls import stratified_joint_derangement
from relational_tfm.encoders.synthetic import EncodedRows
from relational_tfm.protocol import (
    REAL_ALIGNED_ENCODERS,
    REAL_BASE_ENCODERS,
    REAL_CONTROL_ENCODERS,
    SUPPORTED_REAL_ENCODERS,
)

Graph = nx.MultiDiGraph


def _edge_count(graph: Graph, head: int, tail: int, relation: str) -> int:
    return sum(
        str(attributes.get("relation", "r0")) == relation
        for attributes in graph.get_edge_data(head, tail, default={}).values()
    )


def _registered_rows(
    graph: Graph, rows: pd.DataFrame, encoder: str, seed: int
) -> EncodedRows:
    if encoder not in REAL_BASE_ENCODERS:
        raise ValueError(f"unsupported real base encoder: {encoder}")
    nodes = sorted(int(node) for node in graph.nodes())
    order = np.random.default_rng(seed + 9_187).permutation(nodes)
    code = {int(node): index / max(len(nodes) - 1, 1) for index, node in enumerate(order)}
    if encoder == "artifact_ids":
        values = np.asarray(
            [
                [
                    code[int(row.head)],
                    code[int(row.tail)],
                    code[int(row.head)] - code[int(row.tail)],
                ]
                for row in rows.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        return EncodedRows(
            values,
            ("head_code", "tail_code", "code_difference"),
            {"encoder": encoder, "seed": seed},
        )

    relation_edges: dict[str, int] = {}
    relation_heads: dict[str, set[int]] = {}
    relation_tails: dict[str, set[int]] = {}
    for head, tail, attributes in graph.edges(data=True):
        relation = str(attributes.get("relation", "r0"))
        relation_edges[relation] = relation_edges.get(relation, 0) + 1
        relation_heads.setdefault(relation, set()).add(int(head))
        relation_tails.setdefault(relation, set()).add(int(tail))
    encoded = []
    for row in rows.itertuples(index=False):
        head, tail, relation = int(row.head), int(row.tail), str(row.relation)
        head_successors = set(graph.successors(head))
        tail_predecessors = set(graph.predecessors(tail))
        relation_count = relation_edges.get(relation, 0)
        local = [
            float(head == tail),
            float(_edge_count(graph, head, tail, relation) > 0),
            float(_edge_count(graph, tail, head, relation) > 0),
            np.log1p(graph.in_degree(head)),
            np.log1p(graph.out_degree(head)),
            np.log1p(graph.in_degree(tail)),
            np.log1p(graph.out_degree(tail)),
            np.log1p(len(head_successors & set(graph.successors(tail)))),
            np.log1p(len(set(graph.predecessors(head)) & tail_predecessors)),
            np.log1p(
                sum(
                    str(attributes.get("relation", "r0")) == relation
                    for *_, attributes in graph.out_edges(head, data=True)
                )
            ),
            np.log1p(
                sum(
                    str(attributes.get("relation", "r0")) == relation
                    for *_, attributes in graph.in_edges(tail, data=True)
                )
            ),
            np.log1p(relation_count),
            len(relation_heads.get(relation, ())) / max(relation_count, 1),
            len(relation_tails.get(relation, ())) / max(relation_count, 1),
        ]
        if encoder == "typed_paths":
            length_two = len(head_successors & tail_predecessors)
            same_relation_middle = {
                int(middle)
                for _, middle, attributes in graph.out_edges(head, data=True)
                if str(attributes.get("relation", "r0")) == relation
            } & {
                int(middle)
                for middle, _, attributes in graph.in_edges(tail, data=True)
                if str(attributes.get("relation", "r0")) == relation
            }
            local.extend([np.log1p(length_two), np.log1p(len(same_relation_middle))])
        encoded.append(local)
    names = (
        "same_entity",
        "direct_support",
        "reverse_support",
        "head_in_log",
        "head_out_log",
        "tail_in_log",
        "tail_out_log",
        "common_successors_log",
        "common_predecessors_log",
        "head_relation_out_log",
        "tail_relation_in_log",
        "relation_frequency_log",
        "relation_functionality",
        "relation_inverse_functionality",
    )
    if encoder == "typed_paths":
        names += ("path_count_l2_log", "same_relation_path_count_l2_log")
    return EncodedRows(
        np.asarray(encoded, dtype=np.float64),
        names,
        {"encoder": encoder, "seed": seed},
    )


def encode_real_rows(
    graph: Graph,
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
    *,
    aligned_encoder: str,
    control_population: pd.DataFrame,
) -> EncodedRows:
    if encoder not in SUPPORTED_REAL_ENCODERS:
        raise ValueError(f"unsupported real encoder: {encoder}")
    if aligned_encoder not in REAL_ALIGNED_ENCODERS:
        raise ValueError(f"unsupported aligned real encoder: {aligned_encoder}")
    if encoder not in REAL_CONTROL_ENCODERS:
        return _registered_rows(graph, rows, encoder, seed)
    base = _registered_rows(graph, control_population, aligned_encoder, seed)
    rng = np.random.default_rng(
        seed + (71_117 if encoder == "aligned_permutation" else 93_101)
    )
    controlled = base.values.copy()
    if encoder == "aligned_permutation":
        controlled = controlled[rng.permutation(len(controlled))]
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
            controlled,
            strata,
            seed + 81_119,
        )
    else:
        for column in range(controlled.shape[1]):
            controlled[:, column] = controlled[rng.permutation(len(controlled)), column]
        control_metadata = {}
    by_query = {
        int(query_id): controlled[index]
        for index, query_id in enumerate(control_population["query_id"].to_numpy())
    }
    values = np.asarray([by_query[int(value)] for value in rows["query_id"]], dtype=np.float64)
    return EncodedRows(
        values,
        tuple(f"{encoder}_{name}" for name in base.feature_names),
        {
            "encoder": encoder,
            "aligned_source_encoder": aligned_encoder,
            "control": (
                "split_relation_permuted"
                if encoder == "split_relation_permutation"
                else encoder
            ),
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
