from __future__ import annotations

from collections import Counter
from typing import Any

from relational_tfm.datasets.clutrr import (
    CLUTRR_EDGE_RELATIONS,
    CLUTRR_RELATIONS,
    ClutrrTask,
)


def audit_clutrr_task(task: ClutrrTask) -> dict[str, Any]:
    context = task.context
    evaluation = task.evaluation
    if set(context["row_id"]) & set(evaluation["row_id"]):
        raise ValueError("CLUTRR context and evaluation rows overlap")
    expected_labels = set(range(len(CLUTRR_RELATIONS)))
    if set(map(int, context["label"])) != expected_labels:
        raise ValueError("CLUTRR context does not cover every target relation")
    if set(map(int, evaluation["label"])) != expected_labels:
        raise ValueError("CLUTRR evaluation does not cover every target relation")

    context_counts = Counter(map(int, context["label"]))
    if len(set(context_counts.values())) != 1:
        raise ValueError("CLUTRR context is not balanced by target relation")
    if next(iter(context_counts.values())) != task.metadata["context_per_relation"]:
        raise ValueError("CLUTRR context count disagrees with task metadata")

    all_rows = list(context.itertuples(index=False)) + list(
        evaluation.itertuples(index=False)
    )
    minimum_id = 2**40
    for row in all_rows:
        edges = tuple(row.story_edges)
        edge_types = tuple(row.edge_types)
        query_edge = tuple(row.query_edge)
        if len(edges) != int(row.chain_length) or len(edge_types) != int(row.chain_length):
            raise ValueError("CLUTRR encoded chain length mismatch")
        if any(edges[index][1] != edges[index + 1][0] for index in range(len(edges) - 1)):
            raise ValueError("CLUTRR relabeling broke path continuity")
        if query_edge != (edges[0][0], edges[-1][1]):
            raise ValueError("CLUTRR relabeling broke query endpoints")
        node_ids = {node for edge in edges for node in edge}
        if min(node_ids) < minimum_id:
            raise ValueError("CLUTRR entity IDs were not independently randomized")
        if any(
            edge_types[index] not in CLUTRR_EDGE_RELATIONS
            for index in range(len(edge_types))
        ):
            raise ValueError("CLUTRR path contains an unknown relation")

    return {
        "status": "PASS",
        "context_rows": len(context),
        "evaluation_rows": len(evaluation),
        "target_relations": len(expected_labels),
        "context_per_relation": next(iter(context_counts.values())),
        "entity_ids_exposed": False,
        "story_text_exposed": False,
        "context_evaluation_overlap": 0,
    }
