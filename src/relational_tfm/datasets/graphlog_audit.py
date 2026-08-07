from __future__ import annotations

from collections import Counter
from typing import Any

from relational_tfm.datasets.graphlog import GraphLogTask


def audit_graphlog_task(task: GraphLogTask) -> dict[str, Any]:
    context = task.context
    evaluation = task.evaluation
    if set(map(int, context["chain_length"].unique())) - {2, 3}:
        raise ValueError("GraphLog context contains an unregistered chain length")
    if set(map(int, evaluation["chain_length"].unique())) - {4, 5}:
        raise ValueError("GraphLog evaluation contains an unregistered chain length")
    if set(context["row_id"]) & set(evaluation["row_id"]):
        raise ValueError("GraphLog context and evaluation rows overlap")
    if context["query_id"].duplicated().any() or evaluation["query_id"].duplicated().any():
        raise ValueError("GraphLog query IDs are not unique within a split")
    context_counts = Counter(map(int, context["label"]))
    expected = int(task.metadata["context_per_relation"])
    if not context_counts or set(context_counts.values()) != {expected}:
        raise ValueError("GraphLog context is not balanced by relation")
    evaluation_labels = set(map(int, evaluation["label"]))
    if not evaluation_labels <= set(context_counts):
        raise ValueError("GraphLog evaluation contains an unseen class")
    if not context["relation_tokens"].map(len).eq(context["chain_length"]).all():
        raise ValueError("GraphLog context descriptor lengths are inconsistent")
    if not evaluation["relation_tokens"].map(len).eq(evaluation["chain_length"]).all():
        raise ValueError("GraphLog evaluation descriptor lengths are inconsistent")
    return {
        "status": "PASS",
        "world": task.metadata["world"],
        "outer_split": task.metadata["outer_split"],
        "context_rows": len(context),
        "evaluation_rows": len(evaluation),
        "context_class_count": len(context_counts),
        "evaluation_class_count": len(evaluation_labels),
        "context_counts_per_class": sorted(set(context_counts.values())),
        "context_evaluation_disjoint": True,
        "entity_ids_exposed": False,
        "target_relation_exposed": False,
        "symbolic_input_field": "descriptor",
    }
