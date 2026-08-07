from __future__ import annotations

from collections import Counter
from typing import Any

from relational_tfm.datasets.compositional_shift import (
    COMPOSITION_CLASS_COUNT,
    COMPOSITION_TOKENS,
    CompositionalShiftTask,
    compose_tokens,
)


def audit_compositional_shift_task(
    task: CompositionalShiftTask,
) -> dict[str, Any]:
    context = task.context
    evaluation = task.evaluation
    expected_labels = set(range(COMPOSITION_CLASS_COUNT))

    if set(map(int, context["label"])) != expected_labels:
        raise ValueError("D8 context does not cover every class")
    if set(map(int, evaluation["label"])) != expected_labels:
        raise ValueError("D8 evaluation does not cover every class")
    if set(map(int, context["chain_length"])) != {2, 3}:
        raise ValueError("D8 context must contain only lengths two and three")
    if set(map(int, evaluation["chain_length"])) != set(range(4, 11)):
        raise ValueError("D8 evaluation must contain lengths four through ten")

    context_counts = Counter(map(int, context["label"]))
    if len(set(context_counts.values())) != 1:
        raise ValueError("D8 context is not balanced by class")
    if next(iter(context_counts.values())) != task.metadata["context_per_class"]:
        raise ValueError("D8 context size disagrees with metadata")

    expected_per_length = int(task.metadata["evaluation_per_class_per_length"])
    grouped = evaluation.groupby(["chain_length", "label"]).size()
    if len(grouped) != 7 * COMPOSITION_CLASS_COUNT or not grouped.eq(
        expected_per_length
    ).all():
        raise ValueError("D8 evaluation is not balanced by length and class")

    context_sequences = {
        tuple(map(int, tokens)) for tokens in context["relation_tokens"]
    }
    evaluation_sequences = [
        tuple(map(int, tokens)) for tokens in evaluation["relation_tokens"]
    ]
    if context_sequences & set(evaluation_sequences):
        raise ValueError("D8 context and evaluation sequences overlap")
    if len(evaluation_sequences) != len(set(evaluation_sequences)):
        raise ValueError("D8 evaluation sequences are not unique")

    minimum_id = 2**40
    for row in list(context.itertuples(index=False)) + list(
        evaluation.itertuples(index=False)
    ):
        tokens = tuple(map(int, row.relation_tokens))
        edges = tuple(row.story_edges)
        query_edge = tuple(row.query_edge)
        if len(tokens) != int(row.chain_length) or len(edges) != len(tokens):
            raise ValueError("D8 row length is inconsistent")
        if any(token not in COMPOSITION_TOKENS for token in tokens):
            raise ValueError("D8 row contains an unknown relation token")
        if compose_tokens(tokens) != int(row.label):
            raise ValueError("D8 row label disagrees with the group oracle")
        if any(
            edges[index][1] != edges[index + 1][0]
            for index in range(len(edges) - 1)
        ):
            raise ValueError("D8 entity relabeling broke path continuity")
        if query_edge != (edges[0][0], edges[-1][1]):
            raise ValueError("D8 query endpoints disagree with the path")
        if min(node for edge in edges for node in edge) < minimum_id:
            raise ValueError("D8 entity IDs were not independently randomized")

    return {
        "status": "PASS",
        "algebra": "dihedral_d8",
        "context_rows": len(context),
        "evaluation_rows": len(evaluation),
        "classes": COMPOSITION_CLASS_COUNT,
        "context_per_class": next(iter(context_counts.values())),
        "evaluation_per_class_per_length": expected_per_length,
        "evaluation_unique_sequence_fraction": 1.0,
        "context_evaluation_sequence_overlap": 0,
        "entity_ids_exposed": False,
        "target_relation_exposed": False,
        "oracle_label_failures": 0,
    }
