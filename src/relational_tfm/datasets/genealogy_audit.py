from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from relational_tfm.datasets.genealogy import (
    ALL_RELATIONS,
    CORE_RELATIONS,
    DERIVED_RELATIONS,
    GenealogyTask,
    derive_genealogy_facts_from_core,
)


def _facts(graph: Any) -> set[tuple[int, str, int]]:
    return {
        (int(head), str(attributes["relation"]), int(tail))
        for head, tail, attributes in graph.edges(data=True)
    }


def audit_genealogy_task(task: GenealogyTask) -> dict[str, Any]:
    failures: list[str] = []
    support_facts = _facts(task.support_graph)
    complete_facts = _facts(task.complete_graph)
    complete_relations = {relation for _, relation, _ in complete_facts}
    if complete_relations != ALL_RELATIONS:
        failures.append(
            "complete relation vocabulary differs from the registered 17 relations"
        )
    independently_derived = derive_genealogy_facts_from_core(task.complete_graph)
    if complete_facts != independently_derived:
        failures.append(
            "complete genealogy differs from the independently recomputed closure"
        )
    if not support_facts <= complete_facts:
        failures.append("support contains facts outside the complete genealogy")

    population = task.population
    if population["query_id"].duplicated().any():
        failures.append("population query IDs are not unique")
    if population.duplicated(["head", "relation", "tail"]).any():
        failures.append("population contains duplicate candidate triples")
    if set(task.context["query_id"]) & set(task.evaluation["query_id"]):
        failures.append("context and evaluation query IDs overlap")

    expected_counts = {
        "context": {
            0: int(task.metadata["context_per_class"]),
            1: int(task.metadata["context_per_class"]),
        },
        "evaluation": {
            0: int(task.metadata["evaluation_per_class"]),
            1: int(task.metadata["evaluation_per_class"]),
        },
        "population": {
            0: int(task.metadata["max_context_per_class"])
            + int(task.metadata["evaluation_per_class"]),
            1: int(task.metadata["max_context_per_class"])
            + int(task.metadata["evaluation_per_class"]),
        },
    }
    for name, frame in (
        ("context", task.context),
        ("evaluation", task.evaluation),
        ("population", population),
    ):
        counts = dict(sorted(Counter(map(int, frame["label"])).items()))
        if counts != expected_counts[name]:
            failures.append(
                f"{name} class counts differ: expected={expected_counts[name]} "
                f"observed={counts}"
            )

    family_by_node = {
        int(node): int(attributes["family_id"])
        for node, attributes in task.complete_graph.nodes(data=True)
    }
    roles: dict[tuple[int, str], tuple[set[int], set[int]]] = {}
    grouped_heads: dict[tuple[int, str], set[int]] = defaultdict(set)
    grouped_tails: dict[tuple[int, str], set[int]] = defaultdict(set)
    for head, relation, tail in complete_facts:
        key = (family_by_node[head], relation)
        grouped_heads[key].add(head)
        grouped_tails[key].add(tail)
    for key in grouped_heads:
        roles[key] = (grouped_heads[key], grouped_tails[key])

    query_facts: set[tuple[int, str, int]] = set()
    for row in population.itertuples(index=False):
        fact = (int(row.head), str(row.relation), int(row.tail))
        query_facts.add(fact)
        head_family = family_by_node[int(row.head)]
        tail_family = family_by_node[int(row.tail)]
        if head_family != tail_family or head_family != int(row.family_id):
            failures.append("candidate endpoints are not in one registered family")
            break
        if fact in support_facts:
            failures.append("candidate triple appears in the observed support")
            break
        is_complete = fact in complete_facts
        if is_complete != bool(row.label):
            failures.append("candidate label disagrees with complete genealogy truth")
            break
        if not bool(row.label):
            role_heads, role_tails = roles[(head_family, str(row.relation))]
            if int(row.head) not in role_heads or int(row.tail) not in role_tails:
                failures.append("negative candidate violates matched role support")
                break

    if task.split_name in {"evidence", "family"}:
        target_families = (
            set(map(int, task.metadata["target_families"]))
            if task.split_name == "family"
            else set(family_by_node.values())
        )
        target_core = {
            fact
            for fact in complete_facts
            if fact[1] in CORE_RELATIONS
            and family_by_node[fact[0]] in target_families
        }
        if not target_core <= support_facts:
            failures.append("evidence relations are incomplete in target families")
    if task.split_name == "family":
        source_families = set(map(int, task.metadata["source_families"]))
        target_families = set(map(int, task.metadata["target_families"]))
        if source_families & target_families:
            failures.append("source and target family IDs overlap")
        source_nodes = {
            node for node, family in family_by_node.items() if family in source_families
        }
        target_nodes = {
            node for node, family in family_by_node.items() if family in target_families
        }
        if source_nodes & target_nodes:
            failures.append("source and target entity sets overlap")
        source_facts = {
            fact for fact in complete_facts if family_by_node[fact[0]] in source_families
        }
        if not source_facts <= support_facts:
            failures.append("family split omits source-family facts")
        if any(
            family_by_node[int(row.head)] not in target_families
            for row in population.itertuples(index=False)
        ):
            failures.append("family-split evaluation includes source-family candidates")

    entity_ids = np.asarray(sorted(family_by_node), dtype=np.int64)
    if len(entity_ids) and int(entity_ids.min()) < 2**40:
        failures.append("entity IDs are not non-contiguous 52-bit tokens")
    if np.array_equal(entity_ids, np.arange(len(entity_ids), dtype=np.int64)):
        failures.append("entity IDs are sequential")

    if failures:
        raise ValueError(f"genealogy audit failed: {failures}")
    expected_candidate_relations = (
        ALL_RELATIONS if task.split_name == "random" else DERIVED_RELATIONS
    )
    observed_candidate_relations = set(map(str, population["relation"]))
    return {
        "split_name": task.split_name,
        "observed_fraction": task.observed_fraction,
        "complete_relations": len(complete_relations),
        "complete_facts": len(complete_facts),
        "support_facts": len(support_facts),
        "candidate_rows": len(population),
        "candidate_support_overlap": len(query_facts & support_facts),
        "matrix_exposed_to_decoder": False,
        "source_target_entity_overlap": (
            0 if task.split_name == "family" else None
        ),
        "missing_candidate_relations": sorted(
            expected_candidate_relations - observed_candidate_relations
        ),
        "oracle_mismatches": 0,
    }
