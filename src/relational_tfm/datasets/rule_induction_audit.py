from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

from relational_tfm.datasets.rule_induction import RuleInductionTask


def _relation_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes())
    result.add_edges_from(
        (int(head), int(tail))
        for head, tail, attributes in graph.edges(data=True)
        if str(attributes.get("relation", "r0")) == "r0"
    )
    return result


def _expected_positive_closure(task: RuleInductionTask) -> nx.DiGraph:
    support = _relation_graph(task.support_graph)
    rule_name = task.rule_name
    if rule_name == "reflexivity":
        expected = support.copy()
        expected.add_edges_from((node, node) for node in support.nodes())
        return expected
    if rule_name in {"irreflexivity", "antisymmetry"}:
        return support
    if rule_name == "symmetry":
        expected = support.copy()
        expected.add_edges_from((tail, head) for head, tail in support.edges())
        return expected
    if rule_name in {"transitivity", "antisymmetry_transitivity"}:
        if not nx.is_directed_acyclic_graph(support):
            raise ValueError(f"{rule_name} support graph is not acyclic")
        return nx.transitive_closure_dag(support)
    if rule_name == "symmetry_transitivity":
        expected = nx.DiGraph()
        expected.add_nodes_from(support.nodes())
        undirected = support.to_undirected()
        for component in nx.connected_components(undirected):
            if len(component) < 2:
                continue
            expected.add_edges_from(
                (head, tail) for head in component for tail in component
            )
        return expected
    raise ValueError(f"no closure oracle for {rule_name}")


def _oracle_labels(task: RuleInductionTask) -> np.ndarray:
    support = _relation_graph(task.support_graph)
    undirected = support.to_undirected()
    distance_cache: dict[int, dict[int, int]] = {}
    labels = []
    for row in task.population.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        polarity = int(row.query_polarity)
        if task.rule_name == "reflexivity":
            entailed = polarity == 1 and head == tail
        elif task.rule_name == "irreflexivity":
            entailed = polarity == -1 and head == tail
        elif task.rule_name == "symmetry":
            entailed = polarity == 1 and support.has_edge(tail, head)
        elif task.rule_name == "antisymmetry":
            entailed = polarity == -1 and head != tail and support.has_edge(tail, head)
        elif task.rule_name == "transitivity":
            if head not in distance_cache:
                distance_cache[head] = dict(
                    nx.single_source_shortest_path_length(support, head)
                )
            entailed = polarity == 1 and distance_cache[head].get(tail, 0) >= 2
        elif task.rule_name == "symmetry_transitivity":
            entailed = (
                polarity == 1
                and head != tail
                and nx.has_path(undirected, head, tail)
            )
        elif task.rule_name == "antisymmetry_transitivity":
            source = head if polarity == 1 else tail
            target = tail if polarity == 1 else head
            if source not in distance_cache:
                distance_cache[source] = dict(
                    nx.single_source_shortest_path_length(support, source)
                )
            minimum_distance = 2 if polarity == 1 else 1
            entailed = distance_cache[source].get(target, 0) >= minimum_distance
        else:
            raise ValueError(f"no entailment oracle for {task.rule_name}")
        labels.append(int(entailed))
    return np.asarray(labels, dtype=np.int8)


def audit_rule_induction_task(task: RuleInductionTask) -> dict[str, Any]:
    failures: list[str] = []
    population = task.population

    if population["query_id"].duplicated().any():
        failures.append("population query IDs are not unique")
    if population.duplicated(
        ["head", "tail", "relation", "query_polarity"]
    ).any():
        failures.append("population contains duplicate signed queries")
    if set(population["query_polarity"].unique()) - {-1, 1}:
        failures.append("query polarity must be -1 or 1")
    if set(task.context["query_id"]) & set(task.evaluation["query_id"]):
        failures.append("context and evaluation overlap")
    if not set(task.context["query_id"]) <= set(population["query_id"]):
        failures.append("context is not a population subset")
    if not set(task.evaluation["query_id"]) <= set(population["query_id"]):
        failures.append("evaluation is not a population subset")

    max_context = int(task.metadata["max_context_per_class"])
    evaluation_per_class = int(task.metadata["evaluation_per_class"])
    context_per_class = int(task.metadata["context_per_class"])
    expected_counts = {
        "population": {
            0: max_context + evaluation_per_class,
            1: max_context + evaluation_per_class,
        },
        "context": {0: context_per_class, 1: context_per_class},
        "evaluation": {0: evaluation_per_class, 1: evaluation_per_class},
    }
    for name, frame in (
        ("population", population),
        ("context", task.context),
        ("evaluation", task.evaluation),
    ):
        counts = dict(sorted(Counter(map(int, frame["label"])).items()))
        if counts != expected_counts[name]:
            failures.append(
                f"{name} label counts differ: expected={expected_counts[name]} observed={counts}"
            )

    support = _relation_graph(task.support_graph)
    closure = _relation_graph(task.closure_graph)
    support_edges = set(support.edges())
    closure_edges = set(closure.edges())
    if not support_edges <= closure_edges:
        failures.append("positive closure omits observed premises")
    expected_closure = set(_expected_positive_closure(task).edges())
    if closure_edges != expected_closure:
        failures.append(
            "positive closure disagrees with independent rule closure "
            f"(missing={len(expected_closure - closure_edges)}, "
            f"extra={len(closure_edges - expected_closure)})"
        )

    query_edges = {
        (int(row.head), int(row.tail))
        for row in population.itertuples(index=False)
    }
    overlap = support_edges & query_edges
    if overlap:
        failures.append(f"{len(overlap)} candidate conclusions occur among premises")

    entity_ids = np.asarray(sorted(map(int, support.nodes())), dtype=np.int64)
    if len(entity_ids) != int(task.metadata["graph_size"]):
        failures.append("support graph node count differs from graph_size")
    if len(entity_ids) and int(entity_ids.min()) < 2**40:
        failures.append("entity IDs are not non-contiguous 52-bit tokens")
    if np.array_equal(entity_ids, np.arange(len(entity_ids), dtype=np.int64)):
        failures.append("entity IDs are sequential")

    expected_labels = _oracle_labels(task)
    observed_labels = population["label"].to_numpy(dtype=np.int8)
    mismatches = int(np.count_nonzero(expected_labels != observed_labels))
    if mismatches:
        failures.append(
            f"{mismatches} labels disagree with the independent entailment oracle"
        )

    if task.rule_name in {"antisymmetry", "antisymmetry_transitivity"}:
        reciprocal = {
            (head, tail)
            for head, tail in closure_edges
            if head != tail and (tail, head) in closure_edges
        }
        if reciprocal:
            failures.append("antisymmetric closure contains reciprocal distinct facts")
    if task.rule_name in {"transitivity", "antisymmetry_transitivity"}:
        if max(dict(support.in_degree()).values(), default=0) < 2:
            failures.append("DAG lacks converging branches")
        if max(dict(support.out_degree()).values(), default=0) < 2:
            failures.append("DAG lacks diverging branches")

    if failures:
        raise ValueError(
            f"rule-induction audit failed for {task.rule_name}: {failures}"
        )
    return {
        "rule_name": task.rule_name,
        "population_rows": len(population),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "support_edges": len(support_edges),
        "closure_edges": len(closure_edges),
        "oracle_mismatches": 0,
        "support_query_overlap": 0,
        "matrix_exposed_to_decoder": False,
        "support_sha256": task.metadata["support_sha256"],
        "closure_sha256": task.metadata["closure_sha256"],
    }
