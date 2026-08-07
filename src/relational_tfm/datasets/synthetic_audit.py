from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

from relational_tfm.datasets.synthetic import SyntheticTask
from relational_tfm.hashing import object_sha256


def _relation_graph(graph: nx.MultiDiGraph, relation: str) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes())
    result.add_edges_from(
        (head, tail)
        for head, tail, attributes in graph.edges(data=True)
        if str(attributes.get("relation", "r0")) == relation
    )
    return result


def _typed_targets(
    graph: nx.MultiDiGraph, source: int, relations: tuple[str, ...]
) -> set[int]:
    frontier = {source}
    for relation in relations:
        frontier = {
            int(tail)
            for head in frontier
            for _, tail, attributes in graph.out_edges(head, data=True)
            if str(attributes.get("relation", "r0")) == relation
        }
    return frontier


def _relation_role_signature(graph: nx.MultiDiGraph, node: int) -> tuple[int, ...]:
    values = []
    for relation in ("r0", "r1", "r2", "r3"):
        values.extend(
            [
                sum(
                    str(attributes.get("relation", "r0")) == relation
                    for *_, attributes in graph.in_edges(node, data=True)
                ),
                sum(
                    str(attributes.get("relation", "r0")) == relation
                    for *_, attributes in graph.out_edges(node, data=True)
                ),
            ]
        )
    return tuple(values)


def _block_edge_densities(graph: nx.MultiDiGraph) -> dict[str, float]:
    nodes = list(graph.nodes())
    same_possible = 0
    different_possible = 0
    same_edges = 0
    different_edges = 0
    relation_edges = {
        tuple(sorted((int(head), int(tail))))
        for head, tail, attributes in graph.edges(data=True)
        if head != tail and str(attributes.get("relation", "r0")) == "r0"
    }
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            same = graph.nodes[left]["community"] == graph.nodes[right]["community"]
            if same:
                same_possible += 1
                same_edges += tuple(sorted((int(left), int(right)))) in relation_edges
            else:
                different_possible += 1
                different_edges += tuple(sorted((int(left), int(right)))) in relation_edges
    return {
        "same_community_edge_density": same_edges / same_possible,
        "different_community_edge_density": different_edges / different_possible,
    }


def _oracle_labels(task: SyntheticTask) -> np.ndarray:
    graph = task.clean_support_graph
    rows = task.population
    property_name = task.property_name
    relation_r0 = _relation_graph(graph, "r0")
    shortest_paths: dict[int, dict[int, int]] = {}
    role_signatures: dict[int, tuple[int, ...]] = {}
    typed_cache: dict[tuple[int, tuple[str, ...]], set[int]] = {}
    untyped_descendants: dict[int, set[int]] = {}
    simple = nx.DiGraph(graph)
    expected = []
    discovery_mapping = graph.graph.get("discovery_mapping", {})
    discovery_positive_required = int(task.metadata["max_context_per_class"]) + int(
        task.metadata["evaluation_per_class"]
    )
    discovery_noise_period = int(task.metadata.get("label_noise_period", 0))

    for row in rows.itertuples(index=False):
        head = int(row.head)
        tail = int(row.tail)
        if property_name == "reflexivity":
            value = head == tail
        elif property_name == "symmetry":
            value = relation_r0.has_edge(tail, head)
        elif property_name == "antisymmetry":
            value = (
                graph.in_degree(head) == 0
                and graph.out_degree(head) > 0
                and graph.in_degree(tail) > 0
                and graph.out_degree(tail) == 0
            )
        elif property_name == "transitivity":
            if head not in shortest_paths:
                shortest_paths[head] = dict(
                    nx.single_source_shortest_path_length(relation_r0, head)
                )
            distances = shortest_paths[head]
            value = distances.get(tail, 0) >= 2
        elif property_name in {"typed_composition_l2", "typed_composition_l3"}:
            relations = (
                ("r1", "r2")
                if property_name == "typed_composition_l2"
                else ("r1", "r2", "r3")
            )
            cache_key = (head, relations)
            if cache_key not in typed_cache:
                typed_cache[cache_key] = _typed_targets(graph, head, relations)
            targets = typed_cache[cache_key]
            value = tail in targets
            if head not in untyped_descendants:
                untyped_descendants[head] = nx.descendants(simple, head)
            descendants = untyped_descendants[head]
            if tail not in descendants:
                raise ValueError(
                    f"{property_name} query lacks the registered untyped control path"
                )
        elif property_name == "local_structural_roles":
            if head not in role_signatures:
                role_signatures[head] = _relation_role_signature(graph, head)
            if tail not in role_signatures:
                role_signatures[tail] = _relation_role_signature(graph, tail)
            head_signature = role_signatures[head]
            tail_signature = role_signatures[tail]
            value = head_signature == tail_signature
        elif property_name == "block_affinity_homophilic":
            value = graph.nodes[head]["community"] == graph.nodes[tail]["community"]
        elif property_name == "block_affinity_heterophilic":
            value = graph.nodes[head]["community"] != graph.nodes[tail]["community"]
        elif property_name == "discovery_reciprocity":
            support_relation = str(discovery_mapping[str(row.relation)])
            value = any(
                str(attributes.get("relation")) == support_relation
                for attributes in graph.get_edge_data(tail, head, default={}).values()
            )
        elif property_name == "discovery_typed_composition":
            relations = tuple(discovery_mapping[str(row.relation)])
            cache_key = (head, relations)
            if cache_key not in typed_cache:
                typed_cache[cache_key] = _typed_targets(graph, head, relations)
            value = tail in typed_cache[cache_key]
        elif property_name == "discovery_structural_roles":
            head_role, tail_role = discovery_mapping[str(row.relation)]
            value = int(graph.nodes[head]["role"]) == int(head_role) and int(
                graph.nodes[tail]["role"]
            ) == int(tail_role)
        elif property_name == "discovery_block_affinity":
            same_community = graph.nodes[head]["community"] == graph.nodes[tail]["community"]
            value = same_community == bool(discovery_mapping[str(row.relation)])
        else:
            raise ValueError(f"no independent oracle for {property_name}")
        if property_name.startswith("discovery_"):
            query_id = int(row.query_id)
            within_class = (
                query_id
                if query_id < discovery_positive_required
                else query_id - discovery_positive_required
            )
            if within_class % discovery_noise_period == 0:
                value = not value
        expected.append(int(value))
    return np.asarray(expected, dtype=np.int8)


def audit_synthetic_task(task: SyntheticTask) -> dict[str, Any]:
    population = task.population
    context = task.context
    evaluation = task.evaluation
    max_context = int(task.metadata["max_context_per_class"])
    evaluation_per_class = int(task.metadata["evaluation_per_class"])
    evaluation_negative_ratio = int(task.metadata["evaluation_negative_ratio"])
    context_positive_count = int(task.metadata["context_positive_count"])
    context_negative_ratio = int(task.metadata["context_negative_ratio"])

    failures = []
    graph_diagnostics: dict[str, Any] = {}
    if population["query_id"].duplicated().any():
        failures.append("population query IDs are not unique")
    if population.duplicated(["head", "tail", "relation"]).any():
        failures.append("population contains duplicate relational queries")
    if set(context["query_id"]) & set(evaluation["query_id"]):
        failures.append("context and evaluation overlap")
    if not set(context["query_id"]) <= set(population["query_id"]):
        failures.append("context is not a population subset")
    if not set(evaluation["query_id"]) <= set(population["query_id"]):
        failures.append("evaluation is not a population subset")

    expected_counts = {
        "population": {
            0: max_context + evaluation_per_class * evaluation_negative_ratio,
            1: max_context + evaluation_per_class,
        },
        "context": {
            0: context_positive_count * context_negative_ratio,
            1: context_positive_count,
        },
        "evaluation": {
            0: evaluation_per_class * evaluation_negative_ratio,
            1: evaluation_per_class,
        },
    }
    for name, frame in (
        ("population", population),
        ("context", context),
        ("evaluation", evaluation),
    ):
        observed = dict(sorted(Counter(int(value) for value in frame["label"]).items()))
        if observed != expected_counts[name]:
            failures.append(
                f"{name} label counts differ: expected={expected_counts[name]} observed={observed}"
            )

    support_queries = {
        (int(head), int(tail), str(attributes.get("relation", "r0")))
        for head, tail, attributes in task.clean_support_graph.edges(data=True)
    }
    sampled_queries = {
        (int(row.head), int(row.tail), str(row.relation))
        for row in population.itertuples(index=False)
    }
    overlap = support_queries & sampled_queries
    if overlap:
        failures.append(f"{len(overlap)} sampled queries are clean support edges")

    if task.property_name == "local_structural_roles":
        signatures_by_role: dict[int, set[tuple[int, ...]]] = {}
        for node, attributes in task.clean_support_graph.nodes(data=True):
            role = int(attributes["role"])
            signatures_by_role.setdefault(role, set()).add(
                _relation_role_signature(task.clean_support_graph, int(node))
            )
        if any(len(signatures) != 1 for signatures in signatures_by_role.values()):
            failures.append("nodes in the same latent role have different structural signatures")
        distinct_signatures = {next(iter(values)) for values in signatures_by_role.values()}
        if len(distinct_signatures) != len(signatures_by_role):
            failures.append("different latent roles share a structural signature")
        graph_diagnostics["role_count"] = len(signatures_by_role)
        graph_diagnostics["distinct_role_signatures"] = len(distinct_signatures)

    if task.property_name.startswith("block_affinity_"):
        block_densities = _block_edge_densities(task.clean_support_graph)
        graph_diagnostics.update(block_densities)
        same_density = block_densities["same_community_edge_density"]
        different_density = block_densities["different_community_edge_density"]
        if task.property_name.endswith("homophilic") and same_density <= different_density:
            failures.append("realized clean graph is not homophilic")
        if task.property_name.endswith("heterophilic") and different_density <= same_density:
            failures.append("realized clean graph is not heterophilic")

    if task.property_name.startswith("discovery_"):
        mapping = task.clean_support_graph.graph.get("discovery_mapping")
        if not isinstance(mapping, dict) or set(mapping) != {"q0", "q1", "q2", "q3"}:
            failures.append("discovery task has an invalid hidden query-rule mapping")
        elif object_sha256(mapping) != task.metadata.get("latent_rule_sha256"):
            failures.append("discovery hidden-rule hash differs from task metadata")
        if task.metadata.get("encoder_access_to_latent_rule") is not False:
            failures.append("discovery encoder-access declaration is not false")
        graph_diagnostics["label_noise_fraction"] = float(task.metadata["label_noise_fraction"])
        graph_diagnostics["latent_rule_sha256"] = task.metadata["latent_rule_sha256"]

    expected_labels = _oracle_labels(task)
    observed_labels = population["label"].to_numpy(dtype=np.int8)
    mismatch_indices = np.flatnonzero(expected_labels != observed_labels)
    if len(mismatch_indices):
        failures.append(
            f"{len(mismatch_indices)} labels disagree with the independent property oracle"
        )

    if failures:
        raise ValueError(f"synthetic task audit failed for {task.property_name}: {failures}")
    return {
        "property": task.property_name,
        "population_rows": len(population),
        "context_rows": len(context),
        "context_positive_count": context_positive_count,
        "context_negative_ratio": context_negative_ratio,
        "evaluation_negative_ratio": evaluation_negative_ratio,
        "evaluation_rows": len(evaluation),
        "oracle_mismatches": 0,
        "support_query_overlap": 0,
        "noise_regime": task.metadata["noise_regime"],
        "clean_support_sha256": task.metadata["clean_support_sha256"],
        "observed_support_sha256": task.metadata["observed_support_sha256"],
        "context_query_sha256": task.metadata["context_query_sha256"],
        "evaluation_query_sha256": task.metadata["evaluation_query_sha256"],
        "graph_diagnostics": graph_diagnostics,
    }
