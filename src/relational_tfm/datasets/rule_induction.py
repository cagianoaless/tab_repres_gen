from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.hashing import object_sha256

Graph = nx.MultiDiGraph
RuleQuery = tuple[int, int, str, int]

SUPPORTED_RULES = frozenset(
    {
        "reflexivity",
        "irreflexivity",
        "symmetry",
        "antisymmetry",
        "transitivity",
        "symmetry_transitivity",
        "antisymmetry_transitivity",
    }
)


@dataclass(frozen=True)
class RuleInductionTask:
    """A graph-disjoint rule-entailment task.

    The support graph contains only observed positive premises. The closure graph is
    retained for auditing and is never exposed to an encoder. Each tabular query asks
    whether a positive (polarity=1) or negative (polarity=-1) literal is entailed.
    """

    rule_name: str
    support_graph: Graph
    closure_graph: Graph
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, object]


def _query_frame(queries: list[RuleQuery], label: int, offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": np.arange(offset, offset + len(queries), dtype=np.int64),
            "head": np.asarray([head for head, _, _, _ in queries], dtype=np.int64),
            "tail": np.asarray([tail for _, tail, _, _ in queries], dtype=np.int64),
            "relation": [relation for _, _, relation, _ in queries],
            "query_polarity": np.asarray(
                [polarity for _, _, _, polarity in queries], dtype=np.int8
            ),
            "label": np.full(len(queries), label, dtype=np.int8),
        }
    )


def _balanced_frames(
    entailed: list[RuleQuery],
    not_entailed: list[RuleQuery],
    *,
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = max_context_per_class + evaluation_per_class
    if context_per_class > max_context_per_class:
        raise ValueError("context_per_class exceeds max_context_per_class")
    if len(entailed) < required or len(not_entailed) < required:
        raise ValueError(
            f"need {required} candidates per class, found "
            f"{len(entailed)} entailed and {len(not_entailed)} not entailed"
        )

    rng = np.random.default_rng(split_seed)
    entailed = [entailed[int(index)] for index in rng.permutation(len(entailed))[:required]]
    not_entailed = [
        not_entailed[int(index)]
        for index in rng.permutation(len(not_entailed))[:required]
    ]
    positive = _query_frame(entailed, 1, 0)
    negative = _query_frame(not_entailed, 0, required)
    context_pool = pd.concat(
        [
            positive.iloc[:max_context_per_class],
            negative.iloc[:max_context_per_class],
        ],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [
            positive.iloc[max_context_per_class:],
            negative.iloc[max_context_per_class:],
        ],
        ignore_index=True,
    )
    context_ids = set(positive.iloc[:context_per_class]["query_id"]) | set(
        negative.iloc[:context_per_class]["query_id"]
    )
    context = context_pool[context_pool["query_id"].isin(context_ids)].copy()
    context = context.iloc[rng.permutation(len(context))].reset_index(drop=True)
    evaluation = evaluation.iloc[rng.permutation(len(evaluation))].reset_index(drop=True)
    population = pd.concat([context_pool, evaluation], ignore_index=True)
    return context, evaluation, population.sort_values("query_id").reset_index(drop=True)


def _sample_pairs(
    nodes: list[int],
    count: int,
    rng: np.random.Generator,
    predicate: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    attempts = 0
    attempt_limit = max(50_000, count * 1_000)
    while len(selected) < count and attempts < attempt_limit:
        head, tail = rng.choice(nodes, size=2, replace=False)
        pair = (int(head), int(tail))
        attempts += 1
        if predicate(*pair):
            selected.add(pair)
    if len(selected) != count:
        raise ValueError(f"could sample only {len(selected)} of {count} requested pairs")
    return sorted(selected)


def _base_graph(graph_size: int) -> Graph:
    graph = Graph()
    graph.add_nodes_from(range(graph_size))
    return graph


def _reflexivity_candidates(
    graph_size: int,
    required: int,
    rng: np.random.Generator,
    *,
    negative_rule: bool,
) -> tuple[Graph, Graph, list[RuleQuery], list[RuleQuery]]:
    if graph_size < required:
        raise ValueError(f"reflexive rules require at least {required} nodes")
    support = _base_graph(graph_size)
    nodes = list(range(graph_size))
    permutation = [int(node) for node in rng.permutation(nodes)]
    for index, head in enumerate(permutation):
        tail = permutation[(index + 1) % graph_size]
        support.add_edge(head, tail, relation="r0", premise=True)

    polarity = -1 if negative_rule else 1
    entailed = [(node, node, "r0", polarity) for node in nodes]
    non_edges = _sample_pairs(
        nodes,
        required,
        rng,
        lambda head, tail: not support.has_edge(head, tail),
    )
    not_entailed = [(head, tail, "r0", polarity) for head, tail in non_edges]
    closure = support.copy()
    if not negative_rule:
        for node in nodes:
            closure.add_edge(node, node, relation="r0", entailed=True)
    return support, closure, entailed, not_entailed


def _oriented_premise_candidates(
    graph_size: int,
    required: int,
    rng: np.random.Generator,
    *,
    negative_conclusion: bool,
) -> tuple[Graph, Graph, list[RuleQuery], list[RuleQuery]]:
    support = _base_graph(graph_size)
    nodes = list(range(graph_size))
    unordered_pairs: set[tuple[int, int]] = set()
    while len(unordered_pairs) < required:
        left, right = map(int, rng.choice(nodes, size=2, replace=False))
        unordered_pairs.add(tuple(sorted((left, right))))
    premise_pairs = []
    for left, right in sorted(unordered_pairs):
        premise_pairs.append((left, right) if rng.random() < 0.5 else (right, left))
    entailed: list[RuleQuery] = []
    for head, tail in premise_pairs:
        support.add_edge(head, tail, relation="r0", premise=True)
        entailed.append((tail, head, "r0", -1 if negative_conclusion else 1))

    reserved = {(head, tail) for head, tail, _, _ in entailed}
    active_nodes = sorted(
        {node for head, tail in premise_pairs for node in (head, tail)}
    )
    not_entailed_pairs = _sample_pairs(
        active_nodes,
        required,
        rng,
        lambda head, tail: (head, tail) not in reserved
        and not support.has_edge(head, tail)
        and not support.has_edge(tail, head),
    )
    not_entailed = [
        (head, tail, "r0", -1 if negative_conclusion else 1)
        for head, tail in not_entailed_pairs
    ]
    closure = support.copy()
    if not negative_conclusion:
        for head, tail, attributes in list(support.edges(data=True)):
            closure.add_edge(tail, head, **attributes, entailed=True)
    return support, closure, entailed, not_entailed


def _layered_dag(
    graph_size: int,
    rng: np.random.Generator,
    *,
    layer_count: int = 10,
) -> tuple[Graph, dict[int, int]]:
    if graph_size < layer_count * 4:
        raise ValueError(f"layered rule graphs require at least {layer_count * 4} nodes")
    support = _base_graph(graph_size)
    order = [int(node) for node in rng.permutation(graph_size)]
    layers = [list(map(int, part)) for part in np.array_split(order, layer_count)]
    layer_by_node = {
        node: layer_index
        for layer_index, layer in enumerate(layers)
        for node in layer
    }
    for layer_index in range(layer_count - 1):
        targets = layers[layer_index + 1]
        for head in layers[layer_index]:
            sample_size = min(3, len(targets))
            for tail in rng.choice(targets, size=sample_size, replace=False):
                support.add_edge(head, int(tail), relation="r0", premise=True)
    return support, layer_by_node


def _dag_candidate_pools(
    support: Graph,
    layer_by_node: dict[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    simple = nx.DiGraph(support)
    positive_pairs: list[tuple[int, int]] = []
    negative_pairs: list[tuple[int, int]] = []
    for head in simple.nodes():
        distances = nx.single_source_shortest_path_length(simple, head)
        for tail in simple.nodes():
            layer_gap = layer_by_node[int(tail)] - layer_by_node[int(head)]
            if layer_gap < 2:
                continue
            distance = distances.get(tail)
            if distance is not None and distance >= 2:
                positive_pairs.append((int(head), int(tail)))
            elif not nx.has_path(simple, tail, head):
                negative_pairs.append((int(head), int(tail)))
    return positive_pairs, negative_pairs


def _transitivity_candidates(
    graph_size: int,
    required: int,
    rng: np.random.Generator,
) -> tuple[Graph, Graph, list[RuleQuery], list[RuleQuery]]:
    support, layer_by_node = _layered_dag(graph_size, rng)
    positive_pairs, negative_pairs = _dag_candidate_pools(support, layer_by_node)
    if len(positive_pairs) < required or len(negative_pairs) < required:
        raise ValueError(
            "layered DAG lacks enough matched transitive/non-reachable conclusions; "
            f"positive={len(positive_pairs)} negative={len(negative_pairs)}"
        )
    entailed = [(head, tail, "r0", 1) for head, tail in positive_pairs]
    not_entailed = [(head, tail, "r0", 1) for head, tail in negative_pairs]
    closure = nx.MultiDiGraph(nx.transitive_closure_dag(nx.DiGraph(support)))
    nx.set_edge_attributes(closure, "r0", "relation")
    return support, closure, entailed, not_entailed


def _symmetry_transitivity_candidates(
    graph_size: int,
    required: int,
    rng: np.random.Generator,
) -> tuple[Graph, Graph, list[RuleQuery], list[RuleQuery]]:
    component_count = max(4, min(12, graph_size // 16))
    support = _base_graph(graph_size)
    shuffled = [int(node) for node in rng.permutation(graph_size)]
    components = [list(map(int, part)) for part in np.array_split(shuffled, component_count)]
    for component in components:
        order = [int(node) for node in rng.permutation(component)]
        for index in range(1, len(order)):
            earlier = order[int(rng.integers(index))]
            later = order[index]
            if rng.random() < 0.5:
                earlier, later = later, earlier
            support.add_edge(earlier, later, relation="r0", premise=True)

    entailed_pairs = []
    not_entailed_pairs = []
    for left_index, left_component in enumerate(components):
        for head in left_component:
            for tail in left_component:
                if head != tail and not support.has_edge(head, tail):
                    entailed_pairs.append((head, tail))
            for right_component in components[left_index + 1 :]:
                for tail in right_component:
                    not_entailed_pairs.extend(((head, tail), (tail, head)))
    if len(entailed_pairs) < required or len(not_entailed_pairs) < required:
        raise ValueError("component graph lacks enough rule conclusions")

    closure = _base_graph(graph_size)
    for component in components:
        for head in component:
            for tail in component:
                closure.add_edge(head, tail, relation="r0", entailed=True)
    return (
        support,
        closure,
        [(head, tail, "r0", 1) for head, tail in entailed_pairs],
        [(head, tail, "r0", 1) for head, tail in not_entailed_pairs],
    )


def _antisymmetry_transitivity_candidates(
    graph_size: int,
    required: int,
    rng: np.random.Generator,
) -> tuple[Graph, Graph, list[RuleQuery], list[RuleQuery]]:
    support, layer_by_node = _layered_dag(graph_size, rng)
    closure_pairs, incomparable_pairs = _dag_candidate_pools(support, layer_by_node)
    half = required // 2
    positive_literals = [(head, tail, "r0", 1) for head, tail in closure_pairs[:half]]
    negative_literals = [
        (tail, head, "r0", -1)
        for head, tail in closure_pairs[half : half + required - half]
    ]
    not_entailed_positive = [
        (head, tail, "r0", 1) for head, tail in incomparable_pairs[:half]
    ]
    not_entailed_negative = [
        (head, tail, "r0", -1)
        for head, tail in incomparable_pairs[half : half + required - half]
    ]
    if len(positive_literals) + len(negative_literals) < required:
        raise ValueError("DAG lacks enough antisymmetric-transitive conclusions")
    if len(not_entailed_positive) + len(not_entailed_negative) < required:
        raise ValueError("DAG lacks enough antisymmetric-transitive controls")
    closure = nx.MultiDiGraph(nx.transitive_closure_dag(nx.DiGraph(support)))
    nx.set_edge_attributes(closure, "r0", "relation")
    return (
        support,
        closure,
        positive_literals + negative_literals,
        not_entailed_positive + not_entailed_negative,
    )


def _relabel_graphs_and_frames(
    support: Graph,
    closure: Graph,
    frames: tuple[pd.DataFrame, ...],
    seed: int,
) -> tuple[Graph, Graph, tuple[pd.DataFrame, ...]]:
    nodes = np.asarray(sorted(int(node) for node in support.nodes()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    while True:
        labels = rng.integers(2**40, 2**52, size=len(nodes), dtype=np.int64)
        if len(np.unique(labels)) == len(labels):
            break
    mapping = {int(node): int(label) for node, label in zip(nodes, labels, strict=True)}
    relabeled_frames = []
    for frame in frames:
        result = frame.copy()
        result["head"] = result["head"].map(mapping).astype(np.int64)
        result["tail"] = result["tail"].map(mapping).astype(np.int64)
        relabeled_frames.append(result)
    return (
        nx.relabel_nodes(support, mapping, copy=True),
        nx.relabel_nodes(closure, mapping, copy=True),
        tuple(relabeled_frames),
    )


def _graph_hash(graph: Graph) -> str:
    return object_sha256(
        {
            "nodes": sorted(int(node) for node in graph.nodes()),
            "edges": sorted(
                (
                    int(head),
                    int(tail),
                    str(attributes.get("relation", "r0")),
                )
                for head, tail, attributes in graph.edges(data=True)
            ),
        }
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    return object_sha256(
        [
            {
                "query_id": int(row.query_id),
                "head": int(row.head),
                "tail": int(row.tail),
                "relation": str(row.relation),
                "query_polarity": int(row.query_polarity),
                "label": int(row.label),
            }
            for row in frame.sort_values("query_id").itertuples(index=False)
        ]
    )


def generate_rule_induction_task(
    rule_name: str,
    graph_size: int,
    context_per_class: int,
    *,
    max_context_per_class: int | None = None,
    evaluation_per_class: int | None = None,
    graph_seed: int,
    split_seed: int | None = None,
    relabel_seed: int | None = None,
) -> RuleInductionTask:
    """Generate a tabular rule-entailment task without exposing a relation matrix."""

    if rule_name not in SUPPORTED_RULES:
        raise ValueError(f"unsupported rule-induction rule: {rule_name}")
    max_context = max_context_per_class or context_per_class
    evaluation_size = evaluation_per_class or context_per_class
    required = max_context + evaluation_size
    rng = np.random.default_rng(graph_seed)

    if rule_name == "reflexivity":
        generated = _reflexivity_candidates(
            graph_size, required, rng, negative_rule=False
        )
    elif rule_name == "irreflexivity":
        generated = _reflexivity_candidates(
            graph_size, required, rng, negative_rule=True
        )
    elif rule_name == "symmetry":
        generated = _oriented_premise_candidates(
            graph_size, required, rng, negative_conclusion=False
        )
    elif rule_name == "antisymmetry":
        generated = _oriented_premise_candidates(
            graph_size, required, rng, negative_conclusion=True
        )
    elif rule_name == "transitivity":
        generated = _transitivity_candidates(graph_size, required, rng)
    elif rule_name == "symmetry_transitivity":
        generated = _symmetry_transitivity_candidates(graph_size, required, rng)
    else:
        generated = _antisymmetry_transitivity_candidates(graph_size, required, rng)

    support, closure, entailed, not_entailed = generated
    context, evaluation, population = _balanced_frames(
        entailed,
        not_entailed,
        context_per_class=context_per_class,
        max_context_per_class=max_context,
        evaluation_per_class=evaluation_size,
        split_seed=split_seed if split_seed is not None else graph_seed + 1_000_000,
    )
    support, closure, relabeled = _relabel_graphs_and_frames(
        support,
        closure,
        (context, evaluation, population),
        relabel_seed if relabel_seed is not None else graph_seed + 2_000_000,
    )
    context, evaluation, population = relabeled
    metadata: dict[str, object] = {
        "target": "rule_entailment",
        "rule_name": rule_name,
        "graph_size": graph_size,
        "support_edges": support.number_of_edges(),
        "closure_edges": closure.number_of_edges(),
        "context_per_class": context_per_class,
        "max_context_per_class": max_context,
        "evaluation_per_class": evaluation_size,
        "graph_seed": graph_seed,
        "split_seed": split_seed if split_seed is not None else graph_seed + 1_000_000,
        "relabel_seed": (
            relabel_seed if relabel_seed is not None else graph_seed + 2_000_000
        ),
        "entity_relabeling": "independent random non-contiguous 52-bit tokens",
        "matrix_exposed_to_decoder": False,
        "support_sha256": _graph_hash(support),
        "closure_sha256": _graph_hash(closure),
        "context_sha256": _frame_hash(context),
        "evaluation_sha256": _frame_hash(evaluation),
        "population_sha256": _frame_hash(population),
    }
    return RuleInductionTask(
        rule_name=rule_name,
        support_graph=support,
        closure_graph=closure,
        context=context,
        evaluation=evaluation,
        population=population,
        metadata=metadata,
    )
