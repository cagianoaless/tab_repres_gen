from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

import networkx as nx
import numpy as np
import pandas as pd

from relational_tfm.hashing import object_sha256

Graph = nx.MultiDiGraph
Fact = tuple[int, str, int]

CORE_RELATIONS = frozenset({"mother", "father", "son", "daughter"})
DERIVED_RELATIONS = frozenset(
    {
        "husband",
        "wife",
        "sister",
        "brother",
        "grandmother",
        "grandfather",
        "granddaughter",
        "grandson",
        "aunt",
        "uncle",
        "niece",
        "nephew",
        "cousin",
    }
)
ALL_RELATIONS = CORE_RELATIONS | DERIVED_RELATIONS
SUPPORTED_GENEALOGY_SPLITS = frozenset({"random", "evidence", "family"})
REFERENCE_FRACTIONS = frozenset({0.8, 0.4, 0.2, 0.1})


@dataclass(frozen=True)
class GenealogyTask:
    split_name: str
    observed_fraction: float
    support_graph: Graph
    complete_graph: Graph
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, object]


def _add_fact(graph: Graph, head: int, relation: str, tail: int) -> None:
    graph.add_edge(head, tail, relation=relation)


def _facts(graph: Graph) -> set[Fact]:
    return {
        (int(head), str(attributes["relation"]), int(tail))
        for head, tail, attributes in graph.edges(data=True)
    }


def _build_family_template(
    graph: Graph,
    family_id: int,
    start_node: int,
    rng: np.random.Generator,
) -> int:
    next_node = start_node
    parents: set[tuple[int, int]] = set()
    couples: set[tuple[int, int]] = set()

    def person(sex: str, generation: int) -> int:
        nonlocal next_node
        node = next_node
        next_node += 1
        graph.add_node(
            node,
            sex=sex,
            generation=generation,
            family_id=family_id,
        )
        return node

    def family(
        male: int,
        female: int,
        child_sexes: tuple[str, ...],
        generation: int,
    ) -> tuple[int, ...]:
        couples.add((male, female))
        children = tuple(person(sex, generation) for sex in child_sexes)
        for child in children:
            parents.add((male, child))
            parents.add((female, child))
        return children

    paternal_grandfather = person("M", 0)
    paternal_grandmother = person("F", 0)
    maternal_grandfather = person("M", 0)
    maternal_grandmother = person("F", 0)
    paternal_children = family(
        paternal_grandfather,
        paternal_grandmother,
        ("M", "F", "M"),
        1,
    )
    maternal_children = family(
        maternal_grandfather,
        maternal_grandmother,
        ("F", "M", "F"),
        1,
    )
    father, paternal_aunt, paternal_uncle = paternal_children
    mother, maternal_uncle, maternal_aunt = maternal_children
    main_child_sexes = (
        "M",
        "F",
        "M" if int(rng.integers(2)) == 0 else "F",
    )
    family(father, mother, main_child_sexes, 2)

    for relative in (paternal_aunt, maternal_aunt):
        spouse = person("M", 1)
        family(spouse, relative, ("M", "F"), 2)
    for relative in (paternal_uncle, maternal_uncle):
        spouse = person("F", 1)
        family(relative, spouse, ("F", "M"), 2)

    graph.graph.setdefault("parent_pairs", set()).update(parents)
    graph.graph.setdefault("couples", set()).update(couples)
    return next_node


def _core_facts(
    graph: Graph,
    parents: set[tuple[int, int]],
) -> set[Fact]:
    facts: set[Fact] = set()
    for parent, child in parents:
        parent_sex = str(graph.nodes[parent]["sex"])
        child_sex = str(graph.nodes[child]["sex"])
        facts.add((parent, "father" if parent_sex == "M" else "mother", child))
        facts.add((child, "son" if child_sex == "M" else "daughter", parent))
    return facts


def _derive_facts(
    graph: Graph,
    parents: set[tuple[int, int]],
    couples: set[tuple[int, int]],
    *,
    observed_core: set[Fact] | None = None,
) -> set[Fact]:
    facts = set(observed_core) if observed_core is not None else _core_facts(graph, parents)
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    for parent, child in parents:
        children_by_parent[parent].add(child)

    for male, female in couples:
        facts.add((male, "husband", female))
        facts.add((female, "wife", male))

    siblings: set[tuple[int, int]] = set()
    for children in children_by_parent.values():
        for head in children:
            for tail in children:
                if head != tail:
                    siblings.add((head, tail))
    for head, tail in siblings:
        head_sex = graph.nodes[head].get("sex")
        if head_sex in {"M", "F"}:
            relation = "brother" if head_sex == "M" else "sister"
            facts.add((head, relation, tail))

    for grandparent, parent in parents:
        for child in children_by_parent.get(parent, ()):
            grandparent_sex = graph.nodes[grandparent].get("sex")
            child_sex = graph.nodes[child].get("sex")
            if grandparent_sex in {"M", "F"}:
                grand_relation = (
                    "grandfather" if grandparent_sex == "M" else "grandmother"
                )
                facts.add((grandparent, grand_relation, child))
            if child_sex in {"M", "F"}:
                child_relation = (
                    "grandson" if child_sex == "M" else "granddaughter"
                )
                facts.add((child, child_relation, grandparent))

    for relative, parent in siblings:
        for child in children_by_parent.get(parent, ()):
            relative_sex = graph.nodes[relative].get("sex")
            child_sex = graph.nodes[child].get("sex")
            if relative_sex in {"M", "F"}:
                older_relation = "uncle" if relative_sex == "M" else "aunt"
                facts.add((relative, older_relation, child))
            if child_sex in {"M", "F"}:
                younger_relation = "nephew" if child_sex == "M" else "niece"
                facts.add((child, younger_relation, relative))

    for head_parent, tail_parent in siblings:
        for head in children_by_parent.get(head_parent, ()):
            for tail in children_by_parent.get(tail_parent, ()):
                if head != tail:
                    facts.add((head, "cousin", tail))
    return facts


def derive_genealogy_facts_from_core(graph: Graph) -> set[Fact]:
    observed_core: set[Fact] = set()
    parent_pairs: set[tuple[int, int]] = set()
    sex_by_node: dict[int, str] = {}
    for head, relation, tail in _facts(graph):
        if relation in CORE_RELATIONS:
            observed_core.add((head, relation, tail))
        if relation == "father":
            parent_pairs.add((head, tail))
            sex_by_node[head] = "M"
        elif relation == "mother":
            parent_pairs.add((head, tail))
            sex_by_node[head] = "F"
        elif relation == "son":
            parent_pairs.add((tail, head))
            sex_by_node[head] = "M"
        elif relation == "daughter":
            parent_pairs.add((tail, head))
            sex_by_node[head] = "F"
    reconstructed = Graph()
    for node in graph.nodes():
        reconstructed.add_node(
            int(node),
            sex=sex_by_node.get(int(node)),
        )
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    for parent, child in parent_pairs:
        children_by_parent[parent].add(child)
    couples: set[tuple[int, int]] = set()
    parents_by_child: dict[int, set[int]] = defaultdict(set)
    for parent, child in parent_pairs:
        parents_by_child[child].add(parent)
    for parents in parents_by_child.values():
        males = [node for node in parents if reconstructed.nodes[node]["sex"] == "M"]
        females = [node for node in parents if reconstructed.nodes[node]["sex"] == "F"]
        couples.update((male, female) for male in males for female in females)
    return _derive_facts(
        reconstructed,
        parent_pairs,
        couples,
        observed_core=observed_core,
    )


def _frame(
    rows: list[Fact],
    labels: int,
    offset: int,
    family_by_node: dict[int, int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": np.arange(offset, offset + len(rows), dtype=np.int64),
            "head": np.asarray([head for head, _, _ in rows], dtype=np.int64),
            "tail": np.asarray([tail for _, _, tail in rows], dtype=np.int64),
            "relation": [relation for _, relation, _ in rows],
            "query_polarity": np.ones(len(rows), dtype=np.int8),
            "family_id": np.asarray(
                [family_by_node[head] for head, _, _ in rows],
                dtype=np.int32,
            ),
            "label": np.full(len(rows), labels, dtype=np.int8),
        }
    )


def _candidate_frames(
    positive_pool: set[Fact],
    complete_facts: set[Fact],
    family_by_node: dict[int, int],
    *,
    context_per_class: int,
    max_context_per_class: int,
    evaluation_per_class: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = max_context_per_class + evaluation_per_class
    if context_per_class > max_context_per_class:
        raise ValueError("context_per_class exceeds max_context_per_class")

    true_by_family_relation: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    heads: dict[tuple[int, str], set[int]] = defaultdict(set)
    tails: dict[tuple[int, str], set[int]] = defaultdict(set)
    for head, relation, tail in complete_facts:
        key = (family_by_node[head], relation)
        true_by_family_relation[key].add((head, tail))
        heads[key].add(head)
        tails[key].add(tail)

    positives = list(positive_pool)
    rng.shuffle(positives)
    selected_positive: list[Fact] = []
    selected_negative: list[Fact] = []
    used_negative: set[Fact] = set()
    for positive in positives:
        head, relation, _ = positive
        key = (family_by_node[head], relation)
        candidates = [
            (negative_head, relation, negative_tail)
            for negative_head in heads[key]
            for negative_tail in tails[key]
            if negative_head != negative_tail
            and (negative_head, negative_tail) not in true_by_family_relation[key]
            and (negative_head, relation, negative_tail) not in used_negative
        ]
        if not candidates:
            continue
        negative = candidates[int(rng.integers(len(candidates)))]
        selected_positive.append(positive)
        selected_negative.append(negative)
        used_negative.add(negative)
        if len(selected_positive) == required:
            break
    if len(selected_positive) != required:
        raise ValueError(
            f"genealogy has only {len(selected_positive)} matched candidate pairs; "
            f"{required} required"
        )

    positive_frame = _frame(selected_positive, 1, 0, family_by_node)
    negative_frame = _frame(selected_negative, 0, required, family_by_node)
    context_pool = pd.concat(
        [
            positive_frame.iloc[:max_context_per_class],
            negative_frame.iloc[:max_context_per_class],
        ],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [
            positive_frame.iloc[max_context_per_class:],
            negative_frame.iloc[max_context_per_class:],
        ],
        ignore_index=True,
    )
    context_ids = set(positive_frame.iloc[:context_per_class]["query_id"]) | set(
        negative_frame.iloc[:context_per_class]["query_id"]
    )
    context = context_pool[context_pool["query_id"].isin(context_ids)].copy()
    context = context.iloc[rng.permutation(len(context))].reset_index(drop=True)
    evaluation = evaluation.iloc[rng.permutation(len(evaluation))].reset_index(drop=True)
    population = pd.concat([context_pool, evaluation], ignore_index=True)
    return context, evaluation, population.sort_values("query_id").reset_index(drop=True)


def _graph_from_facts(template: Graph, facts: set[Fact]) -> Graph:
    graph = Graph()
    graph.add_nodes_from(
        (int(node), dict(attributes))
        for node, attributes in template.nodes(data=True)
    )
    for head, relation, tail in sorted(facts):
        _add_fact(graph, head, relation, tail)
    return graph


def _relabel(
    support: Graph,
    complete: Graph,
    frames: tuple[pd.DataFrame, ...],
    seed: int,
) -> tuple[Graph, Graph, tuple[pd.DataFrame, ...]]:
    nodes = np.asarray(sorted(map(int, complete.nodes())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    while True:
        labels = rng.integers(2**40, 2**52, size=len(nodes), dtype=np.int64)
        if len(np.unique(labels)) == len(labels):
            break
    mapping = {
        int(node): int(label)
        for node, label in zip(nodes, labels, strict=True)
    }
    relabeled_frames = []
    for frame in frames:
        result = frame.copy()
        result["head"] = result["head"].map(mapping).astype(np.int64)
        result["tail"] = result["tail"].map(mapping).astype(np.int64)
        relabeled_frames.append(result)
    return (
        nx.relabel_nodes(support, mapping, copy=True),
        nx.relabel_nodes(complete, mapping, copy=True),
        tuple(relabeled_frames),
    )


def _hash_graph(graph: Graph) -> str:
    return object_sha256(
        {
            "nodes": sorted(map(int, graph.nodes())),
            "facts": sorted(_facts(graph)),
        }
    )


def generate_genealogy_task(
    split_name: str,
    observed_fraction: float,
    family_count: int,
    context_per_class: int,
    *,
    max_context_per_class: int | None = None,
    evaluation_per_class: int | None = None,
    graph_seed: int,
    split_seed: int | None = None,
    relabel_seed: int | None = None,
) -> GenealogyTask:
    if split_name not in SUPPORTED_GENEALOGY_SPLITS:
        raise ValueError(f"unsupported genealogy split: {split_name}")
    allowed_fractions = (
        REFERENCE_FRACTIONS | {0.0}
        if split_name == "family"
        else REFERENCE_FRACTIONS
    )
    if observed_fraction not in allowed_fractions:
        raise ValueError(
            f"observed fraction {observed_fraction} is invalid for {split_name}"
        )
    if family_count < 5:
        raise ValueError("genealogy tasks require at least five families")

    rng = np.random.default_rng(graph_seed)
    template = Graph()
    next_node = 0
    for family_id in range(family_count):
        next_node = _build_family_template(template, family_id, next_node, rng)
    complete_facts = _derive_facts(
        template,
        set(template.graph["parent_pairs"]),
        set(template.graph["couples"]),
    )
    complete = _graph_from_facts(template, complete_facts)
    family_by_node = {
        int(node): int(attributes["family_id"])
        for node, attributes in template.nodes(data=True)
    }

    family_ids = np.asarray(range(family_count), dtype=np.int32)
    rng.shuffle(family_ids)
    source_count = max(4, int(family_count * 0.8))
    source_families = set(map(int, family_ids[:source_count]))
    target_families = set(map(int, family_ids[source_count:]))
    core = {fact for fact in complete_facts if fact[1] in CORE_RELATIONS}
    derived = complete_facts - core

    def sampled(facts: set[Fact], fraction: float) -> set[Fact]:
        ordered = sorted(facts)
        if not ordered or fraction == 0:
            return set()
        indices = rng.permutation(len(ordered))[: ceil(fraction * len(ordered))]
        return {ordered[int(index)] for index in indices}

    if split_name == "random":
        support_facts = sampled(complete_facts, observed_fraction)
        positive_pool = complete_facts - support_facts
    elif split_name == "evidence":
        support_facts = core | sampled(derived, observed_fraction)
        positive_pool = derived - support_facts
    else:
        source_facts = {
            fact
            for fact in complete_facts
            if family_by_node[fact[0]] in source_families
        }
        target_core = {
            fact for fact in core if family_by_node[fact[0]] in target_families
        }
        target_derived = {
            fact
            for fact in derived
            if family_by_node[fact[0]] in target_families
        }
        support_facts = (
            source_facts
            | target_core
            | sampled(target_derived, observed_fraction)
        )
        positive_pool = target_derived - support_facts

    max_context = max_context_per_class or context_per_class
    evaluation_size = evaluation_per_class or context_per_class
    candidate_rng = np.random.default_rng(
        split_seed if split_seed is not None else graph_seed + 1_000_000
    )
    context, evaluation, population = _candidate_frames(
        positive_pool,
        complete_facts,
        family_by_node,
        context_per_class=context_per_class,
        max_context_per_class=max_context,
        evaluation_per_class=evaluation_size,
        rng=candidate_rng,
    )
    support = _graph_from_facts(template, support_facts)
    support, complete, relabeled_frames = _relabel(
        support,
        complete,
        (context, evaluation, population),
        relabel_seed if relabel_seed is not None else graph_seed + 2_000_000,
    )
    context, evaluation, population = relabeled_frames
    metadata: dict[str, object] = {
        "target": "genealogy_link_truth",
        "split_name": split_name,
        "observed_fraction": observed_fraction,
        "family_count": family_count,
        "source_families": sorted(source_families),
        "target_families": sorted(target_families),
        "context_per_class": context_per_class,
        "max_context_per_class": max_context,
        "evaluation_per_class": evaluation_size,
        "graph_seed": graph_seed,
        "split_seed": (
            split_seed if split_seed is not None else graph_seed + 1_000_000
        ),
        "relabel_seed": (
            relabel_seed if relabel_seed is not None else graph_seed + 2_000_000
        ),
        "entity_relabeling": "independent random non-contiguous 52-bit tokens",
        "matrix_exposed_to_decoder": False,
        "support_sha256": _hash_graph(support),
        "complete_sha256": _hash_graph(complete),
    }
    return GenealogyTask(
        split_name=split_name,
        observed_fraction=observed_fraction,
        support_graph=support,
        complete_graph=complete,
        context=context,
        evaluation=evaluation,
        population=population,
        metadata=metadata,
    )
