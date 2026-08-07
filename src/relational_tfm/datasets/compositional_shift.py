from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.hashing import object_sha256

DIHEDRAL_ROTATIONS = 8
COMPOSITION_CLASS_COUNT = 2 * DIHEDRAL_ROTATIONS
COMPOSITION_CLASS_NAMES = tuple(
    f"{'f' if reflected else 'r'}{rotation}"
    for reflected in range(2)
    for rotation in range(DIHEDRAL_ROTATIONS)
)
COMPOSITION_TOKENS = tuple(range(COMPOSITION_CLASS_COUNT))
GENERATOR_VERSION = "dihedral_d8_length_shift_v1"


@dataclass(frozen=True)
class CompositionalShiftTask:
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, Any]


def decode_element(index: int) -> tuple[int, int]:
    if index < 0 or index >= COMPOSITION_CLASS_COUNT:
        raise ValueError(f"invalid D8 element index: {index}")
    return index % DIHEDRAL_ROTATIONS, index // DIHEDRAL_ROTATIONS


def encode_element(rotation: int, reflected: int) -> int:
    if reflected not in {0, 1}:
        raise ValueError("D8 reflection bit must be zero or one")
    return reflected * DIHEDRAL_ROTATIONS + rotation % DIHEDRAL_ROTATIONS


def multiply_elements(left: int, right: int) -> int:
    left_rotation, left_reflected = decode_element(left)
    right_rotation, right_reflected = decode_element(right)
    signed_right_rotation = (
        right_rotation if left_reflected == 0 else -right_rotation
    )
    return encode_element(
        left_rotation + signed_right_rotation,
        left_reflected ^ right_reflected,
    )


def inverse_element(element: int) -> int:
    rotation, reflected = decode_element(element)
    return encode_element(-rotation if reflected == 0 else rotation, reflected)


def compose_tokens(tokens: tuple[int, ...]) -> int:
    result = 0
    for token in tokens:
        result = multiply_elements(result, int(token))
    return result


def _complete_sequence(prefix: tuple[int, ...], target: int) -> tuple[int, ...]:
    prefix_product = compose_tokens(prefix)
    final_token = multiply_elements(inverse_element(prefix_product), target)
    sequence = prefix + (final_token,)
    if compose_tokens(sequence) != target:
        raise AssertionError("D8 completion oracle returned an invalid sequence")
    return sequence


def _random_nodes(row_key: str, relabel_seed: int, length: int) -> tuple[int, ...]:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{row_key}:{relabel_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    nodes: list[int] = []
    used: set[int] = set()
    while len(nodes) < length + 1:
        candidate = int(rng.integers(2**40, 2**52, endpoint=False))
        if candidate not in used:
            nodes.append(candidate)
            used.add(candidate)
    return tuple(nodes)


def _record(
    *,
    row_key: str,
    split: str,
    tokens: tuple[int, ...],
    target: int,
    relabel_seed: int,
) -> dict[str, Any]:
    nodes = _random_nodes(row_key, relabel_seed, len(tokens))
    edges = tuple(zip(nodes[:-1], nodes[1:], strict=True))
    return {
        "row_id": object_sha256(
            {
                "generator": GENERATOR_VERSION,
                "row_key": row_key,
                "tokens": list(tokens),
                "target": target,
            }
        )[:24],
        "split": split,
        "chain_length": len(tokens),
        "story_edges": edges,
        "relation_tokens": tokens,
        "query_edge": (nodes[0], nodes[-1]),
        "label": target,
        "target_relation": COMPOSITION_CLASS_NAMES[target],
    }


def _context_pool(
    *,
    max_context_per_class: int,
    graph_seed: int,
    relabel_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target in range(COMPOSITION_CLASS_COUNT):
        rng = np.random.default_rng(graph_seed + target * 100_003)
        for index in range(max_context_per_class):
            length = 2 + index % 2
            prefix = tuple(
                int(value)
                for value in rng.integers(
                    0,
                    COMPOSITION_CLASS_COUNT,
                    size=length - 1,
                    endpoint=False,
                )
            )
            tokens = _complete_sequence(prefix, target)
            records.append(
                _record(
                    row_key=f"context:{target}:{index}",
                    split="context",
                    tokens=tokens,
                    target=target,
                    relabel_seed=relabel_seed,
                )
            )
    return pd.DataFrame.from_records(records)


def _evaluation_population(
    *,
    chain_lengths: tuple[int, ...],
    per_class_per_length: int,
    graph_seed: int,
    relabel_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for length in chain_lengths:
        for target in range(COMPOSITION_CLASS_COUNT):
            rng = np.random.default_rng(
                graph_seed + 10_000_019 * length + target * 100_003
            )
            prefixes: set[tuple[int, ...]] = set()
            while len(prefixes) < per_class_per_length:
                prefix = tuple(
                    int(value)
                    for value in rng.integers(
                        0,
                        COMPOSITION_CLASS_COUNT,
                        size=length - 1,
                        endpoint=False,
                    )
                )
                prefixes.add(prefix)
            for index, prefix in enumerate(sorted(prefixes)):
                tokens = _complete_sequence(prefix, target)
                records.append(
                    _record(
                        row_key=f"evaluation:{length}:{target}:{index}",
                        split="evaluation",
                        tokens=tokens,
                        target=target,
                        relabel_seed=relabel_seed + 1,
                    )
                )
    frame = pd.DataFrame.from_records(records)
    order_rng = np.random.default_rng(graph_seed + 90_000_001)
    return frame.iloc[order_rng.permutation(len(frame))].reset_index(drop=True)


def generate_compositional_shift_task(cell: dict[str, Any]) -> CompositionalShiftTask:
    if cell["algebra"] != "dihedral_d8":
        raise ValueError(f"unsupported compositional algebra: {cell['algebra']}")
    train_lengths = tuple(map(int, cell["train_chain_lengths"]))
    evaluation_lengths = tuple(map(int, cell["evaluation_chain_lengths"]))
    if train_lengths != (2, 3) or evaluation_lengths != tuple(range(4, 11)):
        raise ValueError("registered D8 length-shift protocol changed")

    max_context = int(cell["max_context_per_class"])
    context_per_class = int(cell["context_per_class"])
    if context_per_class > max_context:
        raise ValueError("context exceeds registered D8 context pool")

    pool = _context_pool(
        max_context_per_class=max_context,
        graph_seed=int(cell["graph_seed"]),
        relabel_seed=int(cell["relabel_seed"]),
    )
    selected: list[pd.DataFrame] = []
    for label in range(COMPOSITION_CLASS_COUNT):
        group = pool[pool["label"] == label]
        selected.append(group.iloc[:context_per_class])
    context = pd.concat(selected, ignore_index=True)
    context_rng = np.random.default_rng(int(cell["split_seed"]) + 90_000_001)
    context = context.iloc[context_rng.permutation(len(context))].reset_index(drop=True)
    context.insert(0, "query_id", np.arange(len(context), dtype=np.int64))

    evaluation = _evaluation_population(
        chain_lengths=evaluation_lengths,
        per_class_per_length=int(cell["evaluation_per_class_per_length"]),
        graph_seed=int(cell["graph_seed"]) + 1_000_000,
        relabel_seed=int(cell["relabel_seed"]) + 1_000_000,
    )
    evaluation.insert(
        0,
        "query_id",
        np.arange(len(context), len(context) + len(evaluation), dtype=np.int64),
    )

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "algebra": "dihedral_d8",
        "group_order": COMPOSITION_CLASS_COUNT,
        "train_chain_lengths": list(train_lengths),
        "evaluation_chain_lengths": list(evaluation_lengths),
        "context_per_class": context_per_class,
        "max_context_per_class": max_context,
        "evaluation_per_class_per_length": int(
            cell["evaluation_per_class_per_length"]
        ),
        "graph_seed": int(cell["graph_seed"]),
        "relabel_seed": int(cell["relabel_seed"]),
        "split_seed": int(cell["split_seed"]),
        "context_row_ids_sha256": object_sha256(sorted(context["row_id"].tolist())),
        "evaluation_row_ids_sha256": object_sha256(
            sorted(evaluation["row_id"].tolist())
        ),
        "entity_ids_exposed": False,
        "target_relation_exposed": False,
    }
    return CompositionalShiftTask(
        context=context,
        evaluation=evaluation,
        population=evaluation.copy(),
        metadata=metadata,
    )
