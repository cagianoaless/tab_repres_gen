from __future__ import annotations

import ast
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.hashing import file_sha256, object_sha256

CLUTRR_RELATIONS = (
    "aunt",
    "son-in-law",
    "grandfather",
    "brother",
    "sister",
    "father",
    "mother",
    "grandmother",
    "uncle",
    "daughter-in-law",
    "grandson",
    "granddaughter",
    "father-in-law",
    "mother-in-law",
    "nephew",
    "son",
    "daughter",
    "niece",
)
CLUTRR_RELATION_TO_LABEL = {
    relation: index for index, relation in enumerate(CLUTRR_RELATIONS)
}
CLUTRR_EDGE_RELATIONS = (
    "aunt",
    "brother",
    "daughter",
    "father",
    "granddaughter",
    "grandfather",
    "grandmother",
    "grandson",
    "husband",
    "mother",
    "sister",
    "son",
    "uncle",
    "wife",
)
CLUTRR_DATASET_CONFIGS = {
    "gen_train23_test2to10": {
        "train_chain_lengths": (2, 3),
        "evaluation_chain_lengths": tuple(range(2, 11)),
    },
    "gen_train234_context23_test2to10": {
        "train_chain_lengths": (2, 3),
        "evaluation_chain_lengths": tuple(range(2, 11)),
    },
    "gen_train234_test2to10": {
        "train_chain_lengths": (2, 3, 4),
        "evaluation_chain_lengths": tuple(range(2, 11)),
    },
}


@dataclass(frozen=True)
class ClutrrTask:
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, Any]


def _literal(value: str, expected_type: type) -> Any:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"CLUTRR field is not a {expected_type.__name__}")
    return parsed


def _chain_length(task_name: str) -> int:
    prefix = "task_1."
    if not task_name.startswith(prefix):
        raise ValueError(f"unsupported CLUTRR task_name: {task_name}")
    length = int(task_name[len(prefix) :])
    if length < 2 or length > 10:
        raise ValueError(f"unsupported CLUTRR chain length: {length}")
    return length


def _relabel_story(
    row_id: str,
    edges: tuple[tuple[int, int], ...],
    query_edge: tuple[int, int],
    relabel_seed: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    nodes = sorted({node for edge in edges for node in edge} | set(query_edge))
    digest = hashlib.sha256(f"{row_id}:{relabel_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    assigned: set[int] = set()
    mapping: dict[int, int] = {}
    for node in nodes:
        while True:
            candidate = int(rng.integers(2**40, 2**52, endpoint=False))
            if candidate not in assigned:
                break
        assigned.add(candidate)
        mapping[node] = candidate
    return (
        tuple((mapping[head], mapping[tail]) for head, tail in edges),
        (mapping[query_edge[0]], mapping[query_edge[1]]),
    )


@lru_cache(maxsize=16)
def _read_split(
    path: Path,
    *,
    expected_sha256: str,
    allowed_chain_lengths: tuple[int, ...],
    relabel_seed: int,
    split_name: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CLUTRR split is missing: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"CLUTRR checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "target",
            "target_text",
            "task_name",
            "story_edges",
            "edge_types",
            "query_edge",
        }
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"CLUTRR split omits columns: {sorted(required)}")
        for raw in reader:
            length = _chain_length(raw["task_name"])
            if length not in allowed_chain_lengths:
                continue
            relation = raw["target_text"]
            if relation not in CLUTRR_RELATION_TO_LABEL:
                raise ValueError(f"unknown CLUTRR target relation: {relation}")
            label = int(raw["target"])
            if CLUTRR_RELATIONS[label] != relation:
                raise ValueError("CLUTRR integer and text targets disagree")
            raw_edges = tuple(tuple(map(int, edge)) for edge in _literal(raw["story_edges"], list))
            edge_types = tuple(map(str, _literal(raw["edge_types"], list)))
            query_edge = tuple(map(int, _literal(raw["query_edge"], tuple)))
            if (
                len(raw_edges) != length
                or len(edge_types) != length
                or len(query_edge) != 2
            ):
                raise ValueError("CLUTRR chain length disagrees with symbolic fields")
            if any(raw_edges[index][1] != raw_edges[index + 1][0] for index in range(length - 1)):
                raise ValueError("CLUTRR story edges are not an ordered reasoning chain")
            if query_edge != (raw_edges[0][0], raw_edges[-1][1]):
                raise ValueError("CLUTRR query endpoints disagree with the reasoning chain")
            if any(edge_type not in CLUTRR_EDGE_RELATIONS for edge_type in edge_types):
                raise ValueError("CLUTRR path contains an unknown relation")
            edges, relabeled_query = _relabel_story(
                raw["id"], raw_edges, query_edge, relabel_seed
            )
            records.append(
                {
                    "row_id": raw["id"],
                    "split": split_name,
                    "chain_length": length,
                    "story_edges": edges,
                    "edge_types": edge_types,
                    "query_edge": relabeled_query,
                    "label": label,
                    "target_relation": relation,
                }
            )
    if not records:
        raise ValueError(f"CLUTRR split has no registered rows: {path}")
    frame = pd.DataFrame.from_records(records)
    if frame["row_id"].duplicated().any():
        raise ValueError(f"CLUTRR split contains duplicate row IDs: {path}")
    return frame


@lru_cache(maxsize=16)
def _verified_file(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and file_sha256(path) == expected_sha256


def _nested_context(
    training: pd.DataFrame,
    *,
    context_per_relation: int,
    max_context_per_relation: int,
    split_seed: int,
) -> pd.DataFrame:
    if context_per_relation > max_context_per_relation:
        raise ValueError("CLUTRR context exceeds its registered maximum")
    selected: list[pd.DataFrame] = []
    for label in range(len(CLUTRR_RELATIONS)):
        group = training[training["label"] == label]
        if len(group) < max_context_per_relation:
            raise ValueError(
                f"CLUTRR relation {CLUTRR_RELATIONS[label]} has only {len(group)} "
                f"training rows; {max_context_per_relation} required"
            )
        rng = np.random.default_rng(split_seed + label * 10_007)
        order = rng.permutation(len(group))
        selected.append(group.iloc[order[:context_per_relation]])
    context = pd.concat(selected, ignore_index=True)
    order_rng = np.random.default_rng(split_seed + 900_001)
    context = context.iloc[order_rng.permutation(len(context))].reset_index(drop=True)
    context.insert(0, "query_id", np.arange(len(context), dtype=np.int64))
    return context


def _paired_additive_context(
    training: pd.DataFrame,
    *,
    base_context_per_relation: int,
    max_base_context_per_relation: int,
    added_per_relation: int,
    base_chain_lengths: tuple[int, ...],
    added_chain_lengths: tuple[int, ...],
    split_seed: int,
) -> pd.DataFrame:
    """Build paired arms with an identical nested base and fixed added examples."""
    if base_context_per_relation > max_base_context_per_relation:
        raise ValueError("CLUTRR base context exceeds its registered maximum")
    selected: list[pd.DataFrame] = []
    for label in range(len(CLUTRR_RELATIONS)):
        label_rows = training[training["label"] == label]
        base_pool = label_rows[
            label_rows["chain_length"].isin(base_chain_lengths)
        ]
        base_rng = np.random.default_rng(split_seed + label * 10_007)
        base_order = base_rng.permutation(len(base_pool))
        required_base = max_base_context_per_relation + (
            added_per_relation if added_chain_lengths == base_chain_lengths else 0
        )
        if len(base_pool) < required_base:
            raise ValueError(
                f"CLUTRR relation {CLUTRR_RELATIONS[label]} has only "
                f"{len(base_pool)} base rows; {required_base} required"
            )
        base = base_pool.iloc[base_order[:base_context_per_relation]].copy()
        base["context_role"] = "base"

        if added_chain_lengths == base_chain_lengths:
            addition = base_pool.iloc[
                base_order[
                    max_base_context_per_relation:
                    max_base_context_per_relation + added_per_relation
                ]
            ].copy()
        else:
            addition_pool = label_rows[
                label_rows["chain_length"].isin(added_chain_lengths)
            ]
            if len(addition_pool) < added_per_relation:
                raise ValueError(
                    f"CLUTRR relation {CLUTRR_RELATIONS[label]} has only "
                    f"{len(addition_pool)} added-length rows; "
                    f"{added_per_relation} required"
                )
            addition_rng = np.random.default_rng(
                split_seed + label * 10_007 + 700_003
            )
            addition = addition_pool.iloc[
                addition_rng.permutation(len(addition_pool))[:added_per_relation]
            ].copy()
        addition["context_role"] = "addition"
        selected.extend((base, addition))

    context = pd.concat(selected, ignore_index=True)
    order_rng = np.random.default_rng(split_seed + 900_001)
    context = context.iloc[order_rng.permutation(len(context))].reset_index(drop=True)
    context.insert(0, "query_id", np.arange(len(context), dtype=np.int64))
    return context


def _evaluation_rows(test: pd.DataFrame) -> pd.DataFrame:
    evaluation = test.sort_values(["chain_length", "row_id"]).reset_index(drop=True)
    evaluation.insert(0, "query_id", np.arange(len(evaluation), dtype=np.int64))
    return evaluation


def load_clutrr_task(repo_root: Path, cell: dict[str, Any]) -> ClutrrTask:
    config_name = str(cell["dataset_config"])
    try:
        registered = CLUTRR_DATASET_CONFIGS[config_name]
    except KeyError as error:
        raise ValueError(f"unsupported CLUTRR dataset config: {config_name}") from error
    train_lengths = tuple(map(int, cell["train_chain_lengths"]))
    evaluation_lengths = tuple(map(int, cell["evaluation_chain_lengths"]))
    if train_lengths != registered["train_chain_lengths"]:
        raise ValueError("CLUTRR training chain lengths changed")
    if evaluation_lengths != registered["evaluation_chain_lengths"]:
        raise ValueError("CLUTRR evaluation chain lengths changed")

    files = cell["asset_files"]
    for split in ("train", "validation", "test"):
        spec = files.get(split)
        if not isinstance(spec, dict):
            raise ValueError(f"CLUTRR asset specification is missing {split}")
        path = repo_root / str(spec["path"])
        if not _verified_file(path, str(spec["sha256"])):
            raise ValueError(f"CLUTRR {split} asset is absent or has the wrong checksum")

    training = _read_split(
        repo_root / files["train"]["path"],
        expected_sha256=files["train"]["sha256"],
        allowed_chain_lengths=train_lengths,
        relabel_seed=int(cell["relabel_seed"]),
        split_name="train",
    )
    test = _read_split(
        repo_root / files["test"]["path"],
        expected_sha256=files["test"]["sha256"],
        allowed_chain_lengths=evaluation_lengths,
        relabel_seed=int(cell["relabel_seed"]) + 1,
        split_name="test",
    )
    context_sampling = cell.get("context_sampling")
    if context_sampling is None:
        context = _nested_context(
            training,
            context_per_relation=int(cell["context_per_relation"]),
            max_context_per_relation=int(cell["max_context_per_relation"]),
            split_seed=int(cell["split_seed"]),
        )
    else:
        context = _paired_additive_context(
            training,
            base_context_per_relation=int(cell["base_context_per_relation"]),
            max_base_context_per_relation=int(
                cell["max_base_context_per_relation"]
            ),
            added_per_relation=int(context_sampling["added_per_relation"]),
            base_chain_lengths=tuple(map(int, context_sampling["base_chain_lengths"])),
            added_chain_lengths=tuple(
                map(int, context_sampling["added_chain_lengths"])
            ),
            split_seed=int(cell["split_seed"]),
        )
    evaluation = _evaluation_rows(test)
    metadata = {
        "dataset": "CLUTRR/v1",
        "dataset_config": config_name,
        "train_chain_lengths": list(train_lengths),
        "evaluation_chain_lengths": list(evaluation_lengths),
        "context_per_relation": int(cell["context_per_relation"]),
        "context_rows": len(context),
        "evaluation_rows": len(evaluation),
        "context_class_counts": {
            CLUTRR_RELATIONS[int(label)]: int(count)
            for label, count in sorted(Counter(context["label"]).items())
        },
        "context_chain_counts": {
            str(int(length)): int(count)
            for length, count in sorted(Counter(context["chain_length"]).items())
        },
        "evaluation_chain_counts": {
            str(int(length)): int(count)
            for length, count in sorted(Counter(evaluation["chain_length"]).items())
        },
        "asset_sha256": {
            split: str(files[split]["sha256"]) for split in ("train", "validation", "test")
        },
        "task_sha256": object_sha256(
            {
                "context_row_ids": context["row_id"].tolist(),
                "evaluation_row_ids": evaluation["row_id"].tolist(),
                "relabel_seed": int(cell["relabel_seed"]),
            }
        ),
    }
    if context_sampling is not None:
        base_rows = context[context["context_role"] == "base"]
        added_rows = context[context["context_role"] == "addition"]
        metadata.update(
            {
                "context_sampling": context_sampling,
                "base_context_per_relation": int(
                    cell["base_context_per_relation"]
                ),
                "added_context_per_relation": int(
                    context_sampling["added_per_relation"]
                ),
                "base_context_row_ids_sha256": object_sha256(
                    sorted(map(str, base_rows["row_id"]))
                ),
                "added_context_row_ids_sha256": object_sha256(
                    sorted(map(str, added_rows["row_id"]))
                ),
            }
        )
    return ClutrrTask(
        context=context,
        evaluation=evaluation,
        population=evaluation,
        metadata=metadata,
    )


def context_row_ids_by_relation(
    training: pd.DataFrame,
    *,
    max_context_per_relation: int,
    split_seed: int,
) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = defaultdict(tuple)
    for label in range(len(CLUTRR_RELATIONS)):
        group = training[training["label"] == label]
        rng = np.random.default_rng(split_seed + label * 10_007)
        order = rng.permutation(len(group))[:max_context_per_relation]
        result[label] = tuple(map(str, group.iloc[order]["row_id"]))
    return dict(result)
