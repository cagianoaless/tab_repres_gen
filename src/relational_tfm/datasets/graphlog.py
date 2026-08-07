from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.encoders.graphlog import (
    GRAPHLOG_RELATION_INDEX,
    GRAPHLOG_RELATIONS,
)
from relational_tfm.hashing import file_sha256, object_sha256


@dataclass(frozen=True)
class GraphLogTask:
    context: pd.DataFrame
    evaluation: pd.DataFrame
    population: pd.DataFrame
    metadata: dict[str, Any]


@lru_cache(maxsize=4)
def _verified_asset(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"GraphLog asset is missing: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"GraphLog checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return True


@lru_cache(maxsize=4)
def _world_members(path: Path, expected_sha256: str) -> dict[str, dict[str, str]]:
    _verified_asset(path, expected_sha256)
    worlds: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            parts = Path(member).parts
            if (
                len(parts) != 4
                or parts[0] != "graphlog_v1.1"
                or parts[3] not in {"train.jsonl", "valid.jsonl", "test.jsonl"}
            ):
                continue
            outer_split, world = parts[1], parts[2]
            split = parts[3].removesuffix(".jsonl")
            record = worlds.setdefault(world, {"outer_split": outer_split})
            if record["outer_split"] != outer_split:
                raise ValueError(f"GraphLog world {world} appears in two outer splits")
            record[split] = member
    required = {"outer_split", "train", "valid", "test"}
    if len(worlds) != 57 or any(set(record) != required for record in worlds.values()):
        raise ValueError("GraphLog archive does not contain 57 complete worlds")
    return worlds


def _descriptor_matches_graph_path(row: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    edge_relations: dict[tuple[Any, Any], set[str]] = {}
    for edge in row["edges"]:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError("GraphLog row contains an invalid edge")
        edge_relations.setdefault((edge[0], edge[1]), set()).add(str(edge[2]))
    path = row["resolution_path"]
    return all(
        token in edge_relations.get((path[index], path[index + 1]), set())
        for index, token in enumerate(tokens)
    )


def _row_record(
    raw: dict[str, Any],
    *,
    world: str,
    split: str,
    line_index: int,
) -> dict[str, Any]:
    descriptor = raw.get("descriptor")
    target = raw.get("target")
    query = raw.get("query")
    path = raw.get("resolution_path")
    edges = raw.get("edges")
    if (
        not isinstance(descriptor, str)
        or not isinstance(target, str)
        or target not in GRAPHLOG_RELATION_INDEX
        or not isinstance(query, list)
        or len(query) != 3
        or not isinstance(path, list)
        or not isinstance(edges, list)
    ):
        raise ValueError(f"GraphLog {world}/{split} row has an invalid schema")
    tokens = tuple(descriptor.split(","))
    if (
        not tokens
        or any(token not in GRAPHLOG_RELATION_INDEX for token in tokens)
        or len(path) != len(tokens) + 1
        or (query[0], query[1]) != (path[0], path[-1])
        or query[2] != target
    ):
        raise ValueError(f"GraphLog {world}/{split} row is internally inconsistent")
    return {
        "row_id": object_sha256(
            {
                "dataset": "GraphLog/v1.1",
                "world": world,
                "split": split,
                "line_index": line_index,
                "query": query,
                "descriptor": descriptor,
            }
        )[:24],
        "world": world,
        "split": split,
        "chain_length": len(tokens),
        "relation_tokens": tokens,
        "label": GRAPHLOG_RELATION_INDEX[target],
        "target_relation": target,
        "descriptor_graph_path_exact_match": _descriptor_matches_graph_path(
            raw, tokens
        ),
    }


@lru_cache(maxsize=128)
def _load_world(
    path: Path,
    expected_sha256: str,
    world: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    members = _world_members(path, expected_sha256)
    if world not in members:
        raise ValueError(f"unknown GraphLog world: {world}")
    frames = {}
    with zipfile.ZipFile(path) as archive:
        for split in ("train", "test"):
            records = []
            with archive.open(members[world][split]) as handle:
                for line_index, line in enumerate(handle):
                    raw = json.loads(line)
                    records.append(
                        _row_record(
                            raw,
                            world=world,
                            split=split,
                            line_index=line_index,
                        )
                    )
            frames[split] = pd.DataFrame.from_records(records)
    return frames["train"], frames["test"], members[world]["outer_split"]


def _nested_context(
    pool: pd.DataFrame,
    *,
    context_per_relation: int,
    max_context_per_relation: int,
    split_seed: int,
) -> pd.DataFrame:
    selected = []
    labels = sorted(map(int, pool["label"].unique()))
    for label in labels:
        group = pool[pool["label"] == label]
        if len(group) < max_context_per_relation:
            relation = GRAPHLOG_RELATIONS[label]
            raise ValueError(
                f"GraphLog relation {relation} has {len(group)} short rows; "
                f"{max_context_per_relation} required"
            )
        rng = np.random.default_rng(split_seed + label * 10_007)
        order = rng.permutation(len(group))
        selected.append(group.iloc[order[:context_per_relation]])
    context = pd.concat(selected, ignore_index=True)
    rng = np.random.default_rng(split_seed + 900_001)
    context = context.iloc[rng.permutation(len(context))].reset_index(drop=True)
    context.insert(0, "query_id", np.arange(len(context), dtype=np.int64))
    return context


def load_graphlog_task(
    repo_root: Path,
    cell: dict[str, Any],
) -> GraphLogTask:
    asset = cell["asset"]
    path = (repo_root / asset["path"]).resolve()
    expected_sha256 = str(asset["sha256"])
    training, testing, outer_split = _load_world(
        path,
        expected_sha256,
        str(cell["world"]),
    )
    train_lengths = tuple(map(int, cell["train_chain_lengths"]))
    evaluation_lengths = tuple(map(int, cell["evaluation_chain_lengths"]))
    context_pool = training[training["chain_length"].isin(train_lengths)].copy()
    evaluation = testing[testing["chain_length"].isin(evaluation_lengths)].copy()
    if context_pool.empty or evaluation.empty:
        raise ValueError("GraphLog registered length split produced an empty population")

    context = _nested_context(
        context_pool,
        context_per_relation=int(cell["context_per_relation"]),
        max_context_per_relation=int(cell["max_context_per_relation"]),
        split_seed=int(cell["split_seed"]),
    )
    evaluation = evaluation.sort_values("row_id").reset_index(drop=True)
    evaluation.insert(
        0,
        "query_id",
        np.arange(len(context), len(context) + len(evaluation), dtype=np.int64),
    )
    context_labels = set(map(int, context["label"].unique()))
    evaluation_labels = set(map(int, evaluation["label"].unique()))
    if not evaluation_labels <= context_labels:
        missing = sorted(evaluation_labels - context_labels)
        raise ValueError(f"GraphLog OOD labels are absent from context: {missing}")

    metadata = {
        "dataset": "GraphLog/v1.1",
        "asset_sha256": expected_sha256,
        "world": str(cell["world"]),
        "outer_split": outer_split,
        "context_inner_split": "train",
        "evaluation_inner_split": "test",
        "train_chain_lengths": list(train_lengths),
        "evaluation_chain_lengths": list(evaluation_lengths),
        "context_per_relation": int(cell["context_per_relation"]),
        "max_context_per_relation": int(cell["max_context_per_relation"]),
        "context_class_count": len(context_labels),
        "evaluation_class_count": len(evaluation_labels),
        "context_row_ids_sha256": object_sha256(sorted(context["row_id"].tolist())),
        "evaluation_row_ids_sha256": object_sha256(
            sorted(evaluation["row_id"].tolist())
        ),
        "context_descriptor_graph_path_exact_match_fraction": float(
            context["descriptor_graph_path_exact_match"].mean()
        ),
        "evaluation_descriptor_graph_path_exact_match_fraction": float(
            evaluation["descriptor_graph_path_exact_match"].mean()
        ),
        "entity_ids_exposed": False,
        "target_relation_exposed": False,
        "symbolic_input_field": "descriptor",
    }
    return GraphLogTask(
        context=context,
        evaluation=evaluation,
        population=evaluation.copy(),
        metadata=metadata,
    )
