from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.protocol import (
    FROZEN_KG_INTERFACES,
    frozen_kg_asset,
)


@dataclass(frozen=True)
class FrozenKGTask:
    context_values: np.ndarray
    context_labels: np.ndarray
    candidate_values: np.ndarray
    native_ultra_scores: np.ndarray
    filtered: np.ndarray
    target_entities: np.ndarray
    side_codes: np.ndarray
    feature_names: tuple[str, ...]
    categorical_feature_indices: tuple[int, ...]
    metadata: dict[str, Any]


def _asset_root(repo_root: Path, asset_id: str) -> Path:
    override = os.environ.get("RTFM_FROZEN_KG_ASSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    registered = frozen_kg_asset(asset_id)
    return (repo_root / str(registered["asset_root"])).resolve()


@lru_cache(maxsize=2)
def _load_archive(
    asset_root: str,
    dataset: str,
    manifest_sha256: str,
    archive_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(asset_root)
    manifest_path = root / "manifest.json"
    archive_path = root / f"{dataset}.npz"
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("RFMLink manifest bytes do not match the frozen hash")
    if file_sha256(archive_path) != archive_sha256:
        raise ValueError(f"RFMLink archive bytes changed for {dataset}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = manifest["datasets"][dataset]
    if registered["archive_sha256"] != archive_sha256:
        raise ValueError(f"RFMLink manifest/archive mismatch for {dataset}")
    with np.load(archive_path, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    return arrays, manifest


def _nested_context_indices(labels: np.ndarray, seed: int, budget: int) -> np.ndarray:
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng = np.random.default_rng(np.random.SeedSequence([20260714, seed, label, 17]))
        selected.append(indices[rng.permutation(len(indices))][:budget])
    output = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    if len(output) != 2 * budget or int(labels[output].sum()) != budget:
        raise ValueError("RFMLink context selection lost class balance")
    return output


def _affine_parameters(size: int, key: str) -> tuple[int, int]:
    digest = hashlib.sha256(key.encode()).digest()
    candidate = int.from_bytes(digest[:8], "little") % max(size, 1)
    candidate = max(candidate, 1)
    while math.gcd(candidate, size) != 1:
        candidate += 1
        if candidate >= size:
            candidate = 1
    offset = int.from_bytes(digest[8:16], "little") % size
    return candidate, offset


def _permuted_rows(values: np.ndarray, group_size: int, key: str) -> np.ndarray:
    if group_size <= 0 or len(values) % group_size:
        raise ValueError("permutation groups do not divide the RFMLink rows")
    indices = np.arange(len(values), dtype=np.int64)
    groups = indices // group_size
    positions = indices % group_size
    output = np.empty(values.shape, dtype=np.float32)
    for column in range(values.shape[1]):
        multiplier, offset = _affine_parameters(group_size, f"{key}/{column}")
        source_positions = (
            multiplier * positions + offset + groups * (column + 1)
        ) % group_size
        source_rows = groups * group_size + source_positions
        output[:, column] = np.asarray(values[source_rows, column], dtype=np.float32)
    return output


# Per-entity random scalar tokens for the identity-control interface: the KG analog
# of the rule_induction random_entity_tokens encoder. Dataset-scoped and deterministic,
# so context rows and every candidate block of a dataset share one token table; on
# inductive splits the tokens are arbitrary by construction, so any ranking signal a
# decoder extracts from them is memorization, not structure.
def _entity_tokens(dataset: str, num_entities: int) -> np.ndarray:
    digest = hashlib.sha256(f"rfmlink-entity-tokens/{dataset}".encode()).digest()
    seed_words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    rng = np.random.default_rng(
        np.random.SeedSequence([20260725, num_entities, *seed_words])
    )
    return rng.standard_normal(num_entities).astype(np.float32)


def _candidate_token_columns(arrays: dict[str, np.ndarray], dataset: str) -> np.ndarray:
    if "anchor_entities" not in arrays:
        raise ValueError(
            "entity_tokens_only requires an archive with per-row entity identities "
            "(archive schema 2)"
        )
    num_entities = int(arrays["num_entities"])
    tokens = _entity_tokens(dataset, num_entities)
    anchors = np.asarray(arrays["anchor_entities"], dtype=np.int64)
    sides = np.asarray(arrays["side_codes"], dtype=np.int8)
    output = np.empty((len(anchors) * num_entities, 2), dtype=np.float32)
    for direction, (anchor, side) in enumerate(zip(anchors, sides, strict=False)):
        rows = slice(direction * num_entities, (direction + 1) * num_entities)
        if side == 0:  # ranking tails: the head entity is fixed
            output[rows, 0] = tokens[anchor]
            output[rows, 1] = tokens
        else:  # ranking heads: the tail entity is fixed
            output[rows, 0] = tokens
            output[rows, 1] = tokens[anchor]
    return output


def _context_token_columns(
    arrays: dict[str, np.ndarray],
    dataset: str,
    seed: int,
    context_prefix: str | None = None,
) -> np.ndarray:
    prefix = context_prefix or f"context_{seed}"
    key = f"{prefix}_triples"
    if key not in arrays:
        raise ValueError(
            "entity_tokens_only requires an archive with stored context triples "
            "(archive schema 2)"
        )
    tokens = _entity_tokens(dataset, int(arrays["num_entities"]))
    triples = np.asarray(arrays[key], dtype=np.int64)
    return np.stack(
        [tokens[triples[:, 0]], tokens[triples[:, 2]]], axis=1
    ).astype(np.float32)


_DYADIC_B0_COLUMNS = (8, 9, 10)
_DYADIC_B0_NAMES = ("pair_count_log", "reverse_pair_count_log", "same_entity")
_PATH_B0_COLUMNS = (11, 12, 13, 14, 15)
_PATH_B0_NAMES = (
    "common_neighbors_log",
    "two_hop_count_log",
    "start_relation_two_hop_count_log",
    "end_relation_two_hop_count_log",
    "reverse_two_hop_count_log",
)
_LEGACY_DYADIC_B0_COLUMNS = (10, 11)
_LEGACY_DYADIC_B0_NAMES = (
    "pair_other_relation_count",
    "reverse_pair_relation_count",
)
_LEGACY_PATH_B0_COLUMNS = (12, 13, 14, 15)
_LEGACY_PATH_B0_NAMES = (
    "two_hop_count",
    "two_hop_start_relation_count",
    "two_hop_end_relation_count",
    "reverse_two_hop_count",
)


def _candidate_same_entity_column(arrays: dict[str, np.ndarray]) -> np.ndarray:
    num_entities = int(arrays["num_entities"])
    anchors = np.asarray(arrays["anchor_entities"], dtype=np.int64)
    candidates = np.arange(num_entities, dtype=np.int64)
    return np.concatenate(
        [(candidates == anchor).astype(np.float32) for anchor in anchors]
    )[:, None]


def _context_same_entity_column(
    arrays: dict[str, np.ndarray],
    seed: int,
    context_prefix: str | None = None,
) -> np.ndarray:
    prefix = context_prefix or f"context_{seed}"
    triples = np.asarray(arrays[f"{prefix}_triples"], dtype=np.int64)
    return (triples[:, 0] == triples[:, 2]).astype(np.float32)[:, None]


def _named_b0_subset(
    b0: np.ndarray,
    b0_names: tuple[str, ...],
    columns: tuple[int, ...],
    expected_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    actual = tuple(b0_names[i] for i in columns)
    if actual != expected_names:
        raise ValueError(
            f"b0 column layout changed: expected {expected_names} at {columns}, found {actual}"
        )
    return b0[:, list(columns)], actual


def _interface(
    b0: np.ndarray,
    b2v2: np.ndarray,
    interface: str,
    b0_names: tuple[str, ...],
    b2v2_names: tuple[str, ...],
    *,
    graphlet: np.ndarray | None = None,
    graphlet_names: tuple[str, ...] = (),
    permutation_group_size: int | None = None,
    permutation_key: str | None = None,
    token_columns: np.ndarray | None = None,
    same_entity_column: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    if interface not in FROZEN_KG_INTERFACES:
        raise ValueError(f"unsupported RFMLink interface: {interface}")
    if interface == "ultra_score_only":
        return b0[:, 18:19], (b0_names[18],), ()
    if interface == "entity_tokens_only":
        if token_columns is None:
            raise ValueError(
                "the entity-token interface requires per-row entity identities"
            )
        return token_columns, ("head_entity_token", "tail_entity_token"), ()
    if interface == "b0_dyadic_only":
        if tuple(b0_names[index] for index in _DYADIC_B0_COLUMNS) == _DYADIC_B0_NAMES:
            values, names = _named_b0_subset(
                b0, b0_names, _DYADIC_B0_COLUMNS, _DYADIC_B0_NAMES
            )
            return values, names, ()
        values, names = _named_b0_subset(
            b0,
            b0_names,
            _LEGACY_DYADIC_B0_COLUMNS,
            _LEGACY_DYADIC_B0_NAMES,
        )
        if same_entity_column is None or same_entity_column.shape != (len(b0), 1):
            raise ValueError("legacy B0 dyadic evidence requires a same-entity column")
        return (
            np.concatenate([values, same_entity_column], axis=1),
            (*names, "same_entity"),
            (),
        )
    if interface == "b0_paths_only":
        if tuple(b0_names[index] for index in _PATH_B0_COLUMNS) == _PATH_B0_NAMES:
            values, names = _named_b0_subset(
                b0, b0_names, _PATH_B0_COLUMNS, _PATH_B0_NAMES
            )
            return values, names, ()
        values, names = _named_b0_subset(
            b0, b0_names, _LEGACY_PATH_B0_COLUMNS, _LEGACY_PATH_B0_NAMES
        )
        return values, names, ()
    if interface == "b2v2_only":
        return b2v2, b2v2_names, ()
    if interface == "b0_local_only":
        return b0[:, :18], b0_names[:18], (0, 16)
    if interface == "b0_local_no_relation_id":
        return b0[:, 1:18], b0_names[1:18], (15,)
    if interface == "b0_local_ultra":
        return b0, b0_names, (0, 16)
    if interface == "b0_local_ultra_no_relation_id":
        return b0[:, 1:19], b0_names[1:19], (15,)
    if interface == "b2v2_structure_only":
        return (
            np.concatenate([b0[:, :18], b2v2], axis=1),
            (*b0_names[:18], *b2v2_names),
            (0, 16),
        )
    if interface == "b2v2_full":
        return (
            np.concatenate([b0, b2v2], axis=1),
            (*b0_names, *b2v2_names),
            (0, 16),
        )
    if interface in {
        "graphlet_only",
        "b0_local_graphlet",
        "b0_local_graphlet_ultra",
    }:
        if graphlet is None:
            raise ValueError("the graphlet interface requires graphlet columns")
        if graphlet.ndim != 2 or len(graphlet) != len(b0):
            raise ValueError("graphlet rows are not aligned with B0")
        if graphlet.shape[1] != len(graphlet_names):
            raise ValueError("graphlet schema does not match the graphlet matrix")
        if interface == "graphlet_only":
            return graphlet, graphlet_names, ()
        if interface == "b0_local_graphlet":
            return (
                np.concatenate([b0[:, :18], graphlet], axis=1),
                (*b0_names[:18], *graphlet_names),
                (0, 16),
            )
        return (
            np.concatenate([b0, graphlet], axis=1),
            (*b0_names, *graphlet_names),
            (0, 16),
        )
    if interface == "b2v2_permuted_full":
        if permutation_group_size is None or permutation_key is None:
            raise ValueError("the permuted B2v2 interface requires a frozen scope")
        permuted = _permuted_rows(b2v2, permutation_group_size, permutation_key)
        return (
            np.concatenate([b0, permuted], axis=1),
            (*b0_names, *(f"permuted_{name}" for name in b2v2_names)),
            (0, 16),
        )
    raise ValueError(f"unsupported RFMLink interface: {interface}")


@lru_cache(maxsize=2)
def _candidate_interface(
    asset_root: str,
    dataset: str,
    manifest_sha256: str,
    archive_sha256: str,
    interface: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    arrays, manifest = _load_archive(
        asset_root, dataset, manifest_sha256, archive_sha256
    )
    tokens = (
        _candidate_token_columns(arrays, dataset)
        if interface == "entity_tokens_only"
        else None
    )
    same_entity = (
        _candidate_same_entity_column(arrays)
        if interface == "b0_dyadic_only"
        else None
    )
    graphlet = (
        np.asarray(arrays["candidate_graphlet"], dtype=np.float32)
        if "graphlet" in interface
        else None
    )
    return _interface(
        np.asarray(arrays["candidate_b0"], dtype=np.float32),
        np.asarray(arrays["candidate_b2v2"], dtype=np.float32),
        interface,
        tuple(manifest["b0_feature_names"]),
        tuple(manifest["b2v2_feature_names"]),
        graphlet=graphlet,
        graphlet_names=tuple(manifest.get("graphlet_feature_names", ())),
        permutation_group_size=int(arrays["num_entities"]),
        permutation_key=f"{dataset}/test",
        token_columns=tokens,
        same_entity_column=same_entity,
    )


def load_frozen_kg_task(repo_root: Path, cell: dict[str, Any]) -> FrozenKGTask:
    dataset = str(cell["dataset"])
    seed = int(cell["seed"])
    asset_id = str(cell["asset_id"])
    registered_asset = frozen_kg_asset(asset_id)
    if str(cell["asset_manifest_sha256"]) != registered_asset["manifest_sha256"]:
        raise ValueError("cell uses an unregistered frozen KG manifest")
    asset_root = _asset_root(repo_root, asset_id)
    arrays, manifest = _load_archive(
        str(asset_root),
        dataset,
        str(cell["asset_manifest_sha256"]),
        str(cell["archive_sha256"]),
    )
    b0_names = tuple(manifest["b0_feature_names"])
    b2v2_names = tuple(manifest["b2v2_feature_names"])
    context_mode = str(cell.get("context_mode", "original_uniform"))
    context_prefix = (
        f"context_{seed}"
        if context_mode == "original_uniform"
        else f"context_{context_mode}_{seed}"
    )
    context_b0 = np.asarray(arrays[f"{context_prefix}_X"], dtype=np.float32)
    context_b2v2 = np.asarray(arrays[f"{context_prefix}_b2v2"], dtype=np.float32)
    context_graphlet = (
        np.asarray(arrays[f"{context_prefix}_graphlet"], dtype=np.float32)
        if "graphlet" in str(cell["interface"])
        else None
    )
    labels = np.asarray(arrays[f"{context_prefix}_y"], dtype=np.int64)
    selected = _nested_context_indices(labels, seed, int(cell["context_per_class"]))
    context_tokens = (
        _context_token_columns(arrays, dataset, seed, context_prefix)[selected]
        if str(cell["interface"]) == "entity_tokens_only"
        else None
    )
    context_same_entity = (
        _context_same_entity_column(arrays, seed, context_prefix)[selected]
        if str(cell["interface"]) == "b0_dyadic_only"
        else None
    )
    context_values, feature_names, categorical = _interface(
        context_b0[selected],
        context_b2v2[selected],
        str(cell["interface"]),
        b0_names,
        b2v2_names,
        graphlet=(
            context_graphlet[selected]
            if context_graphlet is not None
            else None
        ),
        graphlet_names=tuple(manifest.get("graphlet_feature_names", ())),
        permutation_group_size=len(selected),
        permutation_key=(
            f"{dataset}/context/{seed}/{int(cell['context_per_class'])}"
        ),
        token_columns=context_tokens,
        same_entity_column=context_same_entity,
    )
    candidate_b0 = np.asarray(arrays["candidate_b0"], dtype=np.float32)
    candidate_values, candidate_names, candidate_categorical = _candidate_interface(
        str(asset_root),
        dataset,
        str(cell["asset_manifest_sha256"]),
        str(cell["archive_sha256"]),
        str(cell["interface"]),
    )
    if feature_names != candidate_names or categorical != candidate_categorical:
        raise ValueError("RFMLink context/candidate schemas differ")
    filtered = np.asarray(arrays["filtered"], dtype=np.bool_)
    num_entities = int(arrays["num_entities"])
    if len(candidate_values) != len(filtered) * num_entities:
        raise ValueError("RFMLink candidate blocks are not query aligned")
    if not np.isfinite(context_values).all() or not np.isfinite(candidate_values).all():
        raise ValueError("RFMLink interface contains non-finite values")
    metadata = {
        "dataset": dataset,
        "transfer_regime": cell["transfer_regime"],
        "asset_id": cell["asset_id"],
        "asset_manifest_sha256": cell["asset_manifest_sha256"],
        "archive_sha256": cell["archive_sha256"],
        "query_selection": manifest["query_selection"],
        "query_triples": int(cell["query_triples"]),
        "directional_queries": int(len(filtered)),
        "num_entities": num_entities,
        "num_relations": int(arrays["num_relations"]),
        "context_selection": manifest["context_selection"],
        "context_mode": context_mode,
        "context_indices_sha256": object_sha256(selected.tolist()),
        "candidate_rows": int(len(candidate_values)),
        "candidate_set": "all entities with standard filtered ranking masks",
        "global_feature_graph": "complete observed target training graph",
        "scientific_status": manifest["scientific_status"],
        "source_cluster": manifest["datasets"][dataset].get("source_cluster"),
        "claim_boundary": manifest.get("claim_boundary"),
    }
    return FrozenKGTask(
        context_values=np.asarray(context_values, dtype=np.float32),
        context_labels=labels[selected],
        candidate_values=np.asarray(candidate_values, dtype=np.float32),
        native_ultra_scores=candidate_b0[:, 18].astype(np.float64, copy=False),
        filtered=filtered,
        target_entities=np.asarray(arrays["target_entities"], dtype=np.int64),
        side_codes=np.asarray(arrays["side_codes"], dtype=np.int8),
        feature_names=feature_names,
        categorical_feature_indices=categorical,
        metadata=metadata,
    )
