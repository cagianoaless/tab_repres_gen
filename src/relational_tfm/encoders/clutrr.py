from __future__ import annotations

import numpy as np
import pandas as pd

from relational_tfm.datasets.clutrr import CLUTRR_EDGE_RELATIONS
from relational_tfm.encoders.synthetic import EncodedRows

CLUTRR_ENCODERS = frozenset(
    {
        "path_length_only",
        "relation_multiset",
        "relation_transition_counts",
        "ordered_typed_path",
        "ordered_plus_multiset",
        "aligned_permutation",
        "dimension_matched_noise",
        "within_length_joint_permutation",
        "within_length_independent_permutation",
        "within_length_transition_joint_permutation",
        "within_length_ordered_plus_multiset_joint_permutation",
        "aligned_multiset_with_ordered_block_permutation",
        "aligned_multiset_with_supported_prefix",
    }
)
CLUTRR_MAX_PATH_LENGTH = 10


def _path_length_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(rows["chain_length"], dtype=np.float32)[:, None]
    values /= float(CLUTRR_MAX_PATH_LENGTH)
    return values, ("chain_length_normalized",)


def _multiset_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _path_length_values(rows)
    counts = np.zeros((len(rows), len(CLUTRR_EDGE_RELATIONS)), dtype=np.float32)
    relation_index = {
        relation: index for index, relation in enumerate(CLUTRR_EDGE_RELATIONS)
    }
    for row_index, edge_types in enumerate(rows["edge_types"]):
        for relation in edge_types:
            counts[row_index, relation_index[str(relation)]] += 1.0
    counts /= float(CLUTRR_MAX_PATH_LENGTH)
    names = length_names + tuple(
        f"path_relation_count__{relation}" for relation in CLUTRR_EDGE_RELATIONS
    )
    return np.concatenate([length, counts], axis=1), names


def _ordered_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _path_length_values(rows)
    relation_index = {
        relation: index for index, relation in enumerate(CLUTRR_EDGE_RELATIONS)
    }
    ordered = np.zeros(
        (len(rows), CLUTRR_MAX_PATH_LENGTH * len(CLUTRR_EDGE_RELATIONS)),
        dtype=np.float32,
    )
    for row_index, edge_types in enumerate(rows["edge_types"]):
        for position, relation in enumerate(edge_types):
            ordered[
                row_index,
                position * len(CLUTRR_EDGE_RELATIONS) + relation_index[str(relation)],
            ] = 1.0
    names = length_names + tuple(
        f"path_position_{position + 1}__{relation}"
        for position in range(CLUTRR_MAX_PATH_LENGTH)
        for relation in CLUTRR_EDGE_RELATIONS
    )
    return np.concatenate([length, ordered], axis=1), names


def _transition_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Count directed adjacent relation pairs in coordinates shared across positions."""
    length, length_names = _path_length_values(rows)
    relation_index = {
        relation: index for index, relation in enumerate(CLUTRR_EDGE_RELATIONS)
    }
    relation_count = len(CLUTRR_EDGE_RELATIONS)
    transitions = np.zeros(
        (len(rows), relation_count * relation_count),
        dtype=np.float32,
    )
    for row_index, edge_types in enumerate(rows["edge_types"]):
        for left, right in zip(edge_types, edge_types[1:], strict=False):
            column = relation_index[str(left)] * relation_count + relation_index[str(right)]
            transitions[row_index, column] += 1.0
    transitions /= float(CLUTRR_MAX_PATH_LENGTH - 1)
    names = length_names + tuple(
        f"path_transition__{left}__to__{right}"
        for left in CLUTRR_EDGE_RELATIONS
        for right in CLUTRR_EDGE_RELATIONS
    )
    return np.concatenate([length, transitions], axis=1), names


def _ordered_plus_multiset_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Retain absolute order while exposing a position-shared recovery path."""
    ordered, ordered_names = _ordered_values(rows)
    multiset, multiset_names = _multiset_values(rows)
    return (
        np.concatenate([ordered, multiset[:, 1:]], axis=1),
        ordered_names + multiset_names[1:],
    )


def _aligned_multiset_with_ordered_block_permutation_values(
    rows: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Keep shared counts aligned while permuting only absolute-position rows."""
    ordered, ordered_names = _ordered_values(rows)
    ordered_permuted = _within_length_permutation(
        rows,
        ordered,
        rng,
        independent_columns=False,
    )
    multiset, multiset_names = _multiset_values(rows)
    return (
        np.concatenate([ordered_permuted, multiset[:, 1:]], axis=1),
        ordered_names + multiset_names[1:],
    )


def _aligned_multiset_with_supported_prefix_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Keep aligned positions seen in context and remove only unseen suffix positions."""
    values, names = _ordered_plus_multiset_values(rows)
    relation_count = len(CLUTRR_EDGE_RELATIONS)
    ordered_width = 1 + CLUTRR_MAX_PATH_LENGTH * relation_count
    first_unsupported = 1 + 3 * relation_count
    values[:, first_unsupported:ordered_width] = 0.0
    return values, names


def _within_length_permutation(
    rows: pd.DataFrame,
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    independent_columns: bool,
) -> np.ndarray:
    result = values.copy()
    relation_block = values[:, 1:]
    chain_lengths = np.asarray(rows["chain_length"], dtype=np.int64)
    for chain_length in np.unique(chain_lengths):
        indices = np.flatnonzero(chain_lengths == chain_length)
        if independent_columns:
            result[indices, 1:] = np.column_stack(
                [
                    relation_block[indices[rng.permutation(len(indices))], column]
                    for column in range(relation_block.shape[1])
                ]
            )
        else:
            result[indices, 1:] = relation_block[
                indices[rng.permutation(len(indices))]
            ]
    return result.astype(np.float32, copy=False)


def encode_clutrr_rows(
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
) -> EncodedRows:
    if encoder not in CLUTRR_ENCODERS:
        raise ValueError(f"unsupported CLUTRR encoder: {encoder}")
    required = {"chain_length", "edge_types", "story_edges", "query_edge"}
    if not required <= set(rows.columns):
        raise ValueError(f"CLUTRR rows omit columns: {sorted(required - set(rows.columns))}")

    if encoder == "path_length_only":
        values, feature_names = _path_length_values(rows)
        base = "path_length"
    elif encoder == "relation_multiset":
        values, feature_names = _multiset_values(rows)
        base = "relation_multiset"
    elif encoder in {
        "relation_transition_counts",
        "within_length_transition_joint_permutation",
    }:
        values, feature_names = _transition_values(rows)
        base = "relation_transition_counts"
        if encoder == "within_length_transition_joint_permutation":
            values = _within_length_permutation(
                rows,
                values,
                np.random.default_rng(seed),
                independent_columns=False,
            )
    elif encoder in {
        "ordered_plus_multiset",
        "within_length_ordered_plus_multiset_joint_permutation",
        "aligned_multiset_with_ordered_block_permutation",
        "aligned_multiset_with_supported_prefix",
    }:
        base = "ordered_plus_multiset"
        if encoder == "aligned_multiset_with_ordered_block_permutation":
            values, feature_names = (
                _aligned_multiset_with_ordered_block_permutation_values(
                    rows,
                    np.random.default_rng(seed),
                )
            )
        elif encoder == "aligned_multiset_with_supported_prefix":
            values, feature_names = _aligned_multiset_with_supported_prefix_values(
                rows
            )
        else:
            values, feature_names = _ordered_plus_multiset_values(rows)
        if encoder == "within_length_ordered_plus_multiset_joint_permutation":
            values = _within_length_permutation(
                rows,
                values,
                np.random.default_rng(seed),
                independent_columns=False,
            )
    else:
        values, feature_names = _ordered_values(rows)
        base = "ordered_typed_path"
        rng = np.random.default_rng(seed)
        if encoder == "aligned_permutation":
            values = values[rng.permutation(len(values))]
        elif encoder == "dimension_matched_noise":
            values = np.column_stack(
                [values[rng.permutation(len(values)), column] for column in range(values.shape[1])]
            ).astype(np.float32, copy=False)
        elif encoder == "within_length_joint_permutation":
            values = _within_length_permutation(
                rows,
                values,
                rng,
                independent_columns=False,
            )
        elif encoder == "within_length_independent_permutation":
            values = _within_length_permutation(
                rows,
                values,
                rng,
                independent_columns=True,
            )

    if not np.isfinite(values).all():
        raise ValueError("CLUTRR encoder returned non-finite values")
    return EncodedRows(
        values=np.asarray(values, dtype=np.float32),
        feature_names=feature_names,
        provenance={
            "encoder": encoder,
            "base_representation": base,
            "seed": int(seed),
            "raw_entity_ids_exposed": False,
            "story_text_exposed": False,
            "target_relation_exposed": False,
            "ordered_symbolic_path_exposed": encoder
            in {
                "ordered_typed_path",
                "ordered_plus_multiset",
                "aligned_permutation",
                "dimension_matched_noise",
                "within_length_joint_permutation",
                "within_length_independent_permutation",
                "within_length_ordered_plus_multiset_joint_permutation",
                "aligned_multiset_with_ordered_block_permutation",
                "aligned_multiset_with_supported_prefix",
            },
            "order_sensitive": encoder
            in {
                "ordered_typed_path",
                "relation_transition_counts",
                "ordered_plus_multiset",
                "aligned_permutation",
                "dimension_matched_noise",
                "within_length_joint_permutation",
                "within_length_independent_permutation",
                "within_length_transition_joint_permutation",
                "within_length_ordered_plus_multiset_joint_permutation",
                "aligned_multiset_with_ordered_block_permutation",
                "aligned_multiset_with_supported_prefix",
            },
            "position_specific_coordinates": base
            in {"ordered_typed_path", "ordered_plus_multiset"},
            "position_shared_coordinates": base
            in {"relation_multiset", "relation_transition_counts", "ordered_plus_multiset"},
            "candidate_alignment_preserved": encoder
            not in {
                "aligned_permutation",
                "dimension_matched_noise",
                "within_length_joint_permutation",
                "within_length_independent_permutation",
                "within_length_transition_joint_permutation",
                "within_length_ordered_plus_multiset_joint_permutation",
                "aligned_multiset_with_ordered_block_permutation",
            },
            "chain_length_alignment_preserved": encoder
            not in {"aligned_permutation", "dimension_matched_noise"},
            "permutation_stratified_by_chain_length": encoder
            in {
                "within_length_joint_permutation",
                "within_length_independent_permutation",
                "within_length_transition_joint_permutation",
                "within_length_ordered_plus_multiset_joint_permutation",
                "aligned_multiset_with_ordered_block_permutation",
            },
            "aligned_multiset_block_preserved": encoder
            in {
                "relation_multiset",
                "ordered_plus_multiset",
                "aligned_multiset_with_ordered_block_permutation",
                "aligned_multiset_with_supported_prefix",
            },
            "ordered_block_alignment_preserved": (
                encoder
                not in {
                    "aligned_permutation",
                    "dimension_matched_noise",
                    "within_length_joint_permutation",
                    "within_length_independent_permutation",
                    "within_length_ordered_plus_multiset_joint_permutation",
                    "aligned_multiset_with_ordered_block_permutation",
                }
                if base in {"ordered_typed_path", "ordered_plus_multiset"}
                else None
            ),
            "supported_absolute_positions": (
                3 if encoder == "aligned_multiset_with_supported_prefix" else None
            ),
        },
    )
