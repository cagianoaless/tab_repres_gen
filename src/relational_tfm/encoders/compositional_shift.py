from __future__ import annotations

import numpy as np
import pandas as pd

from relational_tfm.datasets.compositional_shift import (
    COMPOSITION_CLASS_COUNT,
    COMPOSITION_TOKENS,
    decode_element,
)
from relational_tfm.encoders.synthetic import EncodedRows

COMPOSITION_MAX_PATH_LENGTH = 10
COMPOSITION_SHIFT_ENCODERS = frozenset(
    {
        "path_length_only",
        "relation_multiset",
        "coarse_relation_multiset",
        "relation_transition_counts",
        "ordered_typed_path",
        "ordered_plus_multiset",
        "ordered_plus_coarse_multiset",
        "aligned_multiset_with_ordered_block_permutation",
        "aligned_coarse_multiset_with_ordered_block_permutation",
    }
)


def _length_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    values = rows["chain_length"].to_numpy(dtype=np.float32)[:, None]
    values /= float(COMPOSITION_MAX_PATH_LENGTH)
    return values, ("chain_length_normalized",)


def _multiset_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    counts = np.zeros((len(rows), COMPOSITION_CLASS_COUNT), dtype=np.float32)
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for token in tokens:
            counts[row_index, int(token)] += 1.0
    counts /= float(COMPOSITION_MAX_PATH_LENGTH)
    names = length_names + tuple(
        f"relation_count__{token:02d}" for token in COMPOSITION_TOKENS
    )
    return np.concatenate([length, counts], axis=1), names


def _coarse_multiset_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Count eight quotient relation types shared across all path positions."""
    length, length_names = _length_values(rows)
    counts = np.zeros((len(rows), COMPOSITION_CLASS_COUNT // 2), dtype=np.float32)
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for token in tokens:
            rotation, reflected = decode_element(int(token))
            category = reflected * 4 + rotation % 4
            counts[row_index, category] += 1.0
    counts /= float(COMPOSITION_MAX_PATH_LENGTH)
    names = length_names + tuple(
        f"coarse_relation_count__{category:02d}"
        for category in range(COMPOSITION_CLASS_COUNT // 2)
    )
    return np.concatenate([length, counts], axis=1), names


def _ordered_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    ordered = np.zeros(
        (
            len(rows),
            COMPOSITION_MAX_PATH_LENGTH * COMPOSITION_CLASS_COUNT,
        ),
        dtype=np.float32,
    )
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for position, token in enumerate(tokens):
            ordered[
                row_index,
                position * COMPOSITION_CLASS_COUNT + int(token),
            ] = 1.0
    names = length_names + tuple(
        f"position_{position + 1}__relation_{token:02d}"
        for position in range(COMPOSITION_MAX_PATH_LENGTH)
        for token in COMPOSITION_TOKENS
    )
    return np.concatenate([length, ordered], axis=1), names


def _transition_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    transitions = np.zeros(
        (len(rows), COMPOSITION_CLASS_COUNT**2),
        dtype=np.float32,
    )
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for left, right in zip(tokens, tokens[1:], strict=False):
            transitions[
                row_index,
                int(left) * COMPOSITION_CLASS_COUNT + int(right),
            ] += 1.0
    transitions /= float(COMPOSITION_MAX_PATH_LENGTH - 1)
    names = length_names + tuple(
        f"transition__{left:02d}__to__{right:02d}"
        for left in COMPOSITION_TOKENS
        for right in COMPOSITION_TOKENS
    )
    return np.concatenate([length, transitions], axis=1), names


def _ordered_plus_multiset_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    ordered, ordered_names = _ordered_values(rows)
    multiset, multiset_names = _multiset_values(rows)
    return (
        np.concatenate([ordered, multiset[:, 1:]], axis=1),
        ordered_names + multiset_names[1:],
    )


def _ordered_plus_coarse_multiset_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    ordered, ordered_names = _ordered_values(rows)
    multiset, multiset_names = _coarse_multiset_values(rows)
    return (
        np.concatenate([ordered, multiset[:, 1:]], axis=1),
        ordered_names + multiset_names[1:],
    )


def _permute_ordered_block_within_length(
    rows: pd.DataFrame,
    ordered: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = ordered.copy()
    chain_lengths = rows["chain_length"].to_numpy(dtype=np.int64)
    for chain_length in np.unique(chain_lengths):
        indices = np.flatnonzero(chain_lengths == chain_length)
        permutation = rng.permutation(indices)
        result[indices, 1:] = ordered[permutation, 1:]
    return result


def encode_compositional_shift_rows(
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
) -> EncodedRows:
    if encoder not in COMPOSITION_SHIFT_ENCODERS:
        raise ValueError(f"unsupported D8 compositional encoder: {encoder}")
    required = {"chain_length", "relation_tokens", "story_edges", "query_edge"}
    if not required <= set(rows.columns):
        raise ValueError(
            f"D8 compositional rows omit columns: {sorted(required - set(rows.columns))}"
        )

    ordered_alignment_preserved = True
    if encoder == "path_length_only":
        values, names = _length_values(rows)
        base = "path_length"
    elif encoder == "relation_multiset":
        values, names = _multiset_values(rows)
        base = "relation_multiset"
    elif encoder == "coarse_relation_multiset":
        values, names = _coarse_multiset_values(rows)
        base = "coarse_relation_multiset"
    elif encoder == "relation_transition_counts":
        values, names = _transition_values(rows)
        base = "relation_transition_counts"
    elif encoder == "ordered_typed_path":
        values, names = _ordered_values(rows)
        base = "ordered_typed_path"
    elif encoder in {
        "ordered_plus_multiset",
        "aligned_multiset_with_ordered_block_permutation",
    }:
        ordered, ordered_names = _ordered_values(rows)
        multiset, multiset_names = _multiset_values(rows)
        if encoder == "aligned_multiset_with_ordered_block_permutation":
            ordered = _permute_ordered_block_within_length(
                rows,
                ordered,
                np.random.default_rng(seed),
            )
            ordered_alignment_preserved = False
        values = np.concatenate([ordered, multiset[:, 1:]], axis=1)
        names = ordered_names + multiset_names[1:]
        base = "ordered_plus_multiset"
    else:
        ordered, ordered_names = _ordered_values(rows)
        multiset, multiset_names = _coarse_multiset_values(rows)
        if encoder == "aligned_coarse_multiset_with_ordered_block_permutation":
            ordered = _permute_ordered_block_within_length(
                rows,
                ordered,
                np.random.default_rng(seed),
            )
            ordered_alignment_preserved = False
        values = np.concatenate([ordered, multiset[:, 1:]], axis=1)
        names = ordered_names + multiset_names[1:]
        base = "ordered_plus_coarse_multiset"

    if not np.isfinite(values).all():
        raise ValueError("D8 compositional encoder returned non-finite values")
    return EncodedRows(
        values=np.asarray(values, dtype=np.float32),
        feature_names=names,
        provenance={
            "encoder": encoder,
            "base_representation": base,
            "seed": int(seed),
            "raw_entity_ids_exposed": False,
            "target_relation_exposed": False,
            "position_specific_coordinates": base
            in {
                "ordered_typed_path",
                "ordered_plus_multiset",
                "ordered_plus_coarse_multiset",
            },
            "position_shared_coordinates": base
            in {
                "relation_multiset",
                "coarse_relation_multiset",
                "relation_transition_counts",
                "ordered_plus_multiset",
                "ordered_plus_coarse_multiset",
            },
            "candidate_alignment_preserved": ordered_alignment_preserved,
            "shared_multiset_block_preserved": True,
            "ordered_block_alignment_preserved": ordered_alignment_preserved,
            "permutation_stratified_by_chain_length": (
                encoder
                in {
                    "aligned_multiset_with_ordered_block_permutation",
                    "aligned_coarse_multiset_with_ordered_block_permutation",
                }
            ),
        },
    )
