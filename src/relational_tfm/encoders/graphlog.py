from __future__ import annotations

import numpy as np
import pandas as pd

from relational_tfm.encoders.synthetic import EncodedRows

GRAPHLOG_RELATIONS = tuple(
    f"R_{index}_{polarity}"
    for polarity in ("+", "-")
    for index in range(10)
)
GRAPHLOG_RELATION_INDEX = {
    relation: index for index, relation in enumerate(GRAPHLOG_RELATIONS)
}
GRAPHLOG_MAX_PATH_LENGTH = 5
GRAPHLOG_ENCODERS = frozenset(
    {
        "path_length_only",
        "relation_multiset",
        "relation_transition_counts",
        "ordered_typed_path",
        "ordered_plus_multiset",
        "aligned_multiset_with_ordered_block_permutation",
    }
)


def _length_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    values = rows["chain_length"].to_numpy(dtype=np.float32)[:, None]
    values /= float(GRAPHLOG_MAX_PATH_LENGTH)
    return values, ("chain_length_normalized",)


def _multiset_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    counts = np.zeros((len(rows), len(GRAPHLOG_RELATIONS)), dtype=np.float32)
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for token in tokens:
            counts[row_index, GRAPHLOG_RELATION_INDEX[str(token)]] += 1.0
    counts /= float(GRAPHLOG_MAX_PATH_LENGTH)
    names = length_names + tuple(
        f"relation_count__{relation}" for relation in GRAPHLOG_RELATIONS
    )
    return np.concatenate([length, counts], axis=1), names


def _transition_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    relation_count = len(GRAPHLOG_RELATIONS)
    transitions = np.zeros(
        (len(rows), relation_count * relation_count),
        dtype=np.float32,
    )
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for left, right in zip(tokens, tokens[1:], strict=False):
            column = (
                GRAPHLOG_RELATION_INDEX[str(left)] * relation_count
                + GRAPHLOG_RELATION_INDEX[str(right)]
            )
            transitions[row_index, column] += 1.0
    transitions /= float(GRAPHLOG_MAX_PATH_LENGTH - 1)
    names = length_names + tuple(
        f"transition__{left}__to__{right}"
        for left in GRAPHLOG_RELATIONS
        for right in GRAPHLOG_RELATIONS
    )
    return np.concatenate([length, transitions], axis=1), names


def _ordered_values(rows: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    length, length_names = _length_values(rows)
    relation_count = len(GRAPHLOG_RELATIONS)
    ordered = np.zeros(
        (len(rows), GRAPHLOG_MAX_PATH_LENGTH * relation_count),
        dtype=np.float32,
    )
    for row_index, tokens in enumerate(rows["relation_tokens"]):
        for position, token in enumerate(tokens):
            ordered[
                row_index,
                position * relation_count + GRAPHLOG_RELATION_INDEX[str(token)],
            ] = 1.0
    names = length_names + tuple(
        f"position_{position + 1}__{relation}"
        for position in range(GRAPHLOG_MAX_PATH_LENGTH)
        for relation in GRAPHLOG_RELATIONS
    )
    return np.concatenate([length, ordered], axis=1), names


def _ordered_plus_multiset_values(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    ordered, ordered_names = _ordered_values(rows)
    multiset, multiset_names = _multiset_values(rows)
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
        result[indices, 1:] = ordered[rng.permutation(indices), 1:]
    return result


def encode_graphlog_rows(
    rows: pd.DataFrame,
    encoder: str,
    seed: int,
) -> EncodedRows:
    if encoder not in GRAPHLOG_ENCODERS:
        raise ValueError(f"unsupported GraphLog encoder: {encoder}")
    required = {"chain_length", "relation_tokens"}
    if not required <= set(rows.columns):
        raise ValueError(
            f"GraphLog rows omit columns: {sorted(required - set(rows.columns))}"
        )

    ordered_alignment_preserved = True
    if encoder == "path_length_only":
        values, names = _length_values(rows)
        base = "path_length"
    elif encoder == "relation_multiset":
        values, names = _multiset_values(rows)
        base = "relation_multiset"
    elif encoder == "relation_transition_counts":
        values, names = _transition_values(rows)
        base = "relation_transition_counts"
    elif encoder == "ordered_typed_path":
        values, names = _ordered_values(rows)
        base = "ordered_typed_path"
    else:
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

    if not np.isfinite(values).all():
        raise ValueError("GraphLog encoder returned non-finite values")
    return EncodedRows(
        values=np.asarray(values, dtype=np.float32),
        feature_names=names,
        provenance={
            "encoder": encoder,
            "base_representation": base,
            "seed": int(seed),
            "source_field": "descriptor",
            "raw_entity_ids_exposed": False,
            "target_relation_exposed": False,
            "position_specific_coordinates": base
            in {"ordered_typed_path", "ordered_plus_multiset"},
            "position_shared_coordinates": base
            in {
                "relation_multiset",
                "relation_transition_counts",
                "ordered_plus_multiset",
            },
            "candidate_alignment_preserved": ordered_alignment_preserved,
            "shared_multiset_block_preserved": True,
            "ordered_block_alignment_preserved": ordered_alignment_preserved,
            "permutation_stratified_by_chain_length": (
                encoder == "aligned_multiset_with_ordered_block_permutation"
            ),
        },
    )
