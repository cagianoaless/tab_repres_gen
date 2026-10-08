from __future__ import annotations

import numpy as np
import pytest

from relational_tfm.datasets.rfmlink import (
    _candidate_token_columns,
    _context_token_columns,
    _entity_tokens,
    _interface,
    encode_feature_schema,
)

_B0_NAMES = (
    "relation_id",
    "head_total_degree_log",
    "tail_total_degree_log",
    "head_relation_degree_log",
    "tail_relation_degree_log",
    "relation_frequency_log",
    "relation_functionality",
    "relation_inverse_functionality",
    "pair_count_log",
    "reverse_pair_count_log",
    "same_entity",
    "common_neighbors_log",
    "two_hop_count_log",
    "start_relation_two_hop_count_log",
    "end_relation_two_hop_count_log",
    "reverse_two_hop_count_log",
    "same_cluster",
    "cluster_relation_density",
    "ultra_score",
)
_B2V2_NAMES = ("pagerank_head", "pagerank_tail", "heat_similarity", "diffusion_distance")


def test_public_schema_name_preserves_legacy_alias() -> None:
    assert _interface is encode_feature_schema


def test_complete_kg_ladder_is_exact_concatenation() -> None:
    b0 = np.arange(5 * 19, dtype=np.float32).reshape(5, 19)
    global_values = np.arange(5 * 13, dtype=np.float32).reshape(5, 13)
    global_names = tuple(f"global_{index}" for index in range(13))
    local, local_names, categorical = encode_feature_schema(
        b0, global_values, "b0_local_only", _B0_NAMES, global_names
    )
    score, score_names, score_categorical = encode_feature_schema(
        b0, global_values, "ultra_score_only", _B0_NAMES, global_names
    )
    glob, glob_names, glob_categorical = encode_feature_schema(
        b0, global_values, "b2v2_only", _B0_NAMES, global_names
    )
    full, full_names, full_categorical = encode_feature_schema(
        b0, global_values, "b2v2_full", _B0_NAMES, global_names
    )
    assert (local.shape[1], score.shape[1], glob.shape[1], full.shape[1]) == (18, 1, 13, 32)
    np.testing.assert_array_equal(full, np.concatenate([local, score, glob], axis=1))
    assert full_names == local_names + score_names + glob_names
    assert categorical == full_categorical == (0, 16)
    assert score_categorical == glob_categorical == ()


def test_relation_identity_ablation_rejects_mislabeled_column_zero() -> None:
    b0, global_values = _rows(4)
    for variant in ("b0_local_no_relation_id", "b0_local_ultra_no_relation_id"):
        with pytest.raises(ValueError, match="relation"):
            encode_feature_schema(
                b0, global_values, variant, ("wrong", *_B0_NAMES[1:]), _B2V2_NAMES
            )


@pytest.mark.parametrize(
    ("variant", "stop"),
    [("b0_local_no_relation_id", 18), ("b0_local_ultra_no_relation_id", 19)],
)
def test_relation_identity_ablation_accepts_alternative_score_name(
    variant: str, stop: int,
) -> None:
    b0 = np.arange(4 * 19, dtype=np.float32).reshape(4, 19)
    global_values = np.zeros((4, len(_B2V2_NAMES)), dtype=np.float32)
    renamed = (*_B0_NAMES[:18], "frozen_score")
    reference, reference_names, reference_categorical = encode_feature_schema(
        b0, global_values, variant, _B0_NAMES, _B2V2_NAMES
    )
    actual, names, categorical = encode_feature_schema(
        b0, global_values, variant, renamed, _B2V2_NAMES
    )
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(actual, b0[:, 1:stop])
    assert actual.shape == (4, stop - 1)
    assert names == renamed[1:stop]
    assert categorical == reference_categorical == (15,)
    if stop == 19:
        np.testing.assert_array_equal(actual[:, -1], b0[:, 18])
        assert names[:-1] == reference_names[:-1]
    else:
        assert names == reference_names


def _rows(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return (
        rng.standard_normal((n, 19)).astype(np.float32),
        rng.standard_normal((n, 4)).astype(np.float32),
    )


def test_ladder_interfaces_select_the_named_columns() -> None:
    b0, b2v2 = _rows(12)
    values, names, categorical = _interface(b0, b2v2, "b0_dyadic_only", _B0_NAMES, _B2V2_NAMES)
    assert names == ("pair_count_log", "reverse_pair_count_log", "same_entity")
    assert np.array_equal(values, b0[:, [8, 9, 10]])
    assert categorical == ()

    values, names, _ = _interface(b0, b2v2, "b0_paths_only", _B0_NAMES, _B2V2_NAMES)
    assert names == (
        "common_neighbors_log",
        "two_hop_count_log",
        "start_relation_two_hop_count_log",
        "end_relation_two_hop_count_log",
        "reverse_two_hop_count_log",
    )
    assert np.array_equal(values, b0[:, [11, 12, 13, 14, 15]])

    values, names, _ = _interface(b0, b2v2, "b2v2_only", _B0_NAMES, _B2V2_NAMES)
    assert names == _B2V2_NAMES
    assert np.array_equal(values, b2v2)


def test_graphlet_interfaces_preserve_the_declared_schema() -> None:
    b0, b2v2 = _rows(6)
    graphlet = np.arange(6 * 18, dtype=np.float32).reshape(6, 18)
    graphlet_names = tuple(f"triad_{index}" for index in range(18))

    values, names, categorical = _interface(
        b0,
        b2v2,
        "graphlet_only",
        _B0_NAMES,
        _B2V2_NAMES,
        graphlet=graphlet,
        graphlet_names=graphlet_names,
    )
    assert np.array_equal(values, graphlet)
    assert names == graphlet_names
    assert categorical == ()

    values, names, categorical = _interface(
        b0,
        b2v2,
        "b0_local_graphlet_ultra",
        _B0_NAMES,
        _B2V2_NAMES,
        graphlet=graphlet,
        graphlet_names=graphlet_names,
    )
    assert np.array_equal(values, np.concatenate([b0, graphlet], axis=1))
    assert names == (*_B0_NAMES, *graphlet_names)
    assert categorical == (0, 16)


def test_b0_subsets_refuse_a_reordered_layout() -> None:
    b0, b2v2 = _rows(4)
    shuffled = tuple(reversed(_B0_NAMES))
    with pytest.raises(ValueError, match="b0 column layout changed"):
        _interface(b0, b2v2, "b0_dyadic_only", shuffled, _B2V2_NAMES)


def test_legacy_rfmlink_layout_uses_derived_same_entity() -> None:
    b0, b2v2 = _rows(4)
    legacy_names = (
        "relation_id",
        "head_total_out",
        "head_total_in",
        "tail_total_out",
        "tail_total_in",
        "head_relation_out_degree",
        "tail_relation_in_degree",
        "relation_frequency",
        "relation_functionality",
        "relation_inverse_functionality",
        "pair_other_relation_count",
        "reverse_pair_relation_count",
        "two_hop_count",
        "two_hop_start_relation_count",
        "two_hop_end_relation_count",
        "reverse_two_hop_count",
        "same_entity_cluster",
        "cluster_relation_density",
        "ultra_score",
    )
    same_entity = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    values, names, _ = _interface(
        b0,
        b2v2,
        "b0_dyadic_only",
        legacy_names,
        _B2V2_NAMES,
        same_entity_column=same_entity,
    )
    assert names == (
        "pair_other_relation_count",
        "reverse_pair_relation_count",
        "same_entity",
    )
    assert np.array_equal(values[:, :2], b0[:, [10, 11]])
    assert np.array_equal(values[:, 2:], same_entity)

    values, names, _ = _interface(
        b0, b2v2, "b0_paths_only", legacy_names, _B2V2_NAMES
    )
    assert names == (
        "two_hop_count",
        "two_hop_start_relation_count",
        "two_hop_end_relation_count",
        "reverse_two_hop_count",
    )
    assert np.array_equal(values, b0[:, [12, 13, 14, 15]])


def test_entity_tokens_are_dataset_scoped_and_deterministic() -> None:
    first = _entity_tokens("codex_small", 100)
    again = _entity_tokens("codex_small", 100)
    other = _entity_tokens("nell_ind_v1", 100)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_relation_id_ablations_remove_only_the_nominal_identifier() -> None:
    b0 = np.arange(6 * len(_B0_NAMES), dtype=np.float32).reshape(6, -1)
    b2v2 = np.zeros((6, len(_B2V2_NAMES)), dtype=np.float32)

    local, local_names, local_categorical = _interface(
        b0,
        b2v2,
        "b0_local_no_relation_id",
        _B0_NAMES,
        _B2V2_NAMES,
    )
    local_score, local_score_names, local_score_categorical = _interface(
        b0,
        b2v2,
        "b0_local_ultra_no_relation_id",
        _B0_NAMES,
        _B2V2_NAMES,
    )

    np.testing.assert_array_equal(local, b0[:, 1:18])
    np.testing.assert_array_equal(local_score, b0[:, 1:19])
    assert local_names == _B0_NAMES[1:18]
    assert local_score_names == _B0_NAMES[1:19]
    assert local_categorical == (15,)
    assert local_score_categorical == (15,)


def test_candidate_token_columns_follow_direction_semantics() -> None:
    num_entities = 5
    arrays = {
        "num_entities": np.int64(num_entities),
        "anchor_entities": np.array([3, 1], dtype=np.int64),
        "side_codes": np.array([0, 1], dtype=np.int8),
    }
    tokens = _entity_tokens("probe", num_entities)
    columns = _candidate_token_columns(arrays, "probe")
    assert columns.shape == (2 * num_entities, 2)
    # direction 0 ranks tails: head token fixed to entity 3, tail token per candidate
    assert np.allclose(columns[:num_entities, 0], tokens[3])
    assert np.allclose(columns[:num_entities, 1], tokens)
    # direction 1 ranks heads: head token per candidate, tail token fixed to entity 1
    assert np.allclose(columns[num_entities:, 0], tokens)
    assert np.allclose(columns[num_entities:, 1], tokens[1])


def test_context_token_columns_read_the_stored_triples() -> None:
    arrays = {
        "num_entities": np.int64(6),
        "context_101_triples": np.array([[0, 5, 2], [4, 4, 1]], dtype=np.int64),
    }
    tokens = _entity_tokens("probe", 6)
    columns = _context_token_columns(arrays, "probe", 101)
    assert columns.shape == (2, 2)
    assert np.allclose(columns[0], [tokens[0], tokens[2]])
    assert np.allclose(columns[1], [tokens[4], tokens[1]])


def test_token_interfaces_require_schema_two_archives() -> None:
    with pytest.raises(ValueError, match="archive schema 2"):
        _candidate_token_columns({"num_entities": np.int64(3)}, "probe")
    with pytest.raises(ValueError, match="archive schema 2"):
        _context_token_columns({"num_entities": np.int64(3)}, "probe", 101)
    b0, b2v2 = _rows(4)
    with pytest.raises(ValueError, match="entity identities"):
        _interface(b0, b2v2, "entity_tokens_only", _B0_NAMES, _B2V2_NAMES)
