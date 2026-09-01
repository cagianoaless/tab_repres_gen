from __future__ import annotations

import numpy as np

from relational_tfm.evaluation.query_relative import within_query_transform


def test_zscore_is_groupwise_and_zeroes_constants() -> None:
    values = np.asarray([[1, 4], [3, 4], [10, 2], [14, 2]], dtype=np.float32)
    transformed = within_query_transform(
        values, mode="within_query_zscore", group_size=2
    )
    np.testing.assert_allclose(
        transformed,
        np.asarray([[-1, 0], [1, 0], [-1, 0], [1, 0]], dtype=np.float32),
    )


def test_percentiles_use_midranks_and_preserve_order() -> None:
    values = np.asarray([[2], [1], [2], [4]], dtype=np.float32)
    transformed = within_query_transform(
        values, mode="within_query_percentile", group_size=4
    )
    np.testing.assert_allclose(transformed[:, 0], [0.5, 0.0, 0.5, 1.0])
    assert np.array_equal(
        np.sign(values[:, 0, None] - values[None, :, 0]),
        np.sign(transformed[:, 0, None] - transformed[None, :, 0]),
    )


def test_group_ids_are_row_order_invariant() -> None:
    values = np.asarray([[1, 9], [10, 2], [3, 9], [14, 2]], dtype=np.float32)
    groups = np.asarray([7, 8, 7, 8])
    baseline = within_query_transform(
        values, mode="within_query_percentile", group_ids=groups
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = within_query_transform(
        values[permutation],
        mode="within_query_percentile",
        group_ids=groups[permutation],
    )
    np.testing.assert_array_equal(permuted, baseline[permutation])
