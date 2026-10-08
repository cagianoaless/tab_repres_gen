from __future__ import annotations

import numpy as np
import pytest

from relational_tfm.evaluation.query_relative import within_query_transform


@pytest.mark.parametrize("mode", ["within_query_zscore", "within_query_percentile"])
def test_score_only_transform_is_label_free_query_local_and_order_preserving(mode: str) -> None:
    scores = np.array([[1], [3], [3], [9], [-100], [0], [20], [20]], dtype=np.float32)
    transformed = within_query_transform(scores, mode=mode, group_size=4)
    changed = scores.copy()
    changed[4:] = 999
    modified = within_query_transform(changed, mode=mode, group_size=4)
    np.testing.assert_array_equal(transformed[:4], modified[:4])
    for start in (0, 4):
        original = scores[start : start + 4, 0]
        result = transformed[start : start + 4, 0]
        np.testing.assert_array_equal(
            np.sign(original[:, None] - original[None, :]),
            np.sign(result[:, None] - result[None, :]),
        )
    # No labels enter the API; this guarantees input transforms, not preservation
    # of an arbitrary fitted decoder's rankings after training on transformed data.
    assert transformed.shape == scores.shape


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
