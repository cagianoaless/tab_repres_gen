from __future__ import annotations

import numpy as np
import pytest

from relational_tfm.evaluation.classification import collision_conditioned_ceiling


def test_ceiling_decomposes_into_singleton_and_repeated_row_mass() -> None:
    summary = collision_conditioned_ceiling(
        np.asarray([[0], [1], [1], [2], [2], [3]]),
        np.asarray([0, 0, 1, 1, 1, 0]),
    )
    assert summary["singleton_rows"] == 2
    assert summary["nonsingleton_rows"] == 4
    assert summary["singleton_fraction"] == 1 / 3
    assert summary["nonsingleton_fraction"] == 2 / 3
    assert summary["nonsingleton_accuracy_ceiling"] == 3 / 4
    assert summary["ceiling"] == pytest.approx(
        summary["singleton_fraction"]
        + summary["nonsingleton_fraction"]
        * summary["nonsingleton_accuracy_ceiling"]
    )


def test_repeated_row_ceiling_is_undefined_for_injective_rows() -> None:
    summary = collision_conditioned_ceiling(
        np.arange(4, dtype=np.int64).reshape(-1, 1),
        np.asarray([0, 1, 0, 1]),
    )
    assert summary["singleton_fraction"] == 1.0
    assert summary["nonsingleton_fraction"] == 0.0
    assert summary["nonsingleton_accuracy_ceiling"] is None
