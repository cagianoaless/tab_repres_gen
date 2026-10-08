"""Recovered derived evidence, not manuscript-rounded fixtures or scientific reruns."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import verify_historical_gru as gru
from analysis import verify_historical_relation_context as kg

ROOT = Path(__file__).resolve().parents[1]


def test_gru_reconstruction_and_exact_endpoints() -> None:
    result = gru.verify()
    np.testing.assert_allclose(result["ood_means"], gru.OOD_MEANS, atol=1e-15, rtol=0)
    assert result["length_records"] == 135
    assert result["largest_budget_length_accuracy"] == {
        "4": 0.8982456140350877,
        "10": 0.5266106442577031,
    }


@pytest.mark.parametrize("duplicate", [False, True])
def test_gru_rejects_missing_or_duplicate_records(duplicate: bool) -> None:
    frame = pd.read_csv(ROOT / gru.RESULTS / "gru_by_length.csv")
    frame = pd.concat([frame, frame.iloc[:1]]) if duplicate else frame.iloc[:-1]
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        gru.reconstruct(frame)


def test_historical_sixty_categories_and_identity_direction() -> None:
    definitions = kg.contrast_definitions()
    counts = pd.Series([item.category for item in definitions.values()]).value_counts().to_dict()
    assert len(definitions) == 60
    assert counts == kg.CATEGORY_COUNTS
    assert sum(counts.values()) == 12 + 12 + 12 + 6 + 6 + 12
    for decoder, mode, status in itertools.product(
        kg.DECODERS, kg.MODES, ("with_ultra", "without_ultra")
    ):
        terms = definitions[f"relation_id|{decoder}|{status}|{mode}"].terms
        with_id = "b0_local_ultra" if status == "with_ultra" else "b0_local_only"
        without_id = (
            "b0_local_ultra_no_relation_id" if status == "with_ultra" else "b0_local_no_relation_id"
        )
        assert terms == ((1, decoder, with_id, mode), (-1, decoder, without_id, mode))


@pytest.mark.parametrize("duplicate", [False, True])
def test_historical_cells_reject_incompleteness_and_duplicates(duplicate: bool) -> None:
    frame = pd.read_csv(ROOT / kg.RESULTS / "cell_metrics.csv")
    frame = pd.concat([frame, frame.iloc[:1]]) if duplicate else frame.iloc[:-1]
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        kg.validate_cells(frame)


@pytest.mark.parametrize("duplicate", [False, True])
def test_historical_family_matrix_never_drops_incomplete_contrasts(duplicate: bool) -> None:
    frame = pd.read_csv(ROOT / kg.RESULTS / "contrast_by_family.csv")
    combined = frame.loc[frame.direction == "combined"]
    frame = pd.concat([combined, combined.iloc[:1]]) if duplicate else combined.iloc[:-1]
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        kg.family_matrix(frame)


def test_exact_max_t_against_small_independent_enumeration() -> None:
    base = np.array([[0.2, -0.3], [0.4, 0.1], [-0.1, 0.5], [0.3, -0.2]])
    observed, raw, adjusted = kg.exact_max_t(base)
    signed = np.array(list(itertools.product((-1, 1), repeat=4)))[:, :, None] * base
    statistics = signed.mean(axis=1) / (signed.std(axis=1, ddof=1) / np.sqrt(4))
    np.testing.assert_allclose(observed, base.mean(axis=0) / (base.std(axis=0, ddof=1) / 2))
    np.testing.assert_array_equal(raw, (abs(statistics) >= abs(observed) - 1e-12).mean(axis=0))
    expected = (abs(statistics).max(axis=1)[:, None] >= abs(observed) - 1e-12).mean(axis=0)
    np.testing.assert_array_equal(adjusted, expected)
    assert (adjusted >= raw).all()


def test_historical_zero_variance_t_convention() -> None:
    np.testing.assert_array_equal(kg.t_statistic(np.zeros((4, 2))), [0, 0])
    np.testing.assert_array_equal(kg.t_statistic(np.ones((4, 2))), [np.inf, np.inf])


def test_complete_historical_derived_reproduction() -> None:
    result = kg.verify()
    assert result["cells"] == 4104
    assert result["datasets"] == 57
    assert result["provenance_families"] == 16
    assert result["joint_contrasts"] == 60
    assert result["sign_patterns"] == 65536
    assert result["key_effect"] == 0.013937919134311694
    assert result["key_observed_t"] == 3.7901691175274235
    assert result["key_adjusted_p"] == 0.036895751953125
    assert tuple(result["identity_range"]) == kg.IDENTITY_RANGE
    assert result["identity_direction"] == "with identity minus without identity"


def test_historical_and_prospective_contracts_are_explicitly_distinct() -> None:
    prospective = (ROOT / "experiments/ultra4g/results/positive-context/README.md").read_text()
    recovery = (ROOT / kg.RESULTS / "RECOVERY.md").read_text()
    assert "Neither scope reproduces the historical 60-test correction" in prospective
    assert "exact negative rows" in prospective
    assert "resamples negative rows" in recovery
    assert "30-test" in recovery
