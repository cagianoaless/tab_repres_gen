import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "analysis" / "analyze_rfmlink_q64_multiplicity.py"
SPEC = importlib.util.spec_from_file_location("rfmlink_multiplicity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_holm_adjustment_is_monotone_in_sorted_raw_p_values():
    raw = np.asarray([0.04, 0.001, 0.02, 0.2])
    adjusted = MODULE.holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1)


def test_joint_sign_flip_uses_every_exact_family_sign_pattern():
    matrix = pd.DataFrame(
        {
            "provenance_family": ["a", "b", "c", "d"],
            "large": [1.0, 1.1, 0.9, 1.2],
            "small": [0.2, -0.1, 0.1, -0.2],
        }
    )
    result = MODULE.exact_joint_sign_flip(matrix)
    assert set(result["contrast_id"]) == {"large", "small"}
    assert (result["exact_sign_patterns"] == 16).all()
    assert (result["family_count"] == 4).all()
    assert result.loc[result["contrast_id"] == "large", "equal_family_effect"].iloc[0] == 1.05


def test_leave_one_family_out_reports_range_and_sign_stability():
    matrix = pd.DataFrame(
        {
            "provenance_family": ["a", "b", "c"],
            "stable": [1.0, 2.0, 3.0],
            "unstable": [-1.0, -1.0, 3.0],
        }
    )
    estimates, summary = MODULE.leave_one_family_out(matrix)
    assert len(estimates) == 6
    stable = summary.loc[summary["contrast_id"] == "stable"].iloc[0]
    assert stable["minimum_leave_one_family_out_effect"] == 1.5
    assert stable["maximum_leave_one_family_out_effect"] == 2.5
    assert bool(stable["all_leave_one_family_out_same_sign_as_full"])
    unstable = summary.loc[summary["contrast_id"] == "unstable"].iloc[0]
    assert not bool(unstable["all_leave_one_family_out_same_sign_as_full"])


def test_selected_matrix_rejects_mismatched_family_panels():
    rows = []
    for _, contrast, context, direction in MODULE.SELECTED_CONTRASTS:
        families = ["a", "b", "c"]
        if contrast == MODULE.SELECTED_CONTRASTS[-1][1]:
            families = ["a", "b"]
        for family in families:
            rows.append(
                {
                    "contrast": contrast,
                    "context_per_class": context,
                    "direction": direction,
                    "provenance_family": family,
                    "effect": 0.1,
                }
            )
    with np.testing.assert_raises(ValueError):
        MODULE.selected_family_matrix(pd.DataFrame(rows))
