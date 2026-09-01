from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "analysis" / "analyze_ultra4g_controls.py"
SPEC = importlib.util.spec_from_file_location("analyze_ultra4g_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_summary_uses_dataset_then_family_aggregation() -> None:
    rows = []
    for dataset, family, raw, transformed in (
        ("a", "large", 0.1, 0.2),
        ("b", "large", 0.1, 0.2),
        ("c", "small", 0.4, 0.3),
    ):
        for transform, value in (("raw", raw), ("within_query_zscore", transformed)):
            rows.append(
                {
                    "dataset": dataset,
                    "provenance_family": family,
                    "decoder": "logistic",
                    "interface": "b0_local_only",
                    "context_mode": "filtered_uniform",
                    "query_transform": transform,
                    "combined_mrr": value,
                }
            )
    summary, effects = MODULE.paired_summary(
        pd.DataFrame(rows),
        axis="query_transform",
        reference="raw",
        alternatives=("within_query_zscore",),
    )
    assert len(effects) == 3
    assert summary.loc[0, "equal_dataset_effect"] == pytest.approx(
        (0.1 + 0.1 - 0.1) / 3
    )
    assert summary.loc[0, "equal_family_effect"] == pytest.approx(0.0)
    assert summary.loc[0, "bootstrap_draws"] == 10_000
    assert summary.loc[0, "bootstrap_sensitivity_low"] <= 0.0
    assert summary.loc[0, "bootstrap_sensitivity_high"] >= 0.0


def test_exact_joint_max_t_uses_all_family_sign_patterns() -> None:
    rows = []
    for contrast in range(2):
        for family in range(16):
            rows.append(
                {
                    "decoder": "logistic",
                    "interface": str(contrast),
                    "axis": "query_transform",
                    "reference": "raw",
                    "alternative": "within_query_zscore",
                    "dataset": str(family),
                    "provenance_family": str(family),
                    "effect": (family + 1) * (1 if contrast == 0 else -1),
                }
            )
    result = MODULE.exact_joint_max_t(pd.DataFrame(rows))
    assert len(result) == 2
    assert set(result["sign_patterns"]) == {2**16}
    assert set(result["joint_contrasts"]) == {2}
