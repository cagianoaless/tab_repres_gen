from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "analyze_graphlog_rule_overlap.py"
)
SPEC = importlib.util.spec_from_file_location("graphlog_rule_overlap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rule_jaccard_and_exact_sign_sensitivity() -> None:
    left = frozenset({"a", "b", "c"})
    right = frozenset({"b", "c", "d"})
    assert MODULE.rule_jaccard(left, right) == 0.5
    assert MODULE.exact_two_sided_sign_pvalue(0, 0) == 1.0
    assert MODULE.exact_two_sided_sign_pvalue(5, 5) == 1.0
    assert MODULE.exact_two_sided_sign_pvalue(8, 0) == 2.0 / (2**8)


def test_similarity_matrix_is_symmetric_and_ordered() -> None:
    rules = {
        "rule_2": frozenset({"a", "b"}),
        "rule_0": frozenset({"a", "c"}),
        "rule_1": frozenset({"a", "b", "c"}),
    }
    worlds, matrix = MODULE.similarity_matrix(rules)
    assert worlds == ["rule_0", "rule_1", "rule_2"]
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.ones(3))
    assert np.isclose(matrix[0, 1], 2.0 / 3.0)
