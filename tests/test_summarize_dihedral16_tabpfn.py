from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "summarize_dihedral16_tabpfn.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_dihedral16_tabpfn",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bootstrap_interval_contains_constant() -> None:
    lower, upper = MODULE._bootstrap_mean_interval([0.25] * 10)
    assert np.isclose(lower, 0.25)
    assert np.isclose(upper, 0.25)
