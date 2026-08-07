from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "summarize_graphlog_fixed_panel_audit.py"
)
SPEC = importlib.util.spec_from_file_location("graphlog_fixed_panel_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_two_sided_sign_pvalue() -> None:
    assert MODULE.exact_two_sided_sign_pvalue(0, 0) == 1.0
    assert MODULE.exact_two_sided_sign_pvalue(5, 5) == 1.0
    assert MODULE.exact_two_sided_sign_pvalue(10, 0) == 2.0 / (2**10)
    assert MODULE.exact_two_sided_sign_pvalue(0, 10) == 2.0 / (2**10)
