from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "build_rfmlink_negative_robustness_asset.py"
SPEC = importlib.util.spec_from_file_location("structural_context_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selector_filters_truth_and_respects_role_and_partition() -> None:
    candidates = np.zeros((42, 18), dtype=np.float32)
    filtered = np.zeros(42, dtype=np.bool_)
    target = 0
    filtered[41] = True

    candidates[1:41, 6] = 1.0
    candidates[1:40, 16] = 1.0
    candidates[1:41, 3] = np.arange(1, 41)
    candidates[1:41, 4] = np.arange(1, 41)

    entity, audit = MODULE._matched_negative(
        candidates=candidates,
        filtered=filtered,
        target=target,
        side=0,
        rng=np.random.default_rng(7),
    )

    assert entity != target
    assert not filtered[entity]
    assert candidates[entity, 6] > 0
    assert candidates[entity, 16] > 0
    assert entity <= 32
    assert not audit["role_fallback"]
    assert not audit["community_fallback"]


def test_selector_records_fallback_when_role_is_unavailable() -> None:
    candidates = np.zeros((4, 18), dtype=np.float32)
    entity, audit = MODULE._matched_negative(
        candidates=candidates,
        filtered=np.zeros(4, dtype=np.bool_),
        target=0,
        side=1,
        rng=np.random.default_rng(11),
    )

    assert entity in {1, 2, 3}
    assert audit["role_fallback"]
    assert audit["community_fallback"]
