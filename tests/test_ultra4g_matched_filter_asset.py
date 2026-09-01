from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "build_ultra4g_matched_filter_asset.py"
)
SPEC = importlib.util.spec_from_file_location("reviewer_filter_asset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unfiltered_sampling_changes_only_rejection_policy() -> None:
    entities = 4
    source_queries = np.arange(8, dtype=np.int64)
    triples = np.asarray([(0, i, 1) for i in range(4)], dtype=np.int64)
    targets = np.tile([1, 0], 4)
    sides = np.tile([0, 1], 4).astype(np.int8)
    b0 = np.arange(8 * entities * 2, dtype=np.float32).reshape(-1, 2)
    b2 = np.arange(8 * entities * 3, dtype=np.float32).reshape(-1, 3)
    filtered = np.zeros((8, entities), dtype=np.bool_)
    for query, target in enumerate(targets):
        filtered[query, target] = True
        filtered[query, (target + 1) % entities] = True
    arrays, audit = MODULE._unfiltered_rows(
        dataset="synthetic",
        seed=101,
        source_queries=source_queries,
        q512_triples=triples,
        q512_targets=targets,
        q512_sides=sides,
        q512_b0=b0,
        q512_b2v2=b2,
        q512_filtered=filtered,
        num_entities=entities,
    )
    assert arrays["negative_b0"].shape == (8, 2)
    assert arrays["negative_b2v2"].shape == (8, 3)
    assert np.all(arrays["negative_entities"] != targets)
    assert audit["target_entity_collisions"] == 0
    assert audit["all_known_positive_rejection"] is False
    assert audit["known_positive_negatives"] >= 0
