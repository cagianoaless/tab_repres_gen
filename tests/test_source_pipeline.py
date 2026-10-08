from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from relational_tfm.datasets.rfmlink import encode_feature_schema
from reproduction import verify_source_pipeline as source


@pytest.fixture(scope="module")
def local():
    return source.load_file(
        "_test_local18", source.RFMLINK / "rfmlink/vectorized_invariants.py"
    )


@pytest.fixture(scope="module")
def global13():
    return source.load_global()


def test_source_closure_and_reference_identities() -> None:
    assert source.verify() == {
        "source_files": 52, "Local": 18, "score": 1, "Global": 13, "Full": 32,
    }
    reference = json.loads((source.HERE / "REFERENCE_EXECUTION.json").read_text())
    assert reference["relation_context_cpu"]["scientific_plan_sha256"] == (
        "bd836b248fdb2812206087fae0c5ee88f5e9035cb9450c5d6d5da7893e9ccee2"
    )
    assert reference["relation_context_cpu"]["serialized_file_sha256"] != (
        reference["relation_context_cpu"]["scientific_plan_sha256"]
    )


def test_local_direct_formulas_and_candidate_orientation(local) -> None:
    train = np.array([
        [0, 0, 1], [0, 1, 1], [1, 0, 2], [0, 0, 3],
        [3, 0, 2], [2, 1, 0], [3, 0, 4],
    ])
    builder = local.VectorizedInvariantBuilder(train, 5, 2, num_clusters=1)
    row = builder.feature_row(0, 0, 2)
    assert row.shape == (18,)
    assert row[7] == 5
    assert row[8] == np.float32(3 / 5)  # distinct heads / relation triples
    assert row[9] == np.float32(4 / 5)  # distinct tails / relation triples
    assert tuple(row[12:15]) == (2, 2, 2)  # distinct intermediate entities
    assert row[16] == 1
    assert row[17] == np.float32(5 / 25)
    for side in ("tail", "head"):
        block = builder.candidate_block(0, 0, 2, side)
        triples = [[0, 0, entity] if side == "tail" else [entity, 0, 2] for entity in range(5)]
        np.testing.assert_array_equal(block, builder.feature_rows(triples))
    assert builder.feature_row(0, 0, 1)[10] == 1  # exclude the queried relation


def test_local_rejects_duplicate_support_triples(local) -> None:
    with pytest.raises(ValueError, match="unique"):
        local.VectorizedInvariantBuilder(np.array([[0, 0, 1], [0, 0, 1]]), 2, 1)


def test_global_components_scales_and_determinism(global13) -> None:
    train = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0], [3, 0, 4]])
    first, audit = global13.FROZEN.graph_embeddings(train, 6)
    second, _ = global13.FROZEN.graph_embeddings(train, 6)
    queries = np.array([[0, 0, 1], [0, 0, 4], [5, 0, 5]])
    features = global13.b2_rows(queries, first)
    assert features.shape == (3, 13)
    np.testing.assert_array_equal(features, global13.b2_rows(queries, second))
    assert tuple(features[:, 2]) == (1, 0, 1)
    np.testing.assert_allclose(features[0, 3:5], np.log1p(3) / np.log1p(6))
    np.testing.assert_array_equal(features[1, 5:], np.zeros(8))
    np.testing.assert_array_equal(features[2, 5:], np.zeros(8))
    assert np.isclose(first["pagerank"].sum(), 1)
    assert global13.HEAT_SCALES == (0.1, 1.0, 10.0, 100.0)
    assert audit["spectral"]["trivial_modes"].startswith("one excluded")


def test_actual_producer_order_matches_full32_loader(local, global13) -> None:
    b0 = np.arange(3 * 19, dtype=np.float32).reshape(3, 19)
    b2 = np.arange(3 * 13, dtype=np.float32).reshape(3, 13)
    names = tuple(local.FEATURE_NAMES) + ("ultra_score",)
    full, actual_names, categorical = encode_feature_schema(
        b0, b2, "b2v2_full", names, tuple(global13.B2_FEATURE_NAMES)
    )
    np.testing.assert_array_equal(full, np.concatenate([b0, b2], axis=1))
    assert full.shape == (3, 32)
    assert actual_names == names + tuple(global13.B2_FEATURE_NAMES)
    assert categorical == (0, 16)


@pytest.mark.parametrize(
    ("size", "method"), [(512, "dense_eigh"), (513, "eigsh_sm")]
)
def test_original_solver_threshold_and_repeatability(
    global13, monkeypatch: pytest.MonkeyPatch, size: int, method: str,
) -> None:
    monkeypatch.setitem(sys.modules, "features", global13)
    original = source.load_file("_paper_solver_test", source.HYBRID / "test_hybrid_solver.py")
    result = original.solve_twice(size)
    assert result["method"] == method
    assert result["eigenvalues_bitwise_equal"]
    assert result["retained_projectors_equal"]


def test_portable_entry_point_help_requires_no_checkpoints() -> None:
    from reproduction.run_pipeline import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_public_guide_explains_reference_and_prospective_contracts() -> None:
    guide = (Path(__file__).parents[1] / "REPRODUCE.md").read_text()
    assert "60" in guide and "30" in guide
    assert "12/24" in guide
    assert "resamples negative rows" in guide
