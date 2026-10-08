from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.analyze_ultra4g_positive_context as analysis
from analysis.analyze_ultra4g_positive_context import (
    ENCODERS,
    contrast_tables,
    coverage_rows,
    decoder_pairing_identity,
    relation_coverage,
    validate_matrix,
)
from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.protocol import FROZEN_KG_RELATION_CONTEXT_ASSET_ID

MODES = ("pool_random", "evaluation_relation_matched")


def test_decoder_pairing_allows_only_schema_and_deployment_differences() -> None:
    reference = {
        "categorical_features_indices": [0, 16],
        "parameters": {"depth": 6, "cat_features": [0, 16]},
        "checkpoints": [{"path": "worker-a/model.ckpt", "sha256": "a" * 64}],
        "resolved_devices": ["cuda:0"],
    }
    alternative = {
        **reference,
        "categorical_features_indices": [15],
        "parameters": {"depth": 6, "cat_features": [15]},
        "checkpoints": [{"path": "worker-b/model.ckpt", "sha256": "a" * 64}],
        "resolved_devices": ["cuda:1"],
    }
    assert decoder_pairing_identity(reference) == decoder_pairing_identity(alternative)
    assert reference["parameters"]["cat_features"] == [0, 16]  # no mutation
    changed = {**alternative, "parameters": {"depth": 7, "cat_features": [15]}}
    assert decoder_pairing_identity(reference) != decoder_pairing_identity(changed)
    changed = {**alternative, "checkpoints": [{"sha256": "b" * 64}]}
    assert decoder_pairing_identity(reference) != decoder_pairing_identity(changed)


def _matrix() -> pd.DataFrame:
    rows = []
    for dataset, decoder, encoder, mode, seed in itertools.product(
        ("a", "b"), ("logistic", "catboost", "tabpfn"), ENCODERS, MODES, (101, 102, 103)
    ):
        rows.append(
            {
                "dataset": dataset,
                "provenance_family": dataset,
                "decoder": decoder,
                "interface": encoder,
                "context_mode": mode,
                "seed": seed,
                "combined_mrr": 0.2
                + (0.02 if mode == MODES[1] else 0)
                - (0.003 if "no_relation_id" in encoder else 0),
                "context_per_class": 2,
                "archive_sha256": dataset * 64,
                "context_indices_sha256": object_sha256([0, 1, 2, 3]),
            }
        )
    return pd.DataFrame(rows)


def test_coverage_counts_types_mass_and_tv_on_union() -> None:
    result = relation_coverage(np.array([0, 0, 0, 1]), np.array([0, 2]))
    assert result == {
        "relation_type_coverage": 0.5,
        "relation_mass_coverage": 0.75,
        "total_variation_distance": 0.5,
    }
    assert relation_coverage(np.array([1, 1, 2]), np.array([1, 1, 2])) == {
        "relation_type_coverage": 1.0,
        "relation_mass_coverage": 1.0,
        "total_variation_distance": 0.0,
    }


@pytest.mark.parametrize("values", [[], [-1], [0.5], [float("nan")]])
def test_coverage_rejects_invalid_nominal_ids(values: list[float]) -> None:
    with pytest.raises(ValueError):
        relation_coverage(np.array([1]), np.asarray(values))


def test_complete_matrix_and_contrast_directions_are_deterministic() -> None:
    cells = _matrix()
    validate_matrix(cells, MODES, datasets=2, families=2)
    summary, effects = contrast_tables(cells, MODES)
    assert len(summary) == 24
    assert len(effects) == 48
    context = summary.loc[summary.axis == "context_mode"]
    identity = summary.loc[summary.axis == "encoder_variant"]
    assert len(context) == len(identity) == 12
    np.testing.assert_allclose(context.equal_family_effect, 0.02)
    np.testing.assert_allclose(identity.equal_family_effect, -0.003)
    shuffled, _ = contrast_tables(cells.sample(frac=1, random_state=17), MODES)
    pd.testing.assert_frame_equal(summary, shuffled)


def test_nominal_decoder_ids_are_preserved_not_reinterpreted() -> None:
    cells = _matrix()
    cells["decoder"] = cells.decoder.replace(
        {
            "logistic": "logistic_nominal",
            "catboost": "catboost_nominal",
        }
    )
    validate_matrix(cells, MODES, datasets=2, families=2)
    summary, _ = contrast_tables(cells, MODES)
    assert set(summary.decoder) == {"logistic_nominal", "catboost_nominal", "tabpfn"}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "archive", "selection", "budget", "mrr"])
def test_matrix_rejects_unpaired_or_changed_protocol(fault: str) -> None:
    cells = _matrix()
    if fault == "missing":
        cells = cells.iloc[1:]
    elif fault == "duplicate":
        cells = pd.concat([cells, cells.iloc[:1]])
    else:
        column, value = {
            "archive": ("archive_sha256", "changed"),
            "selection": ("context_indices_sha256", "changed"),
            "budget": ("context_per_class", 3),
            "mrr": ("combined_mrr", np.nan),
        }[fault]
        cells.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_matrix(cells, MODES, datasets=2, families=2)


def _coverage_asset(root: Path, *, change_negatives: bool = False) -> pd.DataFrame:
    arrays = {"num_entities": np.array(2)}
    # Equal head/tail masses and 64 triples, each with two candidate rows.
    relations = np.tile(np.repeat([0, 1], 32), 2)
    candidate = np.zeros((256, 19), dtype=np.float32)
    candidate[:, 0] = np.repeat(relations, 2)
    arrays.update(
        {
            "candidate_b0": candidate,
            "side_codes": np.repeat([0, 1], 64),
            "filtered": np.zeros((128, 2), dtype=bool),
        }
    )
    for seed, mode in itertools.product((101, 102, 103), MODES):
        prefix = f"context_{mode}_{seed}"
        x = np.zeros((4, 19), dtype=np.float32)
        x[:, 0] = [0, int(mode == MODES[1]), 0, 0]
        if change_negatives and mode == MODES[1]:
            x[2, 1] = 1
        triples = np.zeros((4, 3), dtype=np.int64)
        triples[:, 1] = x[:, 0]
        arrays.update(
            {
                f"{prefix}_X": x,
                f"{prefix}_triples": triples,
                f"{prefix}_b2v2": np.zeros((4, 13)),
                f"{prefix}_y": np.array([1, 1, 0, 0]),
            }
        )
    archive = root / "a.npz"
    np.savez(archive, **arrays)
    manifest = {"b0_feature_names": ["relation_id", *[f"f{i}" for i in range(18)]]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    cells = _matrix().loc[lambda frame: frame.dataset == "a"].copy()
    cells["archive_sha256"] = file_sha256(archive)
    return cells


def test_coverage_uses_actual_selected_positive_rows_once_per_seed(tmp_path: Path) -> None:
    cells = _coverage_asset(tmp_path)
    coverage = coverage_rows(cells, tmp_path, MODES)
    assert len(coverage) == 6  # not repeated for decoders or identity variants
    pool = coverage.loc[coverage.context_mode == MODES[0]]
    matched = coverage.loc[coverage.context_mode == MODES[1]]
    np.testing.assert_allclose(pool.relation_mass_coverage, 0.5)
    np.testing.assert_allclose(pool.total_variation_distance, 0.5)
    np.testing.assert_allclose(matched.relation_mass_coverage, 1.0)
    np.testing.assert_allclose(matched.total_variation_distance, 0.0)


def test_positive_only_control_rejects_changed_negatives(tmp_path: Path) -> None:
    cells = _coverage_asset(tmp_path, change_negatives=True)
    with pytest.raises(ValueError, match="negative context changed"):
        coverage_rows(cells, tmp_path, MODES)


def test_coverage_checks_archive_and_selection_hashes(tmp_path: Path) -> None:
    cells = _coverage_asset(tmp_path)
    cells["context_indices_sha256"] = "not-the-selection"
    with pytest.raises(ValueError, match="selection differs"):
        coverage_rows(cells, tmp_path, MODES)
    cells["archive_sha256"] = "not-the-archive"
    with pytest.raises(ValueError, match="archive bytes changed"):
        coverage_rows(cells, tmp_path, MODES)


def _result_fixture(root: Path) -> tuple[dict, dict]:
    """Synthetic metrics fixture: unit-tests result loading, not historical evidence."""
    asset = root / "asset"
    asset.mkdir()
    names = ["relation_id", *[f"local_{i}" for i in range(1, 18)], "ultra_score"]
    manifest = {
        "asset_id": FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
        "datasets": {"tiny": {"archive_sha256": "a" * 64}},
        "b0_feature_names": names,
        "b2v2_feature_names": [f"global_{i}" for i in range(13)],
    }
    (asset / "manifest.json").write_text(json.dumps(manifest))
    cell = {
        "cell_id": "fixture",
        "dataset": "tiny",
        "decoder": "catboost",
        "interface": "b0_local_only",
        "context_mode": MODES[0],
        "seed": 101,
        "decoder_seed": 500101,
        "query_triples": 64,
        "context_per_class": 2,
        "asset_id": FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
        "archive_sha256": "a" * 64,
        "asset_manifest_sha256": file_sha256(asset / "manifest.json"),
    }
    shard = root / "run/results/shard-0000"
    shard.mkdir(parents=True)
    ranks = np.full(128, 2.0)
    native_ranks = np.full(128, 3.0)
    np.savez(
        shard / "fixture.predictions.npz",
        ranks=ranks,
        reciprocal_ranks=1 / ranks,
        native_ultra_ranks=native_ranks,
        native_ultra_reciprocal_ranks=1 / native_ranks,
        query_ids=np.arange(128),
        side_codes=np.repeat([0, 1], 64),
        target_entities=np.zeros(128),
    )
    payload = {
        "cell": cell,
        "feature_names": names[:18],
        "feature_count": 18,
        "categorical_feature_indices": [0, 16],
        "context_rows": 4,
        "context_positive_rows": 2,
        "task_metadata": {
            **{
                key: cell[key]
                for key in ("asset_manifest_sha256", "archive_sha256", "context_mode")
            },
            "context_indices_sha256": object_sha256([0, 1, 2, 3]),
        },
        "direction_metrics": {"combined": {"mrr": 0.5}},
        "native_ultra_direction_metrics": {"combined": {"mrr": float((1 / native_ranks).mean())}},
        "decoder_provenance": {"parameters": {"depth": 6}, "categorical_features_indices": [0, 16]},
        "predictions": {"path": "fixture.predictions.npz"},
    }
    plan = {"cells": [cell], "plan_sha256": "fixture-plan"}
    (shard / "fixture.metrics.json").write_text(json.dumps(payload))
    return plan, payload


@pytest.mark.parametrize("fault", [None, "mrr", "schema", "metadata", "native", "incomplete"])
def test_result_loader_checks_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    plan, payload = _result_fixture(tmp_path)
    # Isolate loader logic. Actual plan/shard hash validation has separate tests
    # and the production entry point always invokes those validators.
    monkeypatch.setattr(analysis, "read_plan", lambda path: plan)
    monkeypatch.setattr(
        analysis,
        "verify_shard",
        lambda path: {
            "cell_ids": [] if fault == "incomplete" else ["fixture"],
            "plan_sha256": "fixture-plan",
        },
    )
    if fault == "mrr":
        payload["direction_metrics"]["combined"]["mrr"] = 0.7
    elif fault == "schema":
        payload["categorical_feature_indices"] = [15]
    elif fault == "metadata":
        payload["task_metadata"]["archive_sha256"] = "changed"
    elif fault == "native":
        payload["native_ultra_direction_metrics"]["combined"]["mrr"] = 0.7
    (tmp_path / "run/results/shard-0000/fixture.metrics.json").write_text(json.dumps(payload))
    if fault is None:
        cells = analysis.load_results([tmp_path / "run"], tmp_path / "asset", {"tiny": "family"})
        assert len(cells) == 1
        assert cells.iloc[0].combined_mrr == 0.5
    else:
        with pytest.raises(ValueError):
            analysis.load_results([tmp_path / "run"], tmp_path / "asset", {"tiny": "family"})
