#!/usr/bin/env python3
"""Freeze the metric-free hybrid solver protocol after all numerical gates pass."""

from __future__ import annotations

import json
from pathlib import Path

import features
from protocol import AUDIT_ROOT, FEATURE_ROOT, PROTOCOL, ROOT, STABILITY_AUDIT_ROOT


DEVELOPMENT_ROOT = AUDIT_ROOT / "development_large_stability"
CUTOFF_AUDIT = AUDIT_ROOT / "cutoff_audit.json"
SPECTRAL_AUDIT = AUDIT_ROOT / "spectral_audit.json"
MATERIALIZED_AUDIT = AUDIT_ROOT / "materialized_audit.json"
OUTPUT = AUDIT_ROOT / "frozen_numeric_protocol.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": features.file_sha256(path),
    }


def main() -> None:
    cutoff = load(CUTOFF_AUDIT)
    require(cutoff.get("status") == "PASS", "cutoff audit failed")
    require(
        cutoff.get("labels_or_performance_metrics_accessed") is False,
        "cutoff audit provenance failed",
    )
    require([row["size"] for row in cutoff["records"]] == [511, 512, 513], "cutoff boundary coverage failed")

    spectral = load(SPECTRAL_AUDIT)
    require(spectral.get("status") == "PASS", "spectral audit failed")
    require(spectral.get("datasets_completed") == 57, "spectral dataset coverage failed")
    require(spectral.get("graphs_completed") == 1767, "spectral graph coverage failed")
    require(spectral.get("performance_metrics_accessed") is False, "spectral provenance failed")

    materialized = load(MATERIALIZED_AUDIT)
    require(materialized.get("status") == "PASS", "materialized audit failed")
    require(materialized.get("datasets_completed") == 57, "materialized coverage failed")
    require(materialized.get("performance_metrics_accessed") is False, "materialized provenance failed")

    confirmation_paths = sorted((STABILITY_AUDIT_ROOT / "shards").glob("shard*of2.json"))
    confirmation = [load(path) for path in confirmation_paths]
    require(len(confirmation) == 2, "confirmation stability shards missing")
    require(all(row.get("status") == "COMPLETE" for row in confirmation), "confirmation stability failed")
    require(sum(row["datasets_completed"] for row in confirmation) == 51, "confirmation stability dataset coverage failed")
    require(sum(row["components_completed"] for row in confirmation) == 1657, "confirmation stability component coverage failed")
    require(sum(row["unstable_components"] for row in confirmation) == 0, "confirmation instability detected")
    require(sum(row["graph_errors"] for row in confirmation) == 0, "confirmation graph errors detected")

    development_paths = sorted((DEVELOPMENT_ROOT / "shards").glob("shard*of6.json"))
    development = [load(path) for path in development_paths]
    require(len(development) == 6, "development stability shards missing")
    require(all(row.get("status") == "PASS" for row in development), "development stability failed")
    require(sum(row["datasets_completed"] for row in development) == 6, "development dataset coverage failed")
    require(sum(row["graphs_completed"] for row in development) == 186, "development graph coverage failed")
    require(sum(row["components_completed"] for row in development) == 124, "development component coverage failed")
    require(sum(row["unstable_components"] for row in development) == 0, "development instability detected")
    require(sum(row["graph_errors"] for row in development) == 0, "development graph errors detected")

    metric_paths = list((FEATURE_ROOT.parent / "runs").rglob("metrics.json"))
    require(not metric_paths, "predictive metrics exist inside the metric-free family")

    payload = {
        "status": "FROZEN_BEFORE_PREDICTIVE_METRICS",
        "protocol": PROTOCOL,
        "solver_policy": {
            "components_at_most_512_nodes": "dense_eigh",
            "components_above_512_nodes": {
                "solver": "eigsh",
                "which": "SM",
                "k": 10,
                "deterministic_initialization": True,
                "tolerance": 1e-10,
                "maxiter": "max(10000, 20 * component_size)",
            },
            "truncation": (
                "retain at most eight positive modes; use the ninth positive mode "
                "as look-ahead and back off before any repeated boundary eigenspace"
            ),
            "mandatory_checks": {
                "maximum_residual": features.SOLVER_RESIDUAL_LIMIT,
                "maximum_orthogonality_error": features.ORTHOGONALITY_TOLERANCE,
                "maximum_zero_eigenvalue_error": features.ZERO_EIGENVALUE_TOLERANCE,
            },
            "fallbacks": [],
            "failure_policy": (
                "fail the complete protocol before prediction; never skip a dataset "
                "and never select a fallback using link-prediction metrics"
            ),
        },
        "cutoff_selection_uses_performance_metrics": False,
        "large_component_stability": {
            "confirmation_datasets": 51,
            "confirmation_components": 1657,
            "development_datasets": 6,
            "development_components": 124,
            "unstable_components": 0,
        },
        "artifact_isolation": {
            "new_feature_root": str(FEATURE_ROOT.relative_to(ROOT)),
            "reused_embedding_roots": [],
            "immutable_parent_query_context_and_ultra_assets_reused": True,
        },
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "source_hashes": {
            path.name: features.file_sha256(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*"))
            if path.is_file()
        },
        "audit_artifacts": [
            artifact(CUTOFF_AUDIT),
            artifact(SPECTRAL_AUDIT),
            artifact(MATERIALIZED_AUDIT),
            *[artifact(path) for path in confirmation_paths],
            *[artifact(path) for path in development_paths],
        ],
        "labels_or_performance_metrics_accessed": False,
    }
    features.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
