#!/usr/bin/env python3
"""Registered v2 of the standalone out-of-dataset confirmation suite.

The implementation engine is the immutable v1 runner. This revision changes
only the pre-registered dataset panel and paths after the v1 source audit found
overlapping triples in PharmKG8k. No v1 model or test metric was computed.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ood_low_shot_confirmation as engine  # noqa: E402


PROTOCOL_REVISION = "ood_low_shot_confirmation_v2"
DATASETS = ("countries", "aristov4", "hetionet")
DATASET_RATIONALE = {
    "countries": (
        "small geopolitical boundary-control KG; clean split and unseen during "
        "RFMLink development; not sufficient alone for a general claim"
    ),
    "aristov4": (
        "relation-rich commonsense KG; clean mapped split and unseen during "
        "RFMLink development"
    ),
    "hetionet": (
        "large biomedical KG; clean split, different graph scale, and unseen "
        "during RFMLink development"
    ),
}


engine.PROTOCOL_REVISION = PROTOCOL_REVISION
engine.DATASETS = DATASETS
engine.DATASET_RATIONALE = DATASET_RATIONALE
engine.TUNING_QUERY_TRIPLES = 24
engine.FAMILY_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL_REVISION
engine.DATA_ROOT = engine.FAMILY_ROOT / "immutable_assets"
engine.RUN_ROOT = engine.FAMILY_ROOT / "runs"
engine.TUNING_ROOT = engine.FAMILY_ROOT / "classical_selection"
engine.LOG_ROOT = engine.FAMILY_ROOT / "launcher_logs"
engine.SUMMARY_PATH = engine.FAMILY_ROOT / "summary.json"
engine.SUMMARY_MD = engine.FAMILY_ROOT / "summary.md"
engine.PLAN_PATH = (
    WORKSPACE_ROOT / "results" / "analysis" / f"{PROTOCOL_REVISION}_plan.json"
)


_engine_protocol_specification = engine.protocol_specification


def protocol_specification() -> dict:
    protocol = _engine_protocol_specification()
    protocol["protocol_revision"] = PROTOCOL_REVISION
    protocol["datasets"] = list(DATASETS)
    protocol["dataset_rationale"] = DATASET_RATIONALE
    protocol["source_audit"] = {
        "decision": "GO_WITH_CORRECTIONS",
        "v1_outcome": (
            "stopped before model loading or metric computation because PharmKG8k "
            "contains overlapping mapped splits"
        ),
        "v2_correction": (
            "replace all v1 datasets with three independently audited datasets "
            "having zero within-split duplicates and zero cross-split overlap"
        ),
        "audit_file": str(
            WORKSPACE_ROOT / "results" / "analysis" / "ood_candidate_split_audit.json"
        ),
    }
    protocol["evaluation"]["query_subset"] = (
        "up to 64 deterministic relation-stratified validation/test triples; use "
        "the complete split when it has fewer than 64 triples; each triple contributes "
        "one complete head block and one complete tail block"
    )
    protocol["classical_controls"]["selection"] = (
        "small grids containing only development-selected optima; select once per "
        "dataset/family on seed 101, 50+50 context, up to the first 24 registered "
        "validation triples; freeze for all test cells"
    )
    protocol["decision_rule"]["overall_support"] = (
        "at least two of three untouched datasets pass; Countries is a boundary "
        "control and cannot be the sole non-development evidence"
    )
    return protocol


def source_hashes() -> dict:
    return {
        "runner_v2": engine.file_sha256(Path(__file__)),
        "immutable_engine_v1": engine.file_sha256(
            SCRIPT_DIR / "run_ood_low_shot_confirmation.py"
        ),
        "vectorized_invariants": engine.file_sha256(
            RFMLINK_ROOT / "rfmlink" / "vectorized_invariants.py"
        ),
    }


engine.protocol_specification = protocol_specification
engine.source_hashes = source_hashes


if __name__ == "__main__":
    engine.main()
