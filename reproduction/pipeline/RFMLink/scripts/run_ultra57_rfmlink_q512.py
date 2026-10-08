#!/usr/bin/env python3
"""High-query confirmation of the frozen ULTRA-57 RFMLink protocol.

The only scientific change from ``ultra57_rfmlink_batch1_v1`` is the
deterministic relation-stratified evaluation cap: 512 triples per available
validation/test split instead of 64. Context rows are reused byte-for-byte
from the parent suite because their construction is independent of evaluation
query selection.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import time

import run_ultra57_rfmlink as suite


PROTOCOL = "ultra57_rfmlink_q512_v1"
QUERY_TRIPLES_PER_SPLIT = 512
PARENT_PROTOCOL = "ultra57_rfmlink_batch1_v1"

WORKSPACE_ROOT = suite.WORKSPACE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL
DATA_ROOT = FAMILY_ROOT / "immutable_assets"
RUN_ROOT = FAMILY_ROOT / "runs"
TUNING_ROOT = FAMILY_ROOT / "classical_selection"
LOG_ROOT = FAMILY_ROOT / "launcher_logs"
SUMMARY_PATH = FAMILY_ROOT / "summary.json"
SUMMARY_MD = FAMILY_ROOT / "summary.md"
PLAN_PATH = WORKSPACE_ROOT / "results" / "analysis" / f"{PROTOCOL}_plan.json"
PARENT_ROOT = WORKSPACE_ROOT / "results" / PARENT_PROTOCOL
PARENT_DATA_ROOT = PARENT_ROOT / "immutable_assets"
PARENT_PLAN_PATH = (
    WORKSPACE_ROOT / "results" / "analysis" / f"{PARENT_PROTOCOL}_plan.json"
)
PARENT_SUMMARY_PATH = PARENT_ROOT / "summary.json"

_parent_protocol_specification = suite.protocol_specification
_parent_source_hashes = suite.source_hashes


def source_hashes() -> dict:
    inherited = _parent_source_hashes()
    inherited["parent_runner"] = inherited.pop("runner")
    inherited["parent_launcher"] = inherited.pop("launcher")
    inherited["runner"] = suite.file_sha256(Path(__file__))
    inherited["launcher"] = suite.file_sha256(
        SCRIPT_DIR / "run_ultra57_rfmlink_q512_queue.sh"
    )
    return inherited


def protocol_specification() -> dict:
    payload = copy.deepcopy(_parent_protocol_specification())
    payload["name"] = PROTOCOL
    payload["evaluation"].update({
        "query_selection": (
            "up to 512 deterministic relation-stratified validation/test triples; "
            "use the complete split when fewer than 512 are available"
        ),
        "maximum_directional_queries_per_split": 2 * QUERY_TRIPLES_PER_SPLIT,
        "query_selection_seed": {
            "validation": 20260714,
            "test": 20260715,
        },
        "same_query_rows_across_decoders_and_context_seeds": True,
        "sampled_evaluation_disclosure": (
            "This is complete all-entity filtered ranking for a deterministic "
            "relation-stratified query sample, not full-split evaluation when a "
            "split contains more than 512 triples."
        ),
    })
    payload["confirmation"] = {
        "parent_protocol": PARENT_PROTOCOL,
        "only_scientific_change": (
            "evaluation query cap increased from 64 to 512 triples for both "
            "validation and test assets"
        ),
        "unchanged": [
            "official canonical datasets and split policy",
            "frozen ULTRA-4G checkpoint",
            "400 positive plus 400 negative target context rows",
            "context row identities, labels, features, and ULTRA scores",
            "three context seeds 101, 102, 103",
            "four decoders and TabPFN estimator count",
            "all-entity head and tail candidate ranking",
            "filtered evaluator and validation-only alpha selection",
        ],
        "context_reuse": (
            "Parent context.npz files are SHA-256 verified and hard-linked or "
            "copied byte-for-byte; evaluation assets are newly materialized."
        ),
    }
    payload["dataset_staging"].update({
        "producer_required": False,
        "reason": "all 57 official caches and canonical bundles already exist",
        "parent_canonical_source": (
            f"results/{PARENT_PROTOCOL}/staging/canonical"
        ),
    })
    payload["claim_boundaries"]["official_protocol_equivalence"] = (
        "Dataset files are official. RFMLink uses its standardized 400+400 "
        "target-context and up-to-512-triple deterministic sampled full-ranking "
        "protocol, not ULTRA's full-test evaluation on every dataset."
    )
    return payload


def _configure_paths() -> None:
    suite.PROTOCOL = PROTOCOL
    suite.FAMILY_ROOT = FAMILY_ROOT
    suite.DATA_ROOT = DATA_ROOT
    suite.RUN_ROOT = RUN_ROOT
    suite.TUNING_ROOT = TUNING_ROOT
    suite.LOG_ROOT = LOG_ROOT
    suite.SUMMARY_PATH = SUMMARY_PATH
    suite.SUMMARY_MD = SUMMARY_MD
    suite.PLAN_PATH = PLAN_PATH
    suite.AMENDMENT_PATHS = ()
    suite.protocol_specification = protocol_specification
    suite.source_hashes = source_hashes
    suite.configure_base()
    suite.engine.QUERY_TRIPLES_PER_SPLIT = QUERY_TRIPLES_PER_SPLIT


_configure_paths()
engine = suite.engine


def _atomic_reuse(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        mode = "copy"
    os.replace(temporary, destination)
    return mode


def reuse_parent_context(dataset: str, seed: int, data: dict,
                         model, graph) -> dict:
    del data, model, graph
    destination = engine.context_path(dataset, seed)
    destination_manifest = destination.with_suffix(".json")
    if destination.exists() and destination_manifest.exists():
        manifest = json.loads(destination_manifest.read_text(encoding="utf-8"))
        if suite.file_sha256(destination) != manifest["context_sha256"]:
            raise RuntimeError(f"reused context hash mismatch: {dataset}/{seed}")
        return manifest

    source = PARENT_DATA_ROOT / dataset / "contexts" / str(seed) / "context.npz"
    source_manifest = source.with_suffix(".json")
    if not source.exists() or not source_manifest.exists():
        raise FileNotFoundError(f"parent context is unavailable: {dataset}/{seed}")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected = manifest["context_sha256"]
    if suite.file_sha256(source) != expected:
        raise RuntimeError(f"parent context hash mismatch: {dataset}/{seed}")
    if manifest["dataset"] != dataset or int(manifest["seed"]) != seed:
        raise RuntimeError(f"parent context identity mismatch: {dataset}/{seed}")
    if list(manifest["feature_names"]) != list(engine.FEATURE_NAMES):
        raise RuntimeError(f"parent context schema mismatch: {dataset}/{seed}")

    mode = _atomic_reuse(source, destination)
    reused = {
        **manifest,
        "reused_from": str(source.relative_to(WORKSPACE_ROOT)),
        "reuse_mode": mode,
        "reuse_verified_sha256": expected,
        "evaluation_query_independent": True,
    }
    engine.pilot.atomic_json(destination_manifest, reused)
    return reused


engine.materialize_context = reuse_parent_context


def register() -> dict:
    if PLAN_PATH.exists():
        return json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if list(FAMILY_ROOT.rglob("metrics.json")):
        raise RuntimeError("high-query metrics exist before registration")
    if not PARENT_PLAN_PATH.exists() or not PARENT_SUMMARY_PATH.exists():
        raise FileNotFoundError("parent plan and summary must exist before registration")
    if suite.file_sha256(suite.base.ULTRA_CHECKPOINT) != suite.base.EXPECTED_ULTRA_SHA256:
        raise RuntimeError("ULTRA-4G checkpoint hash mismatch")
    jobs = suite.primary_jobs()
    payload = {
        "status": "REGISTERED_BEFORE_HIGH_QUERY_METRICS",
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_specification(),
        "source_hashes": source_hashes(),
        "parent": {
            "protocol": PARENT_PROTOCOL,
            "plan_sha256": suite.file_sha256(PARENT_PLAN_PATH),
            "summary_sha256_at_registration": suite.file_sha256(PARENT_SUMMARY_PATH),
        },
        "planned_jobs": {
            "total": len(jobs),
            "tabpfn": len(suite.primary_jobs(group="tabpfn")),
            "classical": len(suite.primary_jobs(group="classical")),
            "per_dataset": len(suite.primary_jobs(suite.DATASETS[0])),
        },
        "estimated_runtime_hours": {
            "range": [35, 75],
            "basis": (
                "eight-fold query-cap increase, four parallel TabPFN workers, "
                "byte-identical context reuse, and observed parent-suite timings"
            ),
        },
        "estimated_additional_storage_gib": [90, 140],
    }
    engine.pilot.atomic_json(PLAN_PATH, payload)
    return payload


def validate_registration(require_masking_audit: bool = True) -> dict:
    del require_masking_audit
    if not PLAN_PATH.exists():
        raise FileNotFoundError("high-query suite is not registered")
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if payload["status"] != "REGISTERED_BEFORE_HIGH_QUERY_METRICS":
        raise RuntimeError("invalid high-query registration state")
    if payload["protocol"] != protocol_specification():
        raise RuntimeError("registered high-query protocol changed")
    if payload["source_hashes"] != source_hashes():
        raise RuntimeError("registered high-query source changed")
    audit = json.loads(suite.MASKING_AUDIT_PATH.read_text(encoding="utf-8"))
    if not audit.get("broad_suite_allowed"):
        raise RuntimeError("ULTRA masking audit does not permit the suite")
    return payload


suite.register = register
suite.validate_registration = validate_registration
suite.base.validate_registration = validate_registration
engine.validate_registration = validate_registration


if __name__ == "__main__":
    suite.main()
