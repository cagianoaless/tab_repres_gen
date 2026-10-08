#!/usr/bin/env python3
"""Compute-equivalent revision of the registered OOD confirmation suite.

V2 stopped before metrics because a batch of 64 AristoV4 context negatives
expanded an ULTRA relation-graph intermediate beyond 80 GiB. V3 preserves the
registered scientific protocol and caps only ULTRA triple-scoring micro-batches.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ood_low_shot_confirmation_v2 as v2  # noqa: E402


engine = v2.engine
PROTOCOL_REVISION = "ood_low_shot_confirmation_v3"
MAX_ULTRA_TRIPLE_BATCH_SIZE = 8

engine.PROTOCOL_REVISION = PROTOCOL_REVISION
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


_v2_protocol_specification = engine.protocol_specification
_unbounded_score_triples = engine.score_triples


def bounded_score_triples(*args, batch_size: int = 8, **kwargs):
    """Delegate to the audited scorer with a memory-safe micro-batch."""
    return _unbounded_score_triples(
        *args,
        batch_size=min(int(batch_size), MAX_ULTRA_TRIPLE_BATCH_SIZE),
        **kwargs,
    )


def protocol_specification() -> dict:
    protocol = _v2_protocol_specification()
    protocol["protocol_revision"] = PROTOCOL_REVISION
    protocol["compute_revision"] = {
        "decision": "GO_WITH_COMPUTE_CORRECTION",
        "v2_outcome": (
            "stopped before tuning, smoke tests, model fitting, or metric computation; "
            "ULTRA requested 104.24 GiB while scoring AristoV4 negatives in batches of 64"
        ),
        "only_change": (
            "cap score_triples micro-batches at 8; rows, features, labels, scores, "
            "checkpoint, candidates, and evaluator remain unchanged"
        ),
        "max_ultra_triple_batch_size": MAX_ULTRA_TRIPLE_BATCH_SIZE,
        "expected_semantics": (
            "independent query batching; predictions are mathematically equivalent apart "
            "from possible floating-point roundoff"
        ),
        "subsampling": False,
    }
    return protocol


def source_hashes() -> dict:
    return {
        "runner_v3": engine.file_sha256(Path(__file__)),
        "dataset_revision_v2": engine.file_sha256(
            SCRIPT_DIR / "run_ood_low_shot_confirmation_v2.py"
        ),
        "immutable_engine_v1": engine.file_sha256(
            SCRIPT_DIR / "run_ood_low_shot_confirmation.py"
        ),
        "vectorized_invariants": engine.file_sha256(
            RFMLINK_ROOT / "rfmlink" / "vectorized_invariants.py"
        ),
    }


engine.score_triples = bounded_score_triples
engine.protocol_specification = protocol_specification
engine.source_hashes = source_hashes


if __name__ == "__main__":
    engine.main()
