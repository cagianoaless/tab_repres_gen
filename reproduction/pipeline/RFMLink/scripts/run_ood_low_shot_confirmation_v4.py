#!/usr/bin/env python3
"""80-GB optimized compute revision of the OOD confirmation suite.

The scientific protocol is inherited unchanged. ULTRA triple scoring starts at
micro-batch 32 and retries smaller batches only after an explicit CUDA OOM.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ood_low_shot_confirmation_v3 as v3  # noqa: E402


engine = v3.engine
PROTOCOL_REVISION = "ood_low_shot_confirmation_v4"
MAX_ULTRA_TRIPLE_BATCH_SIZE = 32
ULTRA_BATCH_FALLBACKS = (32, 16, 8, 4, 2, 1)
BATCH_EVENTS: list[dict] = []

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


_v3_protocol_specification = engine.protocol_specification
_unbounded_score_triples = v3._unbounded_score_triples
_prepare_dataset = engine.prepare_dataset


def candidate_batch_sizes(requested: int) -> list[int]:
    maximum = min(int(requested), MAX_ULTRA_TRIPLE_BATCH_SIZE)
    return list(dict.fromkeys([
        maximum,
        *(size for size in ULTRA_BATCH_FALLBACKS if size < maximum),
    ]))


def adaptive_score_triples(*args, batch_size: int = 32, **kwargs):
    """Score identical rows, reducing only the CUDA execution chunk on OOM."""
    requested = int(batch_size)
    attempts = candidate_batch_sizes(requested)
    for index, attempt in enumerate(attempts):
        try:
            result = _unbounded_score_triples(
                *args, batch_size=attempt, **kwargs
            )
            BATCH_EVENTS.append({
                "requested": requested,
                "attempts": attempts[: index + 1],
                "used": attempt,
            })
            return result
        except Exception as error:
            import torch

            if not isinstance(error, torch.cuda.OutOfMemoryError) or index + 1 == len(attempts):
                raise
            print(
                f"ULTRA_MICROBATCH_OOM requested={requested} failed={attempt} "
                f"retry={attempts[index + 1]}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    raise AssertionError("unreachable ULTRA micro-batch retry state")


def prepare_dataset_with_compute_manifest(dataset: str) -> dict:
    BATCH_EVENTS.clear()
    manifest = _prepare_dataset(dataset)
    if BATCH_EVENTS:
        manifest["ultra_triple_scoring_batches"] = list(BATCH_EVENTS)
        engine.pilot.atomic_json(engine.dataset_dir(dataset) / "manifest.json", manifest)
    return manifest


def protocol_specification() -> dict:
    protocol = _v3_protocol_specification()
    protocol["protocol_revision"] = PROTOCOL_REVISION
    protocol["compute_revision"] = {
        "decision": "GO_WITH_80GB_BATCH_OPTIMIZATION",
        "v3_outcome": (
            "manually stopped before metric computation after confirming that the "
            "8-row cap was unnecessarily conservative for the local 80-GB GPU"
        ),
        "only_change": (
            "score_triples starts at micro-batch 32 and retries 16,8,4,2,1 only "
            "after an explicit CUDA OOM; scientific inputs and outputs are unchanged"
        ),
        "max_ultra_triple_batch_size": MAX_ULTRA_TRIPLE_BATCH_SIZE,
        "fallbacks": list(ULTRA_BATCH_FALLBACKS),
        "fallback_logged": True,
        "subsampling": False,
        "expected_semantics": (
            "independent query batching; predictions are mathematically equivalent apart "
            "from possible floating-point roundoff"
        ),
    }
    return protocol


def source_hashes() -> dict:
    return {
        "runner_v4": engine.file_sha256(Path(__file__)),
        "compute_revision_v3": engine.file_sha256(
            SCRIPT_DIR / "run_ood_low_shot_confirmation_v3.py"
        ),
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


engine.score_triples = adaptive_score_triples
engine.prepare_dataset = prepare_dataset_with_compute_manifest
engine.protocol_specification = protocol_specification
engine.source_hashes = source_hashes


if __name__ == "__main__":
    engine.main()
