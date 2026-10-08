#!/usr/bin/env python3
"""Materialize audited hybrid-solver B2v2 sidecars without evaluating them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import features
from protocol import AUDIT_ROOT, FEATURE_ROOT, PANEL, PROTOCOL, ROOT


SPECTRAL_AUDIT_PATH = AUDIT_ROOT / "spectral_audit.json"


def run_shard(shard_id: int, num_shards: int) -> dict:
    if not 0 <= shard_id < num_shards:
        raise ValueError("invalid shard")
    spectral = json.loads(SPECTRAL_AUDIT_PATH.read_text(encoding="utf-8"))
    if spectral.get("status") != "PASS":
        raise RuntimeError("spectral audit has not passed")
    if spectral.get("b2_implementation_sha256") != features.b2_implementation_sha256():
        raise RuntimeError("implementation changed after spectral audit")
    if list((FEATURE_ROOT.parent / "runs").rglob("metrics.json")):
        raise RuntimeError("performance metrics exist inside the metric-free family")
    assigned = [
        dataset for index, dataset in enumerate(PANEL)
        if index % num_shards == shard_id
    ]
    records = []
    failures = []
    for index, dataset in enumerate(assigned, 1):
        print(
            f"MATERIALIZE shard={shard_id}/{num_shards} dataset={dataset} "
            f"{index}/{len(assigned)}",
            flush=True,
        )
        started = time.time()
        try:
            manifest = features.materialize_dataset(dataset)
            records.append({
                "dataset": dataset,
                "status": manifest["status"],
                "manifest": str(
                    (FEATURE_ROOT / dataset / "manifest.json").relative_to(ROOT)
                ),
                "elapsed_seconds": time.time() - started,
            })
        except Exception as error:
            failures.append({"dataset": dataset, "error": repr(error)})
            break
    payload = {
        "status": "COMPLETE" if not failures and len(records) == len(assigned) else "FAIL",
        "protocol": PROTOCOL,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "shard_id": shard_id,
        "num_shards": num_shards,
        "assigned_datasets": assigned,
        "datasets_completed": len(records),
        "failures": failures,
        "spectral_audit_sha256": features.file_sha256(SPECTRAL_AUDIT_PATH),
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "labels_or_performance_metrics_accessed": False,
        "records": records,
    }
    output = FEATURE_ROOT.parent / "materialization" / "shards" / f"shard{shard_id}of{num_shards}.json"
    features.atomic_json(output, payload)
    if payload["status"] != "COMPLETE":
        raise RuntimeError(f"materialization shard failed: {failures}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run_shard(args.shard_id, args.num_shards), indent=2))


if __name__ == "__main__":
    main()
