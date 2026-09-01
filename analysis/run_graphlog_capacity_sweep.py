#!/usr/bin/env python3
"""Run the frozen reviewer-requested GraphLog CatBoost capacity sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from relational_tfm.plan import read_plan
from relational_tfm.runner import verify_shard

EXPECTED_PLAN_SHA256 = "a73c22df3f6ca0f80c4723d049be075fcba07edef27447485c86f113502072f0"
EXPECTED_CELLS = 2_394
DEFAULT_CELLS_PER_SHARD = 10


def run_shard(
    repo: Path,
    plan: Path,
    output: Path,
    shard_index: int,
    cells_per_shard: int,
) -> tuple[int, str]:
    shard_dir = output / "results" / f"shard-{shard_index:04d}"
    manifest = shard_dir / "shard_manifest.json"
    if manifest.exists():
        verify_shard(shard_dir)
        return shard_index, "skipped"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "logs" / f"shard-{shard_index:04d}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{repo / 'src'}:{repo}",
            "RTFM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RTFM_ENCODING_CACHE_DIR": str(output / "encoding_cache"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "relational_tfm.cli",
        "run-shard",
        "--plan",
        str(plan),
        "--shard-index",
        str(shard_index),
        "--cells-per-shard",
        str(cells_per_shard),
        "--output-dir",
        str(shard_dir),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"shard {shard_index} failed; inspect {log_path}")
    verify_shard(shard_dir)
    return shard_index, "completed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cells-per-shard", type=int, default=DEFAULT_CELLS_PER_SHARD)
    args = parser.parse_args()

    repo = args.repo.resolve()
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    plan = read_plan(plan_path)
    if plan["plan_sha256"] != EXPECTED_PLAN_SHA256:
        raise ValueError("capacity-sweep plan hash changed")
    if len(plan["cells"]) != EXPECTED_CELLS:
        raise ValueError("capacity-sweep cell population changed")
    if args.workers <= 0 or args.cells_per_shard <= 0:
        raise ValueError("workers and cells-per-shard must be positive")

    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "encoding_cache").mkdir(exist_ok=True)
    frozen_plan = output / "plan.json"
    if frozen_plan.exists() and frozen_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError("output contains a different frozen plan")
    if not frozen_plan.exists():
        frozen_plan.write_bytes(plan_path.read_bytes())

    shard_count = math.ceil(EXPECTED_CELLS / args.cells_per_shard)
    failures: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_shard,
                repo,
                frozen_plan,
                output,
                shard_index,
                args.cells_per_shard,
            ): shard_index
            for shard_index in range(shard_count)
        }
        for future in as_completed(futures):
            shard_index = futures[future]
            try:
                _, status = future.result()
            except Exception as error:  # noqa: BLE001 - persist every shard failure
                failures.append(
                    {
                        "shard_index": shard_index,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                print(f"FAIL shard={shard_index}: {error}", flush=True)
            else:
                completed += status == "completed"
                skipped += status == "skipped"
                print(f"PASS shard={shard_index} status={status}", flush=True)

    metrics = list((output / "results").glob("shard-*/*.metrics.json"))
    status = {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "expected_cells": EXPECTED_CELLS,
        "observed_metrics": len(metrics),
        "expected_shards": shard_count,
        "completed_shards_this_invocation": completed,
        "skipped_shards_this_invocation": skipped,
        "failures": failures,
        "complete": not failures and len(metrics) == EXPECTED_CELLS,
    }
    temporary = output / ".status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "status.json")
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
