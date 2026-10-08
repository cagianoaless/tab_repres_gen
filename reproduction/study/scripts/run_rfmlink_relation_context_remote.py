#!/usr/bin/env python3
"""Prepare and run distributed relation-context study shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_CELLS = {"cpu": 2_736, "tabpfn": 1_368}
INTERFACES = {
    "b0_local_only", "b0_local_no_relation_id", "b0_local_ultra",
    "b0_local_ultra_no_relation_id",
}
CONTEXT_MODES = {
    "frequency_random_filtered_uniform", "relation_matched_filtered_uniform",
}


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(
    *, repo_root: Path, asset_root: Path, profile: str,
    cache_root: Path | None, threads_per_worker: int,
) -> dict[str, str]:
    output = os.environ.copy()
    output.update({
        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
        "RTFM_DEVICE": "cuda" if profile == "tabpfn" else "cpu",
        "RTFM_FROZEN_KG_ASSET_ROOT": str(asset_root),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "RTFM_CPU_THREADS": str(threads_per_worker),
    })
    if profile == "tabpfn":
        if cache_root is None:
            raise ValueError("TabPFN workers require a model cache")
        output.update({
            "TABPFN_MODEL_CACHE_DIR": str(cache_root),
            "TABPFN_NO_BROWSER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        })
    return output


def prepare(
    *, repo_root: Path, asset_root: Path, output_root: Path, python: Path,
    profile: str, cache_root: Path | None, cells_per_shard: int,
    threads_per_worker: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config = output_root / f"rfmlink_relation_context_{profile}.yaml"
    plan = output_root / "plan.json"
    env = environment(
        repo_root=repo_root, asset_root=asset_root, profile=profile,
        cache_root=cache_root, threads_per_worker=threads_per_worker,
    )
    subprocess.run([
        str(python), "scripts/generate_rfmlink_relation_context_registration.py",
        str(asset_root / "manifest.json"), str(config), "--profile", profile,
    ], cwd=repo_root, env=env, check=True)
    cli = [str(python), "-m", "relational_tfm.cli"]
    subprocess.run(
        [*cli, "plan", "--config", str(config), "--output", str(plan)],
        cwd=repo_root, env=env, check=True,
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    cells = payload["cells"]
    axes = {
        "dataset": {cell["dataset"] for cell in cells},
        "decoder": {cell["decoder"] for cell in cells},
        "interface": {cell["interface"] for cell in cells},
        "context_mode": {cell["context_mode"] for cell in cells},
        "seed": {int(cell["seed"]) for cell in cells},
        "context": {int(cell["context_per_class"]) for cell in cells},
    }
    expected = EXPECTED_CELLS[profile]
    if (
        len(cells) != expected or len(axes["dataset"]) != 57
        or axes["interface"] != INTERFACES or axes["context_mode"] != CONTEXT_MODES
        or axes["seed"] != {101, 102, 103} or axes["context"] != {200}
        or axes["decoder"] != ({"tabpfn"} if profile == "tabpfn" else {"logistic", "catboost"})
    ):
        raise RuntimeError(f"relation-context plan drift: cells={len(cells)}, axes={axes}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    registration = {
        "schema_version": 1,
        "scientific_role": (
            "post-hoc fixed-panel factorial; positive context sampling and relation-ID "
            "availability are crossed while q64 evaluation remains fixed"
        ),
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "asset_manifest_sha256": file_sha256(asset_root / "manifest.json"),
        "config_sha256": file_sha256(config),
        "plan_sha256": payload["plan_sha256"],
        "profile": profile,
        "cells": expected,
        "cells_per_shard": cells_per_shard,
        "shards": (expected + cells_per_shard - 1) // cells_per_shard,
        "axes": {key: sorted(value) for key, value in axes.items()},
        "threads_per_worker": threads_per_worker,
    }
    path = output_root / "registration.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "source_commit", "asset_manifest_sha256", "config_sha256", "plan_sha256",
            "profile", "cells", "cells_per_shard", "shards", "axes",
        ):
            if previous.get(key) != registration[key]:
                raise RuntimeError(f"registration drift at {key}")
    else:
        atomic_json(path, registration)


def run_worker(
    *, repo_root: Path, asset_root: Path, output_root: Path, python: Path,
    profile: str, cache_root: Path | None, cells_per_shard: int,
    threads_per_worker: int, worker_index: int, worker_count: int,
) -> None:
    registration = json.loads(
        (output_root / "registration.json").read_text(encoding="utf-8")
    )
    if registration["profile"] != profile:
        raise ValueError("worker profile differs from registration")
    if registration["cells_per_shard"] != cells_per_shard:
        raise ValueError("worker cells-per-shard differs from registration")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker index must be within worker count")
    env = environment(
        repo_root=repo_root, asset_root=asset_root, profile=profile,
        cache_root=cache_root, threads_per_worker=threads_per_worker,
    )
    cli = [str(python), "-m", "relational_tfm.cli"]
    assigned = list(range(worker_index, int(registration["shards"]), worker_count))
    status_path = output_root / f"status.worker-{worker_index}-of-{worker_count}.json"
    completed = 0
    for shard_index in assigned:
        shard_dir = output_root / "results" / f"shard-{shard_index:04d}"
        verify = [*cli, "verify-shard", "--directory", str(shard_dir)]
        if subprocess.run(verify, cwd=repo_root, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0:
            completed += 1
            continue
        batch_sizes = (20_000, 10_000, 5_000) if profile == "tabpfn" else (0,)
        success = False
        for attempt, batch_size in enumerate(batch_sizes, start=1):
            attempt_env = dict(env)
            attempt_env["RTFM_RESULT_ATTEMPT"] = str(attempt)
            if batch_size:
                attempt_env["RTFM_TABPFN_PREDICTION_BATCH_SIZE"] = str(batch_size)
            command = [
                *cli, "run-shard", "--plan", str(output_root / "plan.json"),
                "--shard-index", str(shard_index), "--cells-per-shard",
                str(cells_per_shard), "--output-dir", str(shard_dir),
            ]
            log = output_root / "logs" / f"shard-{shard_index:04d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{datetime.now(UTC).isoformat()}] {' '.join(command)}\n")
                handle.flush()
                result = subprocess.run(
                    command, cwd=repo_root, env=attempt_env,
                    stdout=handle, stderr=subprocess.STDOUT, check=False,
                )
                handle.write(f"[exit={result.returncode}]\n")
            if result.returncode == 0 and subprocess.run(
                verify, cwd=repo_root, env=attempt_env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False,
            ).returncode == 0:
                success = True
                break
            time.sleep(5)
        if not success:
            atomic_json(status_path, {
                "status": "FAILED", "failed_shard": shard_index,
                "completed_shards": completed, "assigned_shards": len(assigned),
            })
            raise SystemExit(1)
        completed += 1
        atomic_json(status_path, {
            "status": "RUNNING" if completed < len(assigned) else "COMPLETE",
            "profile": profile, "worker_index": worker_index,
            "worker_count": worker_count, "completed_shards": completed,
            "assigned_shards": len(assigned), "suite_shards": registration["shards"],
            "updated_at_utc": datetime.now(UTC).isoformat(),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--profile", choices=("cpu", "tabpfn"), required=True)
    parser.add_argument("--tabpfn-cache", type=Path)
    parser.add_argument("--cells-per-shard", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    values = {
        "repo_root": args.repo_root.resolve(), "asset_root": args.asset_root.resolve(),
        "output_root": args.output_root.resolve(), "python": args.python.resolve(),
        "profile": args.profile,
        "cache_root": args.tabpfn_cache.resolve() if args.tabpfn_cache else None,
        "cells_per_shard": args.cells_per_shard,
        "threads_per_worker": args.threads_per_worker,
    }
    if args.prepare_only:
        prepare(**values)
    else:
        run_worker(**values, worker_index=args.worker_index, worker_count=args.worker_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
