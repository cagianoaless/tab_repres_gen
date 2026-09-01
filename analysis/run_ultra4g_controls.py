#!/usr/bin/env python3
"""Prepare and execute resumable shards for the matched KG protocol controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(SCRIPT_DIR), str(REPO_ROOT)]

from generate_ultra4g_control_registration import (  # noqa: E402
    PROFILES,
    generate_config,
)

CELLS_PER_DATASET = {
    "matched-filter-tabpfn": 18,
    "query-relative-tabpfn": 27,
    "query-relative-cpu": 54,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def environment(
    repo_root: Path, asset_root: Path, cache_root: Path, profile: str
) -> dict[str, str]:
    output = os.environ.copy()
    output.update(
        {
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root / 'analysis'}:{repo_root}",
            "RTFM_FROZEN_KG_ASSET_ROOT": str(asset_root),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    if PROFILES[profile]["resource_class"] == "gpu":
        output.update(
            {
                "RTFM_DEVICE": "cuda",
                "TABPFN_MODEL_CACHE_DIR": str(cache_root),
                "TABPFN_NO_BROWSER": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "RTFM_TABPFN_PREDICTION_BATCH_SIZE": "10000",
            }
        )
    return output


def prepare(
    *,
    repo_root: Path,
    asset_root: Path,
    output_root: Path,
    python: Path,
    cache_root: Path,
    profile: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config = output_root / "study.yaml"
    plan = output_root / "plan.json"
    generate_config(asset_root / "manifest.json", config, profile=profile)
    env = environment(repo_root, asset_root, cache_root, profile)
    subprocess.run(
        [
            str(python),
            "-m",
            "relational_tfm.cli",
            "plan",
            "--config",
            str(config),
            "--output",
            str(plan),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    expected = int(PROFILES[profile]["expected_cells"])
    if len(payload["cells"]) != expected:
        raise RuntimeError(f"expected {expected} cells, found {len(payload['cells'])}")
    cells_per_shard = CELLS_PER_DATASET[profile]
    if expected % cells_per_shard:
        raise RuntimeError("one-dataset shard size does not divide the plan")
    registration = {
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "profile": profile,
        "asset_manifest_sha256": file_sha256(asset_root / "manifest.json"),
        "config_sha256": file_sha256(config),
        "plan_sha256": payload["plan_sha256"],
        "cells": expected,
        "cells_per_shard": cells_per_shard,
        "shards": expected // cells_per_shard,
        "source_hashes": {
            path: file_sha256(repo_root / path)
            for path in (
                "src/relational_tfm/evaluation/query_relative.py",
                "src/relational_tfm/datasets/rfmlink.py",
                "src/relational_tfm/runner.py",
                "analysis/generate_ultra4g_control_registration.py",
                "analysis/run_ultra4g_controls.py",
            )
        },
    }
    registration_path = output_root / "registration.json"
    if registration_path.exists():
        previous = json.loads(registration_path.read_text(encoding="utf-8"))
        for key in (
            "profile",
            "asset_manifest_sha256",
            "config_sha256",
            "plan_sha256",
            "cells",
            "cells_per_shard",
            "source_hashes",
        ):
            if previous.get(key) != registration[key]:
                raise RuntimeError(f"registration drift at {key}")
    else:
        atomic_json(registration_path, registration)


def run_worker(
    *,
    repo_root: Path,
    asset_root: Path,
    output_root: Path,
    python: Path,
    cache_root: Path,
    profile: str,
    worker_index: int,
    worker_count: int,
) -> None:
    registration = json.loads(
        (output_root / "registration.json").read_text(encoding="utf-8")
    )
    if registration["profile"] != profile:
        raise ValueError("worker profile differs from registration")
    env = environment(repo_root, asset_root, cache_root, profile)
    cli = [str(python), "-m", "relational_tfm.cli"]
    completed = 0
    failures: list[dict[str, Any]] = []
    for shard in range(worker_index, int(registration["shards"]), worker_count):
        shard_dir = output_root / "results" / f"shard-{shard:04d}"
        verify = [*cli, "verify-shard", "--directory", str(shard_dir)]
        if subprocess.run(
            verify,
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            completed += 1
            continue
        command = [
            *cli,
            "run-shard",
            "--plan",
            str(output_root / "plan.json"),
            "--shard-index",
            str(shard),
            "--cells-per-shard",
            str(registration["cells_per_shard"]),
            "--output-dir",
            str(shard_dir),
        ]
        log_path = output_root / "logs" / f"shard-{shard:04d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            failures.append({"shard": shard, "returncode": result.returncode})
            atomic_json(
                output_root / f"status.worker-{worker_index}-of-{worker_count}.json",
                {"completed": completed, "failures": failures, "active_shard": shard},
            )
            raise RuntimeError(f"shard {shard} failed")
        completed += 1
        atomic_json(
            output_root / f"status.worker-{worker_index}-of-{worker_count}.json",
            {"completed": completed, "failures": failures, "active_shard": None},
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    kwargs = {
        "repo_root": args.repo_root.resolve(),
        "asset_root": args.asset_root.resolve(),
        "output_root": args.output_root.resolve(),
        "python": args.python.resolve(),
        "cache_root": args.cache_root.resolve(),
        "profile": args.profile,
    }
    if args.prepare:
        prepare(**kwargs)
    else:
        run_worker(
            **kwargs,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
