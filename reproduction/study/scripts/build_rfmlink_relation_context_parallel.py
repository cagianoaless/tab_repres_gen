#!/usr/bin/env python3
"""Build missing relation-context dataset archives with bounded CPU parallelism."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json
from scripts.build_rfmlink_relation_context_asset import build_dataset


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(output_root: Path, dataset: str) -> bool:
    status_path = output_root / f"{dataset}.status.json"
    archive_path = output_root / f"{dataset}.npz"
    if not status_path.is_file() or not archive_path.is_file():
        return False
    status = _load_json(status_path)
    return (
        status.get("status") == "ok"
        and status.get("archive_sha256") == file_sha256(archive_path)
    )


def _one(
    dataset: str,
    reference_root: Path,
    q512_root: Path,
    b2v2_root: Path,
    output_root: Path,
    reference_manifest: dict[str, Any],
) -> dict[str, Any]:
    try:
        status = build_dataset(
            dataset=dataset,
            reference_root=reference_root,
            q512_root=q512_root,
            b2v2_root=b2v2_root,
            output_root=output_root,
            reference_manifest=reference_manifest,
        )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        status = {
            "status": "failed",
            "dataset": dataset,
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_write_json(output_root / f"{dataset}.status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--q512-root", type=Path, required=True)
    parser.add_argument("--b2v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    roots = {
        "reference_root": args.reference_root.resolve(),
        "q512_root": args.q512_root.resolve(),
        "b2v2_root": args.b2v2_root.resolve(),
        "output_root": args.output_root.resolve(),
    }
    manifest = _load_json(roots["reference_root"] / "manifest.json")
    missing = [
        dataset
        for dataset in sorted(manifest["datasets"])
        if not _complete(roots["output_root"], dataset)
    ]
    print(json.dumps({"complete": 57 - len(missing), "missing": len(missing)}), flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_one, dataset, **roots, reference_manifest=manifest): dataset
            for dataset in missing
        }
        for future in as_completed(futures):
            status = future.result()
            print(json.dumps(status, sort_keys=True), flush=True)
            failed += int(status.get("status") != "ok")
    print(json.dumps({"built": len(missing), "failed": failed}), flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
