#!/usr/bin/env python3
"""Coordinate ULTRA-57 dataset staging over a shared filesystem."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ultra57_rfmlink as suite  # noqa: E402


STAGING_ROOT = suite.FAMILY_ROOT / "staging"
MARKER_ROOT = STAGING_ROOT / "datasets"
LOCK_PATH = STAGING_ROOT / ".state.lock"
PRODUCER_LOCK_PATH = STAGING_ROOT / ".producer.lock"
STATE_LOCK_STALE_SECONDS = 600
STATE_LOCK_WAIT_SECONDS = 60
REQUIRED_RUN_FILES = (
    "config.json",
    "metrics.json",
    "timings.json",
    "predictions.parquet",
    "context_manifest.json",
    "run.log",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def state_lock():
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + STATE_LOCK_WAIT_SECONDS
    while True:
        try:
            LOCK_PATH.mkdir()
            break
        except FileExistsError:
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > STATE_LOCK_STALE_SECONDS:
                stale = LOCK_PATH.with_name(
                    f"{LOCK_PATH.name}.stale.{platform.node()}.{os.getpid()}"
                )
                try:
                    LOCK_PATH.rename(stale)
                except (FileNotFoundError, OSError):
                    continue
                for child in stale.iterdir():
                    child.unlink(missing_ok=True)
                stale.rmdir()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for shared state lock {LOCK_PATH}")
            time.sleep(0.05)
    try:
        atomic_json(LOCK_PATH / "owner.json", {
            "host": platform.node(),
            "pid": os.getpid(),
            "acquired_utc": utc_now(),
        })
        yield
    finally:
        (LOCK_PATH / "owner.json").unlink(missing_ok=True)
        LOCK_PATH.rmdir()


def marker_path(slug: str) -> Path:
    if slug not in suite.BY_SLUG:
        raise ValueError(f"unknown ULTRA-57 dataset: {slug}")
    return MARKER_ROOT / f"{slug}.json"


def read_marker(slug: str) -> dict | None:
    path = marker_path(slug)
    return json.loads(path.read_text()) if path.exists() else None


def write_marker(slug: str, payload: dict) -> None:
    payload = {**payload, "dataset": slug, "updated_utc": utc_now()}
    atomic_json(marker_path(slug), payload)


def relative_cache_path(path: str | Path) -> str:
    candidate = Path(path).resolve()
    return str(candidate.relative_to(suite.OFFICIAL_CACHE.resolve()))


def artifact_manifest(dataset) -> list[dict]:
    paths = []
    for attribute in ("raw_paths", "processed_paths"):
        for raw_path in getattr(dataset, attribute, ()):
            path = Path(raw_path)
            if path.is_file():
                paths.append(path)
    unique = sorted(set(paths))
    if not unique:
        raise RuntimeError("official dataset exposed no raw or processed files")
    return [
        {
            "path": relative_cache_path(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in unique
    ]


def verify_artifacts(marker: dict) -> None:
    artifacts = marker.get("artifacts", [])
    if not artifacts:
        raise RuntimeError("READY marker has no artifact manifest")
    for artifact in artifacts:
        path = (suite.OFFICIAL_CACHE / artifact["path"]).resolve()
        path.relative_to(suite.OFFICIAL_CACHE.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")


def verify_canonical_bundle(marker: dict) -> None:
    bundle = marker.get("canonical_bundle")
    if not bundle or not bundle.get("artifacts"):
        raise RuntimeError("READY marker has no canonical preflight bundle")
    expected_policy = suite.json_sha256(suite.canonicalization_policy())
    if bundle.get("policy_sha256") != expected_policy:
        raise RuntimeError("canonical preflight policy mismatch")
    workspace = WORKSPACE_ROOT.resolve()
    for artifact in bundle["artifacts"]:
        path = (workspace / artifact["path"]).resolve()
        path.relative_to(workspace)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["bytes"]:
            raise RuntimeError(f"canonical size mismatch: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"canonical SHA-256 mismatch: {path}")


def preflight_one(slug: str) -> dict:
    bundle = suite.materialize_canonical_bundle(slug)
    verify_canonical_bundle({"canonical_bundle": bundle})
    return bundle


def stage_one(slug: str, attempt: int) -> None:
    with state_lock():
        write_marker(slug, {
            "status": "DOWNLOADING",
            "attempt": attempt,
            "producer_host": platform.node(),
            "producer_pid": os.getpid(),
            "started_utc": utc_now(),
        })
    dataset = suite.instantiate_official_dataset(slug)
    artifacts = artifact_manifest(dataset)
    canonical_bundle = preflight_one(slug)
    with state_lock():
        write_marker(slug, {
            "status": "READY",
            "attempt": attempt,
            "producer_host": platform.node(),
            "producer_pid": os.getpid(),
            "ready_utc": utc_now(),
            "class_name": suite.BY_SLUG[slug].class_name,
            "version": suite.BY_SLUG[slug].version,
            "artifacts": artifacts,
            "canonical_bundle": canonical_bundle,
            "total_bytes": sum(item["bytes"] for item in artifacts),
        })


def preflight_existing(recover_failed: bool = True,
                       only_slug: str | None = None) -> int:
    with state_lock():
        markers = {
            slug: read_marker(slug) for slug in suite.EXECUTION_ORDER
            if only_slug is None or slug == only_slug
        }
    audited = 0
    recovered = 0
    for slug, marker in markers.items():
        if not marker or not marker.get("artifacts"):
            continue
        verify_artifacts(marker)
        try:
            verify_canonical_bundle(marker)
            print(f"PREFLIGHT EXISTS dataset={slug}", flush=True)
            continue
        except (FileNotFoundError, RuntimeError):
            pass
        print(f"PREFLIGHT dataset={slug}", flush=True)
        bundle = preflight_one(slug)
        next_status = marker.get("status")
        history = list(marker.get("failure_history", []))
        if (
            recover_failed
            and next_status == "FAILED"
            and str(marker.get("error", "")).startswith("consumer stage failed: prepare")
        ):
            history.append({
                "failed_utc": marker.get("failed_utc", marker.get("updated_utc")),
                "error": marker.get("error"),
                "recovered_by": "canonical producer preflight",
            })
            next_status = "READY"
            recovered += 1
        with state_lock():
            current = read_marker(slug) or marker
            write_marker(slug, {
                **current,
                "status": next_status,
                "canonical_bundle": bundle,
                "preflight_utc": utc_now(),
                "failure_history": history,
                "error": None if next_status != "FAILED" else current.get("error"),
                "error_type": (
                    None if next_status != "FAILED" else current.get("error_type")
                ),
            })
        audited += 1
        print(f"PREFLIGHT READY dataset={slug} status={next_status}", flush=True)
    print(f"PREFLIGHT COMPLETE audited={audited} recovered={recovered}", flush=True)
    return 0


def reset_interrupted_downloads() -> None:
    with state_lock():
        for slug in suite.EXECUTION_ORDER:
            marker = read_marker(slug)
            if marker and marker.get("status") == "DOWNLOADING":
                write_marker(slug, {
                    **marker,
                    "status": "PENDING",
                    "recovered_after_producer_restart": True,
                })


def cleanup_done_artifacts() -> None:
    cache_root = suite.OFFICIAL_CACHE.resolve()
    with state_lock():
        candidates = [
            (slug, read_marker(slug)) for slug in suite.EXECUTION_ORDER
        ]
    for slug, marker in candidates:
        if not marker or marker.get("status") != "DONE" or marker.get("cache_cleaned"):
            continue
        for artifact in marker.get("artifacts", []):
            path = (cache_root / artifact["path"]).resolve()
            path.relative_to(cache_root)
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent != cache_root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        for artifact in marker.get("canonical_bundle", {}).get("artifacts", []):
            path = (WORKSPACE_ROOT / artifact["path"]).resolve()
            path.relative_to(WORKSPACE_ROOT.resolve())
            path.unlink(missing_ok=True)
        with state_lock():
            current = read_marker(slug) or marker
            write_marker(slug, {**current, "cache_cleaned": True})


def producer(window: int, max_attempts: int, poll_seconds: int,
             cleanup_done: bool) -> int:
    suite.validate_registration(require_masking_audit=False)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    producer_handle = PRODUCER_LOCK_PATH.open("a+")
    try:
        fcntl.flock(producer_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another staging producer is active", file=sys.stderr)
        return 2
    reset_interrupted_downloads()
    preflight_existing(recover_failed=True)
    attempts: dict[str, int] = {}
    while True:
        if cleanup_done:
            cleanup_done_artifacts()
        with state_lock():
            markers = {
                slug: read_marker(slug) for slug in suite.EXECUTION_ORDER
            }
        failed = [
            slug for slug, marker in markers.items()
            if marker and marker.get("status") == "FAILED"
        ]
        if failed:
            print(f"STOP: failed staging markers: {failed}", file=sys.stderr)
            return 3
        if all(
            marker and marker.get("status") == "DONE"
            for marker in markers.values()
        ):
            print("All ULTRA-57 datasets completed", flush=True)
            return 0
        ready = sum(
            bool(marker and marker.get("status") == "READY")
            for marker in markers.values()
        )
        if ready >= window:
            time.sleep(poll_seconds)
            continue
        candidate = next((
            slug for slug in suite.EXECUTION_ORDER
            if not markers[slug]
            or markers[slug].get("status") in ("PENDING",)
        ), None)
        if candidate is None:
            time.sleep(poll_seconds)
            continue
        attempt = attempts.get(candidate, 0) + 1
        attempts[candidate] = attempt
        print(f"STAGING dataset={candidate} attempt={attempt}", flush=True)
        try:
            stage_one(candidate, attempt)
            print(f"READY dataset={candidate}", flush=True)
        except Exception as error:
            detail = {
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "producer_host": platform.node(),
            }
            with state_lock():
                write_marker(candidate, {
                    **detail,
                    "status": "FAILED" if attempt >= max_attempts else "PENDING",
                })
            if attempt >= max_attempts:
                print(
                    f"STOP: dataset={candidate} failed after {attempt} attempts",
                    file=sys.stderr,
                )
                return 4
            time.sleep(min(300, poll_seconds * (2 ** (attempt - 1))))


def validate_completed_runs(slug: str) -> list[dict]:
    manifests = []
    for job in suite.primary_jobs(slug):
        run_dir = suite.base.run_dir(job)
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete run {run_dir}: missing {missing}")
        if (run_dir / "failure.json").exists():
            raise RuntimeError(f"failure marker remains in {run_dir}")
        manifests.append({
            "run": str(run_dir.relative_to(WORKSPACE_ROOT)),
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        })
    return manifests


def claim(slug: str, poll_seconds: int) -> int:
    suite.validate_registration()
    while True:
        with state_lock():
            marker = read_marker(slug)
        if marker and marker.get("status") == "FAILED":
            raise RuntimeError(f"staging failed for {slug}: {marker.get('error')}")
        if marker and marker.get("status") == "DONE":
            validate_completed_runs(slug)
            print(f"ALREADY_DONE dataset={slug}")
            return 0
        if marker and marker.get("status") in ("READY", "COMPUTING"):
            verify_artifacts(marker)
            verify_canonical_bundle(marker)
            with state_lock():
                current = read_marker(slug) or marker
                write_marker(slug, {
                    **current,
                    "status": "COMPUTING",
                    "consumer_host": platform.node(),
                    "consumer_pid": os.getpid(),
                    "claimed_utc": utc_now(),
                })
            print(f"CLAIMED dataset={slug}")
            return 0
        print(
            f"WAITING dataset={slug} status="
            f"{marker.get('status') if marker else 'PENDING'}",
            flush=True,
        )
        time.sleep(poll_seconds)


def complete(slug: str) -> int:
    run_manifests = validate_completed_runs(slug)
    with state_lock():
        marker = read_marker(slug)
        if not marker or marker.get("status") not in ("COMPUTING", "DONE"):
            raise RuntimeError(f"cannot complete {slug} from marker {marker}")
        write_marker(slug, {
            **marker,
            "status": "DONE",
            "completed_utc": utc_now(),
            "run_count": len(run_manifests),
            "runs": run_manifests,
        })
    print(f"DONE dataset={slug} runs={len(run_manifests)}")
    return 0


def fail(slug: str, message: str) -> int:
    with state_lock():
        marker = read_marker(slug) or {}
        write_marker(slug, {
            **marker,
            "status": "FAILED",
            "consumer_host": platform.node(),
            "consumer_pid": os.getpid(),
            "error": message,
            "failed_utc": utc_now(),
        })
    print(f"FAILED dataset={slug} error={message}")
    return 0


def reset(slug: str) -> int:
    with state_lock():
        marker = read_marker(slug)
    if not marker or marker.get("status") != "FAILED":
        raise RuntimeError(f"reset requires a FAILED marker for {slug}: {marker}")
    history = list(marker.get("failure_history", []))
    history.append({
        "failed_utc": marker.get("failed_utc", marker.get("updated_utc")),
        "error": marker.get("error"),
        "error_type": marker.get("error_type"),
    })
    if marker.get("artifacts"):
        verify_artifacts(marker)
        next_status = "READY"
    else:
        next_status = "PENDING"
    with state_lock():
        current = read_marker(slug) or marker
        write_marker(slug, {
            **current,
            "status": next_status,
            "failure_history": history,
            "reset_utc": utc_now(),
            "error": None,
            "error_type": None,
            "traceback": None,
        })
    print(f"RESET dataset={slug} status={next_status}")
    return 0


def status() -> int:
    with state_lock():
        markers = {
            slug: read_marker(slug) for slug in suite.EXECUTION_ORDER
        }
    counts: dict[str, int] = {}
    for marker in markers.values():
        state = marker.get("status") if marker else "PENDING"
        counts[state] = counts.get(state, 0) + 1
    print(json.dumps({"counts": counts, "datasets": markers}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--producer", action="store_true")
    actions.add_argument("--claim", action="store_true")
    actions.add_argument("--complete", action="store_true")
    actions.add_argument("--fail", action="store_true")
    actions.add_argument("--reset", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--preflight-existing", action="store_true")
    parser.add_argument("--dataset", choices=suite.DATASETS)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--message", default="unspecified consumer failure")
    parser.add_argument("--cleanup-done", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.max_attempts < 1 or args.poll_seconds < 1:
        parser.error("window, max-attempts, and poll-seconds must be positive")
    if args.producer:
        return producer(
            args.window, args.max_attempts, args.poll_seconds, args.cleanup_done
        )
    if args.status:
        return status()
    if args.preflight_existing:
        return preflight_existing(recover_failed=True, only_slug=args.dataset)
    if not args.dataset:
        parser.error("--dataset is required for claim/complete/fail/reset")
    if args.claim:
        return claim(args.dataset, args.poll_seconds)
    if args.complete:
        return complete(args.dataset)
    if args.reset:
        return reset(args.dataset)
    return fail(args.dataset, args.message)


if __name__ == "__main__":
    raise SystemExit(main())
