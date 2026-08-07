from __future__ import annotations

import importlib.metadata
import os
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from relational_tfm.hashing import file_sha256, object_sha256

PACKAGE_NAMES = (
    "relational-tfm-study",
    "numpy",
    "pandas",
    "scikit-learn",
    "networkx",
    "scipy",
    "catboost",
    "tabpfn",
    "torch",
    "kubejobs",
)

# This allowlist intentionally excludes credentials and authentication variables.
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HOSTNAME",
    "POD_NAME",
    "POD_NAMESPACE",
    "TABPFN_MODEL_CACHE_DIR",
    "TABPFN_NO_BROWSER",
    "RTFM_CONTAINER_IMAGE",
    "RTFM_CPU_THREADS",
    "RTFM_DEVICE",
    "RTFM_GITHUB_REPOSITORY",
    "RTFM_GPU_CELL_WORKERS",
    "RTFM_MIN_GPU_MEMORY_BYTES",
    "RTFM_IMAGE_COMMIT",
    "RTFM_IMAGE_DIGEST",
    "RTFM_RUNTIME_REQUIREMENTS_SHA256",
    "RTFM_RESULT_ATTEMPT",
    "RTFM_SOURCE_COMMIT",
)


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def repository_state(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _cuda_state() -> dict[str, Any] | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def reset_peak_resources() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        return


def peak_resources() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        max_rss *= 1024

    cuda_devices = []
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
    return {
        "process_max_rss_bytes": max_rss,
        "cpu_user_seconds": usage.ru_utime,
        "cpu_system_seconds": usage.ru_stime,
        "cuda_devices": cuda_devices,
    }


def runtime_state(repo_root: Path, plan_path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "git": repository_state(repo_root),
        "packages": _package_versions(),
        "environment": {
            key: os.environ[key]
            for key in ENVIRONMENT_KEYS
            if key in os.environ and os.environ[key]
        },
        "cuda": _cuda_state(),
    }
    if plan_path is not None:
        payload["plan_path"] = str(plan_path)
        payload["plan_file_sha256"] = file_sha256(plan_path)
    for filename in ("pyproject.toml", "uv.lock"):
        path = repo_root / filename
        if path.is_file():
            payload.setdefault("environment_files", {})[filename] = file_sha256(path)
    return payload


def dump_runtime_state(repo_root: Path, output: Path, plan_path: Path | None = None) -> None:
    from relational_tfm.plan import atomic_write_json

    payload = runtime_state(repo_root, plan_path)
    payload["runtime_state_sha256"] = object_sha256(payload)
    atomic_write_json(output, payload)
