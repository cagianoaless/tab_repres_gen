from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _import_check(module_name: str, required: bool) -> Check:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        return Check(
            name=f"import:{module_name}",
            passed=not required,
            detail=f"missing ({error.name}); required={required}",
        )
    version = getattr(module, "__version__", "unknown")
    return Check(f"import:{module_name}", True, f"version={version}")


def run_preflight(
    repo_root: Path,
    require_gpu: bool,
    require_github_push: bool,
) -> list[Check]:
    checks = [
        Check(
            "python",
            (3, 11) <= sys.version_info[:2] < (3, 13),
            sys.version.split()[0],
        ),
        Check("git", shutil.which("git") is not None, shutil.which("git") or "not found"),
        Check("repo:pyproject", (repo_root / "pyproject.toml").is_file(), str(repo_root)),
        _import_check("numpy", True),
        _import_check("pandas", True),
        _import_check("sklearn", True),
        _import_check("networkx", True),
        _import_check("yaml", True),
        _import_check("catboost", False),
        _import_check("tabpfn", False),
    ]

    github_repository = os.environ.get("RTFM_GITHUB_REPOSITORY", "").strip()
    github_token_present = bool(os.environ.get("GITHUB_TOKEN", "").strip())
    checks.append(
        Check(
            "github-repository",
            bool(github_repository) or not require_github_push,
            github_repository or "RTFM_GITHUB_REPOSITORY is unset",
        )
    )
    checks.append(
        Check(
            "github-token",
            github_token_present or not require_github_push,
            "present" if github_token_present else "GITHUB_TOKEN is unset",
        )
    )
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        minimum_memory = int(os.environ.get("RTFM_MIN_GPU_MEMORY_BYTES", "0"))
        memories = [
            int(torch.cuda.get_device_properties(index).total_memory)
            for index in range(device_count)
        ]
        gpu_passed = (
            cuda_available
            and device_count == 1
            and memories
            and memories[0] >= minimum_memory
        )
        detail = (
            f"available={cuda_available}, count={device_count}, "
            f"memory_bytes={memories}, minimum={minimum_memory}"
        )
    except ModuleNotFoundError:
        gpu_passed = False
        detail = "torch is not installed"
    checks.append(Check("gpu", gpu_passed or not require_gpu, detail))
    return checks
