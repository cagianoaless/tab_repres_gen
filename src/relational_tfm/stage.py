from __future__ import annotations

import gzip
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from relational_tfm.hashing import file_sha256


class StagingError(RuntimeError):
    """Raised when an external asset cannot be staged exactly as registered."""


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise StagingError(f"unsafe path in {archive}: {member.filename}")
        handle.extractall(destination)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _stage_http(repo_root: Path, asset: dict[str, Any], force: bool) -> dict[str, str]:
    target = repo_root / asset["target"]
    expected = asset["sha256"]
    if target.exists() and not force:
        actual = file_sha256(target)
        if actual == expected:
            status = "VERIFIED"
        else:
            raise StagingError(
                f"existing asset has wrong checksum: {target}; expected {expected}, got {actual}"
            )
    else:
        _download(asset["url"], target)
        actual = file_sha256(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise StagingError(
                f"download checksum mismatch for {asset['asset_id']}: "
                f"expected {expected}, got {actual}"
            )
        status = "DOWNLOADED"

    unpack = asset.get("unpack")
    if unpack:
        destination = repo_root / asset["unpack_target"]
        if force and destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if unpack == "zip":
            _safe_extract_zip(target, destination)
        elif unpack == "gzip":
            output = destination / asset["unpacked_name"]
            with gzip.open(target, "rb") as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink)
        else:
            raise StagingError(f"unsupported unpack format: {unpack}")
    return {"asset_id": asset["asset_id"], "status": status, "target": str(target)}


def _git_output(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise StagingError(f"git command failed: {detail}") from error
    return result.stdout.strip()


def _stage_git(repo_root: Path, asset: dict[str, Any], force: bool) -> dict[str, str]:
    target = repo_root / asset["target"]
    expected_commit = asset["commit"]
    if target.exists() and force:
        shutil.rmtree(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _git_output("clone", "--filter=blob:none", "--no-checkout", asset["url"], str(target))
        status = "CLONED"
    else:
        status = "VERIFIED"
    if not (target / ".git").exists():
        raise StagingError(f"git asset target is not a repository: {target}")
    try:
        _git_output("cat-file", "-e", f"{expected_commit}^{{commit}}", cwd=target)
    except StagingError:
        _git_output("fetch", "origin", expected_commit, cwd=target)
    _git_output("checkout", "--detach", expected_commit, cwd=target)
    actual = _git_output("rev-parse", "HEAD", cwd=target)
    if actual != expected_commit:
        raise StagingError(
            f"git commit mismatch for {target}: expected {expected_commit}, got {actual}"
        )
    return {"asset_id": asset["asset_id"], "status": status, "target": str(target)}


def stage_manifest(
    repo_root: Path, manifest_path: Path, selected: set[str], force: bool
) -> list[dict[str, str]]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), list):
        raise StagingError("invalid external asset manifest")
    known = {asset.get("asset_id") for asset in payload["assets"]}
    unknown = selected - known
    if unknown:
        raise StagingError(f"unknown assets requested: {sorted(unknown)}")

    results = []
    for asset in payload["assets"]:
        if selected and asset["asset_id"] not in selected:
            continue
        kind = asset.get("kind")
        if kind == "http":
            results.append(_stage_http(repo_root, asset, force))
        elif kind == "git":
            results.append(_stage_git(repo_root, asset, force))
        else:
            raise StagingError(f"unsupported asset kind for {asset.get('asset_id')}: {kind}")
    return results
