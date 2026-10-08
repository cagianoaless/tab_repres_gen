from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import verify_artifact


def test_manifest_accepts_public_author_names_and_checks_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify_artifact, "ROOT", tmp_path)
    payload = b"Public authors: Alessandro Cagiano; https://github.com/cagiano/\n"
    path = tmp_path / "README.md"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "MANIFEST.sha256").write_text(f"{digest}  README.md\n")
    assert verify_artifact.verify_manifest() == [path]
    path.write_bytes(b"changed release bytes")
    with pytest.raises(AssertionError, match="checksum mismatch"):
        verify_artifact.verify_manifest()


@pytest.mark.parametrize(
    "line",
    [
        "not-a-manifest-line",
        "a" * 64 + "  ../outside",
        "a" * 64 + "  /absolute",
        "wrong-digest  README.md",
        ("a" * 64 + "  README.md\n") * 2,
    ],
)
def test_manifest_rejects_malformed_unsafe_duplicate_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line: str,
) -> None:
    monkeypatch.setattr(verify_artifact, "ROOT", tmp_path)
    (tmp_path / "MANIFEST.sha256").write_text(line + "\n")
    with pytest.raises(AssertionError):
        verify_artifact.manifest_entries()


def test_manifest_refresh_cannot_hide_changed_historical_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify_artifact, "ROOT", tmp_path)
    (tmp_path / "tools").mkdir()
    path = tmp_path / "historical.csv"
    initial = b"effect\n0.1\n"
    snapshot = {"files": {"historical.csv": hashlib.sha256(initial).hexdigest()}}
    (tmp_path / "tools/historical_results.json").write_text(json.dumps(snapshot))
    changed = b"effect\n0.9\n"
    path.write_bytes(changed)
    (tmp_path / "MANIFEST.sha256").write_text(
        f"{hashlib.sha256(changed).hexdigest()}  historical.csv\n"
    )
    verify_artifact.verify_manifest()  # release checksum alone is not enough
    with pytest.raises(AssertionError, match="immutable historical evidence changed"):
        verify_artifact.verify_historical_evidence()


def test_real_primary_endpoints_and_historical_inventory_are_unchanged() -> None:
    verify_artifact.verify_historical_evidence()
    verify_artifact.verify_primary_endpoints()
    verify_artifact.verify_scientific_inventory()
