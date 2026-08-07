#!/usr/bin/env python3
"""Verify publication completeness, manifest integrity, and anonymity checks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    re.compile(r"alessandro|cagiano|molfetta|lorenzo", re.IGNORECASE),
    re.compile(r"/Users/|/home/|172\.20\.|github\.com/cagiano", re.IGNORECASE),
    re.compile(r"overleaf", re.IGNORECASE),
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml", ".toml"}


def fail(message: str) -> None:
    raise AssertionError(message)


def verify_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        fail("MANIFEST.sha256 is missing")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            fail(f"manifest path is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            fail(f"checksum mismatch: {relative}")


def verify_anonymity() -> None:
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.name in {"MANIFEST.sha256", "verify_artifact.py"}
            or path.suffix not in TEXT_SUFFIXES
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                fail(f"anonymity-sensitive string in {path.relative_to(ROOT)}")


def csv_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_scientific_inventory() -> None:
    required = (
        "experiments/clutrr/results/ordered-interference/paired_comparisons.csv",
        "experiments/graphlog/results/decoder-sensitivity/primary_contrasts_by_world.csv",
        "experiments/dihedral16/results/cpu/aggregate.csv",
        "experiments/dihedral16/results/tabpfn/aggregate.csv",
        "experiments/ultra4g/results/q64-encoder-ladder/dataset_metrics.csv",
        "experiments/ultra4g/results/context-robustness/dataset_metrics.csv",
        "experiments/ultra4g/results/logistic-ranking-correction/query_evidence_strata.csv",
        "publication_data/context_robustness_summary.csv",
        "publication_data/query_evidence_absolute.csv",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            fail(f"required publication file is missing: {relative}")

    context_audit = json.loads(
        (ROOT / "experiments/ultra4g/results/context-robustness/audit.json").read_text(encoding="utf-8")
    )
    expected = {"status": "PASS", "cells": 4104, "datasets": 57, "provenance_families": 16}
    for key, value in expected.items():
        if context_audit.get(key) != value:
            fail(f"context robustness {key}: expected {value!r}, found {context_audit.get(key)!r}")

    graphlog = csv_rows("experiments/graphlog/results/decoder-sensitivity/primary_contrasts_by_world.csv")
    if len({row["world"] for row in graphlog}) != 57:
        fail("GraphLog table does not cover 57 worlds")

    ladder = csv_rows("experiments/ultra4g/results/q64-encoder-ladder/dataset_metrics.csv")
    dataset_key = "dataset" if "dataset" in ladder[0] else "dataset_id"
    if len({row[dataset_key] for row in ladder}) != 57:
        fail("ultra_4g ladder does not cover 57 datasets")

    provenance = json.loads((ROOT / "publication_data/context_robustness_provenance.json").read_text())
    if provenance.get("analysis_status") != "complete" or provenance.get("missing_datasets"):
        fail("publication context summary is not the complete, non-imputed analysis")


def main() -> int:
    verify_manifest()
    verify_anonymity()
    verify_scientific_inventory()
    print("Artifact verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Artifact verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
