#!/usr/bin/env python3
"""Verify publication completeness, manifest integrity, and anonymity checks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
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


def manifest_entries() -> list[tuple[str, Path]]:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        fail("MANIFEST.sha256 is missing")
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            fail(f"malformed manifest line {line_number}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            fail(f"unsafe manifest path: {relative}")
        if path in seen:
            fail(f"duplicate manifest path: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"invalid checksum on manifest line {line_number}")
        seen.add(path)
        entries.append((digest, path))
    return entries


def verify_manifest() -> list[Path]:
    entries = manifest_entries()
    for digest, relative in entries:
        path = ROOT / relative
        if not path.is_file():
            fail(f"manifest path is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            fail(f"checksum mismatch: {relative}")

    if (ROOT / ".git").is_dir():
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        tracked = {
            Path(value.decode("utf-8"))
            for value in completed.stdout.split(b"\0")
            if value and value != b"MANIFEST.sha256"
        }
        listed = {relative for _, relative in entries}
        if listed != tracked:
            missing = sorted(map(str, tracked - listed))
            extra = sorted(map(str, listed - tracked))
            fail(f"manifest coverage differs from Git: missing={missing}, extra={extra}")
    return [ROOT / relative for _, relative in entries]


def verify_anonymity(paths: list[Path]) -> None:
    for path in paths:
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
        "experiments/clutrr/results/shortcut-diagnostic/cells.csv",
        "experiments/clutrr/results/collision-conditioned-ceilings/summary.json",
        "experiments/graphlog/results/decoder-sensitivity/primary_contrasts_by_world.csv",
        "experiments/graphlog/results/capacity-sweep/capacity_curve.csv",
        "experiments/graphlog/results/capacity-sweep/summary.json",
        "experiments/dihedral16/results/cpu/aggregate.csv",
        "experiments/dihedral16/results/cpu/primary_seed_contrasts.csv",
        "experiments/dihedral16/results/tabpfn/aggregate.csv",
        "experiments/dihedral16/results/tabpfn/primary_seed_contrasts.csv",
        "experiments/ultra4g/results/q64-encoder-ladder/dataset_metrics.csv",
        "experiments/ultra4g/results/context-robustness/dataset_metrics.csv",
        "experiments/ultra4g/results/logistic-ranking-correction/query_evidence_strata.csv",
        "experiments/ultra4g/results/pointwise-and-filtering-controls/summary.json",
        "experiments/ultra4g/results/pointwise-and-filtering-controls/query_transform_contrasts.csv",
        "experiments/ultra4g/results/pointwise-and-filtering-controls/matched_filter_contrasts.csv",
        "experiments/ultra4g/results/aggregation-sensitivity/summary.json",
        "publication_data/context_robustness_summary.csv",
        "publication_data/query_evidence_absolute.csv",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            fail(f"required publication file is missing: {relative}")

    context_audit = json.loads(
        (ROOT / "experiments/ultra4g/results/context-robustness/audit.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {"status": "PASS", "cells": 4104, "datasets": 57, "provenance_families": 16}
    for key, value in expected.items():
        if context_audit.get(key) != value:
            fail(f"context robustness {key}: expected {value!r}, found {context_audit.get(key)!r}")

    graphlog = csv_rows(
        "experiments/graphlog/results/decoder-sensitivity/primary_contrasts_by_world.csv"
    )
    if len({row["world"] for row in graphlog}) != 57:
        fail("GraphLog table does not cover 57 worlds")

    ladder = csv_rows("experiments/ultra4g/results/q64-encoder-ladder/dataset_metrics.csv")
    dataset_key = "dataset" if "dataset" in ladder[0] else "dataset_id"
    if len({row[dataset_key] for row in ladder}) != 57:
        fail("ultra_4g ladder does not cover 57 datasets")

    collision = json.loads(
        (
            ROOT
            / "experiments/clutrr/results/collision-conditioned-ceilings/summary.json"
        ).read_text(encoding="utf-8")
    )
    collision_by_encoder = {row["encoder"]: row for row in collision["encoders"]}
    multiset = collision_by_encoder["relation_multiset"]["ood_equal_length"]
    if not 0.756 < multiset["singleton_fraction"] < 0.757:
        fail("CLUTRR multiset singleton mass changed")
    if not 0.711 < multiset["nonsingleton_accuracy_ceiling"] < 0.712:
        fail("CLUTRR conditional non-singleton ceiling changed")
    absolute = collision_by_encoder["ordered_typed_path"]["ood_equal_length"]
    if absolute["singleton_fraction"] != 1.0:
        fail("CLUTRR absolute-position endpoint is no longer injective")
    if absolute["nonsingleton_accuracy_ceiling"] is not None:
        fail("CLUTRR injective endpoint has a non-singleton ceiling")

    capacity = json.loads(
        (ROOT / "experiments/graphlog/results/capacity-sweep/summary.json").read_text(
            encoding="utf-8"
        )
    )
    if capacity.get("cells") != 2394 or capacity.get("worlds") != 57:
        fail("GraphLog capacity sweep is incomplete")
    if capacity.get("depths") != list(range(1, 8)):
        fail("GraphLog capacity sweep does not cover depths 1--7")
    if "equal-length macro balanced accuracy" not in capacity.get(
        "primary_effect", ""
    ):
        fail("GraphLog capacity sweep does not use the primary evaluation metric")
    capacity_by_depth = {
        row["depth"]: row for row in capacity.get("capacity_curve", [])
    }
    if not 0.124 < capacity_by_depth.get(7, {}).get("evaluation_shared", 0) < 0.125:
        fail("GraphLog depth-7 multiset endpoint changed")
    if not 0.282 < capacity_by_depth.get(7, {}).get("evaluation_absolute", 0) < 0.283:
        fail("GraphLog depth-7 absolute-position endpoint changed")
    if capacity_by_depth.get(7, {}).get("shared_worlds_at_or_below_chance") != 30:
        fail("GraphLog depth-7 multiset chance count changed")

    controls = json.loads(
        (
            ROOT
            / "experiments/ultra4g/results/pointwise-and-filtering-controls/summary.json"
        ).read_text(encoding="utf-8")
    )
    if controls.get("query_cells") != 4617 or not controls.get("matched_complete"):
        fail("frozen-KG protocol controls are incomplete")
    contamination = controls.get("contamination", {})
    if contamination.get("known_positive_negatives") != 456:
        fail("matched filtering contamination count changed")
    if contamination.get("negative_rows") != 34200:
        fail("matched filtering negative-row count changed")

    aggregation = json.loads(
        (
            ROOT / "experiments/ultra4g/results/aggregation-sensitivity/summary.json"
        ).read_text(encoding="utf-8")
    )
    if aggregation.get("panel", {}).get("provenance_families") != 16:
        fail("aggregation sensitivity does not cover 16 provenance families")
    dataset_counts = aggregation.get("equal_dataset", {})
    observed_counts = (
        dataset_counts.get("wins"),
        dataset_counts.get("ties"),
        dataset_counts.get("losses"),
    )
    if observed_counts != (5, 4, 48):
        fail("aggregation sensitivity win/tie/loss counts changed")

    provenance = json.loads(
        (ROOT / "publication_data/context_robustness_provenance.json").read_text()
    )
    if provenance.get("analysis_status") != "complete" or provenance.get("missing_datasets"):
        fail("publication context summary is not the complete, non-imputed analysis")


def main() -> int:
    paths = verify_manifest()
    verify_anonymity(paths)
    verify_scientific_inventory()
    print("Artifact verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Artifact verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
