#!/usr/bin/env python3
"""Verify release integrity, immutable historical evidence and scientific endpoints."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    if (ROOT / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
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


def verify_historical_evidence() -> None:
    snapshot = json.loads((ROOT / "tools/historical_results.json").read_text())
    for relative, digest in snapshot["files"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail(f"immutable historical evidence changed or missing: {relative}")
    # These are evidence hashes, not a reconstruction of missing execution SHAs.
    # Original 188 entries plus two GRU tables and thirteen KG package files.
    if len(snapshot["files"]) != 203:
        fail("historical result/publication inventory changed")


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
        "experiments/ultra4g/results/positive-context/README.md",
        "analysis/analyze_ultra4g_positive_context.py",
        "analysis/verify_historical_gru.py",
        "analysis/verify_historical_relation_context.py",
        "experiments/clutrr/results/recurrent-gru/README.md",
        "experiments/ultra4g/results/relation-context-historical/RECOVERY.md",
        "TERMINOLOGY.md",
        "PROVENANCE.md",
        "LICENSING.md",
        "reproduction/SOURCE_MANIFEST.json",
        "reproduction/REFERENCE_EXECUTION.json",
        "reproduction/run_pipeline.py",
        "reproduction/verify_source_pipeline.py",
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
        (ROOT / "experiments/clutrr/results/collision-conditioned-ceilings/summary.json").read_text(
            encoding="utf-8"
        )
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
    if "equal-length macro balanced accuracy" not in capacity.get("primary_effect", ""):
        fail("GraphLog capacity sweep does not use the primary evaluation metric")
    capacity_by_depth = {row["depth"]: row for row in capacity.get("capacity_curve", [])}
    if not 0.124 < capacity_by_depth.get(7, {}).get("evaluation_shared", 0) < 0.125:
        fail("GraphLog depth-7 multiset endpoint changed")
    if not 0.282 < capacity_by_depth.get(7, {}).get("evaluation_absolute", 0) < 0.283:
        fail("GraphLog depth-7 absolute-position endpoint changed")
    if capacity_by_depth.get(7, {}).get("shared_worlds_at_or_below_chance") != 30:
        fail("GraphLog depth-7 multiset chance count changed")

    depth_cells = csv_rows("experiments/graphlog/results/capacity-sweep/cells.csv")
    observed_depth_cells = {
        (row["world"], int(row["depth"]), int(row["seed"]), row["encoder"]) for row in depth_cells
    }
    expected_depth_cells = set(
        itertools.product(
            (f"rule_{index}" for index in range(57)),
            range(1, 8),
            (101, 102, 103),
            ("relation_multiset", "ordered_typed_path"),
        )
    )
    if len(depth_cells) != 2394 or observed_depth_cells != expected_depth_cells:
        fail("GraphLog capacity Cartesian matrix is incomplete or duplicated")

    controls = json.loads(
        (
            ROOT / "experiments/ultra4g/results/pointwise-and-filtering-controls/summary.json"
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
        (ROOT / "experiments/ultra4g/results/aggregation-sensitivity/summary.json").read_text(
            encoding="utf-8"
        )
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
    if (
        aggregation["panel"]["datasets"] != 57
        or aggregation["panel"]["practical_tie_band_mrr"] != 0.005
    ):
        fail("KG fixed panel or practical tie band changed")
    for section, expected_mean in (
        ("equal_dataset", -0.03461606091149509),
        ("equal_family", 0.006384521668912228),
    ):
        if not math.isclose(aggregation[section]["mean"], expected_mean, abs_tol=1e-12, rel_tol=0):
            fail(f"Full TabPFN/native ULTRA {section} comparison changed")

    provenance = json.loads(
        (ROOT / "publication_data/context_robustness_provenance.json").read_text()
    )
    if provenance.get("analysis_status") != "complete" or provenance.get("missing_datasets"):
        fail("publication context summary is not the complete, non-imputed analysis")

    for relative in (
        "context_configuration_mrr.csv",
        "context_robustness_provenance.json",
        "context_robustness_summary.csv",
        "graphlog_context_deltas.csv",
        "graphlog_context_scores.csv",
        "property_context_curves.csv",
        "property_encoder_ladder.csv",
        "query_evidence_absolute.csv",
        "query_evidence_provenance.json",
        "ultra_interface_ladder.csv",
    ):
        if not (ROOT / "publication_data" / relative).is_file():
            fail(f"publication input missing: {relative}")


def verify_primary_endpoints() -> None:
    def close(observed: str | float, expected: float, label: str) -> None:
        if not math.isclose(float(observed), expected, abs_tol=1e-12, rel_tol=0):
            fail(f"historical primary endpoint changed: {label}")

    multiset = csv_rows(
        "experiments/clutrr/results/metric-robustness/primary_metric_robustness.csv"
    )
    expected = {
        "logistic": 0.4295186273780921,
        "catboost": 0.27253092665470696,
        "tabpfn": 0.2893819084135285,
    }
    for decoder, value in expected.items():
        (row,) = [
            row for row in multiset if row["decoder"] == decoder and row["metric"] == "accuracy"
        ]
        close(row["relation_multiset"], value, f"CLUTRR multiset/{decoder}")
    interference = csv_rows(
        "experiments/clutrr/results/ordered-interference/paired_comparison_aggregate.csv"
    )
    for decoder, approximate in (("logistic", 0.133), ("catboost", 0.093), ("tabpfn", 0.104)):
        (row,) = [
            row
            for row in interference
            if row["decoder"] == decoder
            and row["context_per_relation"] == "80"
            and row["reference"] == "ordered_plus_multiset"
        ]
        if abs(float(row["seed_mean_difference"]) - approximate) > 0.001:
            fail(f"CLUTRR ordered interference endpoint changed: {decoder}")
    additive = csv_rows("experiments/clutrr/results/additive-context/cpu/summary.csv") + csv_rows(
        "experiments/clutrr/results/additive-context/tabpfn/summary.csv"
    )
    for decoder, value in (
        ("logistic", 0.48917748917748916),
        ("catboost", 0.3722943722943723),
        ("tabpfn", 0.7575757575757575),
    ):
        (row,) = [
            row
            for row in additive
            if row["decoder"] == decoder
            and row["base_context_per_relation"] == "80"
            and row["scope"] == "length4"
            and row["encoder"] == "ordered_typed_path"
        ]
        close(row["mean_effect"], value, f"CLUTRR length-4/{decoder}")
    prefix = json.loads(
        (ROOT / "experiments/clutrr/results/supported-prefix/summary.json").read_text()
    )
    if prefix["maximum_absolute_accuracy_change"] > 1e-14:
        fail("supported-prefix accuracy change is not numerically zero")
    graphlog = csv_rows("experiments/graphlog/results/decoder-sensitivity/primary_contrasts.csv")
    for decoder, approximate in (("logistic", 0.065), ("catboost", -0.157), ("tabpfn", -0.079)):
        (row,) = [
            row
            for row in graphlog
            if row["decoder"] == decoder
            and row["contrast"] == "shared_multiset_minus_absolute"
            and row["context_per_relation"] == "40"
        ]
        if abs(float(row["equal_world_mean_delta"]) - approximate) > 0.001:
            fail(f"GraphLog primary decoder endpoint changed: {decoder}")


def verify_recovered_derived_results() -> None:
    # Invoke repository modules with the same interpreter; no models or raw assets.
    for module in ("analysis.verify_historical_gru", "analysis.verify_historical_relation_context"):
        completed = subprocess.run(
            [sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            fail(f"{module} failed: {completed.stderr or completed.stdout}")
        print(f"{module}: passed")


def main() -> int:
    verify_manifest()
    verify_historical_evidence()
    verify_scientific_inventory()
    verify_primary_endpoints()
    verify_recovered_derived_results()
    completed = subprocess.run(
        [sys.executable, "-m", "reproduction.verify_source_pipeline"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        fail(f"source pipeline failed: {completed.stderr or completed.stdout}")
    print(completed.stdout.strip())
    print("Artifact verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Artifact verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
