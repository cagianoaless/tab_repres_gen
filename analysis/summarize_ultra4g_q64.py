#!/usr/bin/env python3
"""Build a compact, paper-ready summary of the completed ULTRA57 q64 sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DIRECTIONS = ("combined", "head", "tail")
RANK_METRICS = (
    "mrr",
    "hits_at_1",
    "hits_at_3",
    "hits_at_10",
    "hits_at_50",
    "mean_rank",
)
CELL_IDENTITY = (
    "index",
    "cell_id",
    "dataset",
    "source_cluster",
    "transfer_regime",
    "interface",
    "context_per_class",
    "context_rows",
    "seed",
    "feature_count",
    "decoder_fit_status",
    "decoder_prediction_status",
    "context_representation_constant",
    "duration_seconds",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_columns() -> list[str]:
    columns: list[str] = []
    for direction in DIRECTIONS:
        for metric in RANK_METRICS:
            columns.extend(
                (
                    f"{direction}_{metric}",
                    f"native_{direction}_{metric}",
                    f"delta_{direction}_{metric}",
                )
            )
    return columns


def flatten_metric(payload: dict[str, Any]) -> dict[str, Any]:
    cell = payload["cell"]
    metadata = payload["task_metadata"]
    row: dict[str, Any] = {
        "index": int(cell["index"]),
        "cell_id": cell["cell_id"],
        "dataset": cell["dataset"],
        "source_cluster": metadata["source_cluster"],
        "transfer_regime": cell["transfer_regime"],
        "interface": cell["interface"],
        "context_per_class": int(cell["context_per_class"]),
        "context_rows": int(payload["context_rows"]),
        "seed": int(cell["seed"]),
        "feature_count": int(payload["feature_count"]),
        "decoder_fit_status": payload["decoder_fit_status"],
        "decoder_prediction_status": payload["decoder_prediction_status"],
        "context_representation_constant": bool(payload["context_representation_constant"]),
        "duration_seconds": float(payload["duration_seconds"]),
        "feature_schema_sha256": canonical_sha256(payload["feature_names"]),
        "prediction_sha256": payload["predictions"]["sha256"],
        "cell_payload_sha256": payload["cell_payload_sha256"],
    }
    for direction in DIRECTIONS:
        observed = payload["direction_metrics"][direction]
        native = payload["native_ultra_direction_metrics"][direction]
        for metric in RANK_METRICS:
            observed_value = float(observed[metric])
            native_value = float(native[metric])
            if not math.isfinite(observed_value) or not math.isfinite(native_value):
                raise ValueError(f"non-finite {direction}/{metric} in {cell['cell_id']}")
            row[f"{direction}_{metric}"] = observed_value
            row[f"native_{direction}_{metric}"] = native_value
            row[f"delta_{direction}_{metric}"] = observed_value - native_value
    return row


def validate_and_load(
    run_root: Path, *, allow_incomplete: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registration = read_json(run_root / "registration.json")
    plan = read_json(run_root / "plan.json")
    expected_cells = plan["cells"]
    expected_by_id = {cell["cell_id"]: cell for cell in expected_cells}
    if len(expected_by_id) != int(registration["cells"]):
        raise ValueError("plan cell IDs are not unique or registration cell count differs")

    shard_dirs = sorted((run_root / "results").glob("shard-*"))
    expected_shards = int(registration["shards"])
    expected_names = {f"shard-{index:04d}" for index in range(expected_shards)}
    observed_names = {path.name for path in shard_dirs}
    if observed_names != expected_names and not allow_incomplete:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise ValueError(f"incomplete shard set: missing={missing}, extra={extra}")
    if not observed_names <= expected_names:
        extra = sorted(observed_names - expected_names)
        raise ValueError(f"unregistered shard directories: {extra}")

    rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    schemas: dict[str, dict[str, Any]] = {}
    for shard in shard_dirs:
        manifest = read_json(shard / "shard_manifest.json")
        if manifest["plan_sha256"] != registration["plan_sha256"]:
            raise ValueError(f"{shard.name}: plan hash mismatch")
        for cell_id in manifest["cell_ids"]:
            if cell_id in observed_ids:
                raise ValueError(f"duplicate cell result: {cell_id}")
            if cell_id not in expected_by_id:
                raise ValueError(f"unregistered cell result: {cell_id}")
            metric_path = shard / f"{cell_id}.metrics.json"
            if not metric_path.is_file():
                raise ValueError(f"missing metrics file: {metric_path}")
            payload = read_json(metric_path)
            cell = payload["cell"]
            expected = expected_by_id[cell_id]
            for key in (
                "index",
                "cell_id",
                "dataset",
                "interface",
                "context_per_class",
                "seed",
            ):
                if cell[key] != expected[key]:
                    raise ValueError(f"{cell_id}: plan mismatch for {key}")
            if payload.get("status") != "complete":
                raise ValueError(f"{cell_id}: cell status is not complete")
            fit_status = str(payload.get("decoder_fit_status"))
            prediction_status = str(payload.get("decoder_prediction_status"))
            fit_succeeded = fit_status == "succeeded" and prediction_status == "succeeded"
            registered_constant_fallback = (
                fit_status.startswith("rejected_constant_input:")
                and prediction_status == "not_attempted"
                and bool(payload.get("context_representation_constant"))
            )
            if not (fit_succeeded or registered_constant_fallback):
                raise ValueError(
                    f"{cell_id}: unrecognized decoder outcome {fit_status}/{prediction_status}"
                )
            schema = {
                "feature_count": int(payload["feature_count"]),
                "feature_names": payload["feature_names"],
                "feature_schema_sha256": canonical_sha256(payload["feature_names"]),
            }
            interface = cell["interface"]
            if interface in schemas and schemas[interface] != schema:
                raise ValueError(f"{interface}: feature schema changed across cells")
            schemas[interface] = schema
            rows.append(flatten_metric(payload))
            observed_ids.add(cell_id)

    if observed_ids != set(expected_by_id) and not allow_incomplete:
        missing = sorted(set(expected_by_id) - observed_ids)
        raise ValueError(f"missing registered cells: {missing[:20]}")
    missing_ids = sorted(
        set(expected_by_id) - observed_ids,
        key=lambda cell_id: int(expected_by_id[cell_id]["index"]),
    )
    missing_shards = sorted(expected_names - observed_names)
    rows.sort(key=lambda row: int(row["index"]))
    return rows, {
        "registration": registration,
        "schemas": dict(sorted(schemas.items())),
        "expected_cells": expected_cells,
        "observed_cell_ids": observed_ids,
        "missing_cell_ids": missing_ids,
        "missing_shards": missing_shards,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
    value_columns: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item = dict(zip(group_keys, key, strict=False))
        item["n"] = len(members)
        for column in value_columns:
            item[column] = statistics.fmean(float(member[column]) for member in members)
        output.append(item)
    return output


def build_summaries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    values = metric_columns()
    dataset = aggregate(
        rows,
        (
            "dataset",
            "source_cluster",
            "transfer_regime",
            "interface",
            "context_per_class",
        ),
        values,
    )
    source = aggregate(
        dataset,
        ("source_cluster", "transfer_regime", "interface", "context_per_class"),
        values,
    )
    regime = aggregate(
        dataset,
        ("transfer_regime", "interface", "context_per_class"),
        values,
    )
    equal_dataset = aggregate(dataset, ("interface", "context_per_class"), values)
    equal_source = aggregate(source, ("interface", "context_per_class"), values)
    overall: list[dict[str, Any]] = []
    for weighting, summary in (
        ("equal_dataset", equal_dataset),
        ("equal_source_cluster", equal_source),
    ):
        for row in summary:
            overall.append({"weighting": weighting, **row})
    return {
        "dataset_summary.csv": dataset,
        "source_cluster_summary.csv": source,
        "transfer_regime_summary.csv": regime,
        "overall_summary.csv": overall,
    }


def complete_case_summaries(
    rows: list[dict[str, Any]], expected_cells: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
    expected_counts: dict[str, int] = defaultdict(int)
    observed_counts: dict[str, int] = defaultdict(int)
    for cell in expected_cells:
        expected_counts[str(cell["dataset"])] += 1
    for row in rows:
        observed_counts[str(row["dataset"])] += 1
    complete_datasets = sorted(
        dataset
        for dataset, expected in expected_counts.items()
        if observed_counts[dataset] == expected
    )
    incomplete_datasets = sorted(set(expected_counts) - set(complete_datasets))
    complete_rows = [row for row in rows if str(row["dataset"]) in complete_datasets]
    summaries = build_summaries(complete_rows)
    return (
        {f"complete_case_{name}": summary_rows for name, summary_rows in summaries.items()},
        complete_datasets,
        incomplete_datasets,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit = validate_and_load(run_root, allow_incomplete=args.allow_incomplete)

    cell_fields = [
        *CELL_IDENTITY,
        "feature_schema_sha256",
        "prediction_sha256",
        "cell_payload_sha256",
        *metric_columns(),
    ]
    write_csv(output_dir / "cell_metrics.csv", rows, cell_fields)
    completeness_rows = [
        {
            "index": int(cell["index"]),
            "cell_id": cell["cell_id"],
            "dataset": cell["dataset"],
            "transfer_regime": cell["transfer_regime"],
            "interface": cell["interface"],
            "context_per_class": int(cell["context_per_class"]),
            "seed": int(cell["seed"]),
            "observed": cell["cell_id"] in audit["observed_cell_ids"],
        }
        for cell in audit["expected_cells"]
    ]
    write_csv(
        output_dir / "completeness_matrix.csv",
        completeness_rows,
        list(completeness_rows[0]),
    )
    missing_rows = [row for row in completeness_rows if not bool(row["observed"])]
    write_csv(
        output_dir / "missing_cells.csv",
        missing_rows,
        list(completeness_rows[0]),
    )
    summaries = build_summaries(rows)
    complete_summaries, complete_datasets, incomplete_datasets = complete_case_summaries(
        rows, audit["expected_cells"]
    )
    summaries.update(complete_summaries)
    for name, summary_rows in summaries.items():
        fields = list(summary_rows[0])
        write_csv(output_dir / name, summary_rows, fields)

    with (output_dir / "feature_schemas.json").open("w", encoding="utf-8") as handle:
        json.dump(audit["schemas"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    registration = audit["registration"]
    complete = not audit["missing_cell_ids"] and not audit["missing_shards"]
    status = "COMPLETE_AND_AUDITED" if complete else "INCOMPLETE_PUBLISHED_RECOVERY"
    summary = {
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "study_id": "rfmlink_ultra57_q64_gpu_tabpfn_v1",
        "registered_cells": int(registration["cells"]),
        "observed_cells": len(rows),
        "registered_shards": int(registration["shards"]),
        "observed_shards": len(list((run_root / "results").glob("shard-*"))),
        "missing_shards": audit["missing_shards"],
        "missing_cells": len(audit["missing_cell_ids"]),
        "missing_cell_ids": audit["missing_cell_ids"],
        "constant_context_fallback_cells": sum(
            bool(row["context_representation_constant"]) for row in rows
        ),
        "datasets": len({row["dataset"] for row in rows}),
        "complete_case_datasets": complete_datasets,
        "complete_case_dataset_count": len(complete_datasets),
        "incomplete_datasets": incomplete_datasets,
        "source_clusters": len({row["source_cluster"] for row in rows}),
        "interfaces": sorted({row["interface"] for row in rows}),
        "contexts_per_class": sorted({row["context_per_class"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "plan_sha256": registration["plan_sha256"],
        "asset_manifest_sha256": registration["asset_manifest_sha256"],
        "source_commit": registration["source_commit"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    warning = (
        ""
        if complete
        else """
**INCOMPLETE RECOVERY:** this export is missing the shards and cells listed in
`summary.json` and `missing_cells.csv`. Its overall aggregates are descriptive
only and must not be reported as complete-panel estimates. Use
`completeness_matrix.csv` to construct explicitly common, balanced subsets.
The `complete_case_*` tables provide the post-hoc balanced view over datasets
with every registered cell observed; infrastructure-driven exclusions must be
reported explicitly.
"""
    )
    readme = f"""# RFMLink ULTRA57 q64 encoder ladder

Status: **{status}**.
{warning}

- {summary["datasets"]} datasets with {summary["source_clusters"]} registered \
source-cluster labels
- {summary["observed_cells"]} observed TabPFN cells in \
{summary["observed_shards"]} shards
- interfaces: {", ".join(summary["interfaces"])}
- contexts per class: {", ".join(map(str, summary["contexts_per_class"]))}
- seeds: {", ".join(map(str, summary["seeds"]))}
- plan SHA256: `{summary["plan_sha256"]}`
- asset-manifest SHA256: `{summary["asset_manifest_sha256"]}`

`cell_metrics.csv` is the lossless compact metric table. Dataset, source-cluster,
transfer-regime, and overall equal-dataset/equal-source summaries are derived from
it. Raw per-cell metric JSON and prediction NPZ files remain in the sibling shard
directories and are the authoritative source.

All ranking tables contain combined, head, and tail MRR, Hits@1/3/10/50, and mean
rank for TabPFN and native zero-shot ULTRA, plus their paired difference. No result
selection or scientific decision is performed by this summarizer.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    artifact_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_files
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
