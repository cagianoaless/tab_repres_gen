#!/usr/bin/env python3
"""Generate frozen study configs for the matched KG protocol controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(SCRIPT_DIR), str(REPO_ROOT)]

from relational_tfm.protocol import FROZEN_KG_MATCHED_FILTER_ASSET_ID  # noqa: E402

INTERFACES = ("ultra_score_only", "b0_local_only", "b0_local_ultra")
SEEDS = (101, 102, 103)
PROFILES = {
    "matched-filter-tabpfn": {
        "study_id": "rfmlink_ultra57_reviewer_matched_filter_tabpfn_v1",
        "resource_class": "gpu",
        "decoders": ("tabpfn",),
        "context_modes": ("unfiltered_uniform", "filtered_uniform"),
        "query_transforms": None,
        "expected_cells": 1_026,
    },
    "query-relative-tabpfn": {
        "study_id": "rfmlink_ultra57_reviewer_query_relative_tabpfn_v1",
        "resource_class": "gpu",
        "decoders": ("tabpfn",),
        "context_modes": ("filtered_uniform",),
        "query_transforms": (
            "raw",
            "within_query_zscore",
            "within_query_percentile",
        ),
        "expected_cells": 1_539,
    },
    "query-relative-cpu": {
        "study_id": "rfmlink_ultra57_reviewer_query_relative_cpu_v1",
        "resource_class": "cpu",
        "decoders": ("logistic", "catboost"),
        "context_modes": ("filtered_uniform",),
        "query_transforms": (
            "raw",
            "within_query_zscore",
            "within_query_percentile",
        ),
        "expected_cells": 3_078,
    },
}

TABPFN_IDENTITY = {
    "package": "tabpfn",
    "model_version": "V3",
    "n_estimators": 4,
    "fit_mode": "fit_with_cache",
    "inference_precision": "auto",
    "memory_saving_mode": "auto",
    "checkpoint_sha256": (
        "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
    ),
}


def generate_config(
    manifest_path: Path, output_path: Path, *, profile: str
) -> dict[str, Any]:
    specification = PROFILES[profile]
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("asset_id") != FROZEN_KG_MATCHED_FILTER_ASSET_ID:
        raise ValueError("KG control manifest has the wrong asset ID")
    if tuple(manifest.get("context_modes", ())) != (
        "unfiltered_uniform",
        "filtered_uniform",
    ):
        raise ValueError("KG control context modes changed")
    datasets = manifest["datasets"]
    names = sorted(datasets, key=lambda name: (datasets[name]["num_entities"], name))
    payload: dict[str, Any] = {
        "schema_version": 4,
        "study_id": specification["study_id"],
        "status": "development",
        "task_family": "frozen_kg_interface",
        "resource_class": specification["resource_class"],
        "design_role": (
            "reviewer-requested matched filtering control"
            if profile == "matched-filter-tabpfn"
            else "reviewer-requested pointwise versus query-relative control"
        ),
        "asset_id": FROZEN_KG_MATCHED_FILTER_ASSET_ID,
        "asset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "query_triples": 64,
        "decoders": list(specification["decoders"]),
        "context_modes": list(specification["context_modes"]),
        "contexts_per_class": [200],
        "seeds": list(SEEDS),
        "dataset_specs": [
            {
                "dataset": name,
                "transfer_regime": datasets[name]["transfer_regime"],
                "archive_sha256": datasets[name]["archive_sha256"],
                "interfaces": list(INTERFACES),
            }
            for name in names
        ],
    }
    if specification["query_transforms"] is not None:
        payload["query_transforms"] = list(specification["query_transforms"])
    if specification["resource_class"] == "gpu":
        payload["model_identity"] = TABPFN_IDENTITY
    import yaml

    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return {
        "profile": profile,
        "study_id": specification["study_id"],
        "datasets": len(names),
        "expected_cells": specification["expected_cells"],
        "config": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_config(args.manifest, args.output, profile=args.profile),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
