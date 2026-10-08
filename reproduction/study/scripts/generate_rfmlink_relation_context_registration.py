#!/usr/bin/env python3
"""Generate relation-context study configurations from the frozen asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from relational_tfm.protocol import FROZEN_KG_RELATION_CONTEXT_ASSET_ID

INTERFACES = (
    "b0_local_only",
    "b0_local_no_relation_id",
    "b0_local_ultra",
    "b0_local_ultra_no_relation_id",
)
CONTEXT_MODES = (
    "frequency_random_filtered_uniform",
    "relation_matched_filtered_uniform",
)
SEEDS = (101, 102, 103)

TABPFN_IDENTITY = """model_identity:
  package: tabpfn
  model_version: V3
  n_estimators: 4
  fit_mode: fit_with_cache
  inference_precision: auto
  memory_saving_mode: auto
  checkpoint_sha256: d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"""


def generate_config(
    manifest_path: Path,
    output_path: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    if profile not in {"tabpfn", "cpu"}:
        raise ValueError(f"unsupported decoder profile: {profile}")
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("asset_id") != FROZEN_KG_RELATION_CONTEXT_ASSET_ID:
        raise ValueError("relation-context manifest has the wrong asset ID")
    if tuple(manifest.get("context_modes", ())) != CONTEXT_MODES:
        raise ValueError("relation-context modes differ from the frozen design")
    if tuple(manifest.get("allowed_interfaces", ())) != INTERFACES:
        raise ValueError("relation-context interfaces differ from the frozen design")
    if tuple(manifest.get("context_seeds", ())) != SEEDS:
        raise ValueError("relation-context seeds differ from the frozen design")
    if len(manifest.get("datasets", {})) != 57:
        raise ValueError("relation-context panel must contain 57 datasets")

    if profile == "tabpfn":
        study_id = "rfmlink_ultra57_relation_context_gpu_tabpfn_v1"
        resource_class = "gpu"
        decoders = "[tabpfn]"
        identity = TABPFN_IDENTITY.splitlines()
        decoder_count = 1
    else:
        study_id = "rfmlink_ultra57_relation_context_cpu_decoders_v1"
        resource_class = "cpu"
        decoders = "[logistic, catboost]"
        identity = []
        decoder_count = 2

    lines = [
        "schema_version: 4",
        f"study_id: {study_id}",
        "status: development",
        "task_family: frozen_kg_interface",
        f"resource_class: {resource_class}",
        (
            "design_role: post-hoc fixed-panel factorial isolating positive "
            "relation coverage and nominal relation-ID dependence"
        ),
        f"asset_id: {FROZEN_KG_RELATION_CONTEXT_ASSET_ID}",
        f"asset_manifest_sha256: {hashlib.sha256(manifest_bytes).hexdigest()}",
        "query_triples: 64",
        f"decoders: {decoders}",
        *identity,
        "context_modes:",
        *(f"  - {mode}" for mode in CONTEXT_MODES),
        "contexts_per_class: [200]",
        "seeds: [101, 102, 103]",
        "dataset_specs:",
    ]
    for dataset, record in sorted(
        manifest["datasets"].items(),
        key=lambda item: (int(item[1]["num_entities"]), item[0]),
    ):
        lines.extend(
            [
                f"  - dataset: {dataset}",
                f"    transfer_regime: {record['transfer_regime']}",
                f"    archive_sha256: {record['archive_sha256']}",
                "    interfaces:",
                *(f"      - {interface}" for interface in INTERFACES),
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cells = 57 * len(INTERFACES) * len(CONTEXT_MODES) * len(SEEDS) * decoder_count
    return {
        "profile": profile,
        "study_id": study_id,
        "datasets": 57,
        "decoders": decoder_count,
        "cells": cells,
        "config": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("tabpfn", "cpu"), required=True)
    args = parser.parse_args()
    print(json.dumps(generate_config(
        args.manifest, args.output, profile=args.profile
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
