#!/usr/bin/env python3
"""Audit the immutable RFMLink q64 assets without loading candidate matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (101, 102, 103)
BUDGETS = (20, 50, 100, 200)
EXPECTED_DATASETS = 57
EXPECTED_QUERY_TRIPLES = 64
EXPECTED_DIRECTIONAL_QUERIES = 128
EXPECTED_B0_NAMES = (
    "relation_id",
    "head_total_out",
    "head_total_in",
    "tail_total_out",
    "tail_total_in",
    "head_relation_out_degree",
    "tail_relation_in_degree",
    "relation_frequency",
    "relation_functionality",
    "relation_inverse_functionality",
    "pair_other_relation_count",
    "reverse_pair_relation_count",
    "two_hop_count",
    "two_hop_start_relation_count",
    "two_hop_end_relation_count",
    "reverse_two_hop_count",
    "same_entity_cluster",
    "cluster_relation_density",
    "ultra_score",
)
EXPECTED_B2V2_NAMES = (
    "head_pagerank",
    "tail_pagerank",
    "same_connected_component",
    "head_component_log_fraction",
    "tail_component_log_fraction",
    "heat_similarity_t0p1",
    "heat_similarity_t1p0",
    "heat_similarity_t10p0",
    "heat_similarity_t100p0",
    "diffusion_distance_t0p1",
    "diffusion_distance_t1p0",
    "diffusion_distance_t10p0",
    "diffusion_distance_t100p0",
)
INTERFACE_SCHEMAS = {
    "entity_tokens_only": (
        "head_entity_token",
        "tail_entity_token",
    ),
    "b0_dyadic_only": (
        "pair_other_relation_count",
        "reverse_pair_relation_count",
        "same_entity",
    ),
    "b0_paths_only": EXPECTED_B0_NAMES[12:16],
    "b0_local_only": EXPECTED_B0_NAMES[:18],
    "ultra_score_only": EXPECTED_B0_NAMES[18:19],
    "b2v2_only": EXPECTED_B2V2_NAMES,
    "b2v2_structure_only": (*EXPECTED_B0_NAMES[:18], *EXPECTED_B2V2_NAMES),
    "b0_local_ultra": EXPECTED_B0_NAMES,
    "b2v2_full": (*EXPECTED_B0_NAMES, *EXPECTED_B2V2_NAMES),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_context_indices(labels: np.ndarray, seed: int, budget: int) -> np.ndarray:
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng = np.random.default_rng(
            np.random.SeedSequence([20260714, seed, label, 17])
        )
        selected.append(indices[rng.permutation(len(indices))][:budget])
    output = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    if len(output) != 2 * budget or int(labels[output].sum()) != budget:
        raise ValueError("nested context selection lost class balance")
    return output


def _evaluation_triples(arrays: Any) -> set[tuple[int, int, int]]:
    anchors = np.asarray(arrays["anchor_entities"], dtype=np.int64)
    relations = np.asarray(arrays["direction_relations"], dtype=np.int64)
    targets = np.asarray(arrays["target_entities"], dtype=np.int64)
    sides = np.asarray(arrays["side_codes"], dtype=np.int8)
    if not (len(anchors) == len(relations) == len(targets) == len(sides)):
        raise ValueError("direction arrays have inconsistent lengths")
    triples: set[tuple[int, int, int]] = set()
    for anchor, relation, target, side in zip(
        anchors, relations, targets, sides, strict=True
    ):
        if int(side) == 0:
            triples.add((int(anchor), int(relation), int(target)))
        elif int(side) == 1:
            triples.add((int(target), int(relation), int(anchor)))
        else:
            raise ValueError(f"invalid direction code: {side}")
    return triples


def audit(asset_root: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    manifest_path = asset_root / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    if expected_manifest_sha256 and manifest_hash != expected_manifest_sha256:
        raise ValueError("asset manifest does not match the expected SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest["datasets"]
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError(
            f"expected {EXPECTED_DATASETS} datasets, found {len(datasets)}"
        )
    if tuple(manifest["b0_feature_names"]) != EXPECTED_B0_NAMES:
        raise ValueError("B0 feature schema changed")
    if tuple(manifest["b2v2_feature_names"]) != EXPECTED_B2V2_NAMES:
        raise ValueError("B2v2 feature schema changed")
    if "both head and tail directions" not in manifest["query_selection"]:
        raise ValueError("manifest does not guarantee both query directions")
    if "immutable 400+400 contexts" not in manifest["context_selection"]:
        raise ValueError("manifest does not identify the immutable context source")

    selected_cells = 0
    overlap_cells = 0
    direction_counts: dict[str, dict[str, int]] = {}
    negative_eval_overlap_base_rows = 0
    negative_eval_overlap_selected_incidences = 0
    negative_eval_overlap_cells = 0
    duplicate_context_rows = 0
    for dataset, registered in sorted(datasets.items()):
        archive_path = asset_root / registered["archive"]
        if not archive_path.is_file():
            raise ValueError(f"missing archive for {dataset}")
        if registered["archive_sha256"] != _sha256(archive_path):
            raise ValueError(f"archive checksum mismatch for {dataset}")
        with np.load(archive_path, allow_pickle=False) as arrays:
            evaluation = _evaluation_triples(arrays)
            sides = np.asarray(arrays["side_codes"], dtype=np.int8)
            head_count = int(np.sum(sides == 1))
            tail_count = int(np.sum(sides == 0))
            if len(sides) != EXPECTED_DIRECTIONAL_QUERIES:
                raise ValueError(f"{dataset}: unexpected directional query count")
            if head_count != EXPECTED_QUERY_TRIPLES or tail_count != EXPECTED_QUERY_TRIPLES:
                raise ValueError(f"{dataset}: head/tail query imbalance")
            direction_counts[dataset] = {"head": head_count, "tail": tail_count}

            for seed in SEEDS:
                labels = np.asarray(arrays[f"context_{seed}_y"], dtype=np.int64)
                triples = np.asarray(
                    arrays[f"context_{seed}_triples"], dtype=np.int64
                )
                if labels.shape != (800,) or triples.shape != (800, 3):
                    raise ValueError(f"{dataset}/{seed}: context is not 400+400")
                if int(labels.sum()) != 400 or set(np.unique(labels)) != {0, 1}:
                    raise ValueError(f"{dataset}/{seed}: context labels are not balanced")
                if len({tuple(row) for row in triples.tolist()}) != len(triples):
                    duplicate_context_rows += 1
                base_negative_rows = {
                    tuple(map(int, triples[index]))
                    for index in np.flatnonzero(labels == 0)
                }
                negative_eval_overlap_base_rows += len(base_negative_rows & evaluation)

                previous: set[int] = set()
                for budget in BUDGETS:
                    selected = _nested_context_indices(labels, seed, budget)
                    current = set(map(int, selected))
                    if not previous.issubset(current):
                        raise ValueError(
                            f"{dataset}/{seed}/{budget}: context is not nested"
                        )
                    previous = current
                    selected_cells += 1
                    positive_rows = {
                        tuple(map(int, triples[index]))
                        for index in selected
                        if int(labels[index]) == 1
                    }
                    negative_rows = {
                        tuple(map(int, triples[index]))
                        for index in selected
                        if int(labels[index]) == 0
                    }
                    if positive_rows & evaluation:
                        overlap_cells += 1
                    current_negative_overlap = len(negative_rows & evaluation)
                    negative_eval_overlap_selected_incidences += current_negative_overlap
                    if current_negative_overlap:
                        negative_eval_overlap_cells += 1

    if overlap_cells:
        raise ValueError("positive context/evaluation query overlap detected")
    return {
        "schema_version": 1,
        "status": "PASS",
        "asset_id": manifest["asset_id"],
        "manifest_sha256": manifest_hash,
        "dataset_count": len(datasets),
        "query_protocol": {
            "selection": manifest["query_selection"],
            "query_triples_per_dataset": EXPECTED_QUERY_TRIPLES,
            "directions": ["head", "tail"],
            "directional_queries_per_dataset": EXPECTED_DIRECTIONAL_QUERIES,
            "candidate_set": "all entities with standard filtered ranking masks",
        },
        "context_protocol": {
            "source": manifest["context_selection"],
            "source_rows_per_seed": 800,
            "source_positive_rows": 400,
            "source_negative_rows": 400,
            "negative_construction": (
                "uniform-random head and tail entities at the matched positive "
                "relation; source builder does not claim hard or truth-filtered negatives"
            ),
            "nested_budgets_per_class": list(BUDGETS),
            "seeds": list(SEEDS),
            "audited_dataset_seed_budget_cells": selected_cells,
            "positive_context_evaluation_overlap_cells": overlap_cells,
            "base_negative_rows_matching_a_q64_evaluation_positive": (
                negative_eval_overlap_base_rows
            ),
            "selected_negative_overlap_cells": negative_eval_overlap_cells,
            "selected_negative_overlap_incidences_across_nested_budgets": (
                negative_eval_overlap_selected_incidences
            ),
            "dataset_seed_contexts_with_duplicate_triples": duplicate_context_rows,
        },
        "feature_graph": "complete observed target training graph",
        "interface_schemas": {
            name: {
                "dimension": len(columns),
                "columns": list(columns),
            }
            for name, columns in INTERFACE_SCHEMAS.items()
        },
        "direction_counts_sha256": hashlib.sha256(
            json.dumps(direction_counts, sort_keys=True).encode()
        ).hexdigest(),
        "scope": (
            "protocol integrity audit only; it does not validate model predictions "
            "or claim that uniform-random context negatives are hard negatives"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.asset_root, args.expected_manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
