#!/usr/bin/env python3
"""Build a 57-KG asset isolating positive relation-coverage in context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.graphlets import relation_quotas
from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.plan import atomic_write_json
from relational_tfm.protocol import FROZEN_KG_RELATION_CONTEXT_ASSET_ID

CONTEXT_SEEDS = (101, 102, 103)
CONTEXT_TRIPLES = 100
CONTEXT_POSITIVES = 2 * CONTEXT_TRIPLES
CONTEXT_MODES = (
    "frequency_random_filtered_uniform",
    "relation_matched_filtered_uniform",
)
SELECTION_NAMESPACE = 20_260_802


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _seed_for(dataset: str, seed: int, stream: int) -> np.random.SeedSequence:
    digest = hashlib.sha256(dataset.encode()).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    return np.random.SeedSequence([SELECTION_NAMESPACE, seed, stream, *words])


def _artifact(manifest: dict[str, Any], *, split: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["artifacts"]
        if item.get("split") == split and item.get("scope") == "evaluation"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {split}/evaluation B2v2 artifact")
    return matches[0]


def _distribution_tv(reference: np.ndarray, observed: np.ndarray) -> float:
    support = np.union1d(reference, observed)
    reference_mass = np.asarray([(reference == value).mean() for value in support])
    observed_mass = np.asarray([(observed == value).mean() for value in support])
    return float(0.5 * np.abs(reference_mass - observed_mass).sum())


def _coverage(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    supported = set(map(int, np.unique(observed)))
    reference_types = set(map(int, np.unique(reference)))
    type_coverage = len(reference_types & supported) / max(len(reference_types), 1)
    mass_coverage = float(np.mean([int(value) in supported for value in reference]))
    return float(type_coverage), mass_coverage


def _relation_matched_contexts(
    *,
    dataset: str,
    q64_triples: np.ndarray,
    q512_indices: np.ndarray,
    q512_triples: np.ndarray,
    q512_targets: np.ndarray,
    q512_sides: np.ndarray,
    q512_b0: np.ndarray,
    q512_b2v2: np.ndarray,
    q512_filtered: np.ndarray,
    num_entities: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(q512_targets) != 2 * len(q512_triples):
        raise ValueError("q512 directional rows are not paired by triple")
    if not np.array_equal(q512_sides, np.tile([0, 1], len(q512_triples))):
        raise ValueError("q512 directions are not ordered tail/head")
    if q512_filtered.shape != (len(q512_targets), num_entities):
        raise ValueError("q512 filtered mask is not query aligned")
    if q512_b0.shape[0] != len(q512_targets) * num_entities:
        raise ValueError("q512 B0 rows are not query aligned")
    if q512_b2v2.shape[0] != len(q512_targets) * num_entities:
        raise ValueError("q512 B2v2 rows are not query aligned")

    q64_set = {tuple(map(int, triple)) for triple in q64_triples}
    eligible = np.asarray(
        [
            index
            for index, triple in enumerate(q512_triples)
            if tuple(map(int, triple)) not in q64_set
        ],
        dtype=np.int64,
    )
    if len(eligible) < CONTEXT_TRIPLES:
        raise ValueError(f"{dataset}: insufficient q512 positives outside q64")
    pool_relations = q512_triples[eligible, 1].astype(np.int64, copy=False)
    target_relations = q64_triples[:, 1].astype(np.int64, copy=False)
    target_mask = np.isin(pool_relations, np.unique(target_relations))
    target_capacity = int(target_mask.sum())
    if target_capacity >= CONTEXT_TRIPLES:
        quotas = relation_quotas(pool_relations, target_relations, CONTEXT_TRIPLES)
        residual_count = 0
    else:
        quotas = {
            int(relation): int(np.sum(pool_relations == relation))
            for relation in np.unique(pool_relations[target_mask])
        }
        residual_count = CONTEXT_TRIPLES - sum(quotas.values())

    arrays: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for seed in CONTEXT_SEEDS:
        rng = np.random.default_rng(_seed_for(dataset, seed, 0))
        selected: list[int] = []
        for relation, quota in sorted(quotas.items()):
            candidates = eligible[pool_relations == relation]
            selected.extend(rng.permutation(candidates)[:quota].tolist())
        if residual_count:
            selected_set = set(selected)
            residual_pool = np.asarray(
                [index for index in eligible if int(index) not in selected_set],
                dtype=np.int64,
            )
            selected.extend(rng.permutation(residual_pool)[:residual_count].tolist())
        positions = np.asarray(selected, dtype=np.int64)
        rng.shuffle(positions)
        if len(positions) != CONTEXT_TRIPLES or len(np.unique(positions)) != len(positions):
            raise AssertionError("relation-matched positive selection is invalid")
        queries = np.ravel(
            np.stack([2 * positions, 2 * positions + 1], axis=1)
        ).astype(np.int64)

        positive_b0: list[np.ndarray] = []
        positive_b2v2: list[np.ndarray] = []
        positive_triples: list[np.ndarray] = []
        negative_b0: list[np.ndarray] = []
        negative_b2v2: list[np.ndarray] = []
        negative_triples: list[np.ndarray] = []
        negative_entities: list[int] = []
        negative_rng = np.random.default_rng(_seed_for(dataset, seed, 1))
        for query in queries:
            query = int(query)
            triple = np.asarray(q512_triples[query // 2], dtype=np.int64)
            target = int(q512_targets[query])
            side = int(q512_sides[query])
            start = query * num_entities
            target_row = start + target
            positive_b0.append(np.asarray(q512_b0[target_row], dtype=np.float32))
            positive_b2v2.append(np.asarray(q512_b2v2[target_row], dtype=np.float32))
            positive_triples.append(triple)

            valid = np.flatnonzero(~np.asarray(q512_filtered[query], dtype=np.bool_))
            valid = valid[valid != target]
            if len(valid) == 0:
                raise ValueError(f"{dataset}: q512 query {query} has no false candidate")
            entity = int(negative_rng.choice(valid))
            row = start + entity
            negative_b0.append(np.asarray(q512_b0[row], dtype=np.float32))
            negative_b2v2.append(np.asarray(q512_b2v2[row], dtype=np.float32))
            h, r, t = map(int, triple)
            negative_triples.append(
                np.asarray((h, r, entity) if side == 0 else (entity, r, t), dtype=np.int64)
            )
            negative_entities.append(entity)

        prefix = f"context_relation_matched_filtered_uniform_{seed}"
        arrays[f"{prefix}_X"] = np.concatenate(
            [np.stack(positive_b0), np.stack(negative_b0)]
        ).astype(np.float32)
        arrays[f"{prefix}_b2v2"] = np.concatenate(
            [np.stack(positive_b2v2), np.stack(negative_b2v2)]
        ).astype(np.float32)
        arrays[f"{prefix}_y"] = np.concatenate(
            [np.ones(CONTEXT_POSITIVES), np.zeros(CONTEXT_POSITIVES)]
        ).astype(np.int64)
        arrays[f"{prefix}_triples"] = np.concatenate(
            [np.stack(positive_triples), np.stack(negative_triples)]
        ).astype(np.int64)
        arrays[f"{prefix}_source_directional_queries"] = queries
        arrays[f"{prefix}_negative_entities"] = np.asarray(
            negative_entities, dtype=np.int64
        )

        selected_relations = q512_triples[positions, 1]
        type_coverage, mass_coverage = _coverage(target_relations, selected_relations)
        audits[str(seed)] = {
            "selected_q512_triples": CONTEXT_TRIPLES,
            "positive_directional_rows": CONTEXT_POSITIVES,
            "tail_rows": int((q512_sides[queries] == 0).sum()),
            "head_rows": int((q512_sides[queries] == 1).sum()),
            "q64_overlap": int(
                sum(tuple(map(int, q512_triples[index])) in q64_set for index in positions)
            ),
            "evaluation_relation_type_coverage": type_coverage,
            "evaluation_relation_mass_coverage": mass_coverage,
            "relation_distribution_tv": _distribution_tv(
                target_relations, selected_relations
            ),
            "relation_quotas": {str(key): int(value) for key, value in quotas.items()},
            "off_target_relation_fallback_triples": residual_count,
            "selected_split_indices_sha256": object_sha256(
                [int(q512_indices[index]) for index in positions]
            ),
        }
    return arrays, audits


def build_dataset(
    *,
    dataset: str,
    reference_root: Path,
    q512_root: Path,
    b2v2_root: Path,
    output_root: Path,
    reference_manifest: dict[str, Any],
) -> dict[str, Any]:
    reference_path = reference_root / f"{dataset}.npz"
    record = reference_manifest["datasets"][dataset]
    if file_sha256(reference_path) != record["archive_sha256"]:
        raise ValueError(f"reference archive hash mismatch for {dataset}")
    with np.load(reference_path, allow_pickle=False) as handle:
        reference = {name: np.asarray(handle[name]) for name in handle.files}

    base_names = (
        "candidate_b0", "candidate_b2v2", "filtered", "target_entities",
        "side_codes", "selected_split_indices", "selected_triples", "num_entities",
        "num_relations", "anchor_entities", "direction_relations",
    )
    arrays = {name: reference[name] for name in base_names}
    for seed in CONTEXT_SEEDS:
        source = f"context_filtered_uniform_{seed}"
        target = f"context_frequency_random_filtered_uniform_{seed}"
        for suffix in (
            "X", "b2v2", "y", "triples", "source_directional_queries",
            "negative_entities",
        ):
            arrays[f"{target}_{suffix}"] = reference[f"{source}_{suffix}"].copy()

    q512_dir = q512_root / dataset
    split = "valid" if (q512_dir / "valid" / "queries.npz").is_file() else "test"
    query_path = q512_dir / split / "queries.npz"
    with np.load(query_path, allow_pickle=False) as handle:
        q512_indices = np.asarray(handle["selected_split_indices"], dtype=np.int64)
        q512_triples = np.asarray(handle["selected_triples"], dtype=np.int64)
        q512_targets = np.asarray(handle["target_entities"], dtype=np.int64)
        q512_sides = np.asarray(handle["side_codes"], dtype=np.int8)
    q512_b0 = np.load(q512_dir / split / "features.npy", mmap_mode="r")
    q512_filtered = np.load(q512_dir / split / "filtered.npy", mmap_mode="r")
    b2v2_manifest_path = b2v2_root / dataset / "manifest.json"
    b2v2_manifest = _load_json(b2v2_manifest_path)
    b2v2_artifact = _artifact(b2v2_manifest, split=split)
    q512_b2v2 = np.load(b2v2_root / dataset / split / "b2v2.npy", mmap_mode="r")
    if list(q512_b2v2.shape) != list(b2v2_artifact["shape"]):
        raise ValueError(f"B2v2 manifest shape mismatch for {dataset}")

    matched, matched_audits = _relation_matched_contexts(
        dataset=dataset,
        q64_triples=np.asarray(arrays["selected_triples"], dtype=np.int64),
        q512_indices=q512_indices,
        q512_triples=q512_triples,
        q512_targets=q512_targets,
        q512_sides=q512_sides,
        q512_b0=q512_b0,
        q512_b2v2=q512_b2v2,
        q512_filtered=q512_filtered,
        num_entities=int(arrays["num_entities"]),
    )
    arrays.update(matched)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{dataset}.{os.getpid()}.npz"
    np.savez_compressed(temporary, **arrays)
    destination = output_root / f"{dataset}.npz"
    os.replace(temporary, destination)

    context_hashes = {
        name: _array_sha256(value)
        for name, value in arrays.items()
        if name.startswith("context_")
    }
    return {
        "status": "ok",
        "dataset": dataset,
        "archive_sha256": file_sha256(destination),
        "archive_bytes": destination.stat().st_size,
        "transfer_regime": record["transfer_regime"],
        "num_entities": int(arrays["num_entities"]),
        "num_relations": int(arrays["num_relations"]),
        "context_source_split": split,
        "reference_archive_sha256": record["archive_sha256"],
        "q512_query_sha256": file_sha256(query_path),
        "b2v2_manifest_sha256": file_sha256(b2v2_manifest_path),
        "b2v2_evaluation_sha256": b2v2_artifact["sha256"],
        "context_array_hashes": context_hashes,
        "relation_matched_audits": matched_audits,
        "frequency_random_audits": record["seed_audits"],
    }


def finalize_manifest(*, reference_root: Path, output_root: Path) -> dict[str, Any]:
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _load_json(reference_manifest_path)
    datasets: dict[str, Any] = {}
    for dataset in sorted(reference_manifest["datasets"]):
        status = _load_json(output_root / f"{dataset}.status.json")
        if status.get("status") != "ok":
            raise ValueError(f"relation-context asset failed for {dataset}")
        datasets[dataset] = {
            key: value for key, value in status.items() if key not in {"status", "dataset"}
        }
    manifest = {
        "asset_id": FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
        "archive_schema": 4,
        "reference_asset_id": reference_manifest["asset_id"],
        "reference_manifest_sha256": file_sha256(reference_manifest_path),
        "b0_feature_names": reference_manifest["b0_feature_names"],
        "b2v2_feature_names": reference_manifest["b2v2_feature_names"],
        "query_selection": reference_manifest["query_selection"],
        "context_modes": list(CONTEXT_MODES),
        "context_positive_rows": CONTEXT_POSITIVES,
        "context_negative_rows": CONTEXT_POSITIVES,
        "context_seeds": list(CONTEXT_SEEDS),
        "context_selection": (
            "The reference filtered-uniform contexts are copied byte-identically; "
            "the intervention selects 100 validation triples with quotas matching "
            "the q64 evaluation relation distribution. Both arms use paired tail/head "
            "positives and filtered-uniform negative corruptions at 200+200 rows."
        ),
        "scientific_status": "post-hoc fixed-panel context-positive intervention",
        "claim_boundary": (
            "Only positive relation coverage changes; evaluation queries, candidates, "
            "masks, features, context budget, and negative policy are held fixed."
        ),
        "allowed_interfaces": [
            "b0_local_only", "b0_local_no_relation_id", "b0_local_ultra",
            "b0_local_ultra_no_relation_id",
        ],
        "datasets": datasets,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--q512-root", type=Path, required=True)
    parser.add_argument("--b2v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    if args.finalize:
        manifest = finalize_manifest(reference_root=reference_root, output_root=output_root)
        print(json.dumps({
            "asset_id": manifest["asset_id"],
            "datasets": len(manifest["datasets"]),
            "manifest_sha256": file_sha256(output_root / "manifest.json"),
        }, indent=2, sort_keys=True))
        return 0

    reference_manifest = _load_json(reference_root / "manifest.json")
    for dataset in args.dataset or sorted(reference_manifest["datasets"]):
        print(f"=== {dataset} ===", flush=True)
        try:
            status = build_dataset(
                dataset=dataset,
                reference_root=reference_root,
                q512_root=args.q512_root.resolve(),
                b2v2_root=args.b2v2_root.resolve(),
                output_root=output_root,
                reference_manifest=reference_manifest,
            )
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            status = {
                "status": "failed", "dataset": dataset,
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_write_json(output_root / f"{dataset}.status.json", status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if status["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
