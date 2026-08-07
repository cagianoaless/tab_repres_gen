#!/usr/bin/env python3
"""Build filtered-uniform and relation/degree-matched Ultra panel contexts."""

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

from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.plan import atomic_write_json
from relational_tfm.protocol import FROZEN_KG_NEGATIVE_ROBUSTNESS_ASSET_ID

CONTEXT_SEEDS = (101, 102, 103)
CONTEXT_POSITIVES = 200
REFERENCE_MODE = "relation_matched_filtered_uniform"
HARDER_MODE = "relation_matched_role_degree"
CONTEXT_MODES = (REFERENCE_MODE, HARDER_MODE)
SELECTION_NAMESPACE = 20_260_805


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _seed_for(dataset: str, seed: int) -> np.random.SeedSequence:
    digest = hashlib.sha256(dataset.encode()).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    return np.random.SeedSequence([SELECTION_NAMESPACE, seed, *words])


def _source_split(q512_root: Path, dataset: str) -> str:
    for split in ("valid", "test"):
        if (q512_root / dataset / split / "queries.npz").is_file():
            return split
    raise FileNotFoundError(f"{dataset}: q512 query asset is missing")


def _degree_columns(side: int) -> tuple[int, int]:
    return (3, 4) if side == 0 else (1, 2)


def _relative_degree_distance(
    values: np.ndarray,
    target_values: np.ndarray,
    columns: tuple[int, int],
) -> np.ndarray:
    target = np.asarray(target_values[list(columns)], dtype=np.float64)
    scale = np.maximum(np.abs(target), 1.0)
    return np.sum(
        np.abs(np.asarray(values[:, list(columns)], dtype=np.float64) - target) / scale,
        axis=1,
    )


def _matched_negative(
    *,
    candidates: np.ndarray,
    filtered: np.ndarray,
    target: int,
    side: int,
    rng: np.random.Generator,
) -> tuple[int, dict[str, float | int | bool]]:
    valid = np.flatnonzero(~np.asarray(filtered, dtype=np.bool_))
    valid = valid[valid != target]
    if len(valid) == 0:
        raise ValueError("directional query has no filtered negative candidate")

    role_column = 6 if side == 0 else 5
    role_pool = valid[np.asarray(candidates[valid, role_column]) > 0.0]
    role_fallback = len(role_pool) == 0
    if role_fallback:
        role_pool = valid

    community_pool = role_pool[np.asarray(candidates[role_pool, 16]) > 0.0]
    community_fallback = len(community_pool) == 0
    pool = role_pool if community_fallback else community_pool

    target_values = np.asarray(candidates[target], dtype=np.float32)
    columns = _degree_columns(side)
    distances = _relative_degree_distance(candidates[pool], target_values, columns)
    nearest_count = min(32, len(pool))
    nearest_positions = np.argpartition(distances, nearest_count - 1)[:nearest_count]
    nearest_distance = float(np.max(distances[nearest_positions]))
    tied_positions = np.flatnonzero(distances <= nearest_distance + 1e-12)
    selected_position = int(rng.choice(tied_positions))
    entity = int(pool[selected_position])
    return entity, {
        "valid_candidates": int(len(valid)),
        "role_candidates": int(len(role_pool)),
        "community_candidates": int(len(community_pool)),
        "role_fallback": bool(role_fallback),
        "community_fallback": bool(community_fallback),
        "degree_distance": float(distances[selected_position]),
    }


def _build_harder_contexts(
    *,
    dataset: str,
    reference: dict[str, np.ndarray],
    q512_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    split = _source_split(q512_root, dataset)
    q512_dir = q512_root / dataset / split
    with np.load(q512_dir / "queries.npz", allow_pickle=False) as handle:
        q512_triples = np.asarray(handle["selected_triples"], dtype=np.int64)
        q512_targets = np.asarray(handle["target_entities"], dtype=np.int64)
        q512_sides = np.asarray(handle["side_codes"], dtype=np.int8)
    q512_b0 = np.load(q512_dir / "features.npy", mmap_mode="r")
    q512_filtered = np.load(q512_dir / "filtered.npy", mmap_mode="r")
    num_entities = int(reference["num_entities"])
    if q512_b0.shape[0] != len(q512_targets) * num_entities:
        raise ValueError(f"{dataset}: q512 feature rows are not query aligned")
    if q512_filtered.shape != (len(q512_targets), num_entities):
        raise ValueError(f"{dataset}: q512 filtered masks are not query aligned")

    arrays: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    b2v2_width = int(np.asarray(reference["candidate_b2v2"]).shape[1])
    for seed in CONTEXT_SEEDS:
        source_prefix = f"context_{REFERENCE_MODE}_{seed}"
        target_prefix = f"context_{HARDER_MODE}_{seed}"
        source_queries = np.asarray(
            reference[f"{source_prefix}_source_directional_queries"],
            dtype=np.int64,
        )
        source_values = np.asarray(reference[f"{source_prefix}_X"], dtype=np.float32)
        source_triples = np.asarray(
            reference[f"{source_prefix}_triples"],
            dtype=np.int64,
        )
        if len(source_queries) != CONTEXT_POSITIVES:
            raise ValueError(f"{dataset}/{seed}: unexpected positive-row count")
        positive_values = source_values[:CONTEXT_POSITIVES]
        positive_triples = source_triples[:CONTEXT_POSITIVES]
        original_negative_values = source_values[CONTEXT_POSITIVES:]
        negative_values: list[np.ndarray] = []
        negative_triples: list[np.ndarray] = []
        negative_entities: list[int] = []
        degree_distances: list[float] = []
        original_degree_distances: list[float] = []
        role_fallbacks = 0
        community_fallbacks = 0
        pool_sizes: list[int] = []
        rng = np.random.default_rng(_seed_for(dataset, seed))

        for position, query_value in enumerate(source_queries):
            query = int(query_value)
            target = int(q512_targets[query])
            side = int(q512_sides[query])
            start = query * num_entities
            stop = start + num_entities
            candidates = np.asarray(q512_b0[start:stop], dtype=np.float32)
            observed_positive = np.asarray(candidates[target], dtype=np.float32)
            if not np.allclose(observed_positive, positive_values[position]):
                raise ValueError(f"{dataset}/{seed}: positive feature rows changed")
            triple = np.asarray(q512_triples[query // 2], dtype=np.int64)
            if not np.array_equal(triple, positive_triples[position]):
                raise ValueError(f"{dataset}/{seed}: positive triples changed")

            entity, diagnostic = _matched_negative(
                candidates=candidates,
                filtered=q512_filtered[query],
                target=target,
                side=side,
                rng=rng,
            )
            if bool(q512_filtered[query, entity]) or entity == target:
                raise AssertionError("harder context selected a filtered candidate")
            negative_values.append(np.asarray(candidates[entity], dtype=np.float32))
            h, relation, t = map(int, triple)
            negative_triples.append(
                np.asarray(
                    (h, relation, entity) if side == 0 else (entity, relation, t),
                    dtype=np.int64,
                )
            )
            negative_entities.append(entity)
            degree_distances.append(float(diagnostic["degree_distance"]))
            role_fallbacks += int(diagnostic["role_fallback"])
            community_fallbacks += int(diagnostic["community_fallback"])
            pool_sizes.append(
                int(
                    diagnostic["community_candidates"]
                    or diagnostic["role_candidates"]
                    or diagnostic["valid_candidates"]
                )
            )
            columns = _degree_columns(side)
            original_degree_distances.append(
                float(
                    _relative_degree_distance(
                        original_negative_values[position : position + 1],
                        positive_values[position],
                        columns,
                    )[0]
                )
            )

        negative_array = np.stack(negative_values).astype(np.float32)
        labels = np.concatenate(
            [
                np.ones(CONTEXT_POSITIVES, dtype=np.int64),
                np.zeros(CONTEXT_POSITIVES, dtype=np.int64),
            ]
        )
        arrays[f"{target_prefix}_X"] = np.concatenate(
            [positive_values, negative_array]
        ).astype(np.float32)
        arrays[f"{target_prefix}_b2v2"] = np.zeros(
            (len(labels), b2v2_width),
            dtype=np.float32,
        )
        arrays[f"{target_prefix}_y"] = labels
        arrays[f"{target_prefix}_triples"] = np.concatenate(
            [positive_triples, np.stack(negative_triples)]
        ).astype(np.int64)
        arrays[f"{target_prefix}_source_directional_queries"] = source_queries.copy()
        arrays[f"{target_prefix}_negative_entities"] = np.asarray(
            negative_entities,
            dtype=np.int64,
        )
        audits[str(seed)] = {
            "positive_rows_byte_identical_to_reference": True,
            "known_positive_collisions": 0,
            "role_fallbacks": role_fallbacks,
            "community_fallbacks": community_fallbacks,
            "median_sampling_pool_size": float(np.median(pool_sizes)),
            "degree_distance_mean": float(np.mean(degree_distances)),
            "reference_uniform_degree_distance_mean": float(
                np.mean(original_degree_distances)
            ),
            "ultra_score_mean": float(np.mean(negative_array[:, 18])),
            "reference_uniform_ultra_score_mean": float(
                np.mean(original_negative_values[:, 18])
            ),
            "source_directional_queries_sha256": object_sha256(
                source_queries.tolist()
            ),
        }
    return arrays, audits, split


def build_dataset(
    *,
    dataset: str,
    reference_root: Path,
    q512_root: Path,
    output_root: Path,
    reference_manifest: dict[str, Any],
) -> dict[str, Any]:
    reference_path = reference_root / f"{dataset}.npz"
    record = reference_manifest["datasets"][dataset]
    if file_sha256(reference_path) != record["archive_sha256"]:
        raise ValueError(f"{dataset}: reference archive hash mismatch")
    with np.load(reference_path, allow_pickle=False) as handle:
        reference = {name: np.asarray(handle[name]) for name in handle.files}

    arrays = {
        name: value
        for name, value in reference.items()
        if not name.startswith("context_")
    }
    for seed in CONTEXT_SEEDS:
        prefix = f"context_{REFERENCE_MODE}_{seed}"
        for suffix in (
            "X",
            "b2v2",
            "y",
            "triples",
            "source_directional_queries",
            "negative_entities",
        ):
            arrays[f"{prefix}_{suffix}"] = reference[f"{prefix}_{suffix}"].copy()

    harder, audits, split = _build_harder_contexts(
        dataset=dataset,
        reference=reference,
        q512_root=q512_root,
    )
    arrays.update(harder)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{dataset}.{os.getpid()}.npz"
    np.savez_compressed(temporary, **arrays)
    destination = output_root / f"{dataset}.npz"
    os.replace(temporary, destination)
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
        "context_array_hashes": {
            name: _array_sha256(value)
            for name, value in arrays.items()
            if name.startswith("context_")
        },
        "harder_context_audits": audits,
    }


def finalize_manifest(*, reference_root: Path, output_root: Path) -> dict[str, Any]:
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = _load_json(reference_manifest_path)
    datasets: dict[str, Any] = {}
    for dataset in sorted(reference_manifest["datasets"]):
        status = _load_json(output_root / f"{dataset}.status.json")
        if status.get("status") != "ok":
            raise ValueError(f"negative-robustness asset failed for {dataset}")
        datasets[dataset] = {
            key: value
            for key, value in status.items()
            if key not in {"status", "dataset"}
        }
    manifest = {
        "asset_id": FROZEN_KG_NEGATIVE_ROBUSTNESS_ASSET_ID,
        "archive_schema": 5,
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
            "Both arms use the same 200 relation-matched positive directional rows. "
            "The reference draws uniformly from the fully filtered candidate set. "
            "The intervention restricts candidates by observed relation role and "
            "community when supported, then samples among the 32 closest endpoint-"
            "degree matches. The frozen ultra_4g score is never used for selection."
        ),
        "scientific_status": "post-hoc fixed-panel negative-context robustness control",
        "claim_boundary": (
            "Only the negative context candidate changes. Positive rows, context size, "
            "evaluation queries, all-entity candidates, masks, feature rows, and seeds "
            "are held fixed."
        ),
        "allowed_interfaces": [
            "entity_tokens_only",
            "b0_local_only",
            "ultra_score_only",
            "b0_local_ultra",
        ],
        "datasets": datasets,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--q512-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    if args.finalize:
        manifest = finalize_manifest(
            reference_root=reference_root,
            output_root=output_root,
        )
        print(
            json.dumps(
                {
                    "asset_id": manifest["asset_id"],
                    "datasets": len(manifest["datasets"]),
                    "manifest_sha256": file_sha256(output_root / "manifest.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    reference_manifest = _load_json(reference_root / "manifest.json")
    for dataset in args.dataset or sorted(reference_manifest["datasets"]):
        print(f"=== {dataset} ===", flush=True)
        try:
            status = build_dataset(
                dataset=dataset,
                reference_root=reference_root,
                q512_root=args.q512_root.resolve(),
                output_root=output_root,
                reference_manifest=reference_manifest,
            )
        except Exception as error:
            status = {
                "status": "failed",
                "dataset": dataset,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(output_root / f"{dataset}.status.json", status)
            print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        atomic_write_json(output_root / f"{dataset}.status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
