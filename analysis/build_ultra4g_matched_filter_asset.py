#!/usr/bin/env python3
"""Build the matched filtered/unfiltered ULTRA57 context asset."""

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
from relational_tfm.protocol import FROZEN_KG_REVIEWER_MATCHED_FILTER_ASSET_ID

CONTEXT_SEEDS = (101, 102, 103)
CONTEXT_POSITIVES = 200
CONTEXT_MODES = ("unfiltered_uniform", "filtered_uniform")
UNFILTERED_STREAM = 29
SELECTION_NAMESPACE = 20_260_729


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _seed_for(dataset: str, seed: int, stream: int) -> np.random.SeedSequence:
    digest = hashlib.sha256(dataset.encode()).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    return np.random.SeedSequence([SELECTION_NAMESPACE, seed, stream, *words])


def _unfiltered_rows(
    *,
    dataset: str,
    seed: int,
    source_queries: np.ndarray,
    q512_triples: np.ndarray,
    q512_targets: np.ndarray,
    q512_sides: np.ndarray,
    q512_b0: np.ndarray,
    q512_b2v2: np.ndarray,
    q512_filtered: np.ndarray,
    num_entities: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(_seed_for(dataset, seed, UNFILTERED_STREAM))
    negative_b0: list[np.ndarray] = []
    negative_b2v2: list[np.ndarray] = []
    negative_triples: list[np.ndarray] = []
    negative_entities: list[int] = []
    known_positive = 0
    for raw_query in np.asarray(source_queries, dtype=np.int64):
        query = int(raw_query)
        target = int(q512_targets[query])
        entity = int(rng.integers(0, num_entities))
        while entity == target:
            entity = int(rng.integers(0, num_entities))
        row = query * num_entities + entity
        negative_b0.append(np.asarray(q512_b0[row], dtype=np.float32))
        negative_b2v2.append(np.asarray(q512_b2v2[row], dtype=np.float32))
        source_triple = np.asarray(q512_triples[query // 2], dtype=np.int64)
        h, r, t = map(int, source_triple)
        side = int(q512_sides[query])
        negative_triples.append(
            np.asarray(
                (h, r, entity) if side == 0 else (entity, r, t),
                dtype=np.int64,
            )
        )
        negative_entities.append(entity)
        known_positive += int(q512_filtered[query, entity])
    arrays = {
        "negative_b0": np.stack(negative_b0),
        "negative_b2v2": np.stack(negative_b2v2),
        "negative_triples": np.stack(negative_triples),
        "negative_entities": np.asarray(negative_entities, dtype=np.int64),
    }
    audit = {
        "negative_rows": int(len(source_queries)),
        "known_positive_negatives": int(known_positive),
        "known_positive_fraction": float(known_positive / len(source_queries)),
        "target_entity_collisions": 0,
        "sampling": "uniform candidate entity excluding only the target entity",
        "all_known_positive_rejection": False,
    }
    return arrays, audit


def build_dataset(
    *,
    dataset: str,
    hard_root: Path,
    q512_root: Path,
    b2v2_root: Path,
    output_root: Path,
    hard_manifest: dict[str, Any],
) -> dict[str, Any]:
    hard_archive = hard_root / f"{dataset}.npz"
    record = hard_manifest["datasets"][dataset]
    if file_sha256(hard_archive) != record["archive_sha256"]:
        raise ValueError(f"hard-context archive hash mismatch for {dataset}")
    with np.load(hard_archive, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}

    source_split = str(record["context_source_split"])
    query_path = q512_root / dataset / source_split / "queries.npz"
    with np.load(query_path, allow_pickle=False) as handle:
        q512_triples = np.asarray(handle["selected_triples"], dtype=np.int64)
        q512_targets = np.asarray(handle["target_entities"], dtype=np.int64)
        q512_sides = np.asarray(handle["side_codes"], dtype=np.int8)
    q512_b0 = np.load(
        q512_root / dataset / source_split / "features.npy", mmap_mode="r"
    )
    q512_filtered = np.load(
        q512_root / dataset / source_split / "filtered.npy", mmap_mode="r"
    )
    q512_b2v2 = np.load(
        b2v2_root / dataset / source_split / "b2v2.npy", mmap_mode="r"
    )
    num_entities = int(arrays["num_entities"])
    seed_audits: dict[str, Any] = {}
    new_arrays: dict[str, np.ndarray] = {}
    for seed in CONTEXT_SEEDS:
        filtered = f"context_filtered_uniform_{seed}"
        unfiltered = f"context_unfiltered_uniform_{seed}"
        source_queries = np.asarray(
            arrays[f"{filtered}_source_directional_queries"], dtype=np.int64
        )
        generated, audit = _unfiltered_rows(
            dataset=dataset,
            seed=seed,
            source_queries=source_queries,
            q512_triples=q512_triples,
            q512_targets=q512_targets,
            q512_sides=q512_sides,
            q512_b0=q512_b0,
            q512_b2v2=q512_b2v2,
            q512_filtered=q512_filtered,
            num_entities=num_entities,
        )
        positive_b0 = np.asarray(arrays[f"{filtered}_X"][:CONTEXT_POSITIVES])
        positive_b2v2 = np.asarray(
            arrays[f"{filtered}_b2v2"][:CONTEXT_POSITIVES]
        )
        positive_triples = np.asarray(
            arrays[f"{filtered}_triples"][:CONTEXT_POSITIVES]
        )
        new_arrays[f"{unfiltered}_X"] = np.concatenate(
            [positive_b0, generated["negative_b0"]], axis=0
        ).astype(np.float32)
        new_arrays[f"{unfiltered}_b2v2"] = np.concatenate(
            [positive_b2v2, generated["negative_b2v2"]], axis=0
        ).astype(np.float32)
        new_arrays[f"{unfiltered}_triples"] = np.concatenate(
            [positive_triples, generated["negative_triples"]], axis=0
        ).astype(np.int64)
        new_arrays[f"{unfiltered}_y"] = np.asarray(arrays[f"{filtered}_y"]).copy()
        new_arrays[f"{unfiltered}_source_directional_queries"] = source_queries.copy()
        new_arrays[f"{unfiltered}_negative_entities"] = generated[
            "negative_entities"
        ]
        if not np.array_equal(
            new_arrays[f"{unfiltered}_X"][:CONTEXT_POSITIVES], positive_b0
        ):
            raise AssertionError("positive rows changed across filtering arms")
        seed_audits[str(seed)] = {
            **audit,
            "source_directional_queries_sha256": object_sha256(
                source_queries.tolist()
            ),
            "filtered_negative_entities_sha256": object_sha256(
                np.asarray(arrays[f"{filtered}_negative_entities"]).tolist()
            ),
        }

    output_arrays = {**arrays, **new_arrays}
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{dataset}.{os.getpid()}.npz"
    np.savez_compressed(temporary, **output_arrays)
    destination = output_root / f"{dataset}.npz"
    os.replace(temporary, destination)
    return {
        "status": "ok",
        "dataset": dataset,
        "archive_sha256": file_sha256(destination),
        "archive_bytes": destination.stat().st_size,
        "transfer_regime": record["transfer_regime"],
        "num_entities": num_entities,
        "num_relations": int(arrays["num_relations"]),
        "context_source_split": source_split,
        "source_hard_archive_sha256": record["archive_sha256"],
        "q512_query_sha256": file_sha256(query_path),
        "seed_audits": seed_audits,
        "new_array_hashes": {
            name: _array_sha256(value) for name, value in new_arrays.items()
        },
    }


def finalize(*, hard_root: Path, output_root: Path) -> dict[str, Any]:
    hard_manifest_path = hard_root / "manifest.json"
    hard_manifest = json.loads(hard_manifest_path.read_text(encoding="utf-8"))
    datasets: dict[str, Any] = {}
    for dataset in sorted(hard_manifest["datasets"]):
        status = json.loads(
            (output_root / f"{dataset}.status.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "ok":
            raise ValueError(f"matched-filter asset failed for {dataset}")
        datasets[dataset] = {
            key: status[key]
            for key in (
                "archive_sha256",
                "archive_bytes",
                "transfer_regime",
                "num_entities",
                "num_relations",
                "context_source_split",
                "source_hard_archive_sha256",
                "q512_query_sha256",
                "seed_audits",
                "new_array_hashes",
            )
        }
    manifest = {
        "asset_id": FROZEN_KG_REVIEWER_MATCHED_FILTER_ASSET_ID,
        "archive_schema": 4,
        "source_asset_id": hard_manifest["asset_id"],
        "source_manifest_sha256": file_sha256(hard_manifest_path),
        "b0_feature_names": hard_manifest["b0_feature_names"],
        "b2v2_feature_names": hard_manifest["b2v2_feature_names"],
        "query_selection": hard_manifest["query_selection"],
        "context_modes": list(CONTEXT_MODES),
        "context_positive_rows": CONTEXT_POSITIVES,
        "context_negative_rows": CONTEXT_POSITIVES,
        "context_seeds": list(CONTEXT_SEEDS),
        "context_selection": (
            "identical positive directional queries with either uniform candidates "
            "excluding only the target or uniform candidates rejected by the "
            "all-known-positive mask"
        ),
        "scientific_status": "reviewer-requested matched negative-filtering control",
        "claim_boundary": (
            "positive rows, context query identities, seeds, evaluation candidates, "
            "evaluation masks, targets, feature schemas, and decoder settings are "
            "matched; only all-known-positive rejection changes"
        ),
        "datasets": datasets,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--q512-root", type=Path, required=True)
    parser.add_argument("--b2v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    hard_root = args.hard_root.resolve()
    output_root = args.output_root.resolve()
    if args.finalize:
        manifest = finalize(hard_root=hard_root, output_root=output_root)
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
    hard_manifest = json.loads(
        (hard_root / "manifest.json").read_text(encoding="utf-8")
    )
    for dataset in args.dataset or sorted(hard_manifest["datasets"]):
        status_path = output_root / f"{dataset}.status.json"
        try:
            status = build_dataset(
                dataset=dataset,
                hard_root=hard_root,
                q512_root=args.q512_root.resolve(),
                b2v2_root=args.b2v2_root.resolve(),
                output_root=output_root,
                hard_manifest=hard_manifest,
            )
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            status = {
                "status": "failed",
                "dataset": dataset,
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_write_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if status["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
