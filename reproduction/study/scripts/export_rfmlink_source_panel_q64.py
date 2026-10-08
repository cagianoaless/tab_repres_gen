#!/usr/bin/env python3
"""Export the metric-free RFMLink source-family replication panel at q64."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ASSET_ID = "rfmlink_source_panel_q64_v1"
QUERY_BUDGET = 64
QUERY_SEED = 20260715
CONTEXT_SEEDS = (101, 102, 103)

# CoDEx and NELL are excluded because the three-graph bridge uses CoDEx-S and
# NELL-Ind-v1. MetaFam and WDSinger were not in the 14-family q512 confirmation.
SOURCE_FAMILIES = {
    "aristov4": ("aristov4",),
    "conceptnet100k": ("conceptnet100k",),
    "dbpedia100k": ("dbpedia100k",),
    "fb15k237": (
        "fb15k237",
        "fb15k237_10",
        "fb15k237_20",
        "fb15k237_50",
        "fb15k237_ind_v1",
        "fb15k237_ind_v2",
        "fb15k237_ind_v3",
        "fb15k237_ind_v4",
        "fb_ingram_25",
        "fb_ingram_50",
        "fb_ingram_75",
        "fb_ingram_100",
    ),
    "fbnell": ("fbnell",),
    "hetionet": ("hetionet",),
    "hm": ("hm_1k", "hm_3k", "hm_5k", "hm_indigo"),
    "ilpc2022": ("ilpc2022_small", "ilpc2022_large"),
    "wiki_topics": (
        "wiki_mt1_tax",
        "wiki_mt1_health",
        "wiki_mt2_org",
        "wiki_mt2_sci",
        "wiki_mt3_art",
        "wiki_mt3_infra",
        "wiki_mt4_sci",
        "wiki_mt4_health",
    ),
    "wk_ingram": ("wk_ingram_25", "wk_ingram_50", "wk_ingram_75", "wk_ingram_100"),
    "wn18rr": (
        "wn18rr",
        "wn18rr_ind_v1",
        "wn18rr_ind_v2",
        "wn18rr_ind_v3",
        "wn18rr_ind_v4",
    ),
    "yago310": ("yago310",),
}

EXPECTED_REPRESENTATIVES = {
    "aristov4": "aristov4",
    "conceptnet100k": "conceptnet100k",
    "dbpedia100k": "dbpedia100k",
    "fb15k237": "fb15k237_ind_v4",
    "fbnell": "fbnell",
    "hetionet": "hetionet",
    "hm": "hm_3k",
    "ilpc2022": "ilpc2022_large",
    "wiki_topics": "wiki_mt1_tax",
    "wk_ingram": "wk_ingram_50",
    "wn18rr": "wn18rr_ind_v3",
    "yago310": "yago310",
}

B0_FEATURE_NAMES = (
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

B2V2_FEATURE_NAMES = (
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    view = np.ascontiguousarray(array).view(np.uint8)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    for start in range(0, view.size, 16 * 1024 * 1024):
        digest.update(memoryview(view[start : start + 16 * 1024 * 1024]))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def relation_stratified_indices(triples: np.ndarray, size: int, seed: int) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int64)
    if len(triples) <= size:
        return np.arange(len(triples), dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_relation: dict[int, list[int]] = {}
    for relation in np.unique(triples[:, 1]):
        indices = np.flatnonzero(triples[:, 1] == relation)
        by_relation[int(relation)] = indices[rng.permutation(len(indices))].tolist()
    relations = np.asarray(sorted(by_relation), dtype=np.int64)
    relations = relations[rng.permutation(len(relations))]
    selected: list[int] = []
    while len(selected) < size:
        advanced = False
        for relation in relations:
            rows = by_relation[int(relation)]
            if rows:
                selected.append(rows.pop())
                advanced = True
                if len(selected) == size:
                    break
        if not advanced:
            break
    if len(selected) != size:
        raise RuntimeError("relation-stratified selection did not reach q64")
    return np.asarray(selected, dtype=np.int64)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_representatives(parent_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    selected: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for family, members in SOURCE_FAMILIES.items():
        rows = []
        for dataset in members:
            manifest = _load_json(parent_root / dataset / "manifest.json")
            rows.append(
                {
                    "dataset": dataset,
                    "num_entities": int(manifest["num_entities"]),
                    "num_train": int(manifest["num_train"]),
                }
            )
        median_entities = float(statistics.median(row["num_entities"] for row in rows))
        median_train = float(statistics.median(row["num_train"] for row in rows))
        representative = min(
            rows,
            key=lambda row: (
                abs(row["num_entities"] - median_entities),
                abs(row["num_train"] - median_train),
                row["dataset"],
            ),
        )
        selected[family] = representative["dataset"]
        audit[family] = {
            "members": rows,
            "median_num_entities": median_entities,
            "median_num_train": median_train,
            "representative": representative["dataset"],
        }
    if selected != EXPECTED_REPRESENTATIVES:
        raise RuntimeError(f"metric-free representative selection changed: {selected}")
    return selected, audit


def _transfer_regime(parent_manifest: dict[str, Any]) -> str:
    family = str(parent_manifest["ultra57_registry"]["family"])
    if family.startswith("inductive_entity_relation_"):
        return "inductive_entities_relations"
    if family.startswith("inductive_entity_"):
        return "inductive_entities"
    return "transductive"


def _artifact(b2_manifest: dict[str, Any], scope: str, seed: int | None = None) -> dict:
    matches = [
        item
        for item in b2_manifest["artifacts"]
        if item["scope"] == scope
        and item["block"] == "B2v2"
        and (seed is None or int(item.get("seed", -1)) == seed)
        and (scope != "evaluation" or item["split"] == "test")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify one B2v2 {scope} artifact")
    return matches[0]


def _verify(path: Path, expected: str, enabled: bool) -> None:
    if enabled and file_sha256(path) != expected:
        raise RuntimeError(f"source hash mismatch: {path}")


def _copy_query_blocks(
    source: np.ndarray,
    destination: np.ndarray,
    q512_positions: np.ndarray,
    num_entities: int,
) -> None:
    output_query = 0
    for triple_position in q512_positions:
        for side_offset in (0, 1):
            source_query = 2 * int(triple_position) + side_offset
            source_slice = slice(
                source_query * num_entities, (source_query + 1) * num_entities
            )
            destination_slice = slice(
                output_query * num_entities, (output_query + 1) * num_entities
            )
            destination[destination_slice] = source[source_slice]
            output_query += 1


def export_dataset(
    workspace_root: Path,
    output_root: Path,
    source_cluster: str,
    dataset: str,
    verify_source_hashes: bool,
    include_identity: bool = False,
) -> dict[str, Any]:
    parent_root = workspace_root / "results/ultra57_rfmlink_batch1_v1/immutable_assets"
    q512_root = workspace_root / "results/ultra57_rfmlink_q512_v1/immutable_assets"
    b2_root = workspace_root / "results/g2t_q512_hybrid_solver_metric_free_v1/feature_assets"
    parent_dir = parent_root / dataset
    q512_dir = q512_root / dataset
    b2_dir = b2_root / dataset
    parent_manifest_path = parent_dir / "manifest.json"
    b2_manifest_path = b2_dir / "manifest.json"
    parent_manifest = _load_json(parent_manifest_path)
    b2_manifest = _load_json(b2_manifest_path)
    if b2_manifest.get("labels_or_performance_metrics_accessed") is not False:
        raise RuntimeError(f"B2v2 is not metric-free for {dataset}")

    dataset_path = parent_dir / "dataset.npz"
    _verify(dataset_path, parent_manifest["dataset_sha256"], verify_source_hashes)
    with np.load(dataset_path, allow_pickle=False) as handle:
        test = np.asarray(handle["test"], dtype=np.int64)
        num_entities = int(handle["num_entities"])
        num_relations = int(handle["num_relations"])
        num_train = len(handle["train"])
    q64_split_indices = relation_stratified_indices(test, QUERY_BUDGET, QUERY_SEED)

    query_path = parent_dir / "test/queries.npz"
    split_manifest = parent_manifest["split_assets"]["test"]
    _verify(query_path, split_manifest["queries_sha256"], verify_source_hashes)
    with np.load(query_path, allow_pickle=False) as handle:
        frozen_q64_indices = np.asarray(handle["selected_split_indices"], dtype=np.int64)
        q64_targets = np.asarray(handle["target_entities"], dtype=np.int64)
        q64_sides = np.asarray(handle["side_codes"], dtype=np.int8)
    if not np.array_equal(q64_split_indices, frozen_q64_indices):
        raise RuntimeError(f"q64 selection differs from the immutable parent for {dataset}")

    q512_manifest = _load_json(q512_dir / "manifest.json")
    q512_query_path = q512_dir / "test/queries.npz"
    _verify(
        q512_query_path,
        q512_manifest["split_assets"]["test"]["queries_sha256"],
        verify_source_hashes,
    )
    with np.load(q512_query_path, allow_pickle=False) as handle:
        q512_split_indices = np.asarray(handle["selected_split_indices"], dtype=np.int64)
    positions = {int(source): index for index, source in enumerate(q512_split_indices)}
    if not set(q64_split_indices.tolist()) <= set(positions):
        raise RuntimeError(f"q64 is not nested inside the frozen q512 sample for {dataset}")
    q64_positions = np.asarray([positions[int(index)] for index in q64_split_indices])
    parent_features_path = parent_dir / "test/features.npy"
    parent_filtered_path = parent_dir / "test/filtered.npy"
    b2_evaluation = _artifact(b2_manifest, "evaluation")
    b2_features_path = workspace_root / b2_evaluation["path"]
    _verify(
        parent_features_path,
        split_manifest["features_sha256"],
        verify_source_hashes,
    )
    _verify(
        parent_filtered_path,
        split_manifest["filtered_sha256"],
        verify_source_hashes,
    )
    _verify(b2_features_path, b2_evaluation["sha256"], verify_source_hashes)

    source_b0 = np.load(parent_features_path, mmap_mode="r")
    source_b2v2 = np.load(b2_features_path, mmap_mode="r")
    source_filtered = np.load(parent_filtered_path, mmap_mode="r")
    q64_rows = 2 * QUERY_BUDGET * num_entities
    q512_rows = len(q512_split_indices) * 2 * num_entities
    if source_b0.shape != (q64_rows, len(B0_FEATURE_NAMES)):
        raise RuntimeError(f"unexpected B0 shape for {dataset}: {source_b0.shape}")
    if source_b2v2.shape != (q512_rows, len(B2V2_FEATURE_NAMES)):
        raise RuntimeError(f"unexpected B2v2 shape for {dataset}: {source_b2v2.shape}")

    temporary_dir = output_root / ".export_tmp" / dataset
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    q64_b2v2 = np.lib.format.open_memmap(
        temporary_dir / "candidate_b2v2.npy",
        mode="w+",
        dtype=np.float32,
        shape=(q64_rows, len(B2V2_FEATURE_NAMES)),
    )
    _copy_query_blocks(source_b2v2, q64_b2v2, q64_positions, num_entities)
    q64_b2v2.flush()

    selected_triples = test[q64_split_indices]
    arrays: dict[str, np.ndarray] = {
        "candidate_b0": source_b0,
        "candidate_b2v2": q64_b2v2,
        "filtered": np.asarray(source_filtered, dtype=np.bool_),
        "target_entities": q64_targets,
        "side_codes": q64_sides,
        "selected_split_indices": q64_split_indices,
        "selected_triples": selected_triples,
        "num_entities": np.asarray(num_entities, dtype=np.int64),
        "num_relations": np.asarray(num_relations, dtype=np.int64),
    }
    if include_identity:
        if len(q64_sides) != 2 * len(selected_triples):
            raise RuntimeError(f"unexpected directional-query layout for {dataset}")
        triple_rows = np.repeat(np.arange(len(selected_triples)), 2)
        if not np.array_equal(q64_sides, np.tile([0, 1], len(selected_triples))):
            raise RuntimeError(f"unexpected head/tail ordering for {dataset}")
        arrays["anchor_entities"] = np.where(
            q64_sides == 0,
            selected_triples[triple_rows, 0],
            selected_triples[triple_rows, 2],
        ).astype(np.int64)
        arrays["direction_relations"] = selected_triples[
            triple_rows, 1
        ].astype(np.int64)

    context_sources: dict[str, Any] = {}
    for seed in CONTEXT_SEEDS:
        context_path = parent_dir / f"contexts/{seed}/context.npz"
        expected_context_hash = parent_manifest["contexts"][str(seed)]["context_sha256"]
        _verify(context_path, expected_context_hash, verify_source_hashes)
        b2_context = _artifact(b2_manifest, "context", seed)
        b2_context_path = workspace_root / b2_context["path"]
        _verify(b2_context_path, b2_context["sha256"], verify_source_hashes)
        with np.load(context_path, allow_pickle=False) as handle:
            arrays[f"context_{seed}_X"] = np.asarray(handle["X"], dtype=np.float32)
            arrays[f"context_{seed}_y"] = np.asarray(handle["y"], dtype=np.int64)
            arrays[f"context_{seed}_triples"] = np.asarray(
                handle["triples"], dtype=np.int64
            )
            arrays[f"context_{seed}_fold_ids"] = np.asarray(
                handle["fold_ids"], dtype=np.int64
            )
        arrays[f"context_{seed}_b2v2"] = np.asarray(
            np.load(b2_context_path, mmap_mode="r"), dtype=np.float32
        )
        context_sources[str(seed)] = {
            "parent_context_sha256": expected_context_hash,
            "b2v2_sha256": b2_context["sha256"],
        }

    archive_path = output_root / f"{dataset}.npz"
    archive_temporary = temporary_dir / f"{dataset}.npz"
    np.savez_compressed(archive_temporary, **arrays)
    os.replace(archive_temporary, archive_path)
    array_hashes = {name: array_sha256(value) for name, value in arrays.items()}
    archive_sha256 = file_sha256(archive_path)
    del arrays, q64_b2v2, source_b0, source_b2v2, source_filtered
    shutil.rmtree(temporary_dir, ignore_errors=True)

    return {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "array_sha256": array_hashes,
        "source_cluster": source_cluster,
        "transfer_regime": _transfer_regime(parent_manifest),
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train": num_train,
        "test_triples": len(test),
        "q64_positions_within_frozen_q512": q64_positions.tolist(),
        "source_parent_manifest_sha256": file_sha256(parent_manifest_path),
        "source_q512_parent_manifest_sha256": file_sha256(
            q512_dir / "manifest.json"
        ),
        "source_b2v2_manifest_sha256": file_sha256(b2_manifest_path),
        "source_contexts": context_sources,
        "source_test": {
            "query_sha256": split_manifest["queries_sha256"],
            "features_sha256": split_manifest["features_sha256"],
            "filtered_sha256": split_manifest["filtered_sha256"],
            "q512_query_sha256": q512_manifest["split_assets"]["test"][
                "queries_sha256"
            ],
            "b2v2_sha256": b2_evaluation["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rfmlink-workspace",
        type=Path,
        required=True,
        help="Workspace containing results/ultra57_rfmlink_q512_v1",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-source-byte-hashes", action="store_true")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    workspace_root = args.rfmlink_workspace.resolve()
    output_root = args.output_root.resolve()
    parent_root = workspace_root / "results/ultra57_rfmlink_q512_v1/immutable_assets"
    if args.all_datasets:
        selected = {
            path.name: path.name
            for path in sorted(parent_root.iterdir())
            if path.is_dir() and (path / "manifest.json").is_file()
        }
        if len(selected) != 57:
            raise RuntimeError(f"expected 57 immutable datasets, found {len(selected)}")
        selection_audit = {
            "rule": "all 57 immutable ULTRA datasets; no metric-based selection"
        }
    else:
        selected, selection_audit = select_representatives(parent_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = {}
    jobs = []
    for family in sorted(selected):
        dataset = selected[family]
        metadata_path = output_root / f"{dataset}.metadata.json"
        archive_path = output_root / f"{dataset}.npz"
        if metadata_path.is_file() and archive_path.is_file():
            metadata = _load_json(metadata_path)
            if file_sha256(archive_path) == metadata.get("archive_sha256"):
                datasets[dataset] = metadata
                print(f"REUSE dataset={dataset}", flush=True)
                continue
        jobs.append((family, dataset))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                export_dataset,
                workspace_root,
                output_root,
                family,
                dataset,
                not args.skip_source_byte_hashes,
                args.all_datasets,
            ): (family, dataset)
            for family, dataset in jobs
        }
        for future in as_completed(futures):
            family, dataset = futures[future]
            print(f"EXPORT source_cluster={family} dataset={dataset}", flush=True)
            datasets[dataset] = future.result()
            atomic_json(output_root / f"{dataset}.metadata.json", datasets[dataset])
            print(
                f"PASS dataset={dataset} archive_sha256="
                f"{datasets[dataset]['archive_sha256']}",
                flush=True,
            )

    manifest = {
        "asset_id": (
            "rfmlink_frozen_interface_ultra57_v2"
            if args.all_datasets
            else ASSET_ID
        ),
        "archive_schema": 2 if args.all_datasets else 1,
        "scientific_status": "prospective decoder-transfer replication",
        "claim_boundary": (
            "Representatives and all LimiX outcomes were frozen before this run. "
            "Historical TabPFN B0/B2v2 outcomes at q512 were already visible, so "
            "this is not a fully blind feature confirmation."
        ),
        "selection": {
            "rule": (
                "one member per eligible source family minimizing absolute distance "
                "to family-median entity count, then family-median train-edge count, "
                "then lexicographic dataset name"
            ),
            "excluded_development_sources": ["codex", "nell"],
            "audit": selection_audit,
        },
        "query_selection": (
            "exact RFMLink relation_stratified_indices(test, 64, 20260715), "
            "both head and tail directions"
        ),
        "context_selection": (
            "nested class-balanced subsets of the immutable 400+400 contexts"
        ),
        "contexts_per_class": (
            [20, 50, 100, 200]
            if args.all_datasets
            else [20, 50, 100, 200, 400]
        ),
        "context_seeds": list(CONTEXT_SEEDS),
        "b0_feature_names": list(B0_FEATURE_NAMES),
        "b2v2_feature_names": list(B2V2_FEATURE_NAMES),
        "permutation_control": {
            "evaluation": (
                "independent deterministic column-wise affine permutation within "
                "each all-entity query candidate set"
            ),
            "context": (
                "independent deterministic column-wise affine permutation over the "
                "selected rows for each dataset, seed, and context budget"
            ),
            "preserves": "column count and within-scope univariate marginals",
            "breaks": "candidate alignment and cross-column covariance",
        },
        "datasets": datasets,
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(
        f"PASS asset_id={ASSET_ID} datasets={len(datasets)} "
        f"manifest_sha256={file_sha256(output_root / 'manifest.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
