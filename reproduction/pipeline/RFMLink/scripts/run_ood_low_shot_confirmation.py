#!/usr/bin/env python3
"""Stage-1 out-of-dataset confirmation of the frozen RFMLink table.

This is a standalone test of label-efficient adaptation. It is not a matched
comparison against any particular KG method. The representation, TabPFN
checkpoint, context sizes, and evaluator are fixed before target test scores
are observed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_context_intervention_pilot as pilot  # noqa: E402
import run_deterministic_transfer_suite as deterministic  # noqa: E402
from rfmlink.contexts import build_context  # noqa: E402
from rfmlink.decoders import fit_predict_tabpfn3  # noqa: E402
from rfmlink.paths import ULTRA_ROOT  # noqa: E402
from rfmlink.sbm_import import tied_rank_positions  # noqa: E402
from rfmlink.ultra_wrapper import (  # noqa: E402
    build_train_graph,
    load_ultra_model,
    score_all_tails,
    score_triples,
)
from rfmlink.vectorized_invariants import (  # noqa: E402
    FEATURE_NAMES as INVARIANT_NAMES,
    VectorizedInvariantBuilder,
)


PROTOCOL_REVISION = "ood_low_shot_confirmation_v1"
DATASETS = ("pharmkg8k", "dbpedia50", "aristov4")
DATASET_RATIONALE = {
    "pharmkg8k": "dense biomedical KG; unseen during RFMLink development",
    "dbpedia50": "sparse general-domain KG; unseen during RFMLink development",
    "aristov4": "large relation-rich commonsense KG; unseen during RFMLink development",
}
SEEDS = tuple(range(101, 106))
BUDGETS = (5, 20, 200)
LOW_SHOT_BUDGETS = (5, 20)
CAUSAL_BUDGET = 20
QUERY_TRIPLES_PER_SPLIT = 64
TUNING_QUERY_TRIPLES = 32
CONTEXT_POOL_PER_CLASS = 200
NUM_FOLDS = 10
NUM_CLUSTERS = 24
FEATURE_NAMES = [*INVARIANT_NAMES, "ultra_score"]
CATEGORICAL_INDICES = [0, 16]
DECODERS = ("tabpfn_v3", "logistic_frozen", "hgb_frozen", "mlp_frozen")
CAUSAL_DECODERS = ("tabpfn_v3", "logistic_frozen")
CLASSICAL_DECODERS = DECODERS[1:]

FAMILY_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL_REVISION
DATA_ROOT = FAMILY_ROOT / "immutable_assets"
RUN_ROOT = FAMILY_ROOT / "runs"
TUNING_ROOT = FAMILY_ROOT / "classical_selection"
LOG_ROOT = FAMILY_ROOT / "launcher_logs"
SUMMARY_PATH = FAMILY_ROOT / "summary.json"
SUMMARY_MD = FAMILY_ROOT / "summary.md"
PLAN_PATH = WORKSPACE_ROOT / "results" / "analysis" / f"{PROTOCOL_REVISION}_plan.json"
PYKEEN_CACHE = RFMLINK_ROOT / "data" / "ood_confirmation_pykeen"
ULTRA_CHECKPOINT = ULTRA_ROOT / "ckpts" / "ultra_50g.pth"

REQUIRED_RUN_FILES = (
    "config.json",
    "metrics.json",
    "timings.json",
    "query_predictions.parquet",
    "context_manifest.json",
    "run.log",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    vectorized = RFMLINK_ROOT / "rfmlink" / "vectorized_invariants.py"
    return {
        "runner": file_sha256(Path(__file__)),
        "vectorized_invariants": file_sha256(vectorized),
    }


def protocol_specification() -> dict:
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "scientific_status": "confirmatory stage 1; target test scores unseen at registration",
        "research_question": (
            "Does a frozen general-purpose TabPFN provide a repeatable low-label "
            "advantage over matched classical decoders on top of frozen ULTRA and "
            "the frozen 18-column graph table?"
        ),
        "positioning": (
            "standalone study of frozen-model composition and label efficiency; "
            "no method-specific matched-comparison claim"
        ),
        "datasets": list(DATASETS),
        "dataset_rationale": DATASET_RATIONALE,
        "development_datasets_excluded": [
            "umls", "kinships", "nations", "codex-small", "wn18rr", "fb15k237"
        ],
        "seeds": list(SEEDS),
        "budgets_per_class": list(BUDGETS),
        "low_shot_budgets_per_class": list(LOW_SHOT_BUDGETS),
        "causal_budget_per_class": CAUSAL_BUDGET,
        "representation": {
            "columns": FEATURE_NAMES,
            "num_columns": len(FEATURE_NAMES),
            "definition": "18 frozen train-graph invariants plus frozen ULTRA scalar score",
            "raw_entity_ids": False,
            "raw_cluster_ids": False,
            "context_feature_protocol": "10-fold cross-fitting",
            "evaluation_feature_protocol": "full train graph only",
            "vectorized_implementation": (
                "algebraically equivalent candidate-block implementation, validated "
                "against the audited scalar reference before registration"
            ),
        },
        "context": {
            "source": "train split only",
            "pool": "200 positives and 200 train-filtered corrupted negatives per seed",
            "primary_selection": "nested deterministic class-balanced subsets",
            "negative_filtering": "train true triples only",
            "causal_intervention": (
                "correct versus label-swapped relation-matched pairs with identical rows/features"
            ),
        },
        "evaluation": {
            "primary": "filtered all-entity ranking; head, tail, and combined reported separately",
            "query_subset": (
                "64 deterministic relation-stratified validation/test triples; each contributes "
                "one complete head block and one complete tail block"
            ),
            "candidate_subset": False,
            "filter": "train+validation+test true triples, evaluator only",
            "tie_policy": "expected uniform tie breaking",
            "absolute_comparison_to_full_test_tables": False,
        },
        "tabpfn": {
            "version_family": "v3",
            "checkpoint": str(pilot.CHECKPOINT),
            "n_estimators": 8,
            "fit_mode": "fit_with_cache",
            "target_parameter_updates": False,
        },
        "classical_controls": {
            "families": list(CLASSICAL_DECODERS),
            "selection": (
                "small grids containing only development-selected optima; select once per "
                "dataset/family on seed 101, 50+50 context, first 32 registered validation triples; "
                "freeze for all test cells"
            ),
            "test_used_for_selection": False,
        },
        "decision_rule": {
            "dataset_pass": [
                "mean paired combined-MRR delta over seeds and budgets 5,20 >= 0.010",
                "at least 6 of 10 seed-budget deltas are positive",
                "paired-correct minus paired-permuted TabPFN MRR >= 0.020",
                "at least 4 of 5 label-intervention seed deltas are positive",
            ],
            "overall_support": "at least two of three untouched datasets pass",
            "failure": "zero or one datasets pass",
            "interpretation": (
                "A pass supports out-of-dataset label efficiency, not a unique mechanism "
                "or universal link-prediction superiority"
            ),
        },
    }


def all_jobs(group="all") -> list[dict]:
    jobs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for budget in BUDGETS:
                for decoder in DECODERS:
                    jobs.append({
                        "dataset": dataset,
                        "seed": seed,
                        "budget": budget,
                        "condition": "correct_labels",
                        "decoder": decoder,
                    })
            for condition in ("paired_correct_labels", "paired_permuted_labels"):
                for decoder in CAUSAL_DECODERS:
                    jobs.append({
                        "dataset": dataset,
                        "seed": seed,
                        "budget": CAUSAL_BUDGET,
                        "condition": condition,
                        "decoder": decoder,
                    })
    if group == "all":
        return jobs
    if group == "tabpfn":
        return [job for job in jobs if job["decoder"] == "tabpfn_v3"]
    if group == "classical":
        return [job for job in jobs if job["decoder"] != "tabpfn_v3"]
    raise ValueError(group)


def register() -> None:
    if PLAN_PATH.exists():
        raise FileExistsError(f"registration already exists: {PLAN_PATH}")
    payload = {
        "status": "REGISTERED_BEFORE_TARGET_TEST_METRICS",
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hashes": source_hashes(),
        "protocol": protocol_specification(),
        "planned_jobs": {
            "total": len(all_jobs()),
            "tabpfn": len(all_jobs("tabpfn")),
            "classical": len(all_jobs("classical")),
        },
        "estimated_runtime_hours": {
            "asset_materialization": [2.0, 8.0],
            "four_parallel_tabpfn_workers": [5.0, 12.0],
            "four_classical_workers": [1.0, 4.0],
            "warning": "first estimate; update only from observed phase timings, never metrics",
        },
    }
    pilot.atomic_json(PLAN_PATH, payload)
    print(f"REGISTERED {PLAN_PATH}", flush=True)


def validate_registration() -> dict:
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"missing registration: {PLAN_PATH}")
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if payload["source_hashes"] != source_hashes():
        raise RuntimeError("registered source changed; create a new protocol revision")
    if payload["protocol"] != protocol_specification():
        raise RuntimeError("registered protocol changed")
    return payload


def dataset_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset


def split_dir(dataset: str, split: str) -> Path:
    return dataset_dir(dataset) / split


def context_path(dataset: str, seed: int) -> Path:
    return dataset_dir(dataset) / "contexts" / str(seed) / "context.npz"


def _atomic_npy(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def load_pykeen_dataset(name: str) -> dict:
    from pykeen.datasets import get_dataset

    dataset = get_dataset(
        dataset=name,
        dataset_kwargs={
            "cache_root": PYKEEN_CACHE,
            "create_inverse_triples": False,
        },
    )
    splits = {
        "train": dataset.training.mapped_triples.detach().cpu().numpy().astype(np.int64),
        "valid": dataset.validation.mapped_triples.detach().cpu().numpy().astype(np.int64),
        "test": dataset.testing.mapped_triples.detach().cpu().numpy().astype(np.int64),
    }
    for split, triples in splits.items():
        if len(np.unique(triples, axis=0)) != len(triples):
            raise RuntimeError(f"{name}/{split} contains duplicate triples")
    train_set = {tuple(row) for row in splits["train"].tolist()}
    valid_set = {tuple(row) for row in splits["valid"].tolist()}
    test_set = {tuple(row) for row in splits["test"].tolist()}
    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise RuntimeError(f"{name} has overlapping splits")
    return {
        **splits,
        "num_entities": int(dataset.num_entities),
        "num_relations": int(dataset.num_relations),
        "entity_to_id": dataset.training.entity_to_id,
        "relation_to_id": dataset.training.relation_to_id,
    }


def relation_stratified_indices(triples, size: int, seed: int) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int64)
    if len(triples) <= size:
        return np.arange(len(triples), dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_relation = {}
    for relation in np.unique(triples[:, 1]):
        indices = np.flatnonzero(triples[:, 1] == relation)
        by_relation[int(relation)] = indices[rng.permutation(len(indices))].tolist()
    relations = np.asarray(sorted(by_relation), dtype=np.int64)
    relations = relations[rng.permutation(len(relations))]
    selected = []
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
        raise RuntimeError("relation-stratified query selection failed")
    return np.asarray(selected, dtype=np.int64)


def _all_true_maps(data: dict):
    tails = {}
    heads = {}
    for split in ("train", "valid", "test"):
        for h, r, t in data[split].tolist():
            tails.setdefault((h, r), set()).add(t)
            heads.setdefault((r, t), set()).add(h)
    return tails, heads


def materialize_eval_split(dataset: str, split: str, data: dict, builder,
                           model, graph) -> dict:
    root = split_dir(dataset, split)
    root.mkdir(parents=True, exist_ok=True)
    triples = data[split]
    selected = relation_stratified_indices(
        triples,
        QUERY_TRIPLES_PER_SPLIT,
        20260714 + (0 if split == "valid" else 1),
    )
    selected_triples = triples[selected]
    num_entities = data["num_entities"]
    num_queries = 2 * len(selected)
    rows = num_queries * num_entities

    features_path = root / "features.npy"
    filtered_path = root / "filtered.npy"
    features_tmp = root / f".features.npy.{os.getpid()}.tmp"
    filtered_tmp = root / f".filtered.npy.{os.getpid()}.tmp"
    features = np.lib.format.open_memmap(
        features_tmp, mode="w+", dtype=np.float32, shape=(rows, len(FEATURE_NAMES))
    )
    filtered = np.lib.format.open_memmap(
        filtered_tmp, mode="w+", dtype=np.bool_, shape=(num_queries, num_entities)
    )
    tails, heads = _all_true_maps(data)
    pairs = []
    target_entities = []
    side_codes = []
    for triple_index, (h, r, t) in enumerate(selected_triples.tolist()):
        for side in ("tail", "head"):
            query_index = 2 * triple_index + (side == "head")
            start = query_index * num_entities
            stop = start + num_entities
            features[start:stop, :18] = builder.candidate_block(h, r, t, side)
            mask = np.zeros(num_entities, dtype=np.bool_)
            if side == "tail":
                target = t
                alternatives = tails.get((h, r), set())
                pairs.append((h, r))
                side_codes.append(0)
            else:
                target = h
                alternatives = heads.get((r, t), set())
                pairs.append((t, r + data["num_relations"]))
                side_codes.append(1)
            if alternatives:
                mask[np.fromiter(alternatives, dtype=np.int64)] = True
            mask[target] = False
            filtered[query_index] = mask
            target_entities.append(target)

    scores = score_all_tails(model, graph, np.asarray(pairs), batch_size=1)
    features[:, 18] = scores.reshape(-1)
    features.flush()
    filtered.flush()
    del features, filtered, scores
    os.replace(features_tmp, features_path)
    os.replace(filtered_tmp, filtered_path)
    metadata_path = root / "queries.npz"
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            selected_split_indices=selected,
            selected_triples=selected_triples,
            target_entities=np.asarray(target_entities, dtype=np.int64),
            side_codes=np.asarray(side_codes, dtype=np.int8),
        )
    os.replace(temporary, metadata_path)
    return {
        "split": split,
        "source_triples": int(len(triples)),
        "selected_triples": int(len(selected)),
        "queries": int(num_queries),
        "candidate_rows": int(rows),
        "features_sha256": file_sha256(features_path),
        "filtered_sha256": file_sha256(filtered_path),
        "queries_sha256": file_sha256(metadata_path),
        "selection_hash": pilot.array_hash(selected),
    }


def crossfit_context_features(train, context_triples, labels, num_entities,
                              num_relations, seed):
    train = np.asarray(train, dtype=np.int64)
    context_triples = np.asarray(context_triples, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    heldout_by_fold = []
    fold_by_triple = {}
    for fold, (_, heldout) in enumerate(splitter.split(train)):
        heldout_by_fold.append(heldout)
        for row in train[heldout].tolist():
            fold_by_triple[tuple(row)] = fold
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    for index in np.flatnonzero(labels == 1):
        fold_ids[index] = fold_by_triple[tuple(context_triples[index])]
    negatives = np.flatnonzero(labels == 0)
    negative_folds = np.resize(np.arange(NUM_FOLDS), len(negatives))
    rng = np.random.default_rng(seed + 30_000)
    rng.shuffle(negative_folds)
    fold_ids[negatives] = negative_folds
    output = np.empty((len(labels), len(INVARIANT_NAMES)), dtype=np.float32)
    fold_manifest = []
    for fold, heldout in enumerate(heldout_by_fold):
        keep = np.ones(len(train), dtype=np.bool_)
        keep[heldout] = False
        rows = np.flatnonzero(fold_ids == fold)
        builder = VectorizedInvariantBuilder(
            train[keep], num_entities, num_relations,
            num_clusters=NUM_CLUSTERS, seed=seed + fold,
        )
        output[rows] = builder.feature_rows(context_triples[rows])
        fold_manifest.append({
            "fold": fold,
            "graph_triples": int(keep.sum()),
            "context_rows": int(len(rows)),
            "positive_rows": int(np.sum(labels[rows] == 1)),
            "negative_rows": int(np.sum(labels[rows] == 0)),
        })
    return output, fold_ids, fold_manifest


def materialize_context(dataset: str, seed: int, data: dict, model, graph) -> dict:
    path = context_path(dataset, seed)
    if path.exists():
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    triples, labels, source = build_context(
        dataset,
        data["train"],
        data["num_entities"],
        seed,
        CONTEXT_POOL_PER_CLASS,
        CONTEXT_POOL_PER_CLASS,
        negative_filtering_mode="train_only",
    )
    explicit, fold_ids, folds = crossfit_context_features(
        data["train"], triples, labels, data["num_entities"],
        data["num_relations"], seed,
    )
    scores = np.empty(len(labels), dtype=np.float32)
    positive = labels == 1
    scores[positive], _ = score_triples(
        model, graph, triples[positive], batch_size=8,
        capture_representations=False, remove_direct_edges=True,
    )
    scores[~positive], _ = score_triples(
        model, graph, triples[~positive], batch_size=64,
        capture_representations=False, remove_direct_edges=False,
    )
    X = np.concatenate([explicit, scores[:, None]], axis=1).astype(np.float32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, X=X, y=labels, triples=triples, fold_ids=fold_ids
        )
    os.replace(temporary, path)
    valid_test = {
        tuple(row)
        for split in ("valid", "test")
        for row in data[split].tolist()
    }
    negative_rows = triples[labels == 0]
    noise = sum(tuple(row) in valid_test for row in negative_rows.tolist())
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "shape": list(X.shape),
        "feature_names": FEATURE_NAMES,
        "context_sha256": file_sha256(path),
        "feature_hash": pilot.array_hash(X),
        "row_hash": pilot.array_hash(triples),
        "label_hash": pilot.array_hash(labels),
        "source_config": source,
        "negative_label_noise_audit": {
            "valid_or_test_true_triples": int(noise),
            "rate": float(noise / max(len(negative_rows), 1)),
            "audit_only_not_used_for_sampling": True,
        },
        "folds": folds,
    }
    pilot.atomic_json(path.with_suffix(".json"), manifest)
    return manifest


def dataset_complete(dataset: str) -> bool:
    manifest_path = dataset_dir(dataset) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_hashes"] != source_hashes():
            return False
        for split in ("valid", "test"):
            root = split_dir(dataset, split)
            for name in ("features.npy", "filtered.npy", "queries.npz"):
                if not (root / name).exists():
                    return False
        return all(context_path(dataset, seed).exists() for seed in SEEDS)
    except Exception:
        return False


def prepare_dataset(dataset: str) -> dict:
    validate_registration()
    if dataset_complete(dataset):
        print(f"ASSET READY {dataset}", flush=True)
        return json.loads((dataset_dir(dataset) / "manifest.json").read_text())
    started = time.time()
    data = load_pykeen_dataset(dataset)
    root = dataset_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    dataset_npz = root / "dataset.npz"
    temporary = dataset_npz.with_name(f".{dataset_npz.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, train=data["train"], valid=data["valid"], test=data["test"],
            num_entities=np.asarray(data["num_entities"]),
            num_relations=np.asarray(data["num_relations"]),
        )
    os.replace(temporary, dataset_npz)
    builder = VectorizedInvariantBuilder(
        data["train"], data["num_entities"], data["num_relations"],
        num_clusters=NUM_CLUSTERS, seed=7,
    )
    model = load_ultra_model(ULTRA_CHECKPOINT, device="cuda")
    graph = build_train_graph(
        data["train"], data["num_entities"], data["num_relations"], device="cuda"
    )
    split_manifests = {
        split: materialize_eval_split(dataset, split, data, builder, model, graph)
        for split in ("valid", "test")
    }
    contexts = {
        str(seed): materialize_context(dataset, seed, data, model, graph)
        for seed in SEEDS
    }
    del model, graph, builder
    import torch
    torch.cuda.empty_cache()
    manifest = {
        "status": "IMMUTABLE_CONFIRMATORY_INPUT",
        "dataset": dataset,
        "rationale": DATASET_RATIONALE[dataset],
        "source": "PyKEEN 1.11.1 official dataset loader and published splits",
        "num_entities": data["num_entities"],
        "num_relations": data["num_relations"],
        "num_train": int(len(data["train"])),
        "num_valid": int(len(data["valid"])),
        "num_test": int(len(data["test"])),
        "dataset_sha256": file_sha256(dataset_npz),
        "split_assets": split_manifests,
        "contexts": {
            seed: {"context_sha256": value["context_sha256"]}
            for seed, value in contexts.items()
        },
        "source_hashes": source_hashes(),
        "registration": str(PLAN_PATH),
        "checkpoint": str(ULTRA_CHECKPOINT),
        "checkpoint_sha256": file_sha256(ULTRA_CHECKPOINT),
        "materialization_seconds": time.time() - started,
        "packages": deterministic.package_versions(),
        "hardware": pilot.hardware_info(),
    }
    pilot.atomic_json(root / "manifest.json", manifest)
    print(
        f"ASSET WRITTEN {dataset} entities={data['num_entities']} "
        f"train={len(data['train'])} seconds={manifest['materialization_seconds']:.1f}",
        flush=True,
    )
    return manifest


def load_split_asset(dataset: str, split: str) -> dict:
    root = split_dir(dataset, split)
    features = np.load(root / "features.npy", mmap_mode="r")
    filtered = np.load(root / "filtered.npy", mmap_mode="r")
    with np.load(root / "queries.npz", allow_pickle=False) as archive:
        metadata = {name: archive[name] for name in archive.files}
    if features.shape[0] != filtered.shape[0] * filtered.shape[1]:
        raise RuntimeError(f"candidate alignment failed for {dataset}/{split}")
    return {"X": features, "filtered": filtered, **metadata}


def load_context_asset(dataset: str, seed: int) -> dict:
    path = context_path(dataset, seed)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def nested_indices(labels, seed: int, budget: int) -> np.ndarray:
    orders = {}
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng = np.random.default_rng(np.random.SeedSequence([20260714, seed, label, 17]))
        orders[label] = indices[rng.permutation(len(indices))]
    selected = np.sort(np.concatenate([orders[0][:budget], orders[1][:budget]]))
    if np.sum(labels[selected] == 0) != budget or np.sum(labels[selected] == 1) != budget:
        raise AssertionError("context selection lost class balance")
    return selected


def paired_indices(context: dict, seed: int, budget: int):
    labels = context["y"]
    relations = context["triples"][:, 1]
    rng = np.random.default_rng(np.random.SeedSequence([20260714, seed, 911]))
    pairs = []
    for relation in sorted(np.unique(relations)):
        negative = np.flatnonzero((relations == relation) & (labels == 0))
        positive = np.flatnonzero((relations == relation) & (labels == 1))
        negative = negative[rng.permutation(len(negative))]
        positive = positive[rng.permutation(len(positive))]
        pairs.extend((int(n), int(p)) for n, p in zip(negative, positive))
    if len(pairs) < budget:
        raise RuntimeError(f"only {len(pairs)} relation-matched context pairs")
    pairs = [pairs[index] for index in rng.permutation(len(pairs))[:budget]]
    selected = np.asarray([index for pair in pairs for index in pair], dtype=np.int64)
    swap = np.arange(2 * budget).reshape(-1, 2)[:, ::-1].reshape(-1)
    if not np.all(labels[selected] != labels[selected][swap]):
        raise AssertionError("causal intervention did not swap all labels")
    return selected, swap


def context_payload(dataset: str, seed: int, budget: int, condition: str) -> dict:
    context = load_context_asset(dataset, seed)
    if condition == "correct_labels":
        selected = nested_indices(context["y"], seed, budget)
        permutation = np.arange(len(selected))
        policy = "nested_random_class_balanced"
    else:
        selected, swap = paired_indices(context, seed, budget)
        permutation = np.arange(len(selected)) if condition == "paired_correct_labels" else swap
        policy = "relation_matched_pairs"
    original = context["y"][selected]
    return {
        "X": context["X"][selected].astype(np.float32, copy=False),
        "y": original[permutation],
        "original_y": original,
        "triples": context["triples"][selected],
        "selected": selected,
        "permutation": permutation,
        "policy": policy,
    }


def metrics_from_scores(asset: dict, scores) -> tuple[dict, pd.DataFrame]:
    scores = np.asarray(scores, dtype=np.float64)
    filtered = asset["filtered"]
    targets = asset["target_entities"]
    sides = asset["side_codes"]
    num_queries, num_entities = filtered.shape
    if len(scores) != num_queries * num_entities:
        raise ValueError("prediction length does not match candidate blocks")
    records = []
    for query in range(num_queries):
        block = scores[query * num_entities : (query + 1) * num_entities]
        target = int(targets[query])
        valid = ~np.asarray(filtered[query])
        positions = tied_rank_positions(block[valid], float(block[target]))
        records.append({
            "query_index": query,
            "source_triple_index": query // 2,
            "side": "head" if sides[query] else "tail",
            "target_entity": target,
            "target_score": float(block[target]),
            "rank": float(np.mean(positions)),
            "reciprocal_rank": float(np.mean(1.0 / positions)),
            "hits_at_1": float(np.mean(positions <= 1)),
            "hits_at_3": float(np.mean(positions <= 3)),
            "hits_at_10": float(np.mean(positions <= 10)),
        })
    frame = pd.DataFrame(records)
    output = {}
    for side, group in (
        ("head", frame[frame["side"] == "head"]),
        ("tail", frame[frame["side"] == "tail"]),
        ("combined", frame),
    ):
        output[side] = {
            "queries": int(len(group)),
            "mrr": float(group["reciprocal_rank"].mean()),
            "hits_at_1": float(group["hits_at_1"].mean()),
            "hits_at_3": float(group["hits_at_3"].mean()),
            "hits_at_10": float(group["hits_at_10"].mean()),
            "mean_rank": float(group["rank"].mean()),
        }
    return output, frame


CLASSICAL_GRIDS = {
    "logistic_frozen": [{"C": 0.1}, {"C": 1.0}],
    "hgb_frozen": [{
        "learning_rate": 0.05,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    }],
    "mlp_frozen": [
        {"hidden_layer_sizes": [16], "alpha": 0.001},
        {"hidden_layer_sizes": [64], "alpha": 0.01},
    ],
}


def build_classical(decoder: str, params: dict, seed: int, context_rows: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if decoder == "hgb_frozen":
        return HistGradientBoostingClassifier(
            random_state=seed,
            categorical_features=[16],
            max_iter=200,
            **params,
        )
    categorical_names = [FEATURE_NAMES[index] for index in CATEGORICAL_INDICES]
    numeric_names = [name for name in FEATURE_NAMES if name not in categorical_names]
    dense = decoder == "mlp_frozen"
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), categorical_names),
            ("num", StandardScaler(), numeric_names),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )
    if decoder == "logistic_frozen":
        classifier = LogisticRegression(
            max_iter=5000, random_state=seed, solver="lbfgs", **params
        )
    elif decoder == "mlp_frozen":
        classifier = MLPClassifier(
            activation="relu",
            batch_size=min(64, max(2, context_rows)),
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=context_rows >= 50,
            validation_fraction=0.2,
            n_iter_no_change=40,
            random_state=seed,
            alpha=params["alpha"],
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        )
    else:
        raise ValueError(decoder)
    return Pipeline([("pre", preprocessor), ("model", classifier)])


def model_input(decoder: str, matrix):
    if decoder in ("logistic_frozen", "mlp_frozen"):
        return pd.DataFrame(matrix, columns=FEATURE_NAMES)
    return matrix


def positive_probability(model, matrix):
    classes = np.asarray(model.classes_)
    position = np.flatnonzero(classes == 1)
    if len(position) != 1:
        raise RuntimeError("positive class missing")
    return model.predict_proba(matrix)[:, int(position[0])]


def tuning_path(dataset: str, decoder: str) -> Path:
    return TUNING_ROOT / dataset / decoder / "selection.json"


def tune_classical(dataset: str, decoder: str) -> dict:
    output = tuning_path(dataset, decoder)
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    context = context_payload(dataset, 101, 50, "correct_labels")
    valid = load_split_asset(dataset, "valid")
    num_entities = valid["filtered"].shape[1]
    rows = 2 * TUNING_QUERY_TRIPLES * num_entities
    X_valid = valid["X"][:rows]
    subset = {**valid, "X": X_valid, "filtered": valid["filtered"][:2 * TUNING_QUERY_TRIPLES]}
    subset["target_entities"] = valid["target_entities"][:2 * TUNING_QUERY_TRIPLES]
    subset["side_codes"] = valid["side_codes"][:2 * TUNING_QUERY_TRIPLES]
    search = []
    for order, params in enumerate(CLASSICAL_GRIDS[decoder]):
        model = build_classical(decoder, params, 101, len(context["X"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(model_input(decoder, context["X"]), context["y"])
        scores = positive_probability(model, model_input(decoder, X_valid))
        metrics, _ = metrics_from_scores(subset, scores)
        search.append({
            "order": order,
            "params": params,
            "validation_combined_mrr": metrics["combined"]["mrr"],
            "warnings": [str(item.message) for item in caught],
        })
    selected = max(search, key=lambda item: (item["validation_combined_mrr"], -item["order"]))
    payload = {
        "dataset": dataset,
        "decoder": decoder,
        "selection_split": "validation",
        "selection_context": "seed101 50pos+50neg",
        "selection_query_triples": TUNING_QUERY_TRIPLES,
        "test_used": False,
        "search": search,
        "selected_params": selected["params"],
    }
    pilot.atomic_json(output, payload)
    print(f"TUNED {dataset} {decoder} {selected['params']}", flush=True)
    return payload


def run_dir(job: dict, smoke=False) -> Path:
    seed = f"{job['seed']}_smoke" if smoke else str(job["seed"])
    return (
        RUN_ROOT / job["dataset"] / job["condition"] / job["decoder"]
        / f"{job['budget']}pos_{job['budget']}neg" / seed
    )


def run_complete(path: Path) -> bool:
    if not all((path / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    try:
        metrics = json.loads((path / "metrics.json").read_text())
        return np.isfinite(metrics["test"]["combined"]["mrr"])
    except Exception:
        return False


@contextlib.contextmanager
def run_lock(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / ".run.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield True
    except BlockingIOError:
        yield False
    finally:
        handle.close()


def execute_job(job: dict, smoke=False):
    validate_registration()
    output = run_dir(job, smoke)
    if not smoke and run_complete(output):
        return json.loads((output / "metrics.json").read_text())
    with run_lock(output) as acquired:
        if not acquired:
            return None
        if not smoke and run_complete(output):
            return json.loads((output / "metrics.json").read_text())
        started = time.time()
        try:
            context = context_payload(
                job["dataset"], job["seed"], job["budget"], job["condition"]
            )
            test = load_split_asset(job["dataset"], "test")
            valid = load_split_asset(job["dataset"], "valid")
            if smoke:
                num_entities = test["filtered"].shape[1]
                query_count = 2
                test = {
                    **test,
                    "X": test["X"][:query_count * num_entities],
                    "filtered": test["filtered"][:query_count],
                    "target_entities": test["target_entities"][:query_count],
                    "side_codes": test["side_codes"][:query_count],
                }
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            fit_started = time.time()
            if job["decoder"] == "tabpfn_v3":
                predictions, info = fit_predict_tabpfn3(
                    context["X"], context["y"], [test["X"]], job["seed"],
                    categorical_indices=CATEGORICAL_INDICES,
                    fit_mode="fit_with_cache",
                    checkpoint_override=str(pilot.CHECKPOINT),
                    n_estimators=1 if smoke else 8,
                )
                test_scores = predictions[0]
                valid_scores = None
                selected_params = {"n_estimators": 1 if smoke else 8}
            else:
                selection = tune_classical(job["dataset"], job["decoder"])
                selected_params = selection["selected_params"]
                model = build_classical(
                    job["decoder"], selected_params, job["seed"], len(context["X"])
                )
                model.fit(model_input(job["decoder"], context["X"]), context["y"])
                info = {"fit_time": time.time() - fit_started}
                valid_scores = positive_probability(
                    model, model_input(job["decoder"], valid["X"])
                )
                test_scores = positive_probability(
                    model, model_input(job["decoder"], test["X"])
                )
            test_metrics, frame = metrics_from_scores(test, test_scores)
            metrics = {"test": test_metrics}
            if valid_scores is not None and not smoke:
                metrics["valid"], _ = metrics_from_scores(valid, valid_scores)
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            if smoke:
                pilot.atomic_json(output / "smoke_metrics.json", metrics)
                (output / "run.log").write_text("SMOKE OK\n", encoding="utf-8")
                return metrics
            context_manifest = {
                "selection_policy": context["policy"],
                "selected_pool_rows": context["selected"].tolist(),
                "row_hash": pilot.array_hash(context["triples"]),
                "feature_hash": pilot.array_hash(context["X"]),
                "original_label_hash": pilot.array_hash(context["original_y"]),
                "supplied_label_hash": pilot.array_hash(context["y"]),
                "permutation_hash": pilot.array_hash(context["permutation"]),
                "labels_changed": int(np.sum(context["original_y"] != context["y"])),
            }
            config = {
                "experiment_family": PROTOCOL_REVISION,
                "scientific_status": "confirmatory stage 1",
                **job,
                "feature_names": FEATURE_NAMES,
                "num_features": len(FEATURE_NAMES),
                "evaluation": "relation-stratified query subset; all entities; head+tail filtered",
                "selected_hyperparameters": selected_params,
                "checkpoint": str(pilot.CHECKPOINT) if job["decoder"] == "tabpfn_v3" else None,
                "registration": str(PLAN_PATH),
                "source_hashes": source_hashes(),
                "git_commit": pilot.git_commit(),
                "packages": deterministic.package_versions(),
                "hardware": {
                    "hostname": platform.node(),
                    "cpu_count": os.cpu_count(),
                    "gpu": pilot.hardware_info()["gpu"] if job["decoder"] == "tabpfn_v3" else None,
                },
            }
            timings = {
                "fit_seconds": info.get("fit_time"),
                "prediction_seconds": info.get("predict_time"),
                "total_seconds": time.time() - started,
                "peak_gpu_memory_bytes": peak,
            }
            frame["ultra_target_score"] = [
                float(test["X"][row.query_index * test["filtered"].shape[1] + row.target_entity, -1])
                for row in frame.itertuples()
            ]
            pilot.atomic_json(output / "config.json", config)
            pilot.atomic_json(output / "metrics.json", metrics)
            pilot.atomic_json(output / "timings.json", timings)
            pilot.atomic_json(output / "context_manifest.json", context_manifest)
            pilot.atomic_parquet(output / "query_predictions.parquet", frame)
            (output / "run.log").write_text(
                f"OK mrr={test_metrics['combined']['mrr']:.9f}\n", encoding="utf-8"
            )
            (output / "failure.json").unlink(missing_ok=True)
            return metrics
        except Exception as exc:
            failure = {
                "job": job,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            pilot.atomic_json(output / "failure.json", failure)
            (output / "run.log").write_text(failure["traceback"], encoding="utf-8")
            raise


def summarize() -> dict:
    missing = [job for job in all_jobs() if not run_complete(run_dir(job))]
    if missing:
        payload = {"status": "INCOMPLETE", "missing": len(missing), "examples": missing[:10]}
        pilot.atomic_json(SUMMARY_PATH, payload)
        return payload
    rows = []
    for job in all_jobs():
        metrics = json.loads((run_dir(job) / "metrics.json").read_text())
        rows.append({**job, "test_mrr": metrics["test"]["combined"]["mrr"],
                     "valid_mrr": metrics.get("valid", {}).get("combined", {}).get("mrr")})
    frame = pd.DataFrame(rows)
    comparisons = {}
    dataset_passes = {}
    for dataset in DATASETS:
        comparisons[dataset] = {}
        low_differences = []
        for budget in BUDGETS:
            seed_rows = []
            for seed in SEEDS:
                cell = frame[(frame.dataset == dataset) & (frame.seed == seed)
                             & (frame.budget == budget) & (frame.condition == "correct_labels")]
                tabpfn = cell[cell.decoder == "tabpfn_v3"].iloc[0]
                classical = cell[cell.decoder.isin(CLASSICAL_DECODERS)].copy()
                selected = classical.sort_values(["valid_mrr", "decoder"], ascending=[False, True]).iloc[0]
                difference = float(tabpfn.test_mrr - selected.test_mrr)
                seed_rows.append({"seed": seed, "tabpfn_mrr": float(tabpfn.test_mrr),
                                  "selected_classical": selected.decoder,
                                  "classical_mrr": float(selected.test_mrr), "difference": difference})
                if budget in LOW_SHOT_BUDGETS:
                    low_differences.append(difference)
            comparisons[dataset][str(budget)] = {
                "per_seed": seed_rows,
                "mean_difference": float(np.mean([row["difference"] for row in seed_rows])),
                "positive_seeds": int(sum(row["difference"] > 0 for row in seed_rows)),
            }
        causal = []
        for seed in SEEDS:
            correct = frame[(frame.dataset == dataset) & (frame.seed == seed)
                            & (frame.condition == "paired_correct_labels")
                            & (frame.decoder == "tabpfn_v3")].iloc[0].test_mrr
            permuted = frame[(frame.dataset == dataset) & (frame.seed == seed)
                             & (frame.condition == "paired_permuted_labels")
                             & (frame.decoder == "tabpfn_v3")].iloc[0].test_mrr
            causal.append(float(correct - permuted))
        passed = (
            np.mean(low_differences) >= 0.010
            and np.sum(np.asarray(low_differences) > 0) >= 6
            and np.mean(causal) >= 0.020
            and np.sum(np.asarray(causal) > 0) >= 4
        )
        dataset_passes[dataset] = {
            "passed": bool(passed),
            "low_shot_mean_difference": float(np.mean(low_differences)),
            "low_shot_positive_cells": int(np.sum(np.asarray(low_differences) > 0)),
            "label_mean_difference": float(np.mean(causal)),
            "label_positive_seeds": int(np.sum(np.asarray(causal) > 0)),
        }
    pass_count = sum(value["passed"] for value in dataset_passes.values())
    status = "PASS" if pass_count >= 2 else "FAIL"
    payload = {
        "status": "COMPLETE",
        "scientific_verdict": status,
        "dataset_passes": dataset_passes,
        "matched_comparisons": comparisons,
        "supported": status == "PASS",
        "interpretation_limit": (
            "This tests label-efficient frozen-model composition on a registered query "
            "subset; it is not a full-test SOTA comparison or proof of a unique mechanism."
        ),
    }
    pilot.atomic_json(SUMMARY_PATH, payload)
    lines = ["# Out-of-Dataset Low-Shot Confirmation", "", f"Verdict: **{status}**", "",
             "| Dataset | Pass | Low-shot delta | Positive cells | Label delta |", "|---|---:|---:|---:|---:|"]
    for dataset, value in dataset_passes.items():
        lines.append(
            f"| {dataset} | {value['passed']} | {value['low_shot_mean_difference']:+.6f} | "
            f"{value['low_shot_positive_cells']}/10 | {value['label_mean_difference']:+.6f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def smoke() -> None:
    jobs = [
        {"dataset": dataset, "seed": 101, "budget": 5,
         "condition": "correct_labels", "decoder": decoder}
        for dataset in DATASETS for decoder in DECODERS
    ]
    for job in jobs:
        execute_job(job, smoke=True)
        print(f"SMOKE OK {job}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--group", choices=("all", "tabpfn", "classical"), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    elif args.prepare:
        targets = [args.dataset] if args.dataset else DATASETS
        for dataset in targets:
            prepare_dataset(dataset)
    elif args.tune:
        validate_registration()
        for dataset in DATASETS:
            for decoder in CLASSICAL_DECODERS:
                tune_classical(dataset, decoder)
    elif args.smoke:
        validate_registration()
        smoke()
    elif args.run_shard:
        validate_registration()
        jobs = all_jobs(args.group)
        selected = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]
        for index, job in enumerate(selected, 1):
            metrics = execute_job(job)
            print(f"JOB {index}/{len(selected)} {job} mrr={metrics['test']['combined']['mrr']:.6f}", flush=True)
    elif args.summarize:
        print(json.dumps(summarize(), indent=2), flush=True)
    else:
        parser.error("choose an action")


if __name__ == "__main__":
    main()
