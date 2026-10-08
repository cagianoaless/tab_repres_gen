#!/usr/bin/env python3
"""Three-seed RFMLink evaluation on ULTRA's official 57-dataset universe."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT), str(WORKSPACE_ROOT)]

import run_ultra4g_ood_reassessment as base  # noqa: E402
from rfmlink.ultra57_registry import BY_SLUG, ULTRA57  # noqa: E402
from rfmlink.ultra_wrapper import (  # noqa: E402
    build_train_graph,
    import_ultra,
    load_ultra_model,
    score_triples as audited_score_triples,
)


PROTOCOL = "ultra57_rfmlink_batch1_v1"
DATASETS = tuple(spec.slug for spec in ULTRA57)
EXECUTION_ORDER = (
    "aristov4", "hetionet", "yago310",
    "conceptnet100k", "dbpedia100k", "codex_small", "codex_large",
    "wdsinger", "nell23k", "fb15k237_10", "fb15k237_20", "fb15k237_50",
    "fb15k237_ind_v1", "wn18rr_ind_v1", "nell_ind_v1",
    "fb15k237_ind_v2", "wn18rr_ind_v2", "nell_ind_v2",
    "fb15k237_ind_v3", "wn18rr_ind_v3", "nell_ind_v3",
    "fb15k237_ind_v4", "wn18rr_ind_v4", "nell_ind_v4",
    "ilpc2022_small", "ilpc2022_large",
    "fb_ingram_25", "wk_ingram_25", "nl_ingram_0", "nl_ingram_25",
    "fb_ingram_50", "wk_ingram_50", "nl_ingram_50",
    "fb_ingram_75", "wk_ingram_75", "nl_ingram_75",
    "fb_ingram_100", "wk_ingram_100", "nl_ingram_100",
    "hm_1k", "hm_3k", "hm_5k", "hm_indigo",
    "wiki_mt1_tax", "wiki_mt1_health", "wiki_mt2_org", "wiki_mt2_sci",
    "wiki_mt3_art", "wiki_mt3_infra", "wiki_mt4_sci", "wiki_mt4_health",
    "metafam", "fbnell",
    "fb15k237", "wn18rr", "nell995", "codex_medium",
)
if len(EXECUTION_ORDER) != 57 or set(EXECUTION_ORDER) != set(DATASETS):
    raise AssertionError("ULTRA-57 execution order must contain every dataset once")
SEEDS = (101, 102, 103)
PRIMARY_BUDGET = 400
N_ESTIMATORS = 4
DECODERS = ("tabpfn_v3", "logistic_frozen", "hgb_frozen", "mlp_frozen")
PERMUTED_DECODERS: tuple[str, ...] = ()
GLOBAL_ALPHA = 0.65
GLOBAL_ALPHA_SOURCE = (
    "median of the 45 validation-selected TabPFN alphas from the nine-graph "
    "ULTRA-4G development reassessment; frozen before ULTRA-57 metrics"
)
PREDICTION_BATCH_ROWS = int(
    os.environ.get("RFMLINK_TABPFN_PREDICT_BATCH", "50000")
)
if PREDICTION_BATCH_ROWS not in (50_000, 500_000):
    raise RuntimeError(
        "RFMLINK_TABPFN_PREDICT_BATCH must be selected by the registered "
        "50k-versus-500k runtime benchmark"
    )
base.TABPFN_PREDICT_BATCH = PREDICTION_BATCH_ROWS

FAMILY_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL
DATA_ROOT = FAMILY_ROOT / "immutable_assets"
RUN_ROOT = FAMILY_ROOT / "runs"
TUNING_ROOT = FAMILY_ROOT / "classical_selection"
LOG_ROOT = FAMILY_ROOT / "launcher_logs"
SUMMARY_PATH = FAMILY_ROOT / "summary.json"
SUMMARY_MD = FAMILY_ROOT / "summary.md"
PLAN_PATH = WORKSPACE_ROOT / "results" / "analysis" / f"{PROTOCOL}_plan.json"
AMENDMENT_PATHS = (
    WORKSPACE_ROOT / "results" / "analysis"
    / f"{PROTOCOL}_amendment_01_remove_permuted_labels.json",
    WORKSPACE_ROOT / "results" / "analysis"
    / f"{PROTOCOL}_amendment_02_canonical_source_preflight.json",
)
MASKING_AUDIT_PATH = (
    WORKSPACE_ROOT / "results" / "analysis" / "ultra_graph_consistency_audit.json"
)
OFFICIAL_CACHE = RFMLINK_ROOT / "data" / "ultra57_official"
CANONICAL_STAGE_ROOT = FAMILY_ROOT / "staging" / "canonical"

GENERIC_PREPARE = base._prepare_dataset
SOURCE_AUDITS: dict[str, dict] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hashes() -> dict:
    return {
        "runner": file_sha256(Path(__file__)),
        "launcher": file_sha256(SCRIPT_DIR / "run_ultra57_rfmlink_queue.sh"),
        "prediction_batch_benchmark": file_sha256(
            SCRIPT_DIR / "benchmark_tabpfn_prediction_batch.py"
        ),
        "dataset_staging": file_sha256(
            SCRIPT_DIR / "stage_ultra57_datasets.py"
        ),
        "registry": file_sha256(RFMLINK_ROOT / "rfmlink" / "ultra57_registry.py"),
        "active_engine": file_sha256(
            SCRIPT_DIR / "run_ultra4g_ood_reassessment.py"
        ),
        "immutable_engine": file_sha256(
            SCRIPT_DIR / "run_ood_low_shot_confirmation.py"
        ),
        "ultra_wrapper": file_sha256(RFMLINK_ROOT / "rfmlink" / "ultra_wrapper.py"),
        "vectorized_invariants": file_sha256(
            RFMLINK_ROOT / "rfmlink" / "vectorized_invariants.py"
        ),
    }


def canonicalization_policy() -> dict:
    return {
        "policy_version": 2,
        "split_priority": ["train", "valid", "test"],
        "within_split_exact_duplicates": (
            "remove deterministically while preserving first occurrence"
        ),
        "cross_split_exact_overlaps": (
            "retain in highest-priority split and remove from later splits"
        ),
        "reason": (
            "prevent observed target edges and duplicate copies from entering "
            "validation/test or invalidating leave-one-edge-out context scoring"
        ),
        "raw_official_files_modified": False,
        "performance_dependent_decisions": False,
    }


def protocol_specification() -> dict:
    return {
        "name": PROTOCOL,
        "ultra_repository": {
            "path": "RFMLink/external/ULTRA",
            "commit": "427966ad8ed60420eef034063d44f3153addff90",
            "dataset_universe": "57 datasets listed in the pinned official README",
            "automatic_download_cache": str(OFFICIAL_CACHE.relative_to(WORKSPACE_ROOT)),
        },
        "dataset_registry": [spec.as_dict() for spec in ULTRA57],
        "dataset_count": len(ULTRA57),
        "already_in_prior_nine_graph_suite": [
            spec.slug for spec in ULTRA57
            if spec.completed_in_ultra4g_reassessment
        ],
        "note_on_prior_coverage": (
            "Only AristoV4, Hetionet, and YAGO3-10 from the prior nine graphs are "
            "members of the official ULTRA-57 universe. All 57 are rerun here to "
            "obtain a homogeneous three-seed exact-masking protocol."
        ),
        "checkpoint": {
            "name": "ultra_4g.pth",
            "sha256": base.EXPECTED_ULTRA_SHA256,
            "training_graphs": list(base.ULTRA4G_TRAINING_GRAPHS),
            "frozen": True,
            "target_parameter_updates": False,
        },
        "seeds": list(SEEDS),
        "context": {
            "positive_rows": PRIMARY_BUDGET,
            "negative_rows": PRIMARY_BUDGET,
            "crossfit_folds": 10,
            "negative_filtering": "observed target graph only",
            "positive_ultra_masking": (
                "exact one-row leave-one-edge-out; batch_size=1; direct and inverse "
                "target copies removed dynamically"
            ),
            "unmasked_ultra_batch_cap": 8,
            "label_condition": "correct target-context labels only",
        },
        "representation": {
            "columns": list(base.engine.FEATURE_NAMES),
            "num_columns": len(base.engine.FEATURE_NAMES),
            "raw_entity_ids": False,
            "raw_cluster_ids": False,
        },
        "models": {
            "decoders": list(DECODERS),
            "tabpfn_estimators": N_ESTIMATORS,
        },
        "prediction_batching": {
            "candidates": [50_000, 500_000],
            "selection": "runtime and prediction/rank equivalence only",
            "benchmark_dataset": "hetionet",
            "benchmark_queries": 32,
            "minimum_speedup": 1.03,
            "maximum_absolute_prediction_difference": 1e-5,
            "all_query_ranks_must_match": True,
            "maximum_four_worker_reserved_memory_fraction": 0.75,
            "decision_file": (
                "results/analysis/tabpfn_prediction_batch_benchmark.json"
            ),
        },
        "dataset_staging": {
            "transport": "shared-filesystem producer-consumer",
            "ready_window": 5,
            "producer_role": "network-enabled remote host",
            "consumer_role": "network-isolated GPU host",
            "integrity": (
                "atomic markers plus SHA-256 for raw, processed, audit, and canonical "
                "split artifacts"
            ),
            "producer_preflight": (
                "download through pinned ULTRA loaders, canonicalize target splits, "
                "and publish a verified shared bundle before READY"
            ),
            "failure_policy": "three producer attempts then stop; never skip silently",
            "cleanup": "disabled by default; allowed only after validated DONE",
            "pytorch_compatibility": (
                "weights_only=False only for trusted dataset artifacts below the "
                "pinned official ULTRA cache; restores pre-PyTorch-2.6 loader behavior"
            ),
        },
        "dataset_canonicalization": canonicalization_policy(),
        "evaluation": {
            "query_selection": "up to 64 deterministic relation-stratified triples",
            "candidate_set": "all target-graph entities",
            "directions": "head and tail always reported",
            "official_tail_only_primary": [
                spec.slug for spec in ULTRA57
                if spec.official_evaluation == "tail_only"
            ],
            "filtering": "observed target graph plus target validation and test truths",
            "alpha_grid": base.ALPHA_GRID.tolist(),
            "dataset_alpha": (
                "selected by target validation combined MRR where an official target "
                "validation split exists"
            ),
            "global_alpha": GLOBAL_ALPHA,
            "global_alpha_source": GLOBAL_ALPHA_SOURCE,
            "no_target_validation_policy": (
                "HM and MTDEA provide source-graph validation only. Dataset-specific "
                "alpha and direction selection are disabled; the preregistered global "
                "alpha is used for fixed blending."
            ),
            "test_used_for_selection": False,
        },
        "inductive_mapping": {
            "observed_graph": "official inference graph",
            "context_pool": "official inference-graph observed edges",
            "grail_validation": (
                "official valid_ind file retained as target validation rather than "
                "merging valid_ind into the official default test object"
            ),
            "ilpc_ingram_validation": "official inference validation",
            "hm_mtdea_validation": "none on target graph; global alpha policy",
        },
        "claim_boundaries": {
            "checkpoint_training_members": [
                spec.slug for spec in ULTRA57 if spec.ultra4g_training_member
            ],
            "training members reported_separately": True,
            "official_protocol_equivalence": (
                "Dataset files are official. RFMLink uses its standardized 400+400 "
                "target-context and 64-query full-ranking protocol, not ULTRA's full-test "
                "evaluation on every dataset."
            ),
        },
        "dependency": {
            "masking_audit": str(MASKING_AUDIT_PATH.relative_to(WORKSPACE_ROOT)),
            "must_pass_before_metrics": True,
        },
    }


def configure_base() -> None:
    rationale = {
        spec.slug: f"official ULTRA-57 {spec.family} dataset"
        for spec in ULTRA57
    }
    sources = {
        spec.slug: (
            f"official ULTRA dataset class {spec.class_name}"
            + (f" version {spec.version}" if spec.version else "")
        )
        for spec in ULTRA57
    }
    base.PROTOCOL = PROTOCOL
    base.DATASETS = DATASETS
    base.DETAILED_DATASETS = ()
    base.SEEDS = SEEDS
    base.BUDGETS = (PRIMARY_BUDGET,)
    base.SCALING_BUDGETS = ()
    base.PRIMARY_BUDGET = PRIMARY_BUDGET
    base.N_ESTIMATORS = N_ESTIMATORS
    base.DECODERS = DECODERS
    base.PERMUTED_DECODERS = PERMUTED_DECODERS
    base.ABLATION_DECODERS = ()
    base.FAMILY_ROOT = FAMILY_ROOT
    base.DATA_ROOT = DATA_ROOT
    base.RUN_ROOT = RUN_ROOT
    base.TUNING_ROOT = TUNING_ROOT
    base.LOG_ROOT = LOG_ROOT
    base.SUMMARY_PATH = SUMMARY_PATH
    base.SUMMARY_MD = SUMMARY_MD
    base.PLAN_PATH = PLAN_PATH
    base.DATASET_RATIONALE = rationale
    base.DATASET_SOURCES = sources
    base.protocol_specification = protocol_specification
    base.source_hashes = source_hashes
    base.configure_engine()


configure_base()
engine = base.engine


def primary_jobs(dataset: str | None = None, group: str = "all") -> list[dict]:
    targets = [dataset] if dataset else list(DATASETS)
    jobs = []
    for target in targets:
        for seed in SEEDS:
            for decoder in DECODERS:
                jobs.append({
                    "dataset": target,
                    "seed": seed,
                    "budget": PRIMARY_BUDGET,
                    "condition": "correct_labels",
                    "representation": "full",
                    "decoder": decoder,
                })
    if group == "tabpfn":
        jobs = [job for job in jobs if job["decoder"] == "tabpfn_v3"]
    elif group == "classical":
        jobs = [job for job in jobs if job["decoder"] != "tabpfn_v3"]
    elif group != "all":
        raise ValueError(group)
    return jobs


def register() -> dict:
    if PLAN_PATH.exists():
        return json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if list(FAMILY_ROOT.rglob("metrics.json")):
        raise RuntimeError("ULTRA-57 metrics exist before registration")
    if file_sha256(base.ULTRA_CHECKPOINT) != base.EXPECTED_ULTRA_SHA256:
        raise RuntimeError("ULTRA-4G checkpoint hash mismatch")
    payload = {
        "status": "REGISTERED_BEFORE_ULTRA57_METRICS",
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_specification(),
        "source_hashes": source_hashes(),
        "planned_jobs": {
            "total": len(primary_jobs()),
            "tabpfn": len(primary_jobs(group="tabpfn")),
            "classical": len(primary_jobs(group="classical")),
            "per_dataset": len(primary_jobs(DATASETS[0])),
        },
        "estimated_runtime_hours": {
            "range": [20, 45],
            "basis": (
                "57 official graphs, three seeds, six condition/decoder cells per "
                "seed, four parallel TabPFN workers, and observed 9-18 minute large-graph "
                "TabPFN inference"
            ),
        },
    }
    engine.pilot.atomic_json(PLAN_PATH, payload)
    print(json.dumps(payload["planned_jobs"], indent=2), flush=True)
    return payload


def validate_registration(require_masking_audit: bool = True) -> dict:
    if not PLAN_PATH.exists():
        raise FileNotFoundError("ULTRA-57 suite is not registered")
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    amendments = [path for path in AMENDMENT_PATHS if path.exists()]
    if amendments:
        amendment_path = amendments[-1]
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment["base_plan_sha256"] != file_sha256(PLAN_PATH):
            raise RuntimeError("ULTRA-57 base plan changed after amendment")
        previous_path = amendment.get("previous_amendment")
        if previous_path:
            previous = WORKSPACE_ROOT / previous_path
            if amendment["previous_amendment_sha256"] != file_sha256(previous):
                raise RuntimeError("ULTRA-57 amendment chain changed")
        if amendment["protocol_sha256"] != json_sha256(protocol_specification()):
            raise RuntimeError("amended ULTRA-57 protocol changed")
        if amendment["source_hashes"] != source_hashes():
            raise RuntimeError("amended ULTRA-57 source changed")
        payload = amendment
    else:
        if payload["protocol"] != protocol_specification():
            raise RuntimeError("registered ULTRA-57 protocol changed")
        if payload["source_hashes"] != source_hashes():
            raise RuntimeError("registered ULTRA-57 source changed")
    if require_masking_audit:
        if not MASKING_AUDIT_PATH.exists():
            raise RuntimeError("ULTRA masking audit has not completed")
        audit = json.loads(MASKING_AUDIT_PATH.read_text(encoding="utf-8"))
        if not audit.get("broad_suite_allowed"):
            raise RuntimeError("ULTRA masking audit did not permit the broad suite")
    return payload


base.validate_registration = validate_registration
engine.validate_registration = validate_registration


def exact_context_score_triples(*args, batch_size: int = 32, **kwargs):
    if kwargs.get("remove_direct_edges", False):
        batch_size = 1
    else:
        batch_size = min(int(batch_size), 8)
    return audited_score_triples(*args, batch_size=batch_size, **kwargs)


engine.score_triples = exact_context_score_triples


@contextlib.contextmanager
def trusted_official_dataset_loading():
    """Restore the official repo's pre-PyTorch-2.6 load behavior in its cache only."""
    import torch

    original = torch.load
    trusted_root = OFFICIAL_CACHE.resolve()

    def trusted_load(path, *args, **kwargs):
        try:
            candidate = Path(path).resolve()
            candidate.relative_to(trusted_root)
        except (TypeError, ValueError):
            return original(path, *args, **kwargs)
        kwargs.setdefault("weights_only", False)
        return original(path, *args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original


def instantiate_official_dataset(slug: str):
    spec = BY_SLUG[slug]
    import_ultra()
    from ultra import datasets as official

    constructor = getattr(official, spec.class_name)
    kwargs = {"root": str(OFFICIAL_CACHE)}
    if spec.version is not None:
        kwargs["version"] = spec.version
    with trusted_official_dataset_loading():
        return constructor(**kwargs)


def _read_symbolic(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.strip().split()
            if len(fields) != 3:
                raise RuntimeError(f"invalid triple at {path}:{line_number}")
            rows.append((fields[0], fields[1], fields[2]))
    return rows


def _map_symbolic_splits(symbolic: dict[str, list[tuple[str, str, str]]]) -> dict:
    entity_to_id: dict[str, int] = {}
    relation_to_id: dict[str, int] = {}
    mapped = {}
    for split in ("train", "valid", "test"):
        rows = []
        for head, relation, tail in symbolic[split]:
            if head not in entity_to_id:
                entity_to_id[head] = len(entity_to_id)
            if tail not in entity_to_id:
                entity_to_id[tail] = len(entity_to_id)
            if relation not in relation_to_id:
                relation_to_id[relation] = len(relation_to_id)
            rows.append((
                entity_to_id[head], relation_to_id[relation], entity_to_id[tail]
            ))
        mapped[split] = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    mapped.update({
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
    })
    return mapped


def _target_array(data) -> np.ndarray:
    edges = data.target_edge_index.detach().cpu().numpy().T
    relations = data.target_edge_type.detach().cpu().numpy()
    return np.column_stack([edges[:, 0], relations, edges[:, 1]]).astype(
        np.int64, copy=False
    )


def _transductive_data(dataset) -> dict:
    train, valid, test = dataset[0], dataset[1], dataset[2]
    return {
        "train": _target_array(train),
        "valid": _target_array(valid),
        "test": _target_array(test),
        "num_entities": int(max(train.num_nodes, valid.num_nodes, test.num_nodes)),
        "num_relations": int(train.num_relations // 2),
    }


def _inductive_data(spec, dataset) -> tuple[dict, list[Path]]:
    raw_paths = [Path(path) for path in dataset.raw_paths]
    if spec.family == "inductive_entity_grail":
        paths = raw_paths[:3]
        symbolic = {
            "train": _read_symbolic(paths[0]),
            "valid": _read_symbolic(paths[1]),
            "test": _read_symbolic(paths[2]),
        }
    elif spec.target_validation:
        paths = raw_paths[1:4]
        symbolic = {
            "train": _read_symbolic(paths[0]),
            "valid": _read_symbolic(paths[1]),
            "test": _read_symbolic(paths[2]),
        }
    else:
        paths = [raw_paths[1], raw_paths[3]]
        symbolic = {
            "train": _read_symbolic(paths[0]),
            "valid": [],
            "test": _read_symbolic(paths[1]),
        }
    return _map_symbolic_splits(symbolic), paths


def _row_view(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 3)
    dtype = np.dtype([("h", "<i8"), ("r", "<i8"), ("t", "<i8")])
    return rows.view(dtype).reshape(-1)


def _stable_unique_rows(rows: np.ndarray) -> tuple[np.ndarray, int]:
    rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 3)
    if not len(rows):
        return rows, 0
    _, first_indices = np.unique(_row_view(rows), return_index=True)
    keep = np.sort(first_indices)
    return rows[keep], int(len(rows) - len(keep))


def _split_statistics(data: dict) -> dict:
    unique = {
        split: _stable_unique_rows(data[split])[0]
        for split in ("train", "valid", "test")
    }
    views = {split: _row_view(rows) for split, rows in unique.items()}
    return {
        "rows": {split: int(len(data[split])) for split in unique},
        "unique_rows": {split: int(len(unique[split])) for split in unique},
        "duplicates": {
            split: int(len(data[split]) - len(unique[split])) for split in unique
        },
        "overlaps": {
            "train_valid": int(np.intersect1d(views["train"], views["valid"]).size),
            "train_test": int(np.intersect1d(views["train"], views["test"]).size),
            "valid_test": int(np.intersect1d(views["valid"], views["test"]).size),
        },
        "split_hashes": {
            split: engine.pilot.array_hash(data[split]) for split in unique
        },
    }


def canonicalize_target_splits(data: dict) -> tuple[dict, dict]:
    source = _split_statistics(data)
    canonical = {
        **{key: value for key, value in data.items()
           if key not in ("train", "valid", "test")},
    }
    removed_duplicates = {}
    for split in ("train", "valid", "test"):
        canonical[split], removed_duplicates[split] = _stable_unique_rows(data[split])

    train_view = _row_view(canonical["train"])
    valid_view = _row_view(canonical["valid"])
    valid_from_train = np.isin(valid_view, train_view, assume_unique=True)
    canonical["valid"] = canonical["valid"][~valid_from_train]

    test_view = _row_view(canonical["test"])
    test_from_train = np.isin(test_view, train_view, assume_unique=True)
    remaining_test = canonical["test"][~test_from_train]
    test_from_valid = np.isin(
        _row_view(remaining_test), _row_view(canonical["valid"]), assume_unique=True
    )
    canonical["test"] = remaining_test[~test_from_valid]

    after = _split_statistics(canonical)
    if any(after["duplicates"].values()) or any(after["overlaps"].values()):
        raise RuntimeError(f"canonical split integrity failure: {after}")
    report = {
        "policy_version": 2,
        "policy_sha256": json_sha256(
            canonicalization_policy()
        ),
        "source": source,
        "removed": {
            "within_split_duplicates": removed_duplicates,
            "valid_already_in_train": int(valid_from_train.sum()),
            "test_already_in_train": int(test_from_train.sum()),
            "test_already_in_canonical_valid": int(test_from_valid.sum()),
        },
        "canonical": after,
    }
    return canonical, report


def _relative_official_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(OFFICIAL_CACHE.resolve()))
    except ValueError:
        return str(path)


def _audit_dataset(slug: str, data: dict, paths: list[Path],
                   canonicalization: dict) -> dict:
    split_sets = {
        split: {tuple(row) for row in data[split].tolist()}
        for split in ("train", "valid", "test")
    }
    duplicates = {
        split: int(len(data[split]) - len(split_sets[split]))
        for split in split_sets
    }
    overlaps = {
        "train_valid": len(split_sets["train"] & split_sets["valid"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "valid_test": len(split_sets["valid"] & split_sets["test"]),
    }
    if any(duplicates.values()) or any(overlaps.values()):
        raise RuntimeError(
            f"{slug} duplicate/overlap audit failed: "
            f"duplicates={duplicates} overlaps={overlaps}"
        )
    return {
        "dataset": slug,
        "num_entities": int(data["num_entities"]),
        "num_relations": int(data["num_relations"]),
        "num_train": int(len(data["train"])),
        "num_valid": int(len(data["valid"])),
        "num_test": int(len(data["test"])),
        "duplicates": duplicates,
        "overlaps": overlaps,
        "raw_files": [
            {"path": _relative_official_path(path), "sha256": file_sha256(path)}
            for path in paths
        ],
        "mapped_split_digest": base.mapped_split_digest(data),
        "canonicalization": canonicalization,
    }


def _load_and_canonicalize_official(slug: str) -> tuple[dict, dict]:
    spec = BY_SLUG[slug]
    dataset = instantiate_official_dataset(slug)
    if spec.family == "transductive":
        data = _transductive_data(dataset)
        paths = [Path(path) for path in dataset.raw_paths]
    else:
        data, paths = _inductive_data(spec, dataset)
    canonical, report = canonicalize_target_splits(data)
    return canonical, _audit_dataset(slug, canonical, paths, report)


def canonical_bundle_paths(slug: str, root: Path = CANONICAL_STAGE_ROOT) -> tuple[Path, Path]:
    directory = root / slug
    return directory / "dataset.npz", directory / "source_audit.json"


def materialize_canonical_bundle(slug: str,
                                 root: Path = CANONICAL_STAGE_ROOT) -> dict:
    data, audit = _load_and_canonicalize_official(slug)
    dataset_path, audit_path = canonical_bundle_paths(slug, root)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_name(f".{dataset_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            train=data["train"], valid=data["valid"], test=data["test"],
            num_entities=np.asarray(data["num_entities"]),
            num_relations=np.asarray(data["num_relations"]),
        )
    os.replace(temporary, dataset_path)
    audit = {**audit, "canonical_dataset_sha256": file_sha256(dataset_path)}
    engine.pilot.atomic_json(audit_path, audit)
    return {
        "policy_sha256": audit["canonicalization"]["policy_sha256"],
        "mapped_split_digest": audit["mapped_split_digest"],
        "artifacts": [
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (dataset_path, audit_path)
        ],
        "source_counts": audit["canonicalization"]["source"]["rows"],
        "canonical_counts": audit["canonicalization"]["canonical"]["rows"],
        "removed": audit["canonicalization"]["removed"],
    }


def _load_canonical_bundle(slug: str,
                           root: Path = CANONICAL_STAGE_ROOT) -> tuple[dict, dict]:
    dataset_path, audit_path = canonical_bundle_paths(slug, root)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_policy = json_sha256(canonicalization_policy())
    if audit["canonicalization"]["policy_sha256"] != expected_policy:
        raise RuntimeError(f"{slug} canonicalization policy changed")
    if file_sha256(dataset_path) != audit["canonical_dataset_sha256"]:
        raise RuntimeError(f"{slug} canonical dataset hash mismatch")
    with np.load(dataset_path, allow_pickle=False) as payload:
        data = {
            "train": payload["train"],
            "valid": payload["valid"],
            "test": payload["test"],
            "num_entities": int(payload["num_entities"]),
            "num_relations": int(payload["num_relations"]),
        }
    if base.mapped_split_digest(data) != audit["mapped_split_digest"]:
        raise RuntimeError(f"{slug} canonical mapped digest mismatch")
    return data, audit


def load_dataset(slug: str) -> dict:
    dataset_path, audit_path = canonical_bundle_paths(slug)
    if dataset_path.exists() and audit_path.exists():
        data, source_audit = _load_canonical_bundle(slug)
    else:
        data, source_audit = _load_and_canonicalize_official(slug)
    SOURCE_AUDITS[slug] = source_audit
    return data


engine.load_pykeen_dataset = load_dataset


def _special_dataset_complete(slug: str) -> bool:
    root = engine.dataset_dir(slug)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_hashes"] != source_hashes():
            return False
        split = engine.split_dir(slug, "test")
        if not all((split / name).exists() for name in (
            "features.npy", "filtered.npy", "queries.npz"
        )):
            return False
        return all(engine.context_path(slug, seed).exists() for seed in SEEDS)
    except Exception:
        return False


def _prepare_without_target_validation(slug: str, data: dict) -> dict:
    if _special_dataset_complete(slug):
        return json.loads(
            (engine.dataset_dir(slug) / "manifest.json").read_text(encoding="utf-8")
        )
    started = time.time()
    root = engine.dataset_dir(slug)
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "dataset.npz"
    temporary = dataset_path.with_name(f".{dataset_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            train=data["train"], valid=data["valid"], test=data["test"],
            num_entities=np.asarray(data["num_entities"]),
            num_relations=np.asarray(data["num_relations"]),
        )
    os.replace(temporary, dataset_path)
    builder = engine.VectorizedInvariantBuilder(
        data["train"], data["num_entities"], data["num_relations"],
        num_clusters=engine.NUM_CLUSTERS, seed=7,
    )
    model = load_ultra_model(base.ULTRA_CHECKPOINT, device="cuda")
    graph = build_train_graph(
        data["train"], data["num_entities"], data["num_relations"], device="cuda"
    )
    split_manifest = engine.materialize_eval_split(
        slug, "test", data, builder, model, graph
    )
    contexts = {
        str(seed): engine.materialize_context(slug, seed, data, model, graph)
        for seed in SEEDS
    }
    del model, graph, builder
    import torch
    torch.cuda.empty_cache()
    manifest = {
        "status": "IMMUTABLE_ULTRA57_INPUT",
        "dataset": slug,
        "num_entities": data["num_entities"],
        "num_relations": data["num_relations"],
        "num_train": int(len(data["train"])),
        "num_valid": 0,
        "num_test": int(len(data["test"])),
        "dataset_sha256": file_sha256(dataset_path),
        "split_assets": {"test": split_manifest},
        "contexts": {
            seed: {"context_sha256": value["context_sha256"]}
            for seed, value in contexts.items()
        },
        "source_hashes": source_hashes(),
        "registration": str(PLAN_PATH),
        "checkpoint_sha256": base.EXPECTED_ULTRA_SHA256,
        "materialization_seconds": time.time() - started,
        "packages": engine.deterministic.package_versions(),
        "hardware": engine.pilot.hardware_info(),
        "target_validation_available": False,
    }
    engine.pilot.atomic_json(root / "manifest.json", manifest)
    return manifest


def prepare_dataset(slug: str) -> dict:
    validate_registration()
    data = load_dataset(slug)
    source_audit = SOURCE_AUDITS[slug]
    root = engine.dataset_dir(slug)
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source_audit.json"
    if source_path.exists():
        saved = json.loads(source_path.read_text(encoding="utf-8"))
        if saved != source_audit:
            raise RuntimeError(f"{slug} official source changed after freezing")
    else:
        engine.pilot.atomic_json(source_path, source_audit)
    if BY_SLUG[slug].target_validation:
        manifest = GENERIC_PREPARE(slug)
    else:
        manifest = _prepare_without_target_validation(slug, data)
    manifest.update({
        "source_audit": source_audit,
        "ultra57_registry": BY_SLUG[slug].as_dict(),
        "positive_context_ultra_mask_batch_size": 1,
        "ultra4g_training_membership": BY_SLUG[slug].ultra4g_training_member,
    })
    engine.pilot.atomic_json(root / "manifest.json", manifest)
    return manifest


engine.prepare_dataset = prepare_dataset


FROZEN_NO_VALIDATION_PARAMS = {
    decoder: engine.CLASSICAL_GRIDS[decoder][0]
    for decoder in DECODERS if decoder != "tabpfn_v3"
}


def tune(slug: str) -> None:
    validate_registration()
    if BY_SLUG[slug].target_validation:
        for decoder in DECODERS[1:]:
            engine.tune_classical(slug, decoder)
        return
    for decoder, params in FROZEN_NO_VALIDATION_PARAMS.items():
        output = engine.tuning_path(slug, decoder)
        if output.exists():
            continue
        payload = {
            "dataset": slug,
            "decoder": decoder,
            "selection_split": None,
            "selection_context": None,
            "test_used": False,
            "search": [],
            "selected_params": params,
            "selection_policy": (
                "first preregistered classical grid entry; target validation unavailable"
            ),
        }
        engine.pilot.atomic_json(output, payload)


def _blend_scores(asset: dict, direct: np.ndarray, alpha: float) -> np.ndarray:
    ultra = base.query_standardize(asset, asset["X"][:, -1])
    probabilities = np.asarray(direct, dtype=np.float64)
    logits = np.log(
        np.clip(probabilities, 1e-6, 1 - 1e-6)
        / np.clip(1 - probabilities, 1e-6, 1)
    )
    model = base.query_standardize(asset, logits)
    return (1.0 - alpha) * ultra + alpha * model


def _decorate_global_blend(job: dict, metrics: dict) -> dict:
    if "global_fixed_blend" in metrics.get("test", {}):
        return metrics
    for split in ("valid", "test"):
        asset = engine.load_split_asset(job["dataset"], split)
        scores = np.load(base.run_dir(job) / f"{split}_scores.npy", mmap_mode="r")
        blended = _blend_scores(asset, scores, GLOBAL_ALPHA)
        metrics[split]["global_fixed_blend"], _ = engine.metrics_from_scores(
            asset, blended
        )
    metrics["selection"]["global_alpha"] = GLOBAL_ALPHA
    metrics["selection"]["global_alpha_source"] = GLOBAL_ALPHA_SOURCE
    output = base.run_dir(job)
    engine.pilot.atomic_json(output / "metrics.json", metrics)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    config.update({
        "ultra57_registry": BY_SLUG[job["dataset"]].as_dict(),
        "positive_context_ultra_mask_batch_size": 1,
        "global_alpha": GLOBAL_ALPHA,
        "global_alpha_source": GLOBAL_ALPHA_SOURCE,
    })
    engine.pilot.atomic_json(output / "config.json", config)
    return metrics


def _special_prediction_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = frames["ultra"].copy().rename(columns={
        "rank": "ultra_rank",
        "reciprocal_rank": "ultra_reciprocal_rank",
        "target_score": "ultra_target_score",
    })
    for method in ("direct", "fixed_blend"):
        frame = frames[method]
        output[f"{method}_rank"] = frame["rank"].to_numpy()
        output[f"{method}_reciprocal_rank"] = frame["reciprocal_rank"].to_numpy()
        output[f"{method}_target_score"] = frame["target_score"].to_numpy()
    output["split"] = "test"
    return output


def _execute_without_target_validation(job: dict, smoke: bool = False) -> dict | None:
    output = base.run_dir(job, smoke)
    if not smoke and base.run_complete(output):
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with engine.run_lock(output) as acquired:
        if not acquired:
            return None
        if not smoke and base.run_complete(output):
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        started = time.time()
        try:
            context = base.context_payload(
                job["dataset"], job["seed"], job["budget"], job["condition"]
            )
            test = engine.load_split_asset(job["dataset"], "test")
            if smoke:
                test = base.compact_asset(test)
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if job["decoder"] == "tabpfn_v3":
                predictions, info = base.fit_predict_tabpfn3_portable(
                    context["X"], context["y"], [test["X"]], job["seed"],
                    categorical_indices=[0, 16],
                    n_estimators=1 if smoke else N_ESTIMATORS,
                )
                parameters = {"n_estimators": 1 if smoke else N_ESTIMATORS}
            else:
                fit_started = time.time()
                model, parameters = base.build_model(job, len(context["X"]))
                model.fit(base.model_input(job, context["X"]), context["y"])
                fit_seconds = time.time() - fit_started
                predict_started = time.time()
                predictions = [
                    base.classical_predict(model, base.model_input(job, test["X"]))
                ]
                info = {
                    "fit_time": fit_seconds,
                    "predict_time": time.time() - predict_started,
                }
            direct = predictions[0]
            blend = _blend_scores(test, direct, GLOBAL_ALPHA)
            metrics = {
                "selection": {
                    "alpha": GLOBAL_ALPHA,
                    "alpha_source": GLOBAL_ALPHA_SOURCE,
                    "dataset_specific_alpha": None,
                    "dataset_specific_alpha_reason": "target validation unavailable",
                    "test_used_for_selection": False,
                },
                "valid": None,
                "test": {},
            }
            frames = {}
            for method, scores in (
                ("ultra", test["X"][:, -1]),
                ("direct", direct),
                ("fixed_blend", blend),
            ):
                metrics["test"][method], frames[method] = engine.metrics_from_scores(
                    test, scores
                )
            if smoke:
                engine.pilot.atomic_json(output / "smoke_metrics.json", metrics)
                (output / "run.log").write_text("SMOKE OK\n", encoding="utf-8")
                return metrics
            base._atomic_npy(output / "valid_scores.npy", np.zeros(0, dtype=np.float32))
            base._atomic_npy(output / "test_scores.npy", direct)
            config = {
                "experiment_family": PROTOCOL,
                **job,
                "feature_names": list(engine.FEATURE_NAMES),
                "num_features": len(engine.FEATURE_NAMES),
                "selected_hyperparameters": parameters,
                "ultra_checkpoint": str(base.ULTRA_CHECKPOINT),
                "ultra_checkpoint_sha256": base.EXPECTED_ULTRA_SHA256,
                "tabpfn_checkpoint": (
                    str(base.TABPFN_CHECKPOINT)
                    if job["decoder"] == "tabpfn_v3" else None
                ),
                "tabpfn_checkpoint_sha256": (
                    base.EXPECTED_TABPFN_SHA256
                    if job["decoder"] == "tabpfn_v3" else None
                ),
                "evaluation": "complete all-entity head+tail filtered; test only",
                "target_validation_available": False,
                "global_alpha": GLOBAL_ALPHA,
                "global_alpha_source": GLOBAL_ALPHA_SOURCE,
                "positive_context_ultra_mask_batch_size": 1,
                "ultra57_registry": BY_SLUG[job["dataset"]].as_dict(),
                "source_hashes": source_hashes(),
                "packages": engine.deterministic.package_versions(),
                "hardware": {
                    "hostname": platform.node(),
                    "cpu_count": os.cpu_count(),
                    "gpu": (
                        engine.pilot.hardware_info()["gpu"]
                        if job["decoder"] == "tabpfn_v3" else None
                    ),
                },
            }
            context_manifest = {
                "selection_policy": context["policy"],
                "selected_pool_rows": context["selected"].tolist(),
                "row_hash": engine.pilot.array_hash(context["triples"]),
                "feature_hash": engine.pilot.array_hash(context["X"]),
                "original_label_hash": engine.pilot.array_hash(context["original_y"]),
                "supplied_label_hash": engine.pilot.array_hash(context["y"]),
                "permutation_hash": engine.pilot.array_hash(context["permutation"]),
                "labels_changed": int(np.sum(context["original_y"] != context["y"])),
            }
            timings = {
                "fit_seconds": info.get("fit_time"),
                "prediction_seconds": info.get("predict_time"),
                "total_seconds": time.time() - started,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available() and job["decoder"] == "tabpfn_v3"
                    else None
                ),
            }
            engine.pilot.atomic_json(output / "config.json", config)
            engine.pilot.atomic_json(output / "metrics.json", metrics)
            engine.pilot.atomic_json(output / "timings.json", timings)
            engine.pilot.atomic_json(output / "context_manifest.json", context_manifest)
            engine.pilot.atomic_parquet(
                output / "predictions.parquet", _special_prediction_frame(frames)
            )
            (output / "run.log").write_text(
                "OK "
                f"direct_mrr={metrics['test']['direct']['combined']['mrr']:.9f} "
                f"blend_mrr={metrics['test']['fixed_blend']['combined']['mrr']:.9f} "
                f"alpha={GLOBAL_ALPHA:.2f}\n",
                encoding="utf-8",
            )
            (output / "failure.json").unlink(missing_ok=True)
            return metrics
        except Exception as error:
            failure = {
                "job": job,
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            engine.pilot.atomic_json(output / "failure.json", failure)
            (output / "run.log").write_text(failure["traceback"], encoding="utf-8")
            raise


def execute(job: dict, smoke: bool = False) -> dict | None:
    validate_registration()
    if BY_SLUG[job["dataset"]].target_validation:
        metrics = base.execute(job, smoke=smoke)
        if smoke or metrics is None:
            return metrics
        return _decorate_global_blend(job, metrics)
    return _execute_without_target_validation(job, smoke=smoke)


def smoke(slug: str) -> None:
    jobs = [
        job for job in primary_jobs(slug)
        if job["seed"] == SEEDS[0]
        and job["condition"] == "correct_labels"
        and job["decoder"] in ("tabpfn_v3", "hgb_frozen")
    ]
    for job in jobs:
        metrics = execute(job, smoke=True)
        print(
            f"SMOKE OK dataset={slug} decoder={job['decoder']} "
            f"mrr={metrics['test']['direct']['combined']['mrr']:.6f}",
            flush=True,
        )


def summarize() -> dict:
    rows = []
    for job in primary_jobs():
        path = base.run_dir(job) / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        spec = BY_SLUG[job["dataset"]]
        primary_side = "tail" if spec.official_evaluation == "tail_only" else "combined"
        for method, values in metrics["test"].items():
            rows.append({
                **job,
                "method": method,
                "primary_side": primary_side,
                "mrr": values[primary_side]["mrr"],
                "combined_mrr": values["combined"]["mrr"],
                "head_mrr": values["head"]["mrr"],
                "tail_mrr": values["tail"]["mrr"],
                "alpha": metrics["selection"].get("alpha"),
                "global_alpha": metrics["selection"].get(
                    "global_alpha", metrics["selection"].get("alpha")
                    if not spec.target_validation else GLOBAL_ALPHA
                ),
            })
    frame = pd.DataFrame(rows)
    groups = []
    if not frame.empty:
        for keys, group in frame.groupby(
            ["dataset", "condition", "decoder", "method"], sort=True
        ):
            groups.append({
                "dataset": keys[0],
                "condition": keys[1],
                "decoder": keys[2],
                "method": keys[3],
                "seeds": int(len(group)),
                "primary_mrr_mean": float(group["mrr"].mean()),
                "primary_mrr_std": float(group["mrr"].std(ddof=1))
                if len(group) > 1 else None,
                "combined_mrr_mean": float(group["combined_mrr"].mean()),
                "head_mrr_mean": float(group["head_mrr"].mean()),
                "tail_mrr_mean": float(group["tail_mrr"].mean()),
                "alphas": [
                    None if pd.isna(value) else float(value)
                    for value in group["alpha"].tolist()
                ],
                "registry": BY_SLUG[keys[0]].as_dict(),
            })
    failures = [str(path) for path in FAMILY_ROOT.rglob("failure.json")]
    payload = {
        "protocol": PROTOCOL,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planned_jobs": len(primary_jobs()),
        "completed_jobs": sum(
            (base.run_dir(job) / "metrics.json").exists() for job in primary_jobs()
        ),
        "dataset_count_with_results": len(set(frame["dataset"])) if not frame.empty else 0,
        "results": groups,
        "failures": failures,
    }
    engine.pilot.atomic_json(SUMMARY_PATH, payload)
    lines = [
        "# ULTRA-57 RFMLink summary",
        "",
        f"Completed jobs: {payload['completed_jobs']}/{payload['planned_jobs']}",
        f"Datasets with results: {payload['dataset_count_with_results']}/57",
        f"Failures: {len(failures)}",
        "",
        "Dataset-specific validation alphas and the preregistered global alpha "
        "are retained in every run's metrics.json.",
    ]
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_MD.with_name(f".{SUMMARY_MD.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(SUMMARY_MD)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--queue-list", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--group", choices=("all", "tabpfn", "classical"), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.register:
        print(json.dumps(register(), indent=2))
    elif args.validate:
        plan = validate_registration()
        print(f"VALID jobs={plan['planned_jobs']['total']}")
    elif args.list:
        print(json.dumps([spec.as_dict() for spec in ULTRA57], indent=2))
    elif args.queue_list:
        print("\n".join(EXECUTION_ORDER))
    elif args.download:
        validate_registration()
        targets = [args.dataset] if args.dataset else DATASETS
        for slug in targets:
            instantiate_official_dataset(slug)
            print(f"DOWNLOADED {slug}", flush=True)
    elif args.prepare:
        validate_registration()
        targets = [args.dataset] if args.dataset else DATASETS
        for slug in targets:
            manifest = prepare_dataset(slug)
            print(
                f"PREPARED {slug} entities={manifest['num_entities']} "
                f"train={manifest['num_train']}", flush=True,
            )
    elif args.tune:
        if not args.dataset:
            parser.error("--tune requires --dataset")
        tune(args.dataset)
    elif args.smoke:
        if not args.dataset:
            parser.error("--smoke requires --dataset")
        smoke(args.dataset)
    elif args.run_shard:
        if not args.dataset:
            parser.error("--run-shard requires --dataset")
        jobs = primary_jobs(args.dataset, args.group)
        selected = [
            job for index, job in enumerate(jobs)
            if index % args.num_shards == args.shard_id
        ]
        for index, job in enumerate(selected, 1):
            metrics = execute(job)
            print(
                f"JOB {index}/{len(selected)} {job} "
                f"direct={metrics['test']['direct']['combined']['mrr']:.6f} "
                f"blend={metrics['test']['fixed_blend']['combined']['mrr']:.6f}",
                flush=True,
            )
    elif args.summarize:
        print(json.dumps(summarize(), indent=2))
    else:
        parser.error("choose an action")


if __name__ == "__main__":
    main()
