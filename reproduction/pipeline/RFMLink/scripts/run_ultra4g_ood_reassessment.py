#!/usr/bin/env python3
"""Prospective ULTRA-4G reassessment on previously explored RFMLink graphs.

The table, candidate evaluator, target-context construction, and decoder
families are held fixed. Only the frozen ULTRA checkpoint is changed from 50G
to the publicly documented 4G checkpoint. Graphs in, or overlapping with, the
4G training mixture are excluded.
"""

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

import run_ood_low_shot_confirmation_v4 as v4  # noqa: E402
from rfmlink.data_prep import load_processed  # noqa: E402
from rfmlink.decoders import TABPFN_PREDICT_BATCH  # noqa: E402
from rfmlink.sbm_import import (  # noqa: E402
    build_tabpfn_inference_config,
    positive_class_probabilities,
)


engine = v4.engine

PROTOCOL = "ultra4g_ood_reassessment_v1"
DATASETS = (
    "umls",
    "kinships",
    "nations",
    "countries",
    "kg20c",
    "aristov4",
    "hetionet",
    "biokg",
    "yago310",
)
DETAILED_DATASETS = DATASETS[:7]
LOCAL_PROCESSED = frozenset(("umls", "kinships", "nations"))
PYKEEN_DATASETS = frozenset((
    "countries", "aristov4", "hetionet", "biokg", "yago310"
))
SEEDS = (101, 102, 103, 104, 105)
BUDGETS = (20, 200, 400)
SCALING_BUDGETS = (20, 200)
PRIMARY_BUDGET = 400
N_ESTIMATORS = 4
DECODERS = ("tabpfn_v3", "logistic_frozen", "hgb_frozen", "mlp_frozen")
PERMUTED_DECODERS = ("tabpfn_v3", "hgb_frozen")
ABLATION_DECODERS = ("tabpfn_v3", "hgb_frozen")
REPRESENTATIONS = ("full", "structural_only", "ultra_only")
ALPHA_GRID = np.linspace(0.0, 1.0, 21)

ULTRA4G_TRAINING_GRAPHS = ("FB15k237", "WN18RR", "CoDExMedium", "NELL995")
EXCLUDED_EXPLORED_GRAPHS = {
    "fb15k237": "direct ULTRA-4G training graph",
    "wn18rr": "direct ULTRA-4G training graph",
    "codex-small": "excluded because CoDEx-Medium is a 4G training graph and the saved exact-overlap audit found 89.88% of CoDEx-Small test triples in CoDEx-Medium train",
}
ULTRA_CHECKPOINT = RFMLINK_ROOT / "external" / "ULTRA" / "ckpts" / "ultra_4g.pth"
EXPECTED_ULTRA_SHA256 = "48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0"
ULTRA_CHECKPOINT_REPO_PATH = "RFMLink/external/ULTRA/ckpts/ultra_4g.pth"
TABPFN_CHECKPOINT = (
    WORKSPACE_ROOT.parents[1] / "models_ckpt" / "tabpfn-v3-classifier-v3_default.ckpt"
)
EXPECTED_TABPFN_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
TABPFN_CHECKPOINT_DATA_PATH = "data/models_ckpt/tabpfn-v3-classifier-v3_default.ckpt"

FAMILY_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL
DATA_ROOT = FAMILY_ROOT / "immutable_assets"
RUN_ROOT = FAMILY_ROOT / "runs"
TUNING_ROOT = FAMILY_ROOT / "classical_selection"
LOG_ROOT = FAMILY_ROOT / "launcher_logs"
SUMMARY_PATH = FAMILY_ROOT / "summary.json"
SUMMARY_MD = FAMILY_ROOT / "summary.md"
PLAN_PATH = WORKSPACE_ROOT / "results" / "analysis" / f"{PROTOCOL}_plan.json"

REQUIRED_RUN_FILES = (
    "config.json",
    "metrics.json",
    "timings.json",
    "predictions.parquet",
    "context_manifest.json",
    "valid_scores.npy",
    "test_scores.npy",
    "run.log",
)

KG20C_ROOT = RFMLINK_ROOT / "data" / "ood_confirmation_external" / "kg20c"
KG20C_EXPECTED = {
    "num_entities": 16_362,
    "num_relations": 5,
    "num_train": 48_213,
    "num_valid": 3_670,
    "num_test": 3_724,
}

EXPECTED_DATASET_STATS = {
    "umls": (135, 46, 5_216, 652, 661),
    "kinships": (104, 25, 8_544, 1_068, 1_074),
    "nations": (14, 55, 1_592, 199, 201),
    "countries": (271, 2, 1_110, 24, 24),
    "kg20c": (16_362, 5, 48_213, 3_670, 3_724),
    "aristov4": (42_016, 1_593, 242_567, 18_444, 18_414),
    "hetionet": (45_158, 24, 1_800_157, 225_020, 225_020),
    "biokg": (105_524, 17, 1_654_397, 206_800, 206_800),
    "yago310": (123_143, 37, 1_079_040, 4_978, 4_982),
}

# Measured before registration from the exact mapped train/valid/test arrays.
# YAGO3-10 is frozen by its fixed public files and official statistics; its
# mapped-array digest is recorded in the immutable asset manifest after download.
PRE_REGISTRATION_SPLIT_DIGESTS = {
    "umls": "c32757cfc649f574d0d2f33b5303f8d31f67beed5f13c603515886d5d90a032c",
    "kinships": "b4b0d0c1d0c65d0a627cdef8507b1b8d2d337678cec455a3a00c2acda7fbed53",
    "nations": "42af6dcb76dc6faca3425bcb6a3a960af4153657caed22c358e8831025b7fc55",
    "countries": "3e3723715cc832bd3860a2bce68c0977368e1d951fa94d50331639d8c1e71038",
    "kg20c": "29c4dafd43e47e440a260f546c6230fae0a27b7d0f4ce4e1af59d6d6305568bc",
    "aristov4": "fd04d369d9b1317ce75eec3c6fe6b5937d9b9bf5a7a6644a505c55e2ee71dcc7",
    "hetionet": "31fcc084de6a2be855e0da4788bd5879b01266cd52790c13afbee8c0e1c918d4",
    "biokg": "d7bc83d4d1b6af12ae45daca718e00bbdb8c923c7114e75e744624d93bb417a2",
}

DATASET_RATIONALE = {
    "umls": "historical small biomedical development graph; not in ULTRA-4G training",
    "kinships": "historical small relational development graph; not in ULTRA-4G training",
    "nations": "historical small relational development graph; not in ULTRA-4G training",
    "countries": "previously explored geopolitical boundary graph; not in ULTRA-4G training",
    "kg20c": "previously explored external scholarly KG; not in ULTRA-4G training",
    "aristov4": "previously explored large commonsense KG; not in ULTRA-4G training",
    "hetionet": "previously explored large biomedical KG; not in ULTRA-4G training",
    "biokg": (
        "official BioKG v1.0.0 graph with a preregistered deterministic 80/10/10 "
        "split; not in ULTRA-4G training"
    ),
    "yago310": "public fixed YAGO3-10 splits; not in ULTRA-4G training",
}

DATASET_SOURCES = {
    "umls": "frozen local arrays from the standard PyKEEN 1.11.1 UMLS split files",
    "kinships": (
        "frozen local standard Kinships files; data/kg/kinships and RFMLink/data/raw/kinships "
        "are byte-identical to the recorded PyKEEN 1.11.1 files"
    ),
    "nations": "frozen local arrays from the standard PyKEEN 1.11.1 Nations split files",
    "countries": "PyKEEN official Countries fixed splits",
    "kg20c": "official KG20C TSV splits at pinned commit 767c47a780e9dd6b3b336b89afa0a752d06c32ae",
    "aristov4": "PyKEEN AristoV4 mapped public splits",
    "hetionet": "PyKEEN Hetionet mapped public splits",
    "biokg": (
        "BioKG v1.0.0 official release edge list; deterministic PyKEEN 1.11.1 "
        "coverage split with random_state=0"
    ),
    "yago310": "YAGO3-10 fixed train/valid/test files from the public ConvE archive",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    return {
        "runner": file_sha256(Path(__file__)),
        "ood_engine": file_sha256(SCRIPT_DIR / "run_ood_low_shot_confirmation.py"),
        "ood_compute_v4": file_sha256(SCRIPT_DIR / "run_ood_low_shot_confirmation_v4.py"),
        "vectorized_invariants": file_sha256(
            RFMLINK_ROOT / "rfmlink" / "vectorized_invariants.py"
        ),
        "decoders": file_sha256(RFMLINK_ROOT / "rfmlink" / "decoders.py"),
        "contexts": file_sha256(RFMLINK_ROOT / "rfmlink" / "contexts.py"),
    }


def protocol_specification() -> dict:
    return {
        "name": PROTOCOL,
        "scientific_status": (
            "prospective checkpoint-controlled reassessment; the seven detailed graph "
            "outcomes and representation were previously observed with ULTRA-50G, while "
            "BioKG and YAGO3-10 are new checkpoint-control graphs; no ULTRA-4G decoder "
            "metric was observed before this registration"
        ),
        "research_question": (
            "Do the frozen RFMLink table, target labels, and TabPFN/classical decoder "
            "comparisons persist when the encoder is the documented ULTRA-4G model and "
            "every evaluated graph is excluded from its training mixture?"
        ),
        "checkpoint_intervention": {
            "checkpoint_repo_path": ULTRA_CHECKPOINT_REPO_PATH,
            "sha256": EXPECTED_ULTRA_SHA256,
            "training_graphs": list(ULTRA4G_TRAINING_GRAPHS),
            "parameters_frozen": True,
            "target_graph_updates": False,
            "only_encoder_change_from_recent_runs": "ULTRA-50G to ULTRA-4G",
        },
        "datasets": list(DATASETS),
        "detailed_datasets": list(DETAILED_DATASETS),
        "dataset_rationale": DATASET_RATIONALE,
        "dataset_sources": DATASET_SOURCES,
        "expected_dataset_stats": {
            name: list(values) for name, values in EXPECTED_DATASET_STATS.items()
        },
        "pre_registration_split_digests": PRE_REGISTRATION_SPLIT_DIGESTS,
        "excluded_explored_graphs": EXCLUDED_EXPLORED_GRAPHS,
        "claim_limit": (
            "This controls encoder pretraining membership. It is not an untouched "
            "representation-selection experiment for the seven detailed graphs because they "
            "were explored with 50G. BioKG and YAGO3-10 were added before their metrics, but "
            "use the already selected table and decoder protocol."
        ),
        "seeds": list(SEEDS),
        "context": {
            "budgets_per_class": list(BUDGETS),
            "nested": True,
            "maximum_pool": "400 train positives plus 400 train-filtered corrupted negatives",
            "crossfit_folds": 10,
            "negative_filtering": "train true triples only",
            "permuted_condition": (
                "same selected rows, features, order, and relation-wise label marginals; "
                "deterministic positive/negative swaps within relation"
            ),
        },
        "representation": {
            "full": "18 frozen train-graph invariants plus one frozen ULTRA-4G score",
            "structural_only": "same 18 invariants without ULTRA score",
            "ultra_only": "only the frozen ULTRA-4G candidate score",
            "raw_entity_ids": False,
            "raw_cluster_ids": False,
            "evaluation_features": "train graph only",
        },
        "models": {
            "tabpfn_v3": {
                "estimators": N_ESTIMATORS,
                "fit_mode": "fit_with_cache",
                "target_parameter_updates": False,
                "checkpoint_data_path": TABPFN_CHECKPOINT_DATA_PATH,
                "checkpoint_sha256": EXPECTED_TABPFN_SHA256,
                "conda_environment_name_required": False,
                "actual_environment_and_packages_logged": True,
            },
            "classical": list(DECODERS[1:]),
        },
        "composition": {
            "ultra_only": True,
            "direct": True,
            "fixed_blend": {
                "formula": "(1-alpha)*z_query(ULTRA)+alpha*z_query(logit(decoder_probability))",
                "alpha_grid": ALPHA_GRID.tolist(),
                "selection": "combined validation MRR only; smallest alpha breaks ties",
                "test_alpha_sweep": False,
            },
            "direction_selector": (
                "choose ULTRA or decoder independently for head and tail using validation "
                "MRR only, then apply those two fixed choices to test"
            ),
        },
        "evaluation": {
            "query_selection": (
                "up to 64 deterministic relation-stratified validation and test triples"
            ),
            "candidate_set": "all entities for both head and tail",
            "filtering": "train+validation+test true triples in evaluator only",
            "test_used_for_selection": False,
            "primary_metric": "combined filtered MRR; head/tail and Hits@1/3/10 also reported",
        },
        "phases": {
            "primary": (
                "all nine graphs: 400+400 full table, correct/permuted labels, "
                "all matched baselines"
            ),
            "ablation": (
                "seven previously explored graphs: 400+400 structural-only and "
                "ULTRA-score-only for TabPFN and HGB"
            ),
            "scaling": (
                "seven previously explored graphs: 20+20 and 200+200 full table "
                "for all decoders"
            ),
            "scope_reason": (
                "BioKG and YAGO3-10 are large new checkpoint-control graphs and receive "
                "the decisive primary matrix; secondary grids are omitted by a pre-metric "
                "compute rule, not by observed performance"
            ),
        },
    }


def configure_engine() -> None:
    engine.PROTOCOL_REVISION = PROTOCOL
    engine.DATASETS = DATASETS
    engine.DATASET_RATIONALE = DATASET_RATIONALE
    engine.SEEDS = SEEDS
    engine.BUDGETS = BUDGETS
    engine.LOW_SHOT_BUDGETS = BUDGETS
    engine.CAUSAL_BUDGET = PRIMARY_BUDGET
    engine.CONTEXT_POOL_PER_CLASS = PRIMARY_BUDGET
    engine.DECODERS = DECODERS
    engine.CLASSICAL_DECODERS = DECODERS[1:]
    engine.CAUSAL_DECODERS = PERMUTED_DECODERS
    engine.FAMILY_ROOT = FAMILY_ROOT
    engine.DATA_ROOT = DATA_ROOT
    engine.RUN_ROOT = RUN_ROOT
    engine.TUNING_ROOT = TUNING_ROOT
    engine.LOG_ROOT = LOG_ROOT
    engine.SUMMARY_PATH = SUMMARY_PATH
    engine.SUMMARY_MD = SUMMARY_MD
    engine.PLAN_PATH = PLAN_PATH
    engine.ULTRA_CHECKPOINT = ULTRA_CHECKPOINT
    engine.protocol_specification = protocol_specification
    engine.source_hashes = source_hashes


configure_engine()
_load_pykeen_dataset = engine.load_pykeen_dataset


def _audit_mapped_dataset(name: str, data: dict) -> dict:
    splits = {
        split: np.asarray(data[split], dtype=np.int64).reshape(-1, 3)
        for split in ("train", "valid", "test")
    }
    sets = {split: {tuple(row) for row in rows.tolist()} for split, rows in splits.items()}
    duplicates = {split: len(splits[split]) - len(sets[split]) for split in splits}
    overlaps = {
        "train_valid": len(sets["train"] & sets["valid"]),
        "train_test": len(sets["train"] & sets["test"]),
        "valid_test": len(sets["valid"] & sets["test"]),
    }
    if any(duplicates.values()) or any(overlaps.values()):
        raise RuntimeError(
            f"invalid split for {name}: duplicates={duplicates}, overlaps={overlaps}"
        )
    train_entities = set(splits["train"][:, 0]) | set(splits["train"][:, 2])
    heldout_entities = set(np.concatenate([
        splits["valid"][:, 0], splits["valid"][:, 2],
        splits["test"][:, 0], splits["test"][:, 2],
    ]))
    train_relations = set(splits["train"][:, 1])
    heldout_relations = set(np.concatenate([
        splits["valid"][:, 1], splits["test"][:, 1]
    ]))
    if heldout_entities - train_entities or heldout_relations - train_relations:
        raise RuntimeError(f"{name} is not a closed-vocabulary transductive split")
    return {
        "duplicates": duplicates,
        "overlaps": overlaps,
        "num_train": len(splits["train"]),
        "num_valid": len(splits["valid"]),
        "num_test": len(splits["test"]),
    }


def mapped_split_digest(data: dict) -> str:
    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        rows = np.ascontiguousarray(data[split], dtype=np.int64)
        digest.update(split.encode("ascii"))
        digest.update(rows.tobytes())
    return digest.hexdigest()


def validate_expected_dataset(name: str, data: dict) -> None:
    observed = (
        int(data["num_entities"]),
        int(data["num_relations"]),
        len(data["train"]),
        len(data["valid"]),
        len(data["test"]),
    )
    if observed != EXPECTED_DATASET_STATS[name]:
        raise RuntimeError(
            f"{name} statistics changed: expected={EXPECTED_DATASET_STATS[name]} "
            f"observed={observed}"
        )
    expected_digest = PRE_REGISTRATION_SPLIT_DIGESTS.get(name)
    if expected_digest is not None:
        observed_digest = mapped_split_digest(data)
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"{name} mapped split digest changed: expected={expected_digest} "
                f"observed={observed_digest}"
            )


def _load_local_processed(name: str) -> dict:
    loaded = load_processed(name)
    data = {
        "train": loaded["train_triples"],
        "valid": loaded["valid_triples"],
        "test": loaded["test_triples"],
        "num_entities": len(loaded["entities"]),
        "num_relations": len(loaded["relations"]),
        "entity_to_id": loaded["entity_to_id"],
        "relation_to_id": loaded["relation_to_id"],
    }
    _audit_mapped_dataset(name, data)
    return data


def _load_kg20c() -> dict:
    symbolic = {}
    for split in ("train", "valid", "test"):
        path = KG20C_ROOT / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing pinned KG20C split: {path}")
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) != 3 or any(not field for field in fields):
                    raise RuntimeError(f"invalid KG20C TSV row {path}:{line_number}")
                rows.append(tuple(fields))
        symbolic[split] = rows
    sets = {split: set(rows) for split, rows in symbolic.items()}
    if any(len(symbolic[split]) != len(sets[split]) for split in symbolic):
        raise RuntimeError("KG20C contains duplicate triples")
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise RuntimeError("KG20C contains overlapping splits")
    entities = sorted({value for rows in symbolic.values() for h, _, t in rows for value in (h, t)})
    relations = sorted({r for rows in symbolic.values() for _, r, _ in rows})
    entity_to_id = {value: index for index, value in enumerate(entities)}
    relation_to_id = {value: index for index, value in enumerate(relations)}
    data = {
        split: np.asarray([
            (entity_to_id[h], relation_to_id[r], entity_to_id[t])
            for h, r, t in rows
        ], dtype=np.int64)
        for split, rows in symbolic.items()
    }
    data.update({
        "num_entities": len(entities),
        "num_relations": len(relations),
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
    })
    audit = _audit_mapped_dataset("kg20c", data)
    observed = {
        "num_entities": data["num_entities"],
        "num_relations": data["num_relations"],
        "num_train": audit["num_train"],
        "num_valid": audit["num_valid"],
        "num_test": audit["num_test"],
    }
    if observed != KG20C_EXPECTED:
        raise RuntimeError(f"KG20C statistics changed: {observed}")
    return data


def load_dataset(name: str) -> dict:
    if name in LOCAL_PROCESSED:
        data = _load_local_processed(name)
    elif name == "kg20c":
        data = _load_kg20c()
    elif name in PYKEEN_DATASETS:
        data = _load_pykeen_dataset(name)
        _audit_mapped_dataset(name, data)
    else:
        raise ValueError(f"unsupported ULTRA-4G reassessment dataset: {name}")
    validate_expected_dataset(name, data)
    return data


engine.load_pykeen_dataset = load_dataset
_prepare_dataset = engine.prepare_dataset


def prepare_dataset(dataset: str) -> dict:
    data = load_dataset(dataset)
    source_audit = {
        "dataset": dataset,
        "source": DATASET_SOURCES[dataset],
        "num_entities": int(data["num_entities"]),
        "num_relations": int(data["num_relations"]),
        "num_train": int(len(data["train"])),
        "num_valid": int(len(data["valid"])),
        "num_test": int(len(data["test"])),
        "mapped_split_digest": mapped_split_digest(data),
        "completed_before_ultra_scoring": True,
    }
    audit_path = engine.dataset_dir(dataset) / "source_audit.json"
    if audit_path.exists():
        saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if saved_audit != source_audit:
            raise RuntimeError(
                f"{dataset} source audit changed after it was frozen: "
                f"saved={saved_audit} observed={source_audit}"
            )
    else:
        engine.pilot.atomic_json(audit_path, source_audit)
    del data
    manifest = _prepare_dataset(dataset)
    manifest["source"] = DATASET_SOURCES[dataset]
    manifest["source_audit"] = source_audit
    manifest["ultra4g_training_membership"] = False
    manifest["ultra4g_training_graphs"] = list(ULTRA4G_TRAINING_GRAPHS)
    if dataset == "biokg":
        manifest["split_generation"] = {
            "source": "single official BioKG v1.0.0 edge list",
            "ratios": [0.8, 0.1, 0.1],
            "random_state": 0,
            "implementation": "PyKEEN 1.11.1 coverage split",
            "fixed_before_metrics": True,
        }
    engine.pilot.atomic_json(engine.dataset_dir(dataset) / "manifest.json", manifest)
    return manifest


engine.prepare_dataset = prepare_dataset


def job_key(job: dict) -> tuple:
    return (
        DATASETS.index(job["dataset"]),
        SEEDS.index(job["seed"]),
        job["budget"],
        job["condition"],
        job["representation"],
        job["decoder"],
    )


def phase_jobs(phase: str, group: str = "all") -> list[dict]:
    jobs = []
    if phase == "primary":
        for dataset in DATASETS:
            for seed in SEEDS:
                for decoder in DECODERS:
                    jobs.append({
                        "dataset": dataset, "seed": seed, "budget": PRIMARY_BUDGET,
                        "condition": "correct_labels", "representation": "full",
                        "decoder": decoder,
                    })
                for decoder in PERMUTED_DECODERS:
                    jobs.append({
                        "dataset": dataset, "seed": seed, "budget": PRIMARY_BUDGET,
                        "condition": "permuted_labels", "representation": "full",
                        "decoder": decoder,
                    })
    elif phase == "ablation":
        for dataset in DETAILED_DATASETS:
            for seed in SEEDS:
                for representation in ("structural_only", "ultra_only"):
                    for decoder in ABLATION_DECODERS:
                        jobs.append({
                            "dataset": dataset, "seed": seed,
                            "budget": PRIMARY_BUDGET,
                            "condition": "correct_labels",
                            "representation": representation,
                            "decoder": decoder,
                        })
    elif phase == "scaling":
        for dataset in DETAILED_DATASETS:
            for seed in SEEDS:
                for budget in SCALING_BUDGETS:
                    for decoder in DECODERS:
                        jobs.append({
                            "dataset": dataset, "seed": seed, "budget": budget,
                            "condition": "correct_labels", "representation": "full",
                            "decoder": decoder,
                        })
    elif phase == "all":
        jobs = [
            *phase_jobs("primary"),
            *phase_jobs("ablation"),
            *phase_jobs("scaling"),
        ]
    else:
        raise ValueError(phase)
    if group == "tabpfn":
        jobs = [job for job in jobs if job["decoder"] == "tabpfn_v3"]
    elif group == "classical":
        jobs = [job for job in jobs if job["decoder"] != "tabpfn_v3"]
    elif group != "all":
        raise ValueError(group)
    return jobs


def register() -> dict:
    if PLAN_PATH.exists():
        raise FileExistsError(f"registration already exists: {PLAN_PATH}")
    if not ULTRA_CHECKPOINT.exists() or file_sha256(ULTRA_CHECKPOINT) != EXPECTED_ULTRA_SHA256:
        raise RuntimeError("ULTRA-4G checkpoint missing or hash mismatch")
    if not TABPFN_CHECKPOINT.exists() or file_sha256(TABPFN_CHECKPOINT) != EXPECTED_TABPFN_SHA256:
        raise RuntimeError("TabPFN-v3 checkpoint missing or hash mismatch")
    if list(FAMILY_ROOT.rglob("metrics.json")):
        raise RuntimeError("ULTRA-4G metrics exist before registration")
    counts = {
        phase: {
            "total": len(phase_jobs(phase)),
            "tabpfn": len(phase_jobs(phase, "tabpfn")),
            "classical": len(phase_jobs(phase, "classical")),
        }
        for phase in ("primary", "ablation", "scaling")
    }
    payload = {
        "status": "REGISTERED_BEFORE_ULTRA4G_METRICS",
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_specification(),
        "source_hashes": source_hashes(),
        "planned_jobs": counts | {
            "total": sum(row["total"] for row in counts.values()),
            "tabpfn": sum(row["tabpfn"] for row in counts.values()),
            "classical": sum(row["classical"] for row in counts.values()),
        },
        "estimated_runtime_hours": {
            "assets": [1.5, 4.0],
            "primary": [5.0, 12.0],
            "ablation": [3.0, 7.0],
            "scaling": [3.0, 7.0],
            "total": [14.0, 30.0],
            "basis": (
                "measured 4-estimator Hetionet wall time, measured AristoV4/Hetionet "
                "asset preparation, entity-count scaling for BioKG/YAGO3-10, and four "
                "concurrent TabPFN workers"
            ),
        },
    }
    engine.pilot.atomic_json(PLAN_PATH, payload)
    print(json.dumps(payload["planned_jobs"], indent=2), flush=True)
    print("ESTIMATED_RUNTIME_HOURS 14-30", flush=True)
    return payload


def validate_registration() -> dict:
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"missing registration: {PLAN_PATH}")
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if payload["status"] != "REGISTERED_BEFORE_ULTRA4G_METRICS":
        raise RuntimeError("invalid registration status")
    if payload["protocol"] != protocol_specification():
        raise RuntimeError("registered protocol changed")
    if payload["source_hashes"] != source_hashes():
        raise RuntimeError("registered source changed")
    if file_sha256(ULTRA_CHECKPOINT) != EXPECTED_ULTRA_SHA256:
        raise RuntimeError("ULTRA-4G checkpoint hash changed")
    if file_sha256(TABPFN_CHECKPOINT) != EXPECTED_TABPFN_SHA256:
        raise RuntimeError("TabPFN-v3 checkpoint hash changed")
    return payload


def relation_permutation(labels, triples, seed: int, budget: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    triples = np.asarray(triples, dtype=np.int64)
    permutation = np.arange(len(labels), dtype=np.int64)
    for relation in np.unique(triples[:, 1]):
        indices = np.flatnonzero(triples[:, 1] == relation)
        rng = np.random.default_rng(
            np.random.SeedSequence([20260715, seed, int(relation), budget, 4001])
        )
        positive = indices[labels[indices] == 1]
        negative = indices[labels[indices] == 0]
        positive = positive[rng.permutation(len(positive))]
        negative = negative[rng.permutation(len(negative))]
        paired = min(len(positive), len(negative))
        permutation[positive[:paired]] = negative[:paired]
        permutation[negative[:paired]] = positive[:paired]
    supplied = labels[permutation]
    if int(supplied.sum()) != int(labels.sum()):
        raise AssertionError("permutation changed global class balance")
    for relation in np.unique(triples[:, 1]):
        rows = triples[:, 1] == relation
        if int(supplied[rows].sum()) != int(labels[rows].sum()):
            raise AssertionError("permutation changed relation-wise class balance")
    if not np.any(supplied != labels):
        raise RuntimeError("label intervention changed no labels")
    return supplied, permutation


def context_payload(dataset: str, seed: int, budget: int, condition: str) -> dict:
    context = engine.load_context_asset(dataset, seed)
    selected = engine.nested_indices(context["y"], seed, budget)
    original = context["y"][selected].astype(np.int64, copy=False)
    triples = context["triples"][selected]
    if condition == "permuted_labels":
        supplied, permutation = relation_permutation(original, triples, seed, budget)
    elif condition == "correct_labels":
        supplied = original
        permutation = np.arange(len(original), dtype=np.int64)
    else:
        raise ValueError(condition)
    return {
        "X": context["X"][selected].astype(np.float32, copy=False),
        "y": supplied,
        "original_y": original,
        "triples": triples,
        "selected": selected,
        "permutation": permutation,
        "policy": "nested_class_balanced_same_rows",
    }


engine.context_payload = context_payload


def representation_indices(representation: str) -> tuple[list[int], list[int]]:
    if representation == "full":
        return list(range(19)), [0, 16]
    if representation == "structural_only":
        return list(range(18)), [0, 16]
    if representation == "ultra_only":
        return [18], []
    raise ValueError(representation)


def select_columns(matrix, representation: str):
    columns, _ = representation_indices(representation)
    if representation == "full":
        return matrix
    if representation == "structural_only":
        return matrix[:, :18]
    return matrix[:, 18:19]


def feature_names(representation: str) -> list[str]:
    return [engine.FEATURE_NAMES[index] for index in representation_indices(representation)[0]]


def fit_predict_tabpfn3_portable(train_X, train_y, eval_Xs, seed: int,
                                 categorical_indices: list[int],
                                 n_estimators: int):
    """Run the pinned TabPFN checkpoint without assuming a host-specific env path."""
    import torch
    import tabpfn
    from tabpfn import TabPFNClassifier

    if not torch.cuda.is_available():
        raise RuntimeError("TabPFN-v3 requires CUDA for this registered suite")
    if file_sha256(TABPFN_CHECKPOINT) != EXPECTED_TABPFN_SHA256:
        raise RuntimeError("TabPFN-v3 checkpoint missing or hash mismatch")
    train_arr = np.asarray(train_X, dtype=np.float32)
    model_args = {
        "model_path": TABPFN_CHECKPOINT,
        "device": "cuda",
        "n_estimators": n_estimators,
        "random_state": seed,
        "fit_mode": "fit_with_cache",
    }
    if len(train_arr) > 10_000:
        model_args["ignore_pretraining_limits"] = True
    if categorical_indices:
        max_cardinality = max(
            int(np.unique(train_arr[:, index]).size)
            for index in categorical_indices
        )
        model_args["categorical_features_indices"] = categorical_indices
        model_args["inference_config"] = build_tabpfn_inference_config(
            max(max_cardinality, 2)
        )

    fit_started = time.time()
    model = TabPFNClassifier(**model_args)
    model.fit(train_arr, np.asarray(train_y))
    fit_seconds = time.time() - fit_started
    predict_started = time.time()
    predictions = []
    for matrix in eval_Xs:
        matrix = np.asarray(matrix, dtype=np.float32)
        chunks = [
            positive_class_probabilities(
                model, matrix[start:start + TABPFN_PREDICT_BATCH]
            )
            for start in range(0, len(matrix), TABPFN_PREDICT_BATCH)
        ]
        predictions.append(np.concatenate(chunks))
    predict_seconds = time.time() - predict_started
    del model
    torch.cuda.empty_cache()
    return predictions, {
        "model": "tabpfn3",
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "interpreter": sys.executable,
        "tabpfn_version": tabpfn.__version__,
        "checkpoint_path": str(TABPFN_CHECKPOINT),
        "checkpoint_sha256": EXPECTED_TABPFN_SHA256,
        "n_estimators": n_estimators,
        "fit_mode": "fit_with_cache",
        "prediction_batch_rows": TABPFN_PREDICT_BATCH,
        "context_size": int(len(train_arr)),
        "fit_time": fit_seconds,
        "predict_time": predict_seconds,
    }


def run_dir(job: dict, smoke: bool = False) -> Path:
    seed = f"{job['seed']}_smoke" if smoke else str(job["seed"])
    return (
        RUN_ROOT / job["dataset"] / job["condition"] / job["representation"]
        / job["decoder"] / f"{job['budget']}pos_{job['budget']}neg" / seed
    )


def run_complete(path: Path) -> bool:
    if not all((path / name).exists() for name in REQUIRED_RUN_FILES):
        return False
    try:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        return np.isfinite(metrics["test"]["direct"]["combined"]["mrr"])
    except Exception:
        return False


def _atomic_npy(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32))
    os.replace(temporary, path)


def compact_asset(asset: dict, queries: int = 2) -> dict:
    num_entities = asset["filtered"].shape[1]
    return {
        **asset,
        "X": asset["X"][: queries * num_entities],
        "filtered": asset["filtered"][:queries],
        "target_entities": asset["target_entities"][:queries],
        "side_codes": asset["side_codes"][:queries],
    }


def build_hgb(representation: str, params: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    selected_columns, _ = representation_indices(representation)
    # Match the registered full-table HGB baseline: relation_id remains numeric
    # because sklearn's native categorical path is limited to 255 categories.
    categorical = [
        local_index
        for local_index, original_index in enumerate(selected_columns)
        if original_index == 16
    ]
    return HistGradientBoostingClassifier(
        random_state=seed,
        categorical_features=categorical or None,
        max_iter=200,
        **params,
    )


def build_model(job: dict, context_rows: int):
    selection = engine.tune_classical(job["dataset"], job["decoder"])
    params = selection["selected_params"]
    if job["representation"] == "full":
        model = engine.build_classical(job["decoder"], params, job["seed"], context_rows)
    elif job["decoder"] == "hgb_frozen":
        model = build_hgb(job["representation"], params, job["seed"])
    else:
        raise ValueError("only HGB is registered for reduced representations")
    return model, params


def model_input(job: dict, matrix):
    selected = select_columns(matrix, job["representation"])
    if job["representation"] == "full":
        return engine.model_input(job["decoder"], selected)
    return selected


def classical_predict(model, matrix, chunk_size: int = 100_000) -> np.ndarray:
    output = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        output[start:stop] = engine.positive_probability(model, matrix[start:stop])
    return output


def query_standardize(asset: dict, values) -> np.ndarray:
    num_entities = asset["filtered"].shape[1]
    matrix = np.asarray(values, dtype=np.float64).reshape(-1, num_entities)
    keep = ~np.asarray(asset["filtered"], dtype=bool)
    denominator = keep.sum(axis=1)
    mean = np.sum(matrix * keep, axis=1) / denominator
    centered = matrix - mean[:, None]
    scale = np.sqrt(np.sum(centered * centered * keep, axis=1) / denominator)
    scale[scale < 1e-8] = 1.0
    return (centered / scale[:, None]).reshape(-1)


def select_alpha(valid: dict, ultra_z, model_z) -> tuple[float, list[dict]]:
    curve = []
    for alpha in ALPHA_GRID:
        scores = (1.0 - alpha) * ultra_z + alpha * model_z
        metrics, _ = engine.metrics_from_scores(valid, scores)
        curve.append({
            "alpha": float(alpha),
            "combined_mrr": metrics["combined"]["mrr"],
            "head_mrr": metrics["head"]["mrr"],
            "tail_mrr": metrics["tail"]["mrr"],
        })
    best = max(row["combined_mrr"] for row in curve)
    alpha = min(row["alpha"] for row in curve if abs(row["combined_mrr"] - best) <= 1e-12)
    return alpha, curve


def direction_scores(asset: dict, ultra_scores, direct_scores, choices: dict) -> np.ndarray:
    num_entities = asset["filtered"].shape[1]
    output = np.asarray(ultra_scores, dtype=np.float64).copy()
    for query, side_code in enumerate(asset["side_codes"]):
        side = "head" if int(side_code) else "tail"
        if choices[side] == "direct":
            start = query * num_entities
            output[start:start + num_entities] = direct_scores[start:start + num_entities]
    return output


def prediction_frame(method_frames: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    base = method_frames["ultra"].copy().rename(columns={
        "rank": "ultra_rank",
        "reciprocal_rank": "ultra_reciprocal_rank",
        "target_score": "ultra_target_score",
    })
    for method in ("direct", "fixed_blend", "direction_selector"):
        current = method_frames[method]
        base[f"{method}_rank"] = current["rank"].to_numpy()
        base[f"{method}_reciprocal_rank"] = current["reciprocal_rank"].to_numpy()
        base[f"{method}_target_score"] = current["target_score"].to_numpy()
    base["split"] = split
    return base


def execute(job: dict, smoke: bool = False) -> dict | None:
    validate_registration()
    output = run_dir(job, smoke)
    if not smoke and run_complete(output):
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with engine.run_lock(output) as acquired:
        if not acquired:
            return None
        if not smoke and run_complete(output):
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        started = time.time()
        try:
            context = context_payload(
                job["dataset"], job["seed"], job["budget"], job["condition"]
            )
            valid = engine.load_split_asset(job["dataset"], "valid")
            test = engine.load_split_asset(job["dataset"], "test")
            if smoke:
                valid = compact_asset(valid)
                test = compact_asset(test)
            context_X = select_columns(context["X"], job["representation"])
            matrices = [
                select_columns(valid["X"], job["representation"]),
                select_columns(test["X"], job["representation"]),
            ]
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if job["decoder"] == "tabpfn_v3":
                _, categorical = representation_indices(job["representation"])
                predictions, info = fit_predict_tabpfn3_portable(
                    context_X,
                    context["y"],
                    matrices,
                    job["seed"],
                    categorical_indices=categorical,
                    n_estimators=1 if smoke else N_ESTIMATORS,
                )
                parameters = {"n_estimators": 1 if smoke else N_ESTIMATORS}
            else:
                fit_started = time.time()
                model, parameters = build_model(job, len(context_X))
                model.fit(model_input(job, context["X"]), context["y"])
                fit_seconds = time.time() - fit_started
                predict_started = time.time()
                predictions = [
                    classical_predict(model, model_input(job, asset["X"]))
                    for asset in (valid, test)
                ]
                info = {
                    "fit_time": fit_seconds,
                    "predict_time": time.time() - predict_started,
                }

            ultra_raw = [asset["X"][:, -1] for asset in (valid, test)]
            ultra_z = [
                query_standardize(asset, scores)
                for asset, scores in zip((valid, test), ultra_raw)
            ]
            logits = [
                np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
                for scores in predictions
            ]
            model_z = [
                query_standardize(asset, scores)
                for asset, scores in zip((valid, test), logits)
            ]
            alpha, alpha_curve = select_alpha(valid, ultra_z[0], model_z[0])
            blends = [
                (1.0 - alpha) * ultra + alpha * model
                for ultra, model in zip(ultra_z, model_z)
            ]
            valid_ultra_metrics, _ = engine.metrics_from_scores(valid, ultra_raw[0])
            valid_direct_metrics, _ = engine.metrics_from_scores(valid, predictions[0])
            choices = {
                side: (
                    "direct"
                    if valid_direct_metrics[side]["mrr"] > valid_ultra_metrics[side]["mrr"]
                    else "ultra"
                )
                for side in ("head", "tail")
            }
            directional = [
                direction_scores(asset, ultra, direct, choices)
                for asset, ultra, direct in zip((valid, test), ultra_raw, predictions)
            ]
            metrics = {"selection": {
                "alpha": alpha,
                "alpha_curve": alpha_curve,
                "alpha_source": "combined validation MRR only",
                "direction_choices": choices,
                "direction_source": "head/tail validation MRR only",
                "test_sweep_computed": False,
            }}
            frames = {}
            for split, asset, direct, blend, selected_direction in (
                ("valid", valid, predictions[0], blends[0], directional[0]),
                ("test", test, predictions[1], blends[1], directional[1]),
            ):
                metrics[split] = {}
                frames[split] = {}
                for method, scores in (
                    ("ultra", asset["X"][:, -1]),
                    ("direct", direct),
                    ("fixed_blend", blend),
                    ("direction_selector", selected_direction),
                ):
                    metrics[split][method], frames[split][method] = engine.metrics_from_scores(
                        asset, scores
                    )
            if smoke:
                engine.pilot.atomic_json(output / "smoke_metrics.json", metrics)
                (output / "run.log").write_text("SMOKE OK\n", encoding="utf-8")
                return metrics

            _atomic_npy(output / "valid_scores.npy", predictions[0])
            _atomic_npy(output / "test_scores.npy", predictions[1])
            frame = pd.concat([
                prediction_frame(frames["valid"], "valid"),
                prediction_frame(frames["test"], "test"),
            ], ignore_index=True)
            peak = (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() and job["decoder"] == "tabpfn_v3"
                else None
            )
            config = {
                "experiment_family": PROTOCOL,
                **job,
                "feature_names": feature_names(job["representation"]),
                "num_features": len(feature_names(job["representation"])),
                "selected_hyperparameters": parameters,
                "ultra_checkpoint": str(ULTRA_CHECKPOINT),
                "ultra_checkpoint_sha256": EXPECTED_ULTRA_SHA256,
                "ultra_training_graphs": list(ULTRA4G_TRAINING_GRAPHS),
                "tabpfn_checkpoint": (
                    str(TABPFN_CHECKPOINT) if job["decoder"] == "tabpfn_v3" else None
                ),
                "tabpfn_checkpoint_sha256": (
                    EXPECTED_TABPFN_SHA256 if job["decoder"] == "tabpfn_v3" else None
                ),
                "tabpfn_fit_mode": (
                    "fit_with_cache" if job["decoder"] == "tabpfn_v3" else None
                ),
                "score_normalization": (
                    "per-query z-score over unfiltered candidates for fixed blend only; "
                    "direct and ULTRA metrics use raw scores"
                ),
                "evaluation": "relation-stratified subset; complete all-entity head+tail filtered",
                "registration": str(PLAN_PATH),
                "source_hashes": source_hashes(),
                "git_commit": engine.pilot.git_commit(),
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
                "feature_hash": engine.pilot.array_hash(context_X),
                "original_label_hash": engine.pilot.array_hash(context["original_y"]),
                "supplied_label_hash": engine.pilot.array_hash(context["y"]),
                "permutation_hash": engine.pilot.array_hash(context["permutation"]),
                "labels_changed": int(np.sum(context["original_y"] != context["y"])),
            }
            timings = {
                "fit_seconds": info.get("fit_time"),
                "prediction_seconds": info.get("predict_time"),
                "total_seconds": time.time() - started,
                "peak_gpu_memory_bytes": peak,
            }
            engine.pilot.atomic_json(output / "config.json", config)
            engine.pilot.atomic_json(output / "metrics.json", metrics)
            engine.pilot.atomic_json(output / "timings.json", timings)
            engine.pilot.atomic_json(output / "context_manifest.json", context_manifest)
            engine.pilot.atomic_parquet(output / "predictions.parquet", frame)
            (output / "run.log").write_text(
                "OK "
                f"direct_mrr={metrics['test']['direct']['combined']['mrr']:.9f} "
                f"blend_mrr={metrics['test']['fixed_blend']['combined']['mrr']:.9f} "
                f"alpha={alpha:.2f}\n",
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


def smoke_jobs(dataset: str | None = None) -> list[dict]:
    jobs = [
        {
            "dataset": "umls", "seed": 101, "budget": 20,
            "condition": "correct_labels", "representation": "full",
            "decoder": decoder,
        }
        for decoder in DECODERS
    ]
    jobs.extend([
        {
            "dataset": "umls", "seed": 101, "budget": PRIMARY_BUDGET,
            "condition": "permuted_labels", "representation": "full",
            "decoder": decoder,
        }
        for decoder in PERMUTED_DECODERS
    ])
    for representation in ("structural_only", "ultra_only"):
        for decoder in ABLATION_DECODERS:
            jobs.append({
                "dataset": "umls", "seed": 101, "budget": PRIMARY_BUDGET,
                "condition": "correct_labels", "representation": representation,
                "decoder": decoder,
            })
    for target_dataset in DATASETS:
        if target_dataset != "umls":
            jobs.append({
                "dataset": target_dataset, "seed": 101, "budget": 20,
                "condition": "correct_labels", "representation": "full",
                "decoder": "tabpfn_v3",
            })
    if dataset is not None:
        jobs = [job for job in jobs if job["dataset"] == dataset]
    return jobs


def paired(left, right) -> dict:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return {
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "sample_standard_deviation": (
            float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        ),
        "positive_seeds": int(np.sum(differences > 0)),
        "per_seed_difference": {
            str(seed): float(value) for seed, value in zip(SEEDS, differences)
        },
    }


def metric(job: dict, method: str, side: str = "combined") -> float:
    payload = json.loads((run_dir(job) / "metrics.json").read_text(encoding="utf-8"))
    return float(payload["test"][method][side]["mrr"])


def _job(dataset, seed, budget, decoder, condition="correct_labels", representation="full"):
    return {
        "dataset": dataset,
        "seed": seed,
        "budget": budget,
        "condition": condition,
        "representation": representation,
        "decoder": decoder,
    }


def summarize() -> dict:
    missing = [job for job in phase_jobs("all") if not run_complete(run_dir(job))]
    if missing:
        payload = {"status": "INCOMPLETE", "completed": len(phase_jobs("all")) - len(missing), "missing": missing}
        engine.pilot.atomic_json(SUMMARY_PATH, payload)
        return payload
    datasets = {}
    for dataset in DATASETS:
        budget_rows = {}
        dataset_budgets = BUDGETS if dataset in DETAILED_DATASETS else (PRIMARY_BUDGET,)
        for budget in dataset_budgets:
            methods = {}
            for decoder in DECODERS:
                jobs = [_job(dataset, seed, budget, decoder) for seed in SEEDS]
                methods[decoder] = {}
                for method in ("direct", "fixed_blend", "direction_selector"):
                    values = [metric(job, method) for job in jobs]
                    methods[decoder][method] = {
                        "mean_mrr": float(np.mean(values)),
                        "std_mrr": float(np.std(values, ddof=1)),
                        "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
                    }
            ultra = [metric(_job(dataset, seed, budget, "tabpfn_v3"), "ultra") for seed in SEEDS]
            tab = [metric(_job(dataset, seed, budget, "tabpfn_v3"), "direct") for seed in SEEDS]
            tab_blend = [metric(_job(dataset, seed, budget, "tabpfn_v3"), "fixed_blend") for seed in SEEDS]
            selected_classical = []
            selected_names = []
            for seed in SEEDS:
                candidates = []
                for decoder in DECODERS[1:]:
                    job = _job(dataset, seed, budget, decoder)
                    payload = json.loads((run_dir(job) / "metrics.json").read_text(encoding="utf-8"))
                    candidates.append((
                        payload["valid"]["direct"]["combined"]["mrr"],
                        decoder,
                        payload["test"]["direct"]["combined"]["mrr"],
                    ))
                _, name, test_value = max(candidates, key=lambda row: (row[0], row[1]))
                selected_names.append(name)
                selected_classical.append(float(test_value))
            budget_rows[str(budget)] = {
                "ultra_mean_mrr": float(np.mean(ultra)),
                "models": methods,
                "tabpfn_vs_ultra": paired(tab, ultra),
                "tabpfn_blend_vs_ultra": paired(tab_blend, ultra),
                "tabpfn_vs_validation_selected_classical": paired(tab, selected_classical),
                "validation_selected_classical": {
                    "per_seed_decoder": dict(zip(map(str, SEEDS), selected_names)),
                    "mean_test_mrr": float(np.mean(selected_classical)),
                },
            }
        correct = [
            metric(_job(dataset, seed, PRIMARY_BUDGET, "tabpfn_v3"), "direct")
            for seed in SEEDS
        ]
        permuted = [
            metric(_job(dataset, seed, PRIMARY_BUDGET, "tabpfn_v3", "permuted_labels"), "direct")
            for seed in SEEDS
        ]
        hgb_correct = [
            metric(_job(dataset, seed, PRIMARY_BUDGET, "hgb_frozen"), "direct")
            for seed in SEEDS
        ]
        representations = {}
        if dataset in DETAILED_DATASETS:
            for representation in REPRESENTATIONS:
                representations[representation] = {}
                for decoder in ABLATION_DECODERS:
                    values = [
                        metric(_job(dataset, seed, PRIMARY_BUDGET, decoder, representation=representation), "direct")
                        for seed in SEEDS
                    ]
                    representations[representation][decoder] = {
                        "mean_mrr": float(np.mean(values)),
                        "std_mrr": float(np.std(values, ddof=1)),
                    }
        primary = budget_rows[str(PRIMARY_BUDGET)]
        datasets[dataset] = {
            "budgets": budget_rows,
            "label_intervention_tabpfn": paired(correct, permuted),
            "tabpfn_vs_hgb_400": paired(correct, hgb_correct),
            "representations_400": representations,
            "checks": {
                "labels_matter": (
                    paired(correct, permuted)["mean_difference"] >= 0.010
                    and paired(correct, permuted)["positive_seeds"] >= 4
                ),
                "tabpfn_specific_vs_hgb": (
                    paired(correct, hgb_correct)["mean_difference"] >= 0.005
                    and paired(correct, hgb_correct)["positive_seeds"] >= 3
                ),
                "direct_beats_ultra": (
                    primary["tabpfn_vs_ultra"]["mean_difference"] >= 0.005
                    and primary["tabpfn_vs_ultra"]["positive_seeds"] >= 4
                ),
                "blend_beats_ultra": (
                    primary["tabpfn_blend_vs_ultra"]["mean_difference"] >= 0.005
                    and primary["tabpfn_blend_vs_ultra"]["positive_seeds"] >= 4
                ),
            },
        }
    payload = {
        "status": "COMPLETE",
        "protocol": PROTOCOL,
        "checkpoint": {"path": str(ULTRA_CHECKPOINT), "sha256": EXPECTED_ULTRA_SHA256},
        "datasets": datasets,
        "claim_limit": protocol_specification()["claim_limit"],
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    engine.pilot.atomic_json(SUMMARY_PATH, payload)
    lines = [
        "# ULTRA-4G OOD Reassessment",
        "",
        "All evaluated graphs are excluded from the documented ULTRA-4G training mixture.",
        "",
        "| Dataset | ULTRA | TabPFN direct | TabPFN blend | HGB direct | Labels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in datasets.items():
        primary = row["budgets"][str(PRIMARY_BUDGET)]
        lines.append(
            f"| {dataset} | {primary['ultra_mean_mrr']:.6f} | "
            f"{primary['models']['tabpfn_v3']['direct']['mean_mrr']:.6f} | "
            f"{primary['models']['tabpfn_v3']['fixed_blend']['mean_mrr']:.6f} | "
            f"{primary['models']['hgb_frozen']['direct']['mean_mrr']:.6f} | "
            f"{row['label_intervention_tabpfn']['mean_difference']:+.6f} |"
        )
    lines.extend(["", payload["claim_limit"]])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def tune(dataset: str | None = None) -> None:
    validate_registration()
    for target in ([dataset] if dataset else DATASETS):
        for decoder in DECODERS[1:]:
            engine.tune_classical(target, decoder)


def smoke(dataset: str | None = None) -> None:
    for job in smoke_jobs(dataset):
        execute(job, smoke=True)
        print(f"SMOKE OK {job}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--phase", choices=("primary", "ablation", "scaling", "all"), default="all")
    parser.add_argument("--group", choices=("all", "tabpfn", "classical"), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    elif args.validate:
        payload = validate_registration()
        print(f"VALID {payload['status']} jobs={payload['planned_jobs']['total']}")
    elif args.prepare:
        validate_registration()
        targets = [args.dataset] if args.dataset else DATASETS
        for dataset in targets:
            engine.prepare_dataset(dataset)
    elif args.tune:
        tune(args.dataset)
    elif args.smoke:
        validate_registration()
        smoke(args.dataset)
    elif args.run_shard:
        validate_registration()
        jobs = phase_jobs(args.phase, args.group)
        if args.dataset:
            jobs = [job for job in jobs if job["dataset"] == args.dataset]
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
        print(json.dumps(summarize(), indent=2), flush=True)
    else:
        parser.error("choose an action")


if __name__ == "__main__":
    main()
