"""Unified dataset preparation (Module: datasets).

Reuses the audited loader from the parent project (load_kg_dataset — see
results/audit/sbm_tfm_audit.json) for umls/kinships/wn18rr and extends the URL
table with nations and codex-small, which follow the same tab-separated
head\trelation\ttail raw format.

Output layout per dataset (spec-mandated):
    data/processed/<dataset>/entity_to_id.json
    data/processed/<dataset>/relation_to_id.json
    data/processed/<dataset>/train_triples.npy   int64 [n, 3] (h, r, t)
    data/processed/<dataset>/valid_triples.npy
    data/processed/<dataset>/test_triples.npy
    data/processed/<dataset>/dataset_stats.json
"""

import json

import numpy as np

from .paths import DATA_PROCESSED, DATA_RAW
from .sbm_import import DATASET_URLS, all_true_triples, load_kg_dataset

PYKEEN_BASE = "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets"
CODEX_BASE = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples"

KGE_BASE = "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data"

EXTRA_DATASET_URLS = {
    "nations": {
        "train": f"{PYKEEN_BASE}/nations/train.txt",
        "valid": f"{PYKEEN_BASE}/nations/valid.txt",
        "test": f"{PYKEEN_BASE}/nations/test.txt",
    },
    "codex-small": {
        "train": f"{CODEX_BASE}/codex-s/train.txt",
        "valid": f"{CODEX_BASE}/codex-s/valid.txt",
        "test": f"{CODEX_BASE}/codex-s/test.txt",
    },
    "fb15k237": {
        "train": f"{KGE_BASE}/FB15k-237/train.txt",
        "valid": f"{KGE_BASE}/FB15k-237/valid.txt",
        "test": f"{KGE_BASE}/FB15k-237/test.txt",
    },
}

ALL_DATASET_URLS = {**DATASET_URLS, **EXTRA_DATASET_URLS}


def _register_extra_urls():
    """Extend the parent project's URL table in place so load_kg_dataset can
    serve the extra datasets without modifying the audited source file."""
    for name, urls in EXTRA_DATASET_URLS.items():
        DATASET_URLS.setdefault(name, urls)


def load_unified(dataset, download=True):
    """Returns the spec's unified internal format."""
    _register_extra_urls()
    data_dir = DATA_RAW / dataset
    mapped, entities, relations, entity_to_id, relation_to_id = load_kg_dataset(
        data_dir, download, dataset
    )
    splits = {
        split: mapped[split][["h", "r", "t"]].to_numpy(dtype=np.int64)
        for split in ("train", "valid", "test")
    }
    all_true = all_true_triples(mapped)
    return {
        "entities": entities,
        "relations": relations,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "train_triples": splits["train"],
        "valid_triples": splits["valid"],
        "test_triples": splits["test"],
        "all_true_triples": all_true,
    }


def prepare_dataset(dataset, download=True):
    unified = load_unified(dataset, download=download)
    out_dir = DATA_PROCESSED / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "entity_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(unified["entity_to_id"], f, indent=1)
    with (out_dir / "relation_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(unified["relation_to_id"], f, indent=1)
    for split in ("train", "valid", "test"):
        np.save(out_dir / f"{split}_triples.npy", unified[f"{split}_triples"])

    reused = dataset in ("umls", "kinships", "wn18rr")
    stats = {
        "dataset": dataset,
        "num_entities": len(unified["entities"]),
        "num_relations": len(unified["relations"]),
        "num_train": int(len(unified["train_triples"])),
        "num_valid": int(len(unified["valid_triples"])),
        "num_test": int(len(unified["test_triples"])),
        "num_all_true": int(len(unified["all_true_triples"])),
        "split_source": ALL_DATASET_URLS[dataset],
        "preprocessing_notes": (
            "tab/whitespace-separated head-relation-tail text splits; entities and "
            "relations sorted lexicographically over the union of all splits and "
            "mapped to contiguous int ids (deterministic)"
        ),
        "reused_from_sbm_tfm": reused,
        "source_path_if_reused": (
            str("run_umls_complex_vs_tabpfn.py::load_kg_dataset") if reused else
            "load_kg_dataset reused with extra URL registration (rfmlink/data_prep.py)"
        ),
        "duplicate_triples_check": {
            split: int(
                len(unified[f"{split}_triples"])
                - len({tuple(t) for t in unified[f"{split}_triples"].tolist()})
            )
            for split in ("train", "valid", "test")
        },
        "split_overlap_check": {
            "train_valid": int(
                len(
                    {tuple(t) for t in unified["train_triples"].tolist()}
                    & {tuple(t) for t in unified["valid_triples"].tolist()}
                )
            ),
            "train_test": int(
                len(
                    {tuple(t) for t in unified["train_triples"].tolist()}
                    & {tuple(t) for t in unified["test_triples"].tolist()}
                )
            ),
            "valid_test": int(
                len(
                    {tuple(t) for t in unified["valid_triples"].tolist()}
                    & {tuple(t) for t in unified["test_triples"].tolist()}
                )
            ),
        },
    }
    with (out_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_processed(dataset):
    """Load a prepared dataset from data/processed/<dataset>/."""
    out_dir = DATA_PROCESSED / dataset
    with (out_dir / "entity_to_id.json").open(encoding="utf-8") as f:
        entity_to_id = json.load(f)
    with (out_dir / "relation_to_id.json").open(encoding="utf-8") as f:
        relation_to_id = json.load(f)
    splits = {
        split: np.load(out_dir / f"{split}_triples.npy")
        for split in ("train", "valid", "test")
    }
    all_true = {
        tuple(t)
        for split in splits.values()
        for t in split.tolist()
    }
    entities = sorted(entity_to_id, key=entity_to_id.get)
    relations = sorted(relation_to_id, key=relation_to_id.get)
    return {
        "entities": entities,
        "relations": relations,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "train_triples": splits["train"],
        "valid_triples": splits["valid"],
        "test_triples": splits["test"],
        "all_true_triples": all_true,
    }
