#!/usr/bin/env python3
"""Matched transfer suite from deterministic, immutable context artifacts.

This protocol revision supersedes historical-MRR reproduction checks. It
regenerates every compared method from the same canonical bytes and never
overwrites the legacy context-intervention family.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RFMLINK_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RFMLINK_ROOT.parent
sys.path[:0] = [str(SCRIPT_DIR), str(RFMLINK_ROOT)]

import run_context_intervention_pilot as pilot  # noqa: E402
import run_context_transfer_controls as transfer  # noqa: E402
from rfmlink.contexts import load_context  # noqa: E402


PROTOCOL_REVISION = "deterministic_context_v1"
FAMILY_ROOT = WORKSPACE_ROOT / "results" / "context_intervention_deterministic_v1"
ARTIFACT_ROOT = FAMILY_ROOT / "context_artifacts"
RUN_ROOT = FAMILY_ROOT / "runs"
SUMMARY_JSON = FAMILY_ROOT / "summary.json"
SUMMARY_MD = FAMILY_ROOT / "summary.md"
SEEDS = tuple(range(101, 106))
DATASET_CONDITIONS = {
    "kinships": ("correct_labels", "permuted_labels"),
    "codex-small": ("correct_labels",),
}

# Reuse all existing builders/evaluators but isolate every output path.
pilot.SUITE_ROOT = RUN_ROOT


def package_versions() -> dict:
    import pandas as pd
    import sklearn
    import tabpfn
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tabpfn": tabpfn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_scatter": _package_version("torch-scatter"),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_dir(dataset: str, seed: int) -> Path:
    return ARTIFACT_ROOT / dataset / str(seed)


def artifact_complete(path: Path) -> bool:
    return (path / "context_input.npz").exists() and (path / "manifest.json").exists()


def save_artifact(path: Path, payload: dict, metadata: dict, pa, dataset: str, seed: int) -> None:
    artifact = path / "context_input.npz"
    atomic_npz(
        artifact,
        X=np.asarray(payload["X"], dtype=np.float32),
        y=np.asarray(payload["y"], dtype=np.int64),
        triples=np.asarray(payload["triples"], dtype=np.int64),
        indices=np.asarray(payload["indices"], dtype=np.int64),
        orientations=np.asarray(payload["orientations"]),
        original_y=np.asarray(payload["original_y"], dtype=np.int64),
        permutation=np.asarray(payload["permutation"], dtype=np.int64),
    )
    manifest = {
        "status": "CANONICAL",
        "protocol_revision": PROTOCOL_REVISION,
        "dataset": dataset,
        "seed": seed,
        "deterministic_algorithms": True,
        "context_artifact": str(artifact),
        "context_artifact_sha256": file_sha256(artifact),
        "context_feature_hash": pilot.array_hash(payload["X"]),
        "context_shape": list(payload["X"].shape),
        "context_row_identity_hash": pilot.hashlib.sha256(
            json.dumps(
                [
                    (payload["triples"][index].tolist(), str(orientation))
                    for index, orientation in zip(payload["indices"], payload["orientations"])
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_metadata": transfer.candidate_metadata(pa),
        "source_context_config": metadata["source_context_config"],
        "context_build_seconds": metadata["context_build_seconds"],
        "checkpoint_sha256": file_sha256(pilot.CHECKPOINT),
        "package_versions": package_versions(),
        "git_commit": pilot.git_commit(),
        "hardware": pilot.hardware_info(),
    }
    pilot.atomic_json(path / "manifest.json", manifest)


def build_artifact(pa, dataset: str, seed: int) -> None:
    path = artifact_dir(dataset, seed)
    if artifact_complete(path):
        return
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        contexts, metadata = transfer.prepare_contexts(pa, dataset, seed)
        save_artifact(path, contexts["correct_labels"], metadata, pa, dataset, seed)
    finally:
        torch.use_deterministic_algorithms(previous)


def load_artifact(dataset: str, seed: int) -> tuple[dict, dict]:
    path = artifact_dir(dataset, seed)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    artifact = path / "context_input.npz"
    if manifest["protocol_revision"] != PROTOCOL_REVISION:
        raise RuntimeError(f"wrong context protocol in {path}")
    if file_sha256(artifact) != manifest["context_artifact_sha256"]:
        raise RuntimeError(f"context artifact hash mismatch in {path}")
    with np.load(artifact, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    if pilot.array_hash(payload["X"]) != manifest["context_feature_hash"]:
        raise RuntimeError(f"context feature hash mismatch in {path}")
    triples, labels, source_config = load_context(dataset, seed, pilot.SIZE)
    if not np.array_equal(payload["triples"], triples):
        raise RuntimeError(f"source context triples changed for {dataset} seed {seed}")
    if not np.array_equal(payload["original_y"], labels[payload["indices"]]):
        raise RuntimeError(f"source context labels changed for {dataset} seed {seed}")

    relations = payload["triples"][payload["indices"], 1]
    permuted, permutation = pilot.stratified_label_permutation(
        payload["y"], relations, payload["orientations"], seed
    )
    common = {
        "X": payload["X"],
        "triples": payload["triples"],
        "indices": payload["indices"],
        "orientations": payload["orientations"],
        "original_y": payload["original_y"],
    }
    contexts = {
        "correct_labels": {
            **common,
            "y": payload["y"],
            "permutation": np.arange(len(payload["y"]), dtype=np.int64),
        },
        "permuted_labels": {
            **common,
            "y": permuted,
            "permutation": permutation,
        },
    }
    metadata = {
        "context_build_seconds": 0.0,
        "source_context_config": source_config,
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_family": "context_intervention_deterministic_v1",
        "canonical_context_artifact": str(artifact),
        "canonical_context_sha256": manifest["context_artifact_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "package_versions": manifest["package_versions"],
    }
    return contexts, metadata


def aggregate_base(dataset: str, condition: str, decoder: str) -> dict:
    rows = []
    for seed in SEEDS:
        path = transfer.base_dir(dataset, condition, decoder, seed) / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "mrr": metrics["test"]["combined"]["mrr"],
            "head_mrr": metrics["test"]["head"]["mrr"],
            "tail_mrr": metrics["test"]["tail"]["mrr"],
        })
    values = np.asarray([row["mrr"] for row in rows])
    return {
        "per_seed": rows,
        "mean_mrr": float(values.mean()),
        "sample_standard_deviation_mrr": float(values.std(ddof=1)),
    }


def write_summary() -> dict:
    output = {
        "status": "COMPLETE",
        "protocol_revision": PROTOCOL_REVISION,
        "seeds": list(SEEDS),
        "datasets": {},
        "scientific_use": (
            "Matched deterministic transfer evidence; does not reproduce or replace "
            "historical matrices that were not saved."
        ),
    }
    lines = ["# Deterministic Context Transfer", ""]
    for dataset, conditions in DATASET_CONDITIONS.items():
        output["datasets"][dataset] = {}
        for condition in conditions:
            methods = {
                "tabpfn_direct": aggregate_base(dataset, condition, "tabpfn_v3"),
                "hgb_direct": aggregate_base(dataset, condition, "hgb"),
            }
            for method in ("ultra_only", "fixed_blend", "tabpfn_residual", "hgb_residual"):
                methods[method] = transfer.aggregate(dataset, condition, method)
            output["datasets"][dataset][condition] = methods
            lines.append(
                f"- {dataset} / {condition}: ULTRA {methods['ultra_only']['mean_mrr']:.6f}; "
                f"TabPFN direct {methods['tabpfn_direct']['mean_mrr']:.6f}; "
                f"blend {methods['fixed_blend']['mean_mrr']:.6f}; "
                f"TabPFN residual {methods['tabpfn_residual']['mean_mrr']:.6f}; "
                f"HGB residual {methods['hgb_residual']['mean_mrr']:.6f}."
            )
    pilot.atomic_json(SUMMARY_JSON, output)
    tmp = SUMMARY_MD.with_name(f".{SUMMARY_MD.name}.{os.getpid()}.tmp")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, SUMMARY_MD)
    return output


def run_smokes(assets: dict[str, object]) -> None:
    seed = SEEDS[0]
    for dataset, conditions in DATASET_CONDITIONS.items():
        pa = assets[dataset]
        build_artifact(pa, dataset, seed)
        contexts, metadata = load_artifact(dataset, seed)
        for condition in conditions:
            for decoder in ("tabpfn_v3", "hgb"):
                transfer.execute_base(
                    pa,
                    dataset,
                    seed,
                    condition,
                    decoder,
                    contexts[condition],
                    metadata,
                    smoke=True,
                    enforce_historical_reference=False,
                )
                print(
                    f"SMOKE OK {dataset} seed={seed} {condition} {decoder}",
                    flush=True,
                )


def main() -> int:
    started = time.time()
    print(
        "PLAN deterministic_transfer: smokes=6 context_artifacts=10 "
        "tabpfn=15 hgb=15 derived=60 estimated_hours=[1.5,2.0]",
        flush=True,
    )
    assets = {dataset: pilot.PrimaryAssets(dataset) for dataset in DATASET_CONDITIONS}
    run_smokes(assets)
    for dataset, conditions in DATASET_CONDITIONS.items():
        pa = assets[dataset]
        for seed in SEEDS:
            build_artifact(pa, dataset, seed)
            contexts, metadata = load_artifact(dataset, seed)
            for condition in conditions:
                for decoder in ("tabpfn_v3", "hgb"):
                    metrics = transfer.execute_base(
                        pa,
                        dataset,
                        seed,
                        condition,
                        decoder,
                        contexts[condition],
                        metadata,
                        enforce_historical_reference=False,
                    )
                    print(
                        f"OK deterministic base {dataset} seed={seed} {condition} "
                        f"{decoder} mrr={metrics['test']['combined']['mrr']:.6f}",
                        flush=True,
                    )
        for seed in SEEDS:
            for condition in conditions:
                for method, decoder in (
                    ("ultra_only", None),
                    ("fixed_blend", "tabpfn_v3"),
                    ("tabpfn_residual", "tabpfn_v3"),
                    ("hgb_residual", "hgb"),
                ):
                    transfer.save_derived(dataset, condition, method, decoder, seed)
    summary = write_summary()
    summary["total_seconds"] = time.time() - started
    pilot.atomic_json(SUMMARY_JSON, summary)
    print(f"DETERMINISTIC TRANSFER COMPLETE seconds={summary['total_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
