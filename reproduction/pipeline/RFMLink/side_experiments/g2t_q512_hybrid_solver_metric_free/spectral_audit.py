#!/usr/bin/env python3
"""Metric-free audit of hybrid-solver B2v2 graph embeddings and sidecars."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold

import features
from protocol import (
    AUDIT_ROOT,
    FEATURE_ROOT,
    PANEL,
    PROTOCOL,
    ROOT,
    STABILITY_AUDIT_ROOT,
)


DATASET_AUDIT_ROOT = (
    FEATURE_ROOT.parent / "spectral_audit" / "datasets"
)
AUDIT_PATH = AUDIT_ROOT / "spectral_audit.json"
MATERIALIZED_AUDIT_PATH = AUDIT_ROOT / "materialized_audit.json"
INVARIANCE_ATOL = 2e-5


def _probe_rows(train: np.ndarray, embedding: dict) -> np.ndarray:
    rows = train if len(train) <= 4096 else train[
        np.linspace(0, len(train) - 1, 4096, dtype=np.int64)
    ]
    component_rows = []
    labels = embedding["component_labels"]
    for component in np.unique(labels):
        nodes = np.flatnonzero(labels == component)
        if len(nodes):
            component_rows.append((int(nodes[0]), 0, int(nodes[-1])))
    if component_rows:
        rows = np.concatenate([rows, np.asarray(component_rows, dtype=np.int64)])
    return rows


def _transformed_embedding(embedding: dict, mode: str) -> dict:
    output = {name: np.array(value, copy=True) for name, value in embedding.items()}
    vectors = output["spectral_vectors"]
    values = output["spectral_values"]
    valid = output["spectral_valid"]
    if mode == "sign":
        vectors *= np.where(np.arange(vectors.shape[1]) % 2, 1.0, -1.0)
    elif mode == "permutation":
        order = np.arange(vectors.shape[1] - 1, -1, -1)
        output["spectral_vectors"] = vectors[:, order]
        output["spectral_values"] = values[:, order]
        output["spectral_valid"] = valid[:, order]
    elif mode == "retained_eigenspace_rotation":
        labels = output["component_labels"]
        for component in np.unique(labels):
            nodes = np.flatnonzero(labels == component)
            indices = np.flatnonzero(valid[int(nodes[0])]) if len(nodes) else np.zeros(0)
            start = 0
            while start < len(indices):
                stop = start + 1
                while stop < len(indices) and np.isclose(
                    values[int(nodes[0]), indices[start]],
                    values[int(nodes[0]), indices[stop]],
                    rtol=features.DEGENERACY_RTOL,
                    atol=features.DEGENERACY_ATOL,
                ):
                    stop += 1
                group = indices[start:stop]
                if len(group) > 1:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([20260718, int(component), len(group)])
                    )
                    rotation, _ = np.linalg.qr(rng.standard_normal((len(group), len(group))))
                    vectors[np.ix_(nodes, group)] = vectors[np.ix_(nodes, group)] @ rotation
                start = stop
    else:
        raise ValueError(mode)
    return output


def _basis_invariance(train: np.ndarray, embedding: dict) -> dict:
    probes = _probe_rows(train, embedding)
    baseline = features.b2_rows(probes, embedding)
    if not np.isfinite(baseline).all():
        raise RuntimeError("non-finite B2v2 probe features")
    errors = {}
    for mode in ("sign", "permutation", "retained_eigenspace_rotation"):
        changed = features.b2_rows(probes, _transformed_embedding(embedding, mode))
        error = float(np.max(np.abs(changed - baseline), initial=0.0))
        errors[mode] = error
        if error > INVARIANCE_ATOL:
            raise RuntimeError(f"basis invariance failed for {mode}: {error:.3e}")
    return {
        "probe_rows": int(len(probes)),
        "finite": True,
        "maximum_absolute_errors": errors,
        "absolute_tolerance": INVARIANCE_ATOL,
    }


def _boundary_checks(spectral: dict) -> dict:
    records = spectral["backed_off_components"]
    for record in records:
        if not record["backed_off"]:
            raise RuntimeError("backoff record is internally inconsistent")
        if record["retained_positive_modes"] != record["boundary_group_start"]:
            raise RuntimeError("backoff did not stop before the repeated boundary group")
        if record["retained_positive_modes"] >= features.LAPPE_DIMENSIONS:
            raise RuntimeError("backoff retained the unsafe truncation boundary")
        if not np.isclose(
            record["lambda_left"],
            record["lambda_lookahead"],
            rtol=features.DEGENERACY_RTOL,
            atol=features.DEGENERACY_ATOL,
        ):
            raise RuntimeError("backoff was triggered without a repeated boundary")
    histogram = {int(key): int(value) for key, value in spectral["retained_mode_histogram"].items()}
    if any(count < 0 or count > features.LAPPE_DIMENSIONS for count in histogram):
        raise RuntimeError("retained-mode histogram is outside the frozen budget")
    return {
        "safe": True,
        "backed_off_component_count": len(records),
        "retained_mode_histogram": histogram,
        "backed_off_components": records,
    }


def _graphs(train: np.ndarray):
    yield "full_train", train, {"scope": "full_train", "seed": None, "fold": None}
    for seed in (101, 102, 103):
        splitter = KFold(n_splits=features.NUM_FOLDS, shuffle=True, random_state=seed)
        for fold, (_, heldout) in enumerate(splitter.split(train)):
            keep = np.ones(len(train), dtype=bool)
            keep[heldout] = False
            yield (
                f"seed{seed}_fold{fold}",
                train[keep],
                {"scope": "cross_fit", "seed": seed, "fold": fold},
            )


def audit_dataset(dataset: str) -> dict:
    output = DATASET_AUDIT_ROOT / f"{dataset}.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "PASS"
            and payload.get("graphs_completed") == 31
            and payload.get("b2_implementation_sha256")
            == features.b2_implementation_sha256()
        ):
            return payload
    data = features.parent_dataset(dataset)
    train = np.asarray(data["train"], dtype=np.int64)
    num_entities = int(data["num_entities"])
    started = time.time()
    records = []
    for graph_index, (graph_key, graph, scope) in enumerate(_graphs(train), 1):
        print(
            f"AUDIT dataset={dataset} graph={graph_index}/31 key={graph_key}",
            flush=True,
        )
        embedding, manifest = features._embedding_cache(
            dataset, graph_key, graph, num_entities
        )
        finite = {
            name: bool(np.isfinite(value).all())
            for name, value in embedding.items()
            if np.issubdtype(value.dtype, np.number)
        }
        if not all(finite.values()):
            raise RuntimeError(f"non-finite embedding: {dataset}/{graph_key}")
        spectral = manifest["spectral"]
        residual = float(spectral["maximum_solver_residual"])
        orthogonality = float(spectral["maximum_solver_orthogonality_error"])
        zero_error = float(spectral["maximum_zero_eigenvalue_error"])
        if residual > features.SOLVER_RESIDUAL_LIMIT:
            raise RuntimeError(f"residual limit exceeded: {dataset}/{graph_key}")
        if orthogonality > features.ORTHOGONALITY_TOLERANCE:
            raise RuntimeError(f"orthogonality limit exceeded: {dataset}/{graph_key}")
        if zero_error > features.ZERO_EIGENVALUE_TOLERANCE:
            raise RuntimeError(f"zero-mode limit exceeded: {dataset}/{graph_key}")
        if spectral.get("solver_policy") != features.HYBRID_SOLVER_POLICY:
            raise RuntimeError(f"solver policy mismatch: {dataset}/{graph_key}")
        if not set(spectral["solver_counts"]).issubset({"dense_eigh", "eigsh_sm"}):
            raise RuntimeError(f"unexpected solver: {dataset}/{graph_key}")
        records.append({
            "graph_key": graph_key,
            **scope,
            "graph_triples": int(len(graph)),
            "graph_hash": features.array_hash(graph),
            "component_count": manifest["spectral"]["component_count"],
            "solver_counts": spectral["solver_counts"],
            "maximum_solver_residual": residual,
            "maximum_solver_orthogonality_error": orthogonality,
            "maximum_zero_eigenvalue_error": zero_error,
            "finite_embedding_arrays": finite,
            "truncation": _boundary_checks(manifest["spectral"]),
            "basis_invariance": _basis_invariance(graph, embedding),
            "embedding_sha256": manifest["sha256"],
        })
    payload = {
        "status": "PASS",
        "dataset": dataset,
        "audit_scope": "full observed train graph plus all 30 saved cross-fit graphs",
        "validation_or_test_labels_accessed": False,
        "performance_metrics_accessed": False,
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "num_entities": num_entities,
        "num_relations": int(data["num_relations"]),
        "num_train_triples": int(len(train)),
        "train_hash": features.array_hash(train),
        "graphs_planned": 31,
        "graphs_completed": len(records),
        "records": records,
        "maximum_solver_residual": max(row["maximum_solver_residual"] for row in records),
        "maximum_solver_orthogonality_error": max(
            row["maximum_solver_orthogonality_error"] for row in records
        ),
        "maximum_zero_eigenvalue_error": max(
            row["maximum_zero_eigenvalue_error"] for row in records
        ),
        "backed_off_components": sum(
            row["truncation"]["backed_off_component_count"] for row in records
        ),
        "elapsed_seconds": time.time() - started,
    }
    features.atomic_json(output, payload)
    return payload


def _validate_preconditions() -> list[dict]:
    records = []
    for shard_id in range(2):
        path = STABILITY_AUDIT_ROOT / "shards" / f"shard{shard_id}of2.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"large-component stability shard is incomplete: {path}")
        if payload.get("unstable_components") != 0 or payload.get("graph_errors") != 0:
            raise RuntimeError(f"large-component stability shard failed: {path}")
        if payload.get("labels_or_metrics_accessed") is not False:
            raise RuntimeError(f"large-component audit provenance is invalid: {path}")
        records.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": features.file_sha256(path),
            "datasets_completed": payload["datasets_completed"],
            "components_completed": payload["components_completed"],
        })
    if sum(row["datasets_completed"] for row in records) != 51:
        raise RuntimeError("large-component stability audit does not cover 51 datasets")
    if sum(row["components_completed"] for row in records) != 1657:
        raise RuntimeError("large-component stability audit does not cover 1657 components")
    if list((FEATURE_ROOT.parent / "runs").rglob("metrics.json")):
        raise RuntimeError("performance metrics exist inside the metric-free family")
    return records


def _aggregate_dataset_audits() -> dict:
    datasets = []
    failures = []
    for dataset in PANEL:
        path = DATASET_AUDIT_ROOT / f"{dataset}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "PASS" or record.get("graphs_completed") != 31:
                raise RuntimeError("dataset audit is incomplete")
            datasets.append(record)
        except Exception as error:
            failures.append({"dataset": dataset, "error": repr(error)})
    payload = {
        "status": "PASS" if not failures and len(datasets) == len(PANEL) else "FAIL",
        "protocol": PROTOCOL,
        "large_component_stability_audit": _validate_preconditions(),
        "hybrid_solver_policy": features.HYBRID_SOLVER_POLICY,
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "scope": "full observed train plus all cross-fit graphs; no labels or metrics",
        "validation_or_test_labels_accessed": False,
        "performance_metrics_accessed": False,
        "datasets_planned": len(PANEL),
        "datasets_completed": len(datasets),
        "graphs_planned": len(PANEL) * 31,
        "graphs_completed": sum(row["graphs_completed"] for row in datasets),
        "failures": failures,
        "records": [{
            "dataset": row["dataset"],
            "graphs_completed": row["graphs_completed"],
            "maximum_solver_residual": row["maximum_solver_residual"],
            "maximum_solver_orthogonality_error": row[
                "maximum_solver_orthogonality_error"
            ],
            "maximum_zero_eigenvalue_error": row["maximum_zero_eigenvalue_error"],
            "backed_off_components": row["backed_off_components"],
            "dataset_audit": str((DATASET_AUDIT_ROOT / f"{row['dataset']}.json").relative_to(ROOT)),
        } for row in datasets],
        "source_hashes": {
            "features": features.file_sha256(Path(features.__file__)),
            "audit": features.file_sha256(Path(__file__)),
        },
    }
    features.atomic_json(AUDIT_PATH, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"eigengap-safe spectral audit failed: {failures}")
    return payload


def run_audit_shard(shard_id: int, num_shards: int) -> dict:
    _validate_preconditions()
    if not 0 <= shard_id < num_shards:
        raise ValueError("invalid audit shard")
    records = []
    failures = []
    assigned = [
        dataset for index, dataset in enumerate(PANEL)
        if index % num_shards == shard_id
    ]
    for index, dataset in enumerate(assigned, 1):
        print(
            f"AUDIT_SHARD shard={shard_id}/{num_shards} dataset={dataset} "
            f"{index}/{len(assigned)}",
            flush=True,
        )
        try:
            records.append(audit_dataset(dataset))
        except Exception as error:
            failures.append({"dataset": dataset, "error": repr(error)})
            break
    payload = {
        "status": "PASS" if not failures and len(records) == len(assigned) else "FAIL",
        "protocol": PROTOCOL,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "assigned_datasets": assigned,
        "datasets_completed": len(records),
        "graphs_completed": sum(row["graphs_completed"] for row in records),
        "failures": failures,
        "validation_or_test_labels_accessed": False,
        "performance_metrics_accessed": False,
    }
    path = FEATURE_ROOT.parent / "spectral_audit" / "shards" / f"shard{shard_id}of{num_shards}.json"
    features.atomic_json(path, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"spectral audit shard failed: {failures}")
    return payload


def finalize_sharded_audit() -> dict:
    _validate_preconditions()
    return _aggregate_dataset_audits()


def run_full_audit() -> dict:
    _validate_preconditions()
    for index, dataset in enumerate(PANEL, 1):
        print(f"AUDIT_DATASET dataset={dataset} {index}/{len(PANEL)}", flush=True)
        audit_dataset(dataset)
    return _aggregate_dataset_audits()


def _finite_memmap(path: Path, chunk_rows: int = 1_000_000) -> bool:
    values = np.load(path, mmap_mode="r")
    return all(
        np.isfinite(values[start : start + chunk_rows]).all()
        for start in range(0, len(values), chunk_rows)
    )


def finalize_materialized_audit() -> dict:
    spectral = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if spectral.get("status") != "PASS":
        raise RuntimeError("all-graph eigengap-safe audit has not passed")
    failures = []
    records = []
    for dataset in PANEL:
        manifest_path = FEATURE_ROOT / dataset / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "COMPLETE":
                raise RuntimeError("feature manifest is not complete")
            if manifest.get("b2_implementation_sha256") != features.b2_implementation_sha256():
                raise RuntimeError("B2v2 implementation hash mismatch")
            if list((FEATURE_ROOT / dataset).rglob("b1.npy")):
                raise RuntimeError("unexpected B1 artifact in B2v2 confirmation")
            sidecars = list((FEATURE_ROOT / dataset).rglob("b2v2.npy"))
            expected_sidecars = 3 + sum(
                int((features.PARENT_ASSETS / dataset / split / "features.npy").exists())
                for split in ("valid", "test")
            )
            if len(sidecars) != expected_sidecars:
                raise RuntimeError(
                    f"expected {expected_sidecars} B2v2 sidecars, found {len(sidecars)}"
                )
            if not all(_finite_memmap(path) for path in sidecars):
                raise RuntimeError("non-finite B2v2 sidecar")
            embedding_metadata = list(
                (FEATURE_ROOT / dataset / "graph_embeddings").glob("*.json")
            )
            if len(embedding_metadata) != 31:
                raise RuntimeError(
                    f"expected 31 train/cross-fit embeddings, found {len(embedding_metadata)}"
                )
            residuals = []
            for path in embedding_metadata:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                spectral_metadata = metadata["spectral"]
                residual = spectral_metadata["maximum_solver_residual"]
                residuals.append(residual)
                if residual > features.SOLVER_RESIDUAL_LIMIT:
                    raise RuntimeError(f"residual limit exceeded in {path.name}")
                if (
                    spectral_metadata["maximum_solver_orthogonality_error"]
                    > features.ORTHOGONALITY_TOLERANCE
                ):
                    raise RuntimeError(f"orthogonality limit exceeded in {path.name}")
                if (
                    spectral_metadata["maximum_zero_eigenvalue_error"]
                    > features.ZERO_EIGENVALUE_TOLERANCE
                ):
                    raise RuntimeError(f"zero-mode limit exceeded in {path.name}")
            records.append({
                "dataset": dataset,
                "sidecars": len(sidecars),
                "all_features_finite": True,
                "train_and_crossfit_embeddings": len(embedding_metadata),
                "maximum_solver_residual": max(residuals, default=0.0),
            })
        except Exception as error:
            failures.append({"dataset": dataset, "error": repr(error)})
    payload = {
        "status": "PASS" if len(records) == len(PANEL) and not failures else "FAIL",
        "protocol": PROTOCOL,
        "hybrid_solver_policy": features.HYBRID_SOLVER_POLICY,
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "spectral_audit_sha256": features.file_sha256(AUDIT_PATH),
        "datasets_completed": len(records),
        "failures": failures,
        "records": records,
        "validation_or_test_labels_accessed": False,
        "performance_metrics_accessed": False,
    }
    features.atomic_json(MATERIALIZED_AUDIT_PATH, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"materialized audit failed: {failures[:3]}")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-shard", action="store_true")
    parser.add_argument("--finalize-shards", action="store_true")
    parser.add_argument("--finalize-materialized", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.run:
        print(json.dumps(run_full_audit(), indent=2))
    elif args.run_shard:
        print(json.dumps(run_audit_shard(args.shard_id, args.num_shards), indent=2))
    elif args.finalize_shards:
        print(json.dumps(finalize_sharded_audit(), indent=2))
    elif args.finalize_materialized:
        print(json.dumps(finalize_materialized_audit(), indent=2))
    else:
        parser.print_help()
