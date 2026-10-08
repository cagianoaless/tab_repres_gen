#!/usr/bin/env python3
"""Metric-free k/init stability audit for large development components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.model_selection import KFold

import features as hybrid_features
from protocol import SCREEN_DATASETS


SCAN_VERSION = "g2t_q512_hybrid_solver_development_large_stability_v1"
LARGE_COMPONENT_MINIMUM = 513
ZERO_TOLERANCE = 1e-8
ORTHOGONALITY_TOLERANCE = 1e-8
EIGENVALUE_ATOL = 1e-8
EIGENVALUE_RTOL = 1e-6
FEATURE_SCALED_TOLERANCE = 2e-5


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_sequence(train: np.ndarray, folds: int):
    yield "full_train", train
    for seed in (101, 102, 103):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, heldout) in enumerate(splitter.split(train)):
            keep = np.ones(len(train), dtype=bool)
            keep[heldout] = False
            yield f"seed{seed}_fold{fold}", train[keep]


def residuals(matrix, values, vectors):
    errors = matrix @ vectors - vectors * values[np.newaxis, :]
    return np.linalg.norm(errors, axis=0)


def solve(matrix, k: int, seed_words, method: str):
    started = time.time()
    v0 = np.random.default_rng(
        np.random.SeedSequence([20260718, *seed_words, k, 7717])
    ).standard_normal(matrix.shape[0])
    options = {"which": "SM"}
    if method == "SA":
        options = {"which": "SA"}
    elif method == "SHIFT":
        options = {"which": "LM", "sigma": -1e-6}
    try:
        values, vectors = eigsh(
            matrix,
            k=k,
            v0=v0,
            tol=1e-10,
            maxiter=max(10_000, 20 * matrix.shape[0]),
            **options,
        )
        order = np.argsort(values, kind="stable")
        values = np.asarray(values)[order]
        vectors = np.asarray(vectors)[:, order]
        errors = residuals(matrix, values, vectors)
        orthogonality = float(
            np.max(np.abs(vectors.T @ vectors - np.eye(k)), initial=0.0)
        )
        return {
            "status": "PASS",
            "method": method,
            "k": k,
            "values": values,
            "vectors": vectors,
            "maximum_residual": float(errors.max(initial=0.0)),
            "orthogonality_error": orthogonality,
            "elapsed_seconds": time.time() - started,
        }
    except ArpackNoConvergence as error:
        return {
            "status": "NO_CONVERGENCE",
            "method": method,
            "k": k,
            "converged_eigenpairs": int(len(error.eigenvalues)),
            "elapsed_seconds": time.time() - started,
        }
    except Exception as error:
        return {
            "status": "ERROR",
            "method": method,
            "k": k,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }


def serializable_solver(record: dict):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in record.items()
        if key != "vectors"
    }


def retained(features, record: dict):
    values = record["values"][1:]
    vectors = record["vectors"][:, 1:]
    count, boundary = features.FROZEN.eigengap_safe_count(
        values, features.LAPPE_DIMENSIONS
    )
    return values[:count], vectors[:, :count], count, boundary


def sampled_features(features, values, vectors, size: int, seed_words):
    rng = np.random.default_rng(
        np.random.SeedSequence([20260718, *seed_words, 8803])
    )
    count = 256
    left = rng.integers(0, size, size=count)
    right = rng.integers(0, size, size=count)
    left[:3] = [0, 0, size - 1]
    right[:3] = [0, size - 1, size - 1]
    left_vectors = vectors[left]
    right_vectors = vectors[right]
    columns = []
    for scale in features.HEAT_SCALES:
        weights = np.exp(-scale * values)
        columns.append(size * np.sum(weights * left_vectors * right_vectors, axis=1))
        columns.append(
            size
            * np.sum(
                np.square(weights) * np.square(left_vectors - right_vectors), axis=1
            )
        )
    return np.column_stack(columns)


def compare_runs(features, first: dict, second: dict, size: int, seed_words):
    if first["status"] != "PASS" or second["status"] != "PASS":
        return {"status": "SOLVER_FAILURE"}
    first_values, first_vectors, first_count, first_boundary = retained(features, first)
    second_values, second_vectors, second_count, second_boundary = retained(features, second)
    first_ten = first["values"][:10]
    second_ten = second["values"][:10]
    values_stable = bool(
        np.allclose(
            first_ten,
            second_ten,
            atol=EIGENVALUE_ATOL,
            rtol=EIGENVALUE_RTOL,
        )
    )
    count_stable = first_count == second_count
    payload = {
        "values_stable": values_stable,
        "maximum_first_ten_absolute_difference": float(
            np.max(np.abs(first_ten - second_ten), initial=0.0)
        ),
        "retained_count_stable": count_stable,
        "k10_retained_modes": int(first_count),
        "k16_retained_modes": int(second_count),
        "k10_boundary": first_boundary,
        "k16_boundary": second_boundary,
    }
    if count_stable:
        first_features = sampled_features(
            features, first_values, first_vectors, size, seed_words
        )
        second_features = sampled_features(
            features, second_values, second_vectors, size, seed_words
        )
        difference = np.abs(first_features - second_features)
        scaled = difference / np.maximum(1.0, np.abs(second_features))
        payload.update({
            "maximum_sampled_feature_absolute_difference": float(
                difference.max(initial=0.0)
            ),
            "maximum_sampled_feature_scaled_difference": float(
                scaled.max(initial=0.0)
            ),
            "feature_stable": bool(
                scaled.max(initial=0.0) <= FEATURE_SCALED_TOLERANCE
            ),
        })
    else:
        payload["feature_stable"] = False
    payload["status"] = (
        "PASS"
        if values_stable and count_stable and payload["feature_stable"]
        else "FAIL"
    )
    return payload


def scan_component(features, component: int, nodes, matrix):
    size = int(len(nodes))
    seed_words = [component, int(nodes[0]), int(nodes[-1]), size, matrix.nnz]
    k10 = solve(matrix, 10, [*seed_words, 10], "SM")
    k16 = solve(matrix, 16, [*seed_words, 16], "SM")
    comparison = compare_runs(features, k10, k16, size, seed_words)
    basic_checks = {
        "k10_zero_present": bool(
            k10["status"] == "PASS" and abs(k10["values"][0]) <= ZERO_TOLERANCE
        ),
        "k16_zero_present": bool(
            k16["status"] == "PASS" and abs(k16["values"][0]) <= ZERO_TOLERANCE
        ),
        "k10_residual_valid": bool(
            k10["status"] == "PASS"
            and k10["maximum_residual"] <= features.LOBPCG_RESIDUAL_LIMIT
        ),
        "k16_residual_valid": bool(
            k16["status"] == "PASS"
            and k16["maximum_residual"] <= features.LOBPCG_RESIDUAL_LIMIT
        ),
        "k10_orthogonality_valid": bool(
            k10["status"] == "PASS"
            and k10["orthogonality_error"] <= ORTHOGONALITY_TOLERANCE
        ),
        "k16_orthogonality_valid": bool(
            k16["status"] == "PASS"
            and k16["orthogonality_error"] <= ORTHOGONALITY_TOLERANCE
        ),
    }
    stable = all(basic_checks.values()) and comparison["status"] == "PASS"
    cross_checks = []
    if not stable:
        cross_checks.append(solve(matrix, 16, [*seed_words, 16, 1], "SA"))
        cross_checks.append(solve(matrix, 24, [*seed_words, 24, 2], "SM"))
        if size <= 20_000:
            cross_checks.append(solve(matrix, 16, [*seed_words, 16, 3], "SHIFT"))
    return {
        "status": "PASS" if stable else "UNSTABLE",
        "component": int(component),
        "component_size": size,
        "laplacian_nnz": int(matrix.nnz),
        "basic_checks": basic_checks,
        "comparison": comparison,
        "k10": serializable_solver(k10),
        "k16": serializable_solver(k16),
        "cross_checks": [serializable_solver(record) for record in cross_checks],
    }


def scan_dataset(dataset: str, output_path: Path, provenance: dict):
    previous = None
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and previous.get("provenance") == provenance:
            return previous
        if previous.get("provenance") != provenance:
            previous = None
    records = [] if previous is None else previous.get("records", [])
    completed = {record["graph_key"] for record in records}
    data = hybrid_features.parent_dataset(dataset)
    train = np.asarray(data["train"], dtype=np.int64)
    for graph_key, graph in graph_sequence(train, hybrid_features.NUM_FOLDS):
        if graph_key in completed:
            continue
        directed = hybrid_features.FROZEN._binary_csr(
            graph[:, 0], graph[:, 2], int(data["num_entities"])
        )
        adjacency = directed.maximum(directed.T).astype(np.float64)
        adjacency.setdiag(0.0)
        adjacency.eliminate_zeros()
        laplacian = csgraph.laplacian(adjacency, normed=True).tocsr()
        component_count, labels = csgraph.connected_components(
            adjacency, directed=False, return_labels=True
        )
        component_sizes = np.bincount(labels, minlength=component_count)
        targets = np.flatnonzero(component_sizes >= LARGE_COMPONENT_MINIMUM)
        print(
            f"STABILITY dataset={dataset} graph={graph_key} "
            f"components={len(targets)}",
            flush=True,
        )
        component_records = []
        try:
            for target in targets:
                component = int(target)
                nodes = np.flatnonzero(labels == component)
                component_records.append(
                    scan_component(
                        hybrid_features,
                        component,
                        nodes,
                        laplacian[nodes][:, nodes].tocsr(),
                    )
                )
            graph_status = (
                "PASS"
                if all(record["status"] == "PASS" for record in component_records)
                else "UNSTABLE"
            )
            graph_record = {
                "status": graph_status,
                "graph_key": graph_key,
                "components": component_records,
            }
        except Exception as error:
            graph_record = {
                "status": "ERROR",
                "graph_key": graph_key,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "components": component_records,
            }
        records.append(graph_record)
        atomic_json(output_path, {
            "status": "PARTIAL",
            "dataset": dataset,
            "provenance": provenance,
            "graphs_completed": len(records),
            "records": records,
        })
    components = [component for graph in records for component in graph["components"]]
    unstable_components = sum(c["status"] != "PASS" for c in components)
    graph_errors = sum(g["status"] == "ERROR" for g in records)
    payload = {
        "status": "PASS" if not unstable_components and not graph_errors else "FAIL",
        "dataset": dataset,
        "provenance": provenance,
        "graphs_completed": len(records),
        "components_completed": len(components),
        "unstable_components": unstable_components,
        "graph_errors": graph_errors,
        "records": records,
    }
    atomic_json(output_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard")

    panel = SCREEN_DATASETS
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(panel))
        if unknown:
            raise ValueError(f"unknown datasets: {unknown}")
        panel = args.datasets
    assigned = [
        dataset
        for index, dataset in enumerate(panel)
        if index % args.num_shards == args.shard_id
    ]
    provenance = {
        "scan_version": SCAN_VERSION,
        "scanner_sha256": file_sha256(Path(__file__).resolve()),
        "hybrid_features_sha256": file_sha256(Path(hybrid_features.__file__)),
        "b2_implementation_sha256": hybrid_features.b2_implementation_sha256(),
        "large_component_minimum": LARGE_COMPONENT_MINIMUM,
        "zero_tolerance": ZERO_TOLERANCE,
        "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
        "eigenvalue_atol": EIGENVALUE_ATOL,
        "eigenvalue_rtol": EIGENVALUE_RTOL,
        "feature_scaled_tolerance": FEATURE_SCALED_TOLERANCE,
        "labels_or_metrics_accessed": False,
    }
    records = []
    heartbeat = args.output_root / "shards" / f"shard{args.shard_id}.heartbeat"
    for index, dataset in enumerate(assigned, 1):
        atomic_json(heartbeat, {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "dataset": dataset,
            "dataset_index": index,
            "datasets_assigned": len(assigned),
            "epoch": time.time(),
        })
        payload = scan_dataset(
            dataset,
            args.output_root / "datasets" / f"{dataset}.json",
            provenance,
        )
        records.append({
            "dataset": dataset,
            "graphs_completed": payload["graphs_completed"],
            "components_completed": payload["components_completed"],
            "unstable_components": payload["unstable_components"],
            "graph_errors": payload["graph_errors"],
        })
    unstable_components = sum(r["unstable_components"] for r in records)
    graph_errors = sum(r["graph_errors"] for r in records)
    summary = {
        "status": (
            "PASS"
            if all(r["dataset"] for r in records)
            and not unstable_components
            and not graph_errors
            else "FAIL"
        ),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "hostname": socket.gethostname(),
        "provenance": provenance,
        "datasets_completed": len(records),
        "graphs_completed": sum(r["graphs_completed"] for r in records),
        "components_completed": sum(r["components_completed"] for r in records),
        "unstable_components": unstable_components,
        "graph_errors": graph_errors,
        "records": records,
        "labels_or_metrics_accessed": False,
    }
    atomic_json(
        args.output_root / "shards" / f"shard{args.shard_id}of{args.num_shards}.json",
        summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
