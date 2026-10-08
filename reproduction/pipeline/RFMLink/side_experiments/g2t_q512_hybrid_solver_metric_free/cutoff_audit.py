#!/usr/bin/env python3
"""Metric-free numerical audit of the frozen 512-node hybrid cutoff."""

from __future__ import annotations

import json
import resource
import time

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

import features


def cycle_laplacian(size: int):
    nodes = np.arange(size, dtype=np.int64)
    neighbors = (nodes + 1) % size
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * size, dtype=np.float64),
            (
                np.concatenate([nodes, neighbors]),
                np.concatenate([neighbors, nodes]),
            ),
        ),
        shape=(size, size),
    ).tocsr()
    return csgraph.laplacian(adjacency, normed=True).tocsr()


def sampled_features(values, vectors, size: int):
    rng = np.random.default_rng(np.random.SeedSequence([20260718, size, 511513]))
    left = rng.integers(0, size, size=256)
    right = rng.integers(0, size, size=256)
    columns = []
    for scale in features.HEAT_SCALES:
        weights = np.exp(-scale * values)
        columns.append(size * np.sum(weights * vectors[left] * vectors[right], axis=1))
        columns.append(
            size
            * np.sum(
                np.square(weights) * np.square(vectors[left] - vectors[right]),
                axis=1,
            )
        )
    return np.column_stack(columns)


def audit_size(size: int) -> dict:
    laplacian = cycle_laplacian(size)
    seed_words = [0, 0, size - 1]
    started = time.perf_counter()
    policy_values, policy_vectors, policy_solver = (
        features._component_positive_eigenpairs(laplacian, 8, seed_words)
    )
    policy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    dense_values, dense_vectors = np.linalg.eigh(laplacian.toarray())
    dense_seconds = time.perf_counter() - started
    order = np.argsort(dense_values, kind="stable")
    dense_positive = np.asarray(dense_values)[order][1:]
    dense_positive_vectors = np.asarray(dense_vectors)[:, order][:, 1:]
    dense_count, dense_boundary = features.FROZEN.eigengap_safe_count(dense_positive, 8)
    dense_retained_values = dense_positive[:dense_count]
    dense_retained_vectors = dense_positive_vectors[:, :dense_count]

    counts_equal = len(policy_values) == dense_count
    eigenvalues_equal = bool(
        counts_equal
        and np.allclose(
            policy_values,
            dense_retained_values,
            rtol=1e-6,
            atol=1e-8,
        )
    )
    projector_error = None
    feature_error = None
    if counts_equal:
        projector_error = float(
            np.max(
                np.abs(
                    policy_vectors @ policy_vectors.T
                    - dense_retained_vectors @ dense_retained_vectors.T
                ),
                initial=0.0,
            )
        )
        policy_features = sampled_features(policy_values, policy_vectors, size)
        dense_features = sampled_features(
            dense_retained_values, dense_retained_vectors, size
        )
        feature_error = float(
            np.max(
                np.abs(policy_features - dense_features)
                / np.maximum(1.0, np.abs(dense_features)),
                initial=0.0,
            )
        )
    expected_method = "dense_eigh" if size <= features.DENSE_COMPONENT_LIMIT else "eigsh_sm"
    passed = bool(
        policy_solver["method"] == expected_method
        and counts_equal
        and eigenvalues_equal
        and projector_error is not None
        and projector_error <= 2e-5
        and feature_error is not None
        and feature_error <= 2e-5
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "size": size,
        "expected_method": expected_method,
        "observed_method": policy_solver["method"],
        "retained_modes": len(policy_values),
        "dense_retained_modes": dense_count,
        "counts_equal": counts_equal,
        "eigenvalues_equal": eigenvalues_equal,
        "maximum_projector_absolute_error": projector_error,
        "maximum_feature_scaled_error": feature_error,
        "policy_seconds": policy_seconds,
        "dense_seconds": dense_seconds,
        "dense_matrix_bytes": size * size * np.dtype(np.float64).itemsize,
        "dense_eigenvector_bytes": size * size * np.dtype(np.float64).itemsize,
        "policy_solver": policy_solver,
        "dense_boundary": dense_boundary,
    }


def main() -> None:
    records = [audit_size(size) for size in (511, 512, 513)]
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "cutoff": features.DENSE_COMPONENT_LIMIT,
        "cutoff_rationale": {
            "selection_uses_performance_metrics": False,
            "dense_complexity": {"memory": "O(n^2)", "time": "O(n^3)"},
            "engineering_reason": (
                "power-of-two conservative exact-solver ceiling above every suspicious "
                "component found in the metric-free census (maximum 365 nodes)"
            ),
        },
        "records": records,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "labels_or_performance_metrics_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
