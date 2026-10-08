#!/usr/bin/env python3
"""Fast metric-free checks for the hybrid B2v2 eigensolver."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

import features


def cycle_laplacian(size: int):
    rows = np.arange(size, dtype=np.int64)
    columns = (rows + 1) % size
    adjacency = sparse.coo_matrix(
        (np.ones(2 * size), (np.concatenate([rows, columns]), np.concatenate([columns, rows]))),
        shape=(size, size),
    ).tocsr()
    return csgraph.laplacian(adjacency, normed=True).tocsr()


def solve_twice(size: int) -> dict:
    matrix = cycle_laplacian(size)
    seed_words = [0, 0, size - 1]
    first = features._component_positive_eigenpairs(matrix, 8, seed_words)
    second = features._component_positive_eigenpairs(matrix, 8, seed_words)
    values_equal = bool(np.array_equal(first[0], second[0]))
    projectors_equal = bool(
        np.allclose(
            first[1] @ first[1].T,
            second[1] @ second[1].T,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    if not values_equal or not projectors_equal:
        raise RuntimeError(f"non-deterministic solver output at size {size}")
    return {
        "size": size,
        "method": first[2]["method"],
        "retained_modes": int(len(first[0])),
        "maximum_residual": first[2]["maximum_residual"],
        "orthogonality_error": first[2]["orthogonality_error"],
        "zero_eigenvalue_error": first[2]["zero_eigenvalue_error"],
        "eigenvalues_bitwise_equal": values_equal,
        "retained_projectors_equal": projectors_equal,
    }


def strict_loader_check() -> dict:
    dataset = "wn18rr_ind_v1"
    loaded = {
        "dataset": sorted(features.parent_dataset(dataset)),
        "context": sorted(features.parent_context(dataset, 101)),
        "test": sorted(features.parent_split(dataset, "test")),
    }
    expected = {
        "dataset": ["num_entities", "num_relations", "train"],
        "context": ["X", "fold_ids", "triples"],
        "test": ["X", "selected_triples", "side_codes"],
    }
    if loaded != expected:
        raise RuntimeError(f"unexpected metric-free loader surface: {loaded}")
    return loaded


def main() -> None:
    dense = solve_twice(512)
    sparse_result = solve_twice(513)
    if dense["method"] != "dense_eigh" or sparse_result["method"] != "eigsh_sm":
        raise RuntimeError("hybrid threshold dispatch failed")
    payload = {
        "status": "PASS",
        "hybrid_solver_policy": features.HYBRID_SOLVER_POLICY,
        "b2_implementation_sha256": features.b2_implementation_sha256(),
        "threshold_checks": [dense, sparse_result],
        "loader_keys": strict_loader_check(),
        "labels_or_performance_metrics_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
