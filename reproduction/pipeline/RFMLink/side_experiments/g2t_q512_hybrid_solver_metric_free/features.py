#!/usr/bin/env python3
"""Metric-free B2v2 materialization with the audited hybrid eigensolver."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import KFold

from protocol import (
    FAMILY_ROOT,
    FEATURE_ROOT,
    PANEL,
    PARENT_ASSETS,
    ROOT,
    SCREEN_FEATURES_PATH,
    SEEDS,
)


def _load_frozen_module():
    spec = importlib.util.spec_from_file_location(
        "_g2t_q512_frozen_eigengap_safe_features", SCREEN_FEATURES_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(SCREEN_FEATURES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FROZEN = _load_frozen_module()
FROZEN.PANEL = PANEL
FROZEN.FAMILY_ROOT = FAMILY_ROOT
FROZEN.FEATURE_ROOT = FEATURE_ROOT

NUM_FOLDS = FROZEN.NUM_FOLDS
LAPPE_DIMENSIONS = FROZEN.LAPPE_DIMENSIONS
HEAT_SCALES = FROZEN.HEAT_SCALES
LOBPCG_RESIDUAL_LIMIT = FROZEN.LOBPCG_RESIDUAL_LIMIT
SOLVER_RESIDUAL_LIMIT = LOBPCG_RESIDUAL_LIMIT
DEGENERACY_RTOL = FROZEN.DEGENERACY_RTOL
DEGENERACY_ATOL = FROZEN.DEGENERACY_ATOL
BASE_FEATURE_NAMES = FROZEN.BASE_FEATURE_NAMES
B2_FEATURE_NAMES = FROZEN.B2_FEATURE_NAMES
BLOCK_FEATURE_NAMES = {
    "B0": list(BASE_FEATURE_NAMES),
    "B2v2": [*BASE_FEATURE_NAMES, *B2_FEATURE_NAMES],
}

DENSE_COMPONENT_LIMIT = 512
EIGSH_TOLERANCE = 1e-10
EIGSH_MINIMUM_MAXITER = 10_000
EIGSH_MAXITER_PER_NODE = 20
ZERO_EIGENVALUE_TOLERANCE = 1e-8
ORTHOGONALITY_TOLERANCE = 1e-8
HYBRID_SOLVER_POLICY = {
    "dense_component_limit": DENSE_COMPONENT_LIMIT,
    "large_component_solver": "scipy.sparse.linalg.eigsh",
    "which": "SM",
    "requested_eigenpairs": LAPPE_DIMENSIONS + 2,
    "tolerance": EIGSH_TOLERANCE,
    "minimum_maxiter": EIGSH_MINIMUM_MAXITER,
    "maxiter_per_node": EIGSH_MAXITER_PER_NODE,
    "residual_limit": SOLVER_RESIDUAL_LIMIT,
    "zero_eigenvalue_tolerance": ZERO_EIGENVALUE_TOLERANCE,
    "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
}

file_sha256 = FROZEN.file_sha256
array_hash = FROZEN.array_hash
atomic_json = FROZEN.atomic_json
atomic_array = FROZEN.atomic_array
directed_pagerank = FROZEN.directed_pagerank
b2_rows = FROZEN.b2_rows


def parent_dataset(dataset: str) -> dict:
    """Load only the observed train graph and cardinalities."""
    path = PARENT_ASSETS / dataset / "dataset.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            "train": archive["train"],
            "num_entities": archive["num_entities"],
            "num_relations": archive["num_relations"],
        }


def parent_context(dataset: str, seed: int) -> dict:
    """Load context inputs without materializing the label array ``y``."""
    path = PARENT_ASSETS / dataset / "contexts" / str(seed) / "context.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            "X": archive["X"],
            "triples": archive["triples"],
            "fold_ids": archive["fold_ids"],
        }


def parent_split(dataset: str, split: str) -> dict:
    """Load query structure without reading ``target_entities``."""
    root = PARENT_ASSETS / dataset / split
    features = np.load(root / "features.npy", mmap_mode="r")
    with np.load(root / "queries.npz", allow_pickle=False) as archive:
        return {
            "X": features,
            "selected_triples": archive["selected_triples"],
            "side_codes": archive["side_codes"],
        }


def _residuals(matrix, eigenvalues, eigenvectors) -> np.ndarray:
    errors = matrix @ eigenvectors - eigenvectors * eigenvalues[np.newaxis, :]
    return np.linalg.norm(errors, axis=0)


def _component_positive_eigenpairs(laplacian, dimensions: int, seed_words: list[int]):
    size = int(laplacian.shape[0])
    requested = min(dimensions + 2, size)
    if size <= DENSE_COMPONENT_LIMIT:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        method = "dense_eigh"
        iterations = None
    else:
        seed = np.random.SeedSequence(
            [
                20260718,
                *seed_words,
                size,
                int(laplacian.nnz),
                requested,
                requested,
                7717,
            ]
        )
        initial = np.random.default_rng(seed).standard_normal(size)
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=requested,
            which="SM",
            v0=initial,
            tol=EIGSH_TOLERANCE,
            maxiter=max(EIGSH_MINIMUM_MAXITER, EIGSH_MAXITER_PER_NODE * size),
        )
        method = "eigsh_sm"
        iterations = None
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)[order]
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)[:, order]
    residual = float(_residuals(laplacian, eigenvalues, eigenvectors).max(initial=0.0))
    orthogonality = float(
        np.max(
            np.abs(eigenvectors.T @ eigenvectors - np.eye(eigenvectors.shape[1])),
            initial=0.0,
        )
    )
    zero_error = float(abs(eigenvalues[0]))
    if residual > SOLVER_RESIDUAL_LIMIT:
        raise RuntimeError(
            f"component {method} residual {residual:.3e} exceeds "
            f"{SOLVER_RESIDUAL_LIMIT:.3e}"
        )
    if zero_error > ZERO_EIGENVALUE_TOLERANCE:
        raise RuntimeError(
            f"component {method} omitted the zero mode: {zero_error:.3e}"
        )
    if orthogonality > ORTHOGONALITY_TOLERANCE:
        raise RuntimeError(
            f"component {method} orthogonality error {orthogonality:.3e} exceeds "
            f"{ORTHOGONALITY_TOLERANCE:.3e}"
        )
    positive_values = eigenvalues[1:]
    positive_vectors = eigenvectors[:, 1:]
    retained, boundary = FROZEN.eigengap_safe_count(positive_values, dimensions)
    solver = {
        "method": method,
        "component_size": size,
        "requested_eigenpairs": requested,
        "iterations": iterations,
        "maximum_residual": residual,
        "orthogonality_error": orthogonality,
        "zero_eigenvalue_error": zero_error,
        "eigengap_safe_boundary": boundary,
    }
    return positive_values[:retained], positive_vectors[:, :retained], solver


def component_spectral_encoding(triples, num_entities: int,
                                dimensions: int = LAPPE_DIMENSIONS):
    triples = np.asarray(triples, dtype=np.int64)
    directed = FROZEN._binary_csr(triples[:, 0], triples[:, 2], num_entities)
    adjacency = directed.maximum(directed.T).astype(np.float64)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    laplacian = csgraph.laplacian(adjacency, normed=True).tocsr()
    component_count, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_sizes = np.bincount(labels, minlength=component_count).astype(np.int64)
    vectors = np.zeros((num_entities, dimensions), dtype=np.float32)
    values = np.zeros((num_entities, dimensions), dtype=np.float32)
    valid = np.zeros((num_entities, dimensions), dtype=np.uint8)
    solver_counts: dict[str, int] = {}
    maximum_residual = 0.0
    maximum_orthogonality_error = 0.0
    maximum_zero_eigenvalue_error = 0.0
    retained_mode_histogram: dict[str, int] = {}
    backed_off_components = []
    for component in range(component_count):
        nodes = np.flatnonzero(labels == component)
        if len(nodes) <= 1:
            continue
        eigenvalues, eigenvectors, solver = _component_positive_eigenpairs(
            laplacian[nodes][:, nodes].tocsr(),
            dimensions,
            [component, int(nodes[0]), int(nodes[-1])],
        )
        count = len(eigenvalues)
        vectors[nodes, :count] = eigenvectors[:, :count].astype(np.float32)
        values[nodes, :count] = eigenvalues[:count].astype(np.float32)
        valid[nodes, :count] = 1
        solver_counts[solver["method"]] = solver_counts.get(solver["method"], 0) + 1
        maximum_residual = max(maximum_residual, solver["maximum_residual"])
        maximum_orthogonality_error = max(
            maximum_orthogonality_error, solver["orthogonality_error"]
        )
        maximum_zero_eigenvalue_error = max(
            maximum_zero_eigenvalue_error, solver["zero_eigenvalue_error"]
        )
        retained_mode_histogram[str(count)] = retained_mode_histogram.get(str(count), 0) + 1
        boundary = solver["eigengap_safe_boundary"]
        if boundary["backed_off"]:
            backed_off_components.append({
                "component": int(component),
                "component_size": int(len(nodes)),
                **boundary,
            })
    return {
        "component_labels": labels.astype(np.int32),
        "component_sizes": component_sizes[labels].astype(np.int32),
        "spectral_vectors": vectors,
        "spectral_values": values,
        "spectral_valid": valid,
    }, {
        "component_count": int(component_count),
        "largest_component_fraction": float(component_sizes.max() / num_entities),
        "isolated_components": int(np.sum(component_sizes == 1)),
        "solver_counts": solver_counts,
        "maximum_solver_residual": maximum_residual,
        "maximum_solver_orthogonality_error": maximum_orthogonality_error,
        "maximum_zero_eigenvalue_error": maximum_zero_eigenvalue_error,
        "solver_policy": HYBRID_SOLVER_POLICY,
        "trivial_modes": "one excluded independently within every connected component",
        "positive_mode_policy": (
            "retain at most eight modes; solve one look-ahead mode and back off "
            "before any repeated eigenspace crossing the truncation boundary"
        ),
        "retained_mode_histogram": retained_mode_histogram,
        "backed_off_component_count": len(backed_off_components),
        "backed_off_components": backed_off_components,
        "degeneracy_rtol": DEGENERACY_RTOL,
        "degeneracy_atol": DEGENERACY_ATOL,
    }


FROZEN._component_positive_eigenpairs = _component_positive_eigenpairs
FROZEN.component_spectral_encoding = component_spectral_encoding
graph_embeddings = FROZEN.graph_embeddings


def frozen_source_sha256() -> str:
    return file_sha256(SCREEN_FEATURES_PATH)


def b2_implementation_sha256() -> str:
    functions = (
        FROZEN.directed_pagerank,
        FROZEN.eigengap_safe_count,
        _component_positive_eigenpairs,
        component_spectral_encoding,
        FROZEN.graph_embeddings,
        FROZEN.b2_rows,
        parent_dataset,
        parent_context,
        parent_split,
        FROZEN._embedding_cache,
        FROZEN._split_b2v2,
        _context_b2v2_sidecar,
        materialize_dataset,
    )
    payload = [
        repr(tuple(FROZEN.B2_FEATURE_NAMES)),
        repr(FROZEN.LAPPE_DIMENSIONS),
        repr(tuple(FROZEN.HEAT_SCALES)),
        repr(FROZEN.DEGENERACY_RTOL),
        repr(FROZEN.DEGENERACY_ATOL),
        repr(FROZEN.NUM_FOLDS),
        json.dumps(HYBRID_SOLVER_POLICY, sort_keys=True),
    ]
    payload.extend(inspect.getsource(function) for function in functions)
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


@contextlib.contextmanager
def atomic_claim(path: Path):
    claim = path.with_name(path.name + ".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
    except FileExistsError as error:
        raise RuntimeError(f"active cross-host claim exists: {claim}") from error
    try:
        atomic_json(
            claim / "owner.json",
            {"hostname": os.uname().nodename, "pid": os.getpid()},
        )
        yield
    finally:
        (claim / "owner.json").unlink(missing_ok=True)
        try:
            claim.rmdir()
        except FileNotFoundError:
            pass


def _embedding_cache(dataset: str, graph_key: str, triples, num_entities: int):
    return FROZEN._embedding_cache(dataset, graph_key, triples, num_entities)


def full_train_embedding(dataset: str, triples, num_entities: int):
    return _embedding_cache(dataset, "full_train", triples, num_entities)


def _context_b2v2_sidecar(dataset: str, seed: int, data: dict) -> dict:
    context = parent_context(dataset, seed)
    row_count = len(context["triples"])
    if context["X"].shape != (800, 19) or row_count != 800:
        raise RuntimeError(f"unexpected parent context shape: {dataset}/{seed}")
    b2 = np.empty((row_count, len(B2_FEATURE_NAMES)), dtype=np.float32)
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    fold_records = []
    for fold, (_, heldout) in enumerate(splitter.split(data["train"])):
        keep = np.ones(len(data["train"]), dtype=bool)
        keep[heldout] = False
        graph_triples = data["train"][keep]
        rows = np.flatnonzero(context["fold_ids"] == fold)
        embedding, embedding_manifest = _embedding_cache(
            dataset,
            f"seed{seed}_fold{fold}",
            graph_triples,
            int(data["num_entities"]),
        )
        b2[rows] = b2_rows(context["triples"][rows], embedding)
        fold_records.append(
            {
                "fold": fold,
                "context_rows": int(len(rows)),
                "graph_triples": int(len(graph_triples)),
                "graph_hash": array_hash(graph_triples),
                "embedding_sha256": embedding_manifest["sha256"],
                "maximum_solver_residual": embedding_manifest["spectral"][
                    "maximum_solver_residual"
                ],
            }
        )
        print(
            f"FEATURES dataset={dataset} seed={seed} fold={fold + 1}/{NUM_FOLDS} "
            f"rows={len(rows)}",
            flush=True,
        )
    if not np.isfinite(b2).all():
        raise RuntimeError(f"non-finite context B2v2 sidecar: {dataset}/{seed}")
    output = FEATURE_ROOT / dataset / "contexts" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "b2v2.npy"
    atomic_array(path, b2)
    return FROZEN._record_artifact(
        path,
        b2,
        {
            "dataset": dataset,
            "seed": seed,
            "scope": "context",
            "block": "B2v2",
            "feature_names": B2_FEATURE_NAMES,
            "parent_context_sha256": file_sha256(
                PARENT_ASSETS / dataset / "contexts" / str(seed) / "context.npz"
            ),
            "parent_row_hash": array_hash(context["triples"]),
            "parent_fold_hash": array_hash(context["fold_ids"]),
            "labels_accessed_for_feature_construction": False,
            "folds": fold_records,
        },
    )


def materialize_dataset(dataset: str) -> dict:
    if dataset not in PANEL:
        raise ValueError(f"dataset is not in the frozen metric-free panel: {dataset}")
    completed_manifest = FEATURE_ROOT / dataset / "manifest.json"
    parent_path = PARENT_ASSETS / dataset / "dataset.npz"
    implementation_hash = b2_implementation_sha256()
    if completed_manifest.exists():
        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if (
            completed.get("status") == "COMPLETE"
            and completed.get("parent_dataset_sha256") == file_sha256(parent_path)
            and completed.get("b2_implementation_sha256") == implementation_hash
        ):
            return completed
    with atomic_claim(FEATURE_ROOT / dataset / ".materialize"):
        if completed_manifest.exists():
            completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
            if (
                completed.get("status") == "COMPLETE"
                and completed.get("parent_dataset_sha256") == file_sha256(parent_path)
                and completed.get("b2_implementation_sha256") == implementation_hash
            ):
                return completed
        data = parent_dataset(dataset)
        artifacts = []
        full_embedding, full_manifest = full_train_embedding(
            dataset, data["train"], int(data["num_entities"])
        )
        for split in ("valid", "test"):
            if not (PARENT_ASSETS / dataset / split / "features.npy").exists():
                continue
            asset = parent_split(dataset, split)
            artifacts.append(FROZEN._split_b2v2(dataset, split, asset, full_embedding))
        for seed in SEEDS:
            artifacts.append(_context_b2v2_sidecar(dataset, seed, data))
        manifest = {
            "status": "COMPLETE",
            "dataset": dataset,
            "num_entities": int(data["num_entities"]),
            "num_relations": int(data["num_relations"]),
            "num_train": int(len(data["train"])),
            "parent_dataset_sha256": file_sha256(parent_path),
            "parent_train_hash": array_hash(data["train"]),
            "frozen_screen_features_sha256": frozen_source_sha256(),
            "b2_implementation_sha256": implementation_hash,
            "hybrid_solver_policy": HYBRID_SOLVER_POLICY,
            "labels_or_performance_metrics_accessed": False,
            "feature_blocks": BLOCK_FEATURE_NAMES,
            "full_embedding": full_manifest,
            "artifacts": artifacts,
        }
        atomic_json(completed_manifest, manifest)
        return manifest


def load_sidecar(dataset: str, scope: str, block: str, seed: int | None = None):
    if block != "B2v2":
        raise ValueError("the metric-free side experiment exposes only B2v2")
    if scope == "context":
        if seed is None:
            raise ValueError("context sidecar requires a seed")
        path = FEATURE_ROOT / dataset / "contexts" / str(seed) / "b2v2.npy"
    else:
        path = FEATURE_ROOT / dataset / scope / "b2v2.npy"
    return np.load(path, mmap_mode="r")


def feature_chunks(
    dataset: str,
    scope: str,
    block: str,
    start: int,
    stop: int,
    seed: int | None = None,
) -> np.ndarray:
    if block != "B2v2":
        raise ValueError(block)
    parent_path = (
        PARENT_ASSETS / dataset / "contexts" / str(seed) / "context.npz"
        if scope == "context"
        else PARENT_ASSETS / dataset / scope / "features.npy"
    )
    if scope == "context":
        with np.load(parent_path, allow_pickle=False) as archive:
            base = archive["X"][start:stop]
    else:
        base = np.load(parent_path, mmap_mode="r")[start:stop]
    sidecar = load_sidecar(dataset, scope, "B2v2", seed)[start:stop]
    return np.concatenate(
        [np.asarray(base, dtype=np.float32), np.asarray(sidecar, dtype=np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)
