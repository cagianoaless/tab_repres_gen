#!/usr/bin/env python3
"""Eigengap-safe, leakage-controlled feature sidecars for the q512 panel."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import lobpcg
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[3]
RFMLINK_ROOT = ROOT / "RFMLink"
PARENT_ROOT = ROOT / "results" / "ultra57_rfmlink_q512_v1"
PARENT_ASSETS = PARENT_ROOT / "immutable_assets"
FAMILY_ROOT = ROOT / "results" / "g2t_q512_eigengap_safe_screen_v2"
FEATURE_ROOT = FAMILY_ROOT / "feature_assets"

PANEL = (
    "codex_small",
    "nell_ind_v1",
    "nl_ingram_75",
    "ilpc2022_small",
    "wdsinger",
    "metafam",
)
SEEDS = (101, 102, 103)
NUM_FOLDS = 10
LAPPE_DIMENSIONS = 8
HEAT_SCALES = (0.1, 1.0, 10.0, 100.0)
LOBPCG_MAX_ITERATIONS = 1_000
LOBPCG_RESIDUAL_LIMIT = 5e-5
DENSE_COMPONENT_LIMIT = 64
DEGENERACY_ATOL = 1e-7
DEGENERACY_RTOL = 1e-5

BASE_FEATURE_NAMES = [
    "relation_id",
    "head_total_out",
    "head_total_in",
    "tail_total_out",
    "tail_total_in",
    "head_relation_out_degree",
    "tail_relation_in_degree",
    "relation_frequency",
    "relation_functionality",
    "relation_inverse_functionality",
    "pair_other_relation_count",
    "reverse_pair_relation_count",
    "two_hop_count",
    "two_hop_start_relation_count",
    "two_hop_end_relation_count",
    "reverse_two_hop_count",
    "same_entity_cluster",
    "cluster_relation_density",
    "ultra_score",
]
B1_FEATURE_NAMES = [
    "query_direction",
    "head_relation_out_degree_query_rank_pct",
    "tail_relation_in_degree_query_rank_pct",
    "two_hop_count_query_rank_pct",
    "two_hop_start_relation_count_query_rank_pct",
    "two_hop_end_relation_count_query_rank_pct",
    "reverse_two_hop_count_query_rank_pct",
]
B2_FEATURE_NAMES = [
    "head_pagerank",
    "tail_pagerank",
    "same_connected_component",
    "head_component_log_fraction",
    "tail_component_log_fraction",
    *[f"heat_similarity_t{str(scale).replace('.', 'p')}" for scale in HEAT_SCALES],
    *[f"diffusion_distance_t{str(scale).replace('.', 'p')}" for scale in HEAT_SCALES],
]
BLOCK_FEATURE_NAMES = {
    "B0": BASE_FEATURE_NAMES,
    "B1": [*BASE_FEATURE_NAMES, *B1_FEATURE_NAMES],
    "B2v2": [*BASE_FEATURE_NAMES, *B2_FEATURE_NAMES],
    "B3v2": [*BASE_FEATURE_NAMES, *B1_FEATURE_NAMES, *B2_FEATURE_NAMES],
}
RAW_RANK_COLUMNS = (5, 6, 12, 13, 14, 15)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_array(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32))
    os.replace(temporary, path)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


def rank_percentiles(values) -> np.ndarray:
    """Average-tie ranks divided by candidate count, matching prior RFMLink ranks."""
    values = np.asarray(values)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    right = np.cumsum(counts)
    left = right - counts
    ranks = (left + 1 + right) / 2.0
    return (ranks[inverse] / len(values)).astype(np.float32)


def deterministic_context_directions(row_count: int, seed: int) -> np.ndarray:
    """Assign balanced directions without consulting row values or labels."""
    if row_count % 2:
        raise ValueError("balanced direction assignment requires an even row count")
    rng = np.random.default_rng(np.random.SeedSequence([20260717, seed, 1201]))
    order = rng.permutation(row_count)
    output = np.zeros(row_count, dtype=np.int8)
    output[order[row_count // 2 :]] = 1
    return output


def _binary_csr(rows, columns, num_entities: int) -> sparse.csr_matrix:
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(num_entities, num_entities),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()
    return matrix


def _dense_vector(value, size: int) -> np.ndarray:
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=np.float32).reshape(size)
    return np.asarray(value, dtype=np.float32).reshape(size)


class QueryRankBuilder:
    """Exact query-relative versions of six frozen invariant columns."""

    def __init__(self, triples, num_entities: int):
        self.triples = np.asarray(triples, dtype=np.int64)
        self.num_entities = int(num_entities)
        h, r, t = self.triples.T
        self.adjacency = _binary_csr(h, t, self.num_entities)
        self.adjacency_csc = self.adjacency.tocsc()
        self._relation_rows: dict[int, np.ndarray] = {
            int(relation): np.flatnonzero(r == relation) for relation in np.unique(r)
        }
        self._relations: dict[int, tuple[sparse.csr_matrix, sparse.csc_matrix]] = {}

    def relation_adjacency(self, relation: int):
        relation = int(relation)
        if relation not in self._relations:
            indices = self._relation_rows.get(relation, np.zeros(0, dtype=np.int64))
            rows = self.triples[indices, 0] if len(indices) else indices
            columns = self.triples[indices, 2] if len(indices) else indices
            matrix = _binary_csr(rows, columns, self.num_entities)
            self._relations[relation] = matrix, matrix.tocsc()
        return self._relations[relation]

    def candidate_vectors(self, h: int, r: int, t: int, side: int) -> np.ndarray:
        h, r, t = int(h), int(r), int(t)
        relation, relation_csc = self.relation_adjacency(r)
        n = self.num_entities
        if side == 0:
            vectors = [
                np.full(n, relation.indptr[h + 1] - relation.indptr[h], dtype=np.float32),
                np.diff(relation_csc.indptr).astype(np.float32),
                _dense_vector(self.adjacency.getrow(h) @ self.adjacency, n),
                _dense_vector(relation.getrow(h) @ self.adjacency, n),
                _dense_vector(self.adjacency.getrow(h) @ relation, n),
                _dense_vector(self.adjacency @ self.adjacency_csc.getcol(h), n),
            ]
        elif side == 1:
            vectors = [
                np.diff(relation.indptr).astype(np.float32),
                np.full(n, relation_csc.indptr[t + 1] - relation_csc.indptr[t], dtype=np.float32),
                _dense_vector(self.adjacency @ self.adjacency_csc.getcol(t), n),
                _dense_vector(relation @ self.adjacency_csc.getcol(t), n),
                _dense_vector(self.adjacency @ relation_csc.getcol(t), n),
                _dense_vector(self.adjacency.getrow(t) @ self.adjacency, n),
            ]
        else:
            raise ValueError(f"invalid side code: {side}")
        return np.vstack(vectors)

    def row_features(self, triple, side: int, expected_raw=None) -> np.ndarray:
        h, r, t = map(int, triple)
        vectors = self.candidate_vectors(h, r, t, side)
        target = t if side == 0 else h
        raw = vectors[:, target]
        if expected_raw is not None and not np.array_equal(raw, np.asarray(expected_raw)):
            raise RuntimeError(
                f"cross-fit rank reconstruction mismatch triple={(h, r, t)} "
                f"side={side} rebuilt={raw.tolist()} expected={np.asarray(expected_raw).tolist()}"
            )
        return np.asarray(
            [rank_percentiles(vector)[target] for vector in vectors], dtype=np.float32
        )


def directed_pagerank(triples, num_entities: int, damping: float = 0.85,
                      tolerance: float = 1e-10, max_iterations: int = 200):
    triples = np.asarray(triples, dtype=np.int64)
    adjacency = _binary_csr(triples[:, 0], triples[:, 2], num_entities).astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    transition = sparse.diags(inverse) @ adjacency
    dangling = degree == 0
    values = np.full(num_entities, 1.0 / num_entities, dtype=np.float64)
    converged = False
    for iteration in range(1, max_iterations + 1):
        updated = damping * (
            np.asarray(values @ transition).reshape(-1)
            + values[dangling].sum() / num_entities
        ) + (1.0 - damping) / num_entities
        error = float(np.abs(updated - values).sum())
        values = updated
        if error <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"PageRank did not converge in {max_iterations} iterations")
    values /= values.sum()
    return values.astype(np.float32), {"iterations": iteration, "l1_error": error}


def eigengap_safe_count(positive_eigenvalues, dimensions: int = LAPPE_DIMENSIONS) -> tuple[int, dict]:
    """Keep at most ``dimensions`` modes without splitting a repeated eigenspace."""
    values = np.asarray(positive_eigenvalues, dtype=np.float64)
    if len(values) <= dimensions:
        return len(values), {
            "requested_positive_modes": int(dimensions),
            "available_positive_modes": int(len(values)),
            "retained_positive_modes": int(len(values)),
            "backed_off": False,
            "boundary_group_start": None,
            "lambda_left": None,
            "lambda_lookahead": None,
            "boundary_gap": None,
        }
    left = float(values[dimensions - 1])
    right = float(values[dimensions])
    repeated = bool(np.isclose(left, right, rtol=DEGENERACY_RTOL, atol=DEGENERACY_ATOL))
    retained = dimensions
    group_start = None
    if repeated:
        group_start = dimensions - 1
        while group_start > 0 and np.isclose(
            values[group_start - 1],
            values[dimensions],
            rtol=DEGENERACY_RTOL,
            atol=DEGENERACY_ATOL,
        ):
            group_start -= 1
        retained = group_start
    return retained, {
        "requested_positive_modes": int(dimensions),
        "available_positive_modes": int(len(values)),
        "retained_positive_modes": int(retained),
        "backed_off": repeated,
        "boundary_group_start": int(group_start) if group_start is not None else None,
        "lambda_left": left,
        "lambda_lookahead": right,
        "boundary_gap": right - left,
        "degeneracy_rtol": DEGENERACY_RTOL,
        "degeneracy_atol": DEGENERACY_ATOL,
    }


def _component_positive_eigenpairs(laplacian, dimensions: int, seed_words: list[int]):
    size = laplacian.shape[0]
    # One trivial mode plus at most eight positive modes and one look-ahead.
    requested = min(dimensions + 2, size)
    if size <= DENSE_COMPONENT_LIMIT:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        solver = {
            "method": "dense_eigh",
            "component_size": int(size),
            "maximum_residual": 0.0,
        }
    else:
        initial = np.random.default_rng(
            np.random.SeedSequence([20260718, *seed_words, size, laplacian.nnz])
        ).standard_normal((size, requested))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eigenvalues, eigenvectors, residual_history = lobpcg(
                laplacian,
                initial,
                largest=False,
                tol=1e-5,
                maxiter=LOBPCG_MAX_ITERATIONS,
                retResidualNormsHistory=True,
            )
        residuals = np.asarray(residual_history[-1], dtype=np.float64)
        maximum_residual = float(residuals.max(initial=0.0))
        if maximum_residual > LOBPCG_RESIDUAL_LIMIT:
            raise RuntimeError(
                f"component LOBPCG residual {maximum_residual:.3e} exceeds "
                f"{LOBPCG_RESIDUAL_LIMIT:.3e}"
            )
        solver = {
            "method": "lobpcg_smallest",
            "component_size": int(size),
            "iterations": int(len(residual_history) - 1),
            "maximum_residual": maximum_residual,
            "warnings": [str(item.message) for item in caught],
        }
    order = np.argsort(eigenvalues, kind="stable")
    # Each connected component has exactly one trivial normalized-Laplacian mode.
    positive_values = np.asarray(eigenvalues)[order][1:]
    positive_vectors = np.asarray(eigenvectors)[:, order][:, 1:]
    retained, boundary = eigengap_safe_count(positive_values, dimensions)
    eigenvalues = positive_values[:retained]
    eigenvectors = positive_vectors[:, :retained]
    solver["eigengap_safe_boundary"] = boundary
    return eigenvalues, eigenvectors, solver


def component_spectral_encoding(triples, num_entities: int,
                                dimensions: int = LAPPE_DIMENSIONS):
    triples = np.asarray(triples, dtype=np.int64)
    directed = _binary_csr(triples[:, 0], triples[:, 2], num_entities)
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
    retained_mode_histogram: dict[str, int] = {}
    backed_off_components = []
    for component in range(component_count):
        nodes = np.flatnonzero(labels == component)
        if len(nodes) <= 1:
            continue
        component_laplacian = laplacian[nodes][:, nodes].tocsr()
        eigenvalues, eigenvectors, solver = _component_positive_eigenpairs(
            component_laplacian,
            dimensions,
            [component, int(nodes[0]), int(nodes[-1])],
        )
        count = len(eigenvalues)
        vectors[nodes, :count] = eigenvectors[:, :count].astype(np.float32)
        values[nodes, :count] = eigenvalues[:count].astype(np.float32)
        valid[nodes, :count] = 1
        solver_counts[solver["method"]] = solver_counts.get(solver["method"], 0) + 1
        maximum_residual = max(maximum_residual, solver["maximum_residual"])
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
        "maximum_lobpcg_residual": maximum_residual,
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


def graph_embeddings(triples, num_entities: int):
    pagerank, pagerank_info = directed_pagerank(triples, num_entities)
    spectral, spectral_info = component_spectral_encoding(triples, num_entities)
    return {"pagerank": pagerank, **spectral}, {
        "pagerank": pagerank_info,
        "spectral": spectral_info,
        "spectral_dimensions_per_component": LAPPE_DIMENSIONS,
        "heat_scales": list(HEAT_SCALES),
        "spectral_graph": "undirected relation-collapsed simple train graph",
        "spectral_pair_features": (
            "component-size-scaled heat similarity and diffusion distance; "
            "basis invariant because every retained repeated eigenspace is complete"
        ),
        "pagerank_graph": "directed relation-collapsed binary train graph",
    }


def b2_rows(triples, embedding) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int64)
    head = triples[:, 0]
    tail = triples[:, 2]
    labels = embedding["component_labels"]
    sizes = embedding["component_sizes"].astype(np.float64)
    same_component = labels[head] == labels[tail]
    graph_size = max(1, len(labels))
    denominator = np.log1p(graph_size)
    vectors_head = embedding["spectral_vectors"][head].astype(np.float64)
    vectors_tail = embedding["spectral_vectors"][tail].astype(np.float64)
    values = embedding["spectral_values"][head].astype(np.float64)
    valid = embedding["spectral_valid"][head].astype(np.float64)
    heat_similarities = []
    diffusion_distances = []
    for scale in HEAT_SCALES:
        weights = np.exp(-scale * values) * valid
        heat = sizes[head] * np.sum(
            weights * vectors_head * vectors_tail, axis=1
        )
        distance = sizes[head] * np.sum(
            np.square(weights) * np.square(vectors_head - vectors_tail), axis=1
        )
        heat_similarities.append(np.where(same_component, heat, 0.0))
        diffusion_distances.append(np.where(same_component, distance, 0.0))
    return np.column_stack([
        embedding["pagerank"][head],
        embedding["pagerank"][tail],
        same_component.astype(np.float32),
        np.log1p(sizes[head]) / denominator,
        np.log1p(sizes[tail]) / denominator,
        *heat_similarities,
        *diffusion_distances,
    ]).astype(np.float32)


def parent_dataset(dataset: str) -> dict:
    path = PARENT_ASSETS / dataset / "dataset.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def parent_context(dataset: str, seed: int) -> dict:
    path = PARENT_ASSETS / dataset / "contexts" / str(seed) / "context.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def parent_split(dataset: str, split: str) -> dict:
    root = PARENT_ASSETS / dataset / split
    features = np.load(root / "features.npy", mmap_mode="r")
    with np.load(root / "queries.npz", allow_pickle=False) as archive:
        metadata = {name: archive[name] for name in archive.files}
    return {"X": features, **metadata}


def _artifact_complete(path: Path, shape: tuple[int, int]) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(path, mmap_mode="r")
        return (
            tuple(values.shape) == shape
            and metadata["sha256"] == file_sha256(path)
            and metadata["shape"] == list(shape)
        )
    except Exception:
        return False


def _record_artifact(path: Path, values, extra: dict) -> dict:
    payload = {
        **extra,
        "path": str(path.relative_to(ROOT)),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": file_sha256(path),
        "array_hash": array_hash(values),
    }
    atomic_json(path.with_suffix(".json"), payload)
    return payload


def _embedding_cache(dataset: str, graph_key: str, triples, num_entities: int):
    path = FEATURE_ROOT / dataset / "graph_embeddings" / f"{graph_key}.npz"
    metadata_path = path.with_suffix(".json")
    graph_hash = array_hash(np.asarray(triples, dtype=np.int64))
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("graph_hash") == graph_hash and metadata.get("sha256") == file_sha256(path):
            with np.load(path, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}, metadata
    embedding, details = graph_embeddings(triples, num_entities)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **embedding)
    os.replace(temporary, path)
    metadata = {
        "path": str(path.relative_to(ROOT)),
        "graph_key": graph_key,
        "graph_hash": graph_hash,
        "graph_triples": int(len(triples)),
        "num_entities": int(num_entities),
        "sha256": file_sha256(path),
        **details,
    }
    atomic_json(metadata_path, metadata)
    return embedding, metadata


def _context_sidecars(dataset: str, seed: int, data: dict) -> list[dict]:
    context = parent_context(dataset, seed)
    if context["X"].shape != (800, 19):
        raise RuntimeError(f"unexpected parent context shape: {dataset}/{seed}")
    directions = deterministic_context_directions(len(context["y"]), seed)
    b1 = np.empty((len(context["y"]), len(B1_FEATURE_NAMES)), dtype=np.float32)
    b1[:, 0] = directions
    b2 = np.empty((len(context["y"]), len(B2_FEATURE_NAMES)), dtype=np.float32)
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    fold_heldout = [heldout for _, heldout in splitter.split(data["train"])]
    fold_records = []
    for fold, heldout in enumerate(fold_heldout):
        keep = np.ones(len(data["train"]), dtype=bool)
        keep[heldout] = False
        graph_triples = data["train"][keep]
        rows = np.flatnonzero(context["fold_ids"] == fold)
        rank_builder = QueryRankBuilder(graph_triples, int(data["num_entities"]))
        for row in rows:
            expected = context["X"][row, list(RAW_RANK_COLUMNS)]
            b1[row, 1:] = rank_builder.row_features(
                context["triples"][row], int(directions[row]), expected_raw=expected
            )
        embedding, embedding_manifest = _embedding_cache(
            dataset, f"seed{seed}_fold{fold}", graph_triples, int(data["num_entities"])
        )
        b2[rows] = b2_rows(context["triples"][rows], embedding)
        fold_records.append({
            "fold": fold,
            "context_rows": int(len(rows)),
            "graph_triples": int(len(graph_triples)),
            "graph_hash": array_hash(graph_triples),
            "embedding_sha256": embedding_manifest["sha256"],
        })
        print(
            f"FEATURES dataset={dataset} seed={seed} fold={fold + 1}/{NUM_FOLDS} "
            f"rows={len(rows)}",
            flush=True,
        )
    if not np.isfinite(b1).all() or not np.isfinite(b2).all():
        raise RuntimeError(f"non-finite context sidecar: {dataset}/{seed}")
    output = FEATURE_ROOT / dataset / "contexts" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for block, values, names in (("b1", b1, B1_FEATURE_NAMES), ("b2v2", b2, B2_FEATURE_NAMES)):
        path = output / f"{block}.npy"
        atomic_array(path, values)
        artifacts.append(_record_artifact(path, values, {
            "dataset": dataset,
            "seed": seed,
            "scope": "context",
            "block": block.upper(),
            "feature_names": names,
            "parent_context_sha256": file_sha256(
                PARENT_ASSETS / dataset / "contexts" / str(seed) / "context.npz"
            ),
            "parent_row_hash": array_hash(context["triples"]),
            "parent_label_hash": array_hash(context["y"]),
            "parent_fold_hash": array_hash(context["fold_ids"]),
            "direction_hash": array_hash(directions),
            "direction_policy": (
                "deterministic exactly-balanced row-position assignment; "
                "independent of triples, features, and labels"
            ),
            "folds": fold_records,
        }))
    return artifacts


def _split_b1(dataset: str, split: str, asset: dict) -> dict:
    num_queries = len(asset["side_codes"])
    num_entities = int(asset["X"].shape[0] // num_queries)
    shape = (asset["X"].shape[0], len(B1_FEATURE_NAMES))
    path = FEATURE_ROOT / dataset / split / "b1.npy"
    if _artifact_complete(path, shape):
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    for query, side in enumerate(asset["side_codes"]):
        start, stop = query * num_entities, (query + 1) * num_entities
        output[start:stop, 0] = int(side)
        for target_column, source_column in enumerate(RAW_RANK_COLUMNS, 1):
            output[start:stop, target_column] = rank_percentiles(
                asset["X"][start:stop, source_column]
            )
    output.flush()
    del output
    os.replace(temporary, path)
    values = np.load(path, mmap_mode="r")
    return _record_artifact(path, values, {
        "dataset": dataset,
        "split": split,
        "scope": "evaluation",
        "block": "B1",
        "feature_names": B1_FEATURE_NAMES,
        "queries": num_queries,
        "num_entities": num_entities,
        "rank_reference": "all candidate entities before evaluator filtering",
        "parent_feature_sha256": file_sha256(PARENT_ASSETS / dataset / split / "features.npy"),
        "parent_query_sha256": file_sha256(PARENT_ASSETS / dataset / split / "queries.npz"),
    })


def _split_b2v2(dataset: str, split: str, asset: dict, embedding) -> dict:
    num_queries = len(asset["side_codes"])
    num_entities = int(asset["X"].shape[0] // num_queries)
    shape = (asset["X"].shape[0], len(B2_FEATURE_NAMES))
    path = FEATURE_ROOT / dataset / split / "b2v2.npy"
    if _artifact_complete(path, shape):
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    triples = asset["selected_triples"]
    for query, side in enumerate(asset["side_codes"]):
        triple = triples[query // 2]
        h, _, t = map(int, triple)
        start, stop = query * num_entities, (query + 1) * num_entities
        if int(side) == 0:
            rows = np.column_stack([
                np.full(num_entities, h, dtype=np.int64),
                np.zeros(num_entities, dtype=np.int64),
                np.arange(num_entities, dtype=np.int64),
            ])
        else:
            rows = np.column_stack([
                np.arange(num_entities, dtype=np.int64),
                np.zeros(num_entities, dtype=np.int64),
                np.full(num_entities, t, dtype=np.int64),
            ])
        output[start:stop] = b2_rows(rows, embedding)
    output.flush()
    del output
    os.replace(temporary, path)
    values = np.load(path, mmap_mode="r")
    return _record_artifact(path, values, {
        "dataset": dataset,
        "split": split,
        "scope": "evaluation",
        "block": "B2v2",
        "feature_names": B2_FEATURE_NAMES,
        "queries": num_queries,
        "num_entities": num_entities,
        "parent_feature_sha256": file_sha256(PARENT_ASSETS / dataset / split / "features.npy"),
        "parent_query_sha256": file_sha256(PARENT_ASSETS / dataset / split / "queries.npz"),
    })


def materialize_dataset(dataset: str) -> dict:
    if dataset not in PANEL:
        raise ValueError(f"dataset is not in the frozen panel: {dataset}")
    with file_lock(FEATURE_ROOT / dataset / ".materialize.lock"):
        completed_manifest = FEATURE_ROOT / dataset / "manifest.json"
        if completed_manifest.exists():
            completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
            parent_path = PARENT_ASSETS / dataset / "dataset.npz"
            if (
                completed.get("status") == "COMPLETE"
                and completed.get("parent_dataset_sha256") == file_sha256(parent_path)
            ):
                return completed
        data = parent_dataset(dataset)
        if list(map(int, np.unique(data["train"][:, 1]))) and int(data["num_entities"]) <= 1:
            raise RuntimeError("invalid parent dataset")
        artifacts = []
        full_graph_embedding, full_embedding_manifest = _embedding_cache(
            dataset, "full_train", data["train"], int(data["num_entities"])
        )
        for split in ("valid", "test"):
            if not (PARENT_ASSETS / dataset / split / "features.npy").exists():
                continue
            asset = parent_split(dataset, split)
            artifacts.append(_split_b1(dataset, split, asset))
            artifacts.append(_split_b2v2(dataset, split, asset, full_graph_embedding))
        for seed in SEEDS:
            artifacts.extend(_context_sidecars(dataset, seed, data))
        manifest = {
            "status": "COMPLETE",
            "dataset": dataset,
            "num_entities": int(data["num_entities"]),
            "num_relations": int(data["num_relations"]),
            "num_train": int(len(data["train"])),
            "parent_dataset_sha256": file_sha256(PARENT_ASSETS / dataset / "dataset.npz"),
            "parent_train_hash": array_hash(data["train"]),
            "feature_blocks": BLOCK_FEATURE_NAMES,
            "full_embedding": full_embedding_manifest,
            "artifacts": artifacts,
        }
        atomic_json(FEATURE_ROOT / dataset / "manifest.json", manifest)
        return manifest


def load_sidecar(dataset: str, scope: str, block: str, seed: int | None = None):
    if scope == "context":
        if seed is None:
            raise ValueError("context sidecar requires a seed")
        path = FEATURE_ROOT / dataset / "contexts" / str(seed) / f"{block.lower()}.npy"
    else:
        path = FEATURE_ROOT / dataset / scope / f"{block.lower()}.npy"
    return np.load(path, mmap_mode="r")


def feature_chunks(dataset: str, scope: str, block: str, start: int, stop: int,
                   seed: int | None = None) -> np.ndarray:
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
    pieces = [np.asarray(base, dtype=np.float32)]
    if block in ("B1", "B3v2"):
        pieces.append(np.asarray(load_sidecar(dataset, scope, "B1", seed)[start:stop]))
    if block in ("B2v2", "B3v2"):
        pieces.append(np.asarray(load_sidecar(dataset, scope, "B2v2", seed)[start:stop]))
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
