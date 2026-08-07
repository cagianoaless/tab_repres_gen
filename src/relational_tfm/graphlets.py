from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

TRIAD_STATES = ("out_only", "in_only", "reciprocal")
TRIAD_FEATURE_NAMES = tuple(
    [
        f"triad_{left}_{right}_log1p"
        for left in TRIAD_STATES
        for right in TRIAD_STATES
    ]
    + [
        f"triad_{left}_{right}_cosine"
        for left in TRIAD_STATES
        for right in TRIAD_STATES
    ]
)


@dataclass(frozen=True)
class DirectedTriadIndex:
    states: tuple[sparse.csr_matrix, ...]
    states_csc: tuple[sparse.csc_matrix, ...]
    row_degrees: tuple[np.ndarray, ...]
    column_degrees: tuple[np.ndarray, ...]
    undirected: sparse.csr_matrix


def directed_triad_index(
    edge_index: np.ndarray,
    num_nodes: int,
) -> DirectedTriadIndex:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge_index contains an entity outside num_nodes")

    keep = edges[0] != edges[1]
    data = np.ones(int(keep.sum()), dtype=np.float32)
    adjacency = sparse.csr_matrix(
        (data, (edges[0, keep], edges[1, keep])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()

    reciprocal = adjacency.multiply(adjacency.T).tocsr()
    reciprocal.data[:] = 1.0
    out_only = (adjacency - reciprocal).tocsr()
    out_only.eliminate_zeros()
    in_only = out_only.T.tocsr()
    states = (out_only, in_only, reciprocal)
    states_csc = tuple(matrix.tocsc() for matrix in states)
    row_degrees = tuple(
        np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
        for matrix in states
    )
    column_degrees = tuple(
        np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
        for matrix in states
    )

    undirected = (adjacency + adjacency.T).tocsr()
    undirected.data[:] = 1.0
    undirected.eliminate_zeros()
    return DirectedTriadIndex(
        states=states,
        states_csc=states_csc,
        row_degrees=row_degrees,
        column_degrees=column_degrees,
        undirected=undirected,
    )


def _pack_features(
    counts: np.ndarray,
    denominators: np.ndarray,
) -> np.ndarray:
    normalized = np.divide(
        counts,
        denominators,
        out=np.zeros_like(counts, dtype=np.float32),
        where=denominators > 0,
    )
    return np.concatenate(
        [np.log1p(counts).astype(np.float32), normalized.astype(np.float32)],
        axis=-1,
    )


def pair_triad_features(
    index: DirectedTriadIndex,
    heads: np.ndarray,
    tails: np.ndarray,
) -> np.ndarray:
    heads = np.asarray(heads, dtype=np.int64)
    tails = np.asarray(tails, dtype=np.int64)
    if heads.shape != tails.shape or heads.ndim != 1:
        raise ValueError("heads and tails must be aligned one-dimensional arrays")
    num_nodes = index.states[0].shape[0]
    if heads.size and (
        heads.min() < 0
        or tails.min() < 0
        or heads.max() >= num_nodes
        or tails.max() >= num_nodes
    ):
        raise ValueError("query endpoints fall outside the triad index")

    counts = np.zeros((len(heads), 9), dtype=np.float32)
    denominators = np.zeros_like(counts)
    column = 0
    for left, left_degrees in zip(
        index.states,
        index.row_degrees,
        strict=True,
    ):
        for right_index, right in enumerate(index.states_csc):
            for row, (head, tail) in enumerate(zip(heads, tails, strict=True)):
                left_neighbors = left.indices[
                    left.indptr[head] : left.indptr[head + 1]
                ]
                right_sources = right.indices[
                    right.indptr[tail] : right.indptr[tail + 1]
                ]
                counts[row, column] = np.intersect1d(
                    left_neighbors,
                    right_sources,
                    assume_unique=True,
                ).size
            denominators[:, column] = np.sqrt(
                left_degrees[heads]
                * index.column_degrees[right_index][tails]
            )
            column += 1
    return _pack_features(counts, denominators)


def candidate_triad_features(
    index: DirectedTriadIndex,
    *,
    anchor: int,
    side: int,
) -> np.ndarray:
    num_nodes = index.states[0].shape[0]
    if not 0 <= anchor < num_nodes:
        raise ValueError("anchor is outside the triad index")
    if side not in {0, 1}:
        raise ValueError("side must be 0 (tail candidates) or 1 (head candidates)")

    counts = np.zeros((num_nodes, 9), dtype=np.float32)
    denominators = np.zeros_like(counts)
    column = 0
    for left_index, left in enumerate(index.states):
        for right_index, right in enumerate(index.states):
            if side == 0:
                product = left.getrow(anchor) @ right
                counts[:, column] = np.asarray(product.toarray()).ravel()
                denominators[:, column] = np.sqrt(
                    index.row_degrees[left_index][anchor]
                    * index.column_degrees[right_index]
                )
            else:
                product = left @ right.getcol(anchor)
                counts[:, column] = np.asarray(product.toarray()).ravel()
                denominators[:, column] = np.sqrt(
                    index.row_degrees[left_index]
                    * index.column_degrees[right_index][anchor]
                )
            column += 1
    return _pack_features(counts, denominators)


def two_hop_entities(index: DirectedTriadIndex, anchor: int) -> np.ndarray:
    adjacency = index.undirected
    if not 0 <= anchor < adjacency.shape[0]:
        raise ValueError("anchor is outside the triad index")
    one_hop = adjacency.indices[
        adjacency.indptr[anchor] : adjacency.indptr[anchor + 1]
    ]
    if len(one_hop) == 0:
        return np.empty(0, dtype=np.int64)
    two_hop = (adjacency.getrow(anchor) @ adjacency).indices
    output = np.unique(np.concatenate([one_hop, two_hop])).astype(np.int64)
    return output[output != anchor]


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    median = np.median(values, axis=0)
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    scale = upper - lower
    scale[scale < 1e-8] = 1.0
    return ((values - median) / scale).astype(np.float32)


def facility_location_indices(
    values: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    values = robust_scale(values)
    rows = len(values)
    if count < 0 or count > rows:
        raise ValueError("facility-location count is outside the candidate pool")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == rows:
        return np.arange(rows, dtype=np.int64)

    squared = np.maximum(
        np.sum(values * values, axis=1)[:, None]
        + np.sum(values * values, axis=1)[None, :]
        - 2.0 * values @ values.T,
        0.0,
    )
    positive = squared[squared > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-squared / max(bandwidth, 1e-8))
    jitter = rng.uniform(0.0, 1e-12, size=rows)

    selected: list[int] = []
    covered = np.zeros(rows, dtype=np.float64)
    available = np.ones(rows, dtype=np.bool_)
    for _ in range(count):
        gains = np.maximum(covered[:, None], similarity).sum(axis=0)
        gains -= covered.sum()
        gains[~available] = -np.inf
        choice = int(np.argmax(gains + jitter))
        selected.append(choice)
        available[choice] = False
        covered = np.maximum(covered, similarity[:, choice])
    return np.asarray(selected, dtype=np.int64)


def relation_quotas(
    pool_relations: np.ndarray,
    target_relations: np.ndarray,
    count: int,
) -> dict[int, int]:
    pool_relations = np.asarray(pool_relations, dtype=np.int64)
    target_relations = np.asarray(target_relations, dtype=np.int64)
    if count <= 0 or count > len(pool_relations):
        raise ValueError("relation quota count is outside the positive pool")

    relations, target_counts = np.unique(target_relations, return_counts=True)
    capacities = {
        int(relation): int((pool_relations == relation).sum())
        for relation in relations
    }
    eligible = np.asarray(
        [relation for relation in relations if capacities[int(relation)] > 0],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise ValueError("no target relation occurs in the positive pool")
    weights = {
        int(relation): int(target_counts[relations == relation][0])
        for relation in eligible
    }
    quotas = {int(relation): 0 for relation in eligible}
    for _ in range(count):
        candidates = [
            int(relation)
            for relation in eligible
            if quotas[int(relation)] < capacities[int(relation)]
        ]
        if not candidates:
            raise ValueError("target-relation positive pool cannot fill the budget")
        total_weight = sum(weights[relation] for relation in candidates)
        choice = max(
            candidates,
            key=lambda relation: (
                (weights[relation] * count / total_weight) - quotas[relation],
                capacities[relation] - quotas[relation],
                -relation,
            ),
        )
        quotas[choice] += 1
    return quotas
