"""Vectorized implementation of RFMLink's frozen 18 structural invariants.

The scalar reference is ``run_umls_complex_vs_tabpfn.triple_feature_row``.
This module keeps the same definitions but materializes complete candidate
blocks without constructing one Python dictionary per candidate entity.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


FEATURE_NAMES = [
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
]


def _binary_csr(rows, cols, shape):
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()
    return matrix


def _dense_vector(value, size):
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=np.float32).reshape(size)
    return np.asarray(value, dtype=np.float32).reshape(size)


class VectorizedInvariantBuilder:
    """Train-only graph statistics with exact candidate-block formulas."""

    def __init__(
        self,
        train_triples,
        num_entities,
        num_relations,
        num_clusters=24,
        seed=7,
    ):
        train = np.asarray(train_triples, dtype=np.int64)
        if train.ndim != 2 or train.shape[1] != 3:
            raise ValueError("train_triples must have shape [n, 3]")
        if len(np.unique(train, axis=0)) != len(train):
            raise ValueError("training triples must be unique")
        self.train = train
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.num_clusters = min(int(num_clusters), self.num_entities)
        self.seed = int(seed)

        h, r, t = train.T
        self.adjacency = _binary_csr(
            h, t, (self.num_entities, self.num_entities)
        )
        self.adjacency_csc = self.adjacency.tocsc()
        self.out_degree = np.diff(self.adjacency.indptr).astype(np.float32)
        self.in_degree = np.diff(self.adjacency_csc.indptr).astype(np.float32)

        pair_counts = sparse.coo_matrix(
            (np.ones(len(train), dtype=np.float32), (h, t)),
            shape=(self.num_entities, self.num_entities),
        ).tocsr()
        pair_counts.sum_duplicates()
        pair_counts.sort_indices()
        self.pair_counts = pair_counts
        self.pair_counts_csc = pair_counts.tocsc()

        self.relation_count = np.bincount(
            r, minlength=self.num_relations
        ).astype(np.float32)
        self.relation_heads = np.zeros(self.num_relations, dtype=np.float32)
        self.relation_tails = np.zeros(self.num_relations, dtype=np.float32)
        for relation in np.unique(r):
            mask = r == relation
            self.relation_heads[relation] = np.unique(h[mask]).size
            self.relation_tails[relation] = np.unique(t[mask]).size
        denominator = np.maximum(self.relation_count, 1.0)
        self.relation_functionality = self.relation_heads / denominator
        self.relation_inverse_functionality = self.relation_tails / denominator
        self.relation_functionality[self.relation_count == 0] = 0.0
        self.relation_inverse_functionality[self.relation_count == 0] = 0.0

        self._relation_rows = defaultdict(list)
        for index, relation in enumerate(r.tolist()):
            self._relation_rows[int(relation)].append(index)
        self._relation_cache = {}

        signature_rows = np.concatenate([h, t])
        signature_cols = np.concatenate([r, self.num_relations + r])
        signatures = sparse.coo_matrix(
            (
                np.ones(2 * len(train), dtype=np.float64),
                (signature_rows, signature_cols),
            ),
            shape=(self.num_entities, 2 * self.num_relations),
        ).tocsr()
        signatures.sum_duplicates()
        signatures = normalize(signatures, norm="l2", axis=1, copy=False)
        self.entity_clusters = KMeans(
            n_clusters=self.num_clusters,
            n_init=50,
            random_state=self.seed,
        ).fit_predict(signatures)
        self.cluster_sizes = np.bincount(
            self.entity_clusters, minlength=self.num_clusters
        ).astype(np.float32)
        self._cluster_relation_cache = {}

    def relation_adjacency(self, relation):
        relation = int(relation)
        if relation not in self._relation_cache:
            indices = np.asarray(self._relation_rows.get(relation, []), dtype=np.int64)
            if len(indices):
                rows = self.train[indices, 0]
                cols = self.train[indices, 2]
            else:
                rows = cols = np.zeros(0, dtype=np.int64)
            csr = _binary_csr(
                rows, cols, (self.num_entities, self.num_entities)
            )
            self._relation_cache[relation] = (csr, csr.tocsc())
        return self._relation_cache[relation]

    def cluster_relation_density(self, relation):
        relation = int(relation)
        if relation not in self._cluster_relation_cache:
            counts = np.zeros(
                (self.num_clusters, self.num_clusters), dtype=np.float32
            )
            indices = np.asarray(self._relation_rows.get(relation, []), dtype=np.int64)
            if len(indices):
                rows = self.train[indices]
                np.add.at(
                    counts,
                    (
                        self.entity_clusters[rows[:, 0]],
                        self.entity_clusters[rows[:, 2]],
                    ),
                    1.0,
                )
            denom = np.maximum(
                self.cluster_sizes[:, None] * self.cluster_sizes[None, :], 1.0
            )
            self._cluster_relation_cache[relation] = counts / denom
        return self._cluster_relation_cache[relation]

    @staticmethod
    def _intersection_size(left, right):
        return float(np.intersect1d(left, right, assume_unique=True).size)

    def feature_row(self, h, r, t):
        h, r, t = int(h), int(r), int(t)
        rel, rel_csc = self.relation_adjacency(r)
        h_out = self.adjacency.indices[
            self.adjacency.indptr[h] : self.adjacency.indptr[h + 1]
        ]
        h_in = self.adjacency_csc.indices[
            self.adjacency_csc.indptr[h] : self.adjacency_csc.indptr[h + 1]
        ]
        t_out = self.adjacency.indices[
            self.adjacency.indptr[t] : self.adjacency.indptr[t + 1]
        ]
        t_in = self.adjacency_csc.indices[
            self.adjacency_csc.indptr[t] : self.adjacency_csc.indptr[t + 1]
        ]
        hr = rel.indices[rel.indptr[h] : rel.indptr[h + 1]]
        rt = rel_csc.indices[rel_csc.indptr[t] : rel_csc.indptr[t + 1]]
        pair_count = float(self.pair_counts[h, t])
        relation_present = float(rel[h, t] != 0)
        reverse_pair = float(self.pair_counts[t, h])
        hc, tc = self.entity_clusters[[h, t]]
        density = self.cluster_relation_density(r)
        return np.asarray(
            [
                r,
                len(h_out),
                len(h_in),
                len(t_out),
                len(t_in),
                len(hr),
                len(rt),
                self.relation_count[r],
                self.relation_functionality[r],
                self.relation_inverse_functionality[r],
                pair_count - relation_present,
                reverse_pair,
                self._intersection_size(h_out, t_in),
                self._intersection_size(hr, t_in),
                self._intersection_size(h_out, rt),
                self._intersection_size(t_out, h_in),
                float(hc == tc),
                density[hc, tc],
            ],
            dtype=np.float32,
        )

    def feature_rows(self, triples):
        triples = np.asarray(triples, dtype=np.int64)
        return np.vstack([self.feature_row(*row) for row in triples]).astype(
            np.float32, copy=False
        )

    def candidate_block(self, h, r, t, side):
        """Return all-entity features for one ``tail`` or ``head`` query."""
        h, r, t = int(h), int(r), int(t)
        n = self.num_entities
        rel, rel_csc = self.relation_adjacency(r)
        output = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        output[:, 0] = float(r)

        if side == "tail":
            output[:, 1] = self.out_degree[h]
            output[:, 2] = self.in_degree[h]
            output[:, 3] = self.out_degree
            output[:, 4] = self.in_degree
            output[:, 5] = float(rel.indptr[h + 1] - rel.indptr[h])
            output[:, 6] = np.diff(rel_csc.indptr)
            pair = _dense_vector(self.pair_counts.getrow(h), n)
            relation_pair = _dense_vector(rel.getrow(h), n)
            output[:, 10] = pair - relation_pair
            output[:, 11] = _dense_vector(self.pair_counts_csc.getcol(h), n)
            output[:, 12] = _dense_vector(self.adjacency.getrow(h) @ self.adjacency, n)
            output[:, 13] = _dense_vector(rel.getrow(h) @ self.adjacency, n)
            output[:, 14] = _dense_vector(self.adjacency.getrow(h) @ rel, n)
            output[:, 15] = _dense_vector(
                self.adjacency @ self.adjacency_csc.getcol(h), n
            )
            hc = self.entity_clusters[h]
            candidate_clusters = self.entity_clusters
            output[:, 16] = candidate_clusters == hc
            output[:, 17] = self.cluster_relation_density(r)[hc, candidate_clusters]
        elif side == "head":
            output[:, 1] = self.out_degree
            output[:, 2] = self.in_degree
            output[:, 3] = self.out_degree[t]
            output[:, 4] = self.in_degree[t]
            output[:, 5] = np.diff(rel.indptr)
            output[:, 6] = float(rel_csc.indptr[t + 1] - rel_csc.indptr[t])
            output[:, 10] = _dense_vector(self.pair_counts_csc.getcol(t), n) - _dense_vector(
                rel_csc.getcol(t), n
            )
            output[:, 11] = _dense_vector(self.pair_counts.getrow(t), n)
            output[:, 12] = _dense_vector(
                self.adjacency @ self.adjacency_csc.getcol(t), n
            )
            output[:, 13] = _dense_vector(rel @ self.adjacency_csc.getcol(t), n)
            output[:, 14] = _dense_vector(self.adjacency @ rel_csc.getcol(t), n)
            output[:, 15] = _dense_vector(self.adjacency.getrow(t) @ self.adjacency, n)
            tc = self.entity_clusters[t]
            candidate_clusters = self.entity_clusters
            output[:, 16] = candidate_clusters == tc
            output[:, 17] = self.cluster_relation_density(r)[candidate_clusters, tc]
        else:
            raise ValueError(f"unknown query side: {side!r}")

        output[:, 7] = self.relation_count[r]
        output[:, 8] = self.relation_functionality[r]
        output[:, 9] = self.relation_inverse_functionality[r]
        return output
