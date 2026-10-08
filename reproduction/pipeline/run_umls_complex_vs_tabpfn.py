import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_sbm_probability_experiments import (
    build_tabpfn_inference_config,
    positive_class_probabilities,
)


UMLS_URLS = {
    "train": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/umls/train.txt",
    "valid": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/umls/valid.txt",
    "test": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/umls/test.txt",
}
KINSHIPS_URLS = {
    "train": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/kinships/train.txt",
    "valid": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/kinships/valid.txt",
    "test": "https://raw.githubusercontent.com/pykeen/pykeen/v1.11.1/src/pykeen/datasets/kinships/test.txt",
}
WN18RR_URLS = {
    "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/train.txt",
    "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/valid.txt",
    "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/test.txt",
}
DATASET_URLS = {
    "umls": UMLS_URLS,
    "kinships": KINSHIPS_URLS,
    "wn18rr": WN18RR_URLS,
}

KG_FEATURE_COLUMNS = [
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
    "head_cluster",
    "tail_cluster",
    "same_entity_cluster",
    "cluster_relation_density",
]
ID_FEATURE_COLUMNS = ["head_id", "tail_id"]
CATEGORICAL_FEATURE_COLUMNS = [
    "head_id",
    "tail_id",
    "relation_id",
    "head_cluster",
    "tail_cluster",
    "same_entity_cluster",
    "query_direction",
]
FEATURE_SETS = ("kg_context", "kg_context_ids")
FEATURE_GROUPS = {
    "ids": ID_FEATURE_COLUMNS,
    "relation": [
        "relation_id",
        "relation_frequency",
        "relation_functionality",
        "relation_inverse_functionality",
    ],
    "entity_role": [
        "head_total_out",
        "head_total_in",
        "tail_total_out",
        "tail_total_in",
        "head_relation_out_degree",
        "tail_relation_in_degree",
    ],
    "pair": [
        "pair_other_relation_count",
        "reverse_pair_relation_count",
    ],
    "two_hop": [
        "two_hop_count",
        "two_hop_start_relation_count",
        "two_hop_end_relation_count",
        "reverse_two_hop_count",
    ],
    "clusters": [
        "head_cluster",
        "tail_cluster",
        "same_entity_cluster",
        "cluster_relation_density",
    ],
    "cluster_invariants": [
        "same_entity_cluster",
        "cluster_relation_density",
    ],
    "query_direction": ["query_direction"],
    "role_signatures": [],
    "anchor_distances": [],
    "path_patterns": [],
}
STRUCTURAL_FEATURE_GROUPS = (
    "relation",
    "entity_role",
    "pair",
    "two_hop",
    "clusters",
)
PATH_FEATURE_MODES = ("raw", "binary", "log1p")
PATH_VOCABULARY_MODES = ("top_train", "hashed")
TABULAR_TRAIN_FEATURE_PROTOCOLS = (
    "in_graph",
    "leave_one_triple_out",
    "kfold_cross_fit",
)
TABULAR_CONTEXT_MODES = (
    "sampled",
    "query_sampled",
    "one_vs_all",
)
COMPLEX_TRAINING_MODES = (
    "fixed_negative",
    "symmetric_lcwa",
    "multilabel_lcwa",
    "reciprocal_lcwa",
)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deduplicate(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_feature_groups(raw_groups):
    if raw_groups is None:
        return None
    groups = [group.strip() for group in raw_groups.split(",") if group.strip()]
    if not groups:
        raise ValueError("--feature-groups was provided but no groups were parsed")
    if groups == ["all"]:
        return ["ids", *STRUCTURAL_FEATURE_GROUPS, "path_patterns"]
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(
            f"unsupported feature groups {unknown}; "
            f"valid groups are {sorted(FEATURE_GROUPS)}"
        )
    return groups


def columns_from_feature_groups(feature_groups, stats):
    columns = []
    for group in feature_groups:
        if group == "path_patterns":
            columns.extend(stats.path_feature_columns)
        elif group == "role_signatures":
            columns.extend(stats.role_signature_feature_columns)
        elif group == "anchor_distances":
            columns.extend(stats.anchor_distance_feature_columns)
        else:
            columns.extend(FEATURE_GROUPS[group])
    return deduplicate(columns)


def columns_from_legacy_feature_set(feature_set, stats):
    base_feature_columns = KG_FEATURE_COLUMNS + stats.path_feature_columns
    if feature_set == "kg_context":
        return base_feature_columns
    if feature_set == "kg_context_ids":
        return ID_FEATURE_COLUMNS + base_feature_columns
    raise ValueError(f"unsupported feature set: {feature_set}")


def maybe_download(data_dir, urls=UMLS_URLS):
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, url in urls.items():
        path = data_dir / f"{split}.txt"
        if path.exists():
            continue
        print(f"Downloading {url} -> {path}", flush=True)
        urllib.request.urlretrieve(url, path)


def read_split(data_dir, split):
    path = data_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; re-run with --download")
    return pd.read_csv(
        path,
        sep="\t",
        names=["head", "relation", "tail"],
        dtype=str,
    )


def load_kg_dataset(data_dir, download, dataset):
    if dataset not in DATASET_URLS:
        raise ValueError(
            f"unsupported dataset {dataset!r}; choose from {sorted(DATASET_URLS)}"
        )
    if download:
        maybe_download(data_dir, DATASET_URLS[dataset])
    splits = {split: read_split(data_dir, split) for split in ["train", "valid", "test"]}

    entities = sorted(
        set(pd.concat([frame["head"] for frame in splits.values()]))
        | set(pd.concat([frame["tail"] for frame in splits.values()]))
    )
    relations = sorted(set(pd.concat([frame["relation"] for frame in splits.values()])))
    entity_to_id = {entity: index for index, entity in enumerate(entities)}
    relation_to_id = {relation: index for index, relation in enumerate(relations)}

    mapped = {}
    for split, frame in splits.items():
        mapped_frame = frame.copy()
        mapped_frame["h"] = mapped_frame["head"].map(entity_to_id).astype(int)
        mapped_frame["r"] = mapped_frame["relation"].map(relation_to_id).astype(int)
        mapped_frame["t"] = mapped_frame["tail"].map(entity_to_id).astype(int)
        mapped[split] = mapped_frame

    return mapped, entities, relations, entity_to_id, relation_to_id


def load_umls(data_dir, download):
    return load_kg_dataset(data_dir, download, dataset="umls")


def augment_splits_with_reciprocals(splits, relations):
    num_direct_relations = len(relations)
    augmented = {}
    for split, frame in splits.items():
        direct = frame.copy().reset_index(drop=True)
        direct["reciprocal_group"] = np.arange(len(direct), dtype=np.int64)

        inverse = direct.copy()
        inverse["head"] = direct["tail"]
        inverse["tail"] = direct["head"]
        inverse["relation"] = direct["relation"] + "__inverse"
        inverse["h"] = direct["t"]
        inverse["r"] = direct["r"] + num_direct_relations
        inverse["t"] = direct["h"]
        augmented[split] = pd.concat([direct, inverse], ignore_index=True)

    inverse_relation_ids = {
        relation_id: relation_id + num_direct_relations
        for relation_id in range(num_direct_relations)
    }
    augmented_relations = [
        *relations,
        *(f"{relation}__inverse" for relation in relations),
    ]
    return augmented, augmented_relations, inverse_relation_ids


class KGStats:
    def __init__(
        self,
        train,
        num_entities,
        num_relations,
        num_entity_clusters,
        seed,
        path_feature_mode,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.path_feature_mode = path_feature_mode
        self.train_triples = {
            (int(row.h), int(row.r), int(row.t))
            for row in train[["h", "r", "t"]].itertuples(index=False)
        }
        if len(self.train_triples) != len(train):
            raise ValueError(
                "KGStats requires unique training triples for reversible "
                "leave-one-triple-out feature extraction"
            )
        self.out_neighbors = [set() for _ in range(num_entities)]
        self.in_neighbors = [set() for _ in range(num_entities)]
        self.out_by_hr = defaultdict(set)
        self.in_by_rt = defaultdict(set)
        self.out_rel_targets = [defaultdict(set) for _ in range(num_entities)]
        self.in_rel_sources = [defaultdict(set) for _ in range(num_entities)]
        self.pair_relations = defaultdict(set)
        self.relation_count = np.zeros(num_relations, dtype=np.float64)
        self.relation_heads = [set() for _ in range(num_relations)]
        self.relation_tails = [set() for _ in range(num_relations)]

        for h, r, t in train[["h", "r", "t"]].itertuples(index=False):
            h = int(h)
            r = int(r)
            t = int(t)
            self.out_neighbors[h].add(t)
            self.in_neighbors[t].add(h)
            self.out_by_hr[(h, r)].add(t)
            self.in_by_rt[(r, t)].add(h)
            self.out_rel_targets[h][r].add(t)
            self.in_rel_sources[t][r].add(h)
            self.pair_relations[(h, t)].add(r)
            self.relation_count[r] += 1.0
            self.relation_heads[r].add(h)
            self.relation_tails[r].add(t)

        self.relation_functionality = np.zeros(num_relations, dtype=np.float64)
        self.relation_inverse_functionality = np.zeros(num_relations, dtype=np.float64)
        for relation in range(num_relations):
            count = self.relation_count[relation]
            if count > 0:
                self.relation_functionality[relation] = (
                    len(self.relation_heads[relation]) / count
                )
                self.relation_inverse_functionality[relation] = (
                    len(self.relation_tails[relation]) / count
                )

        self.entity_clusters = self._cluster_entities(
            train=train,
            num_entity_clusters=min(num_entity_clusters, num_entities),
            seed=seed,
        )
        self.cluster_sizes = np.bincount(
            self.entity_clusters,
            minlength=min(num_entity_clusters, num_entities),
        ).astype(np.float64)
        self.cluster_relation_counts = defaultdict(float)
        for h, r, t in train[["h", "r", "t"]].itertuples(index=False):
            hc = int(self.entity_clusters[int(h)])
            tc = int(self.entity_clusters[int(t)])
            self.cluster_relation_counts[(hc, int(r), tc)] += 1.0
        self.selected_path_patterns = []
        self.path_feature_columns = []
        self.path_vocabulary_mode = "top_train"
        self.num_hashed_path_features = 0
        self.num_role_signature_bins = 0
        self.role_signature_feature_columns = []
        self.outgoing_role_signatures = None
        self.incoming_role_signatures = None
        self.num_anchors = 0
        self.anchor_max_distance = 0
        self.anchor_entities = []
        self.anchor_distance_feature_columns = []
        self.from_anchor_distances = None
        self.to_anchor_distances = None

    def configure_extended_features(
        self,
        num_role_signature_bins=0,
        num_anchors=0,
        anchor_max_distance=3,
    ):
        self._configure_role_signatures(num_role_signature_bins)
        self._configure_anchor_distances(num_anchors, anchor_max_distance)

    def _configure_role_signatures(self, num_bins):
        if num_bins < 0:
            raise ValueError("num_role_signature_bins cannot be negative")
        self.num_role_signature_bins = num_bins
        if num_bins == 0:
            return

        outgoing = np.zeros((self.num_entities, num_bins), dtype=np.float64)
        incoming = np.zeros((self.num_entities, num_bins), dtype=np.float64)
        for (h, r), targets in self.out_by_hr.items():
            index = stable_relation_hash(r, num_bins)
            outgoing[h, index] += len(targets)
        for (r, t), sources in self.in_by_rt.items():
            index = stable_relation_hash(r, num_bins)
            incoming[t, index] += len(sources)
        self.outgoing_role_signatures = np.log1p(outgoing)
        self.incoming_role_signatures = np.log1p(incoming)
        self.role_signature_feature_columns = [
            *[f"head_out_role_{index}" for index in range(num_bins)],
            *[f"head_in_role_{index}" for index in range(num_bins)],
            *[f"tail_out_role_{index}" for index in range(num_bins)],
            *[f"tail_in_role_{index}" for index in range(num_bins)],
        ]

    @staticmethod
    def _bounded_distances(start, adjacency, max_distance):
        unreachable = max_distance + 1
        distances = np.full(len(adjacency), unreachable, dtype=np.float64)
        distances[start] = 0.0
        frontier = {start}
        for distance in range(1, max_distance + 1):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier = {
                node for node in next_frontier if distances[node] == unreachable
            }
            if not next_frontier:
                break
            distances[list(next_frontier)] = float(distance)
            frontier = next_frontier
        return distances

    def _configure_anchor_distances(self, num_anchors, max_distance):
        if num_anchors < 0:
            raise ValueError("num_anchors cannot be negative")
        if num_anchors == 0:
            return
        if max_distance < 1:
            raise ValueError("anchor_max_distance must be positive")

        self.num_anchors = min(num_anchors, self.num_entities)
        self.anchor_max_distance = max_distance
        degrees = [
            len(self.out_neighbors[node]) + len(self.in_neighbors[node])
            for node in range(self.num_entities)
        ]
        self.anchor_entities = sorted(
            range(self.num_entities),
            key=lambda node: (-degrees[node], node),
        )[: self.num_anchors]
        self.from_anchor_distances = np.column_stack(
            [
                self._bounded_distances(anchor, self.out_neighbors, max_distance)
                for anchor in self.anchor_entities
            ]
        )
        self.to_anchor_distances = np.column_stack(
            [
                self._bounded_distances(anchor, self.in_neighbors, max_distance)
                for anchor in self.anchor_entities
            ]
        )
        self.anchor_distance_feature_columns = [
            *[f"head_from_anchor_{index}" for index in range(self.num_anchors)],
            *[f"head_to_anchor_{index}" for index in range(self.num_anchors)],
            *[f"tail_from_anchor_{index}" for index in range(self.num_anchors)],
            *[f"tail_to_anchor_{index}" for index in range(self.num_anchors)],
        ]

    def extended_feature_values(self, h, t):
        values = {}
        if self.num_role_signature_bins:
            arrays = (
                self.outgoing_role_signatures[h],
                self.incoming_role_signatures[h],
                self.outgoing_role_signatures[t],
                self.incoming_role_signatures[t],
            )
            flattened = np.concatenate(arrays)
            values.update(
                zip(self.role_signature_feature_columns, flattened, strict=True)
            )
        if self.num_anchors:
            arrays = (
                self.from_anchor_distances[h],
                self.to_anchor_distances[h],
                self.from_anchor_distances[t],
                self.to_anchor_distances[t],
            )
            flattened = np.concatenate(arrays)
            values.update(
                zip(self.anchor_distance_feature_columns, flattened, strict=True)
            )
        return values

    def _refresh_relation_summary(self, relation):
        count = self.relation_count[relation]
        if count <= 0:
            self.relation_functionality[relation] = 0.0
            self.relation_inverse_functionality[relation] = 0.0
            return
        self.relation_functionality[relation] = (
            len(self.relation_heads[relation]) / count
        )
        self.relation_inverse_functionality[relation] = (
            len(self.relation_tails[relation]) / count
        )

    def remove_triple(self, h, r, t):
        triple = (int(h), int(r), int(t))
        h, r, t = triple
        if triple not in self.train_triples:
            raise ValueError(f"cannot remove absent training triple {triple}")

        self.train_triples.remove(triple)
        self.out_by_hr[(h, r)].remove(t)
        self.in_by_rt[(r, t)].remove(h)
        self.out_rel_targets[h][r].remove(t)
        self.in_rel_sources[t][r].remove(h)
        self.pair_relations[(h, t)].remove(r)

        if not self.out_by_hr[(h, r)]:
            self.relation_heads[r].remove(h)
        if not self.in_by_rt[(r, t)]:
            self.relation_tails[r].remove(t)
        if not self.pair_relations[(h, t)]:
            self.out_neighbors[h].remove(t)
            self.in_neighbors[t].remove(h)

        self.relation_count[r] -= 1.0
        hc = int(self.entity_clusters[h])
        tc = int(self.entity_clusters[t])
        self.cluster_relation_counts[(hc, r, tc)] -= 1.0
        self._refresh_relation_summary(r)

    def add_triple(self, h, r, t):
        triple = (int(h), int(r), int(t))
        h, r, t = triple
        if triple in self.train_triples:
            raise ValueError(f"cannot add existing training triple {triple}")

        self.train_triples.add(triple)
        self.out_neighbors[h].add(t)
        self.in_neighbors[t].add(h)
        self.out_by_hr[(h, r)].add(t)
        self.in_by_rt[(r, t)].add(h)
        self.out_rel_targets[h][r].add(t)
        self.in_rel_sources[t][r].add(h)
        self.pair_relations[(h, t)].add(r)
        self.relation_heads[r].add(h)
        self.relation_tails[r].add(t)
        self.relation_count[r] += 1.0
        hc = int(self.entity_clusters[h])
        tc = int(self.entity_clusters[t])
        self.cluster_relation_counts[(hc, r, tc)] += 1.0
        self._refresh_relation_summary(r)

    @contextmanager
    def without_triple(self, h, r, t):
        self.remove_triple(h, r, t)
        try:
            yield
        finally:
            self.add_triple(h, r, t)

    def configure_path_features(self, train, max_patterns, vocabulary_mode):
        self.path_vocabulary_mode = vocabulary_mode
        if vocabulary_mode == "hashed":
            self.selected_path_patterns = []
            self.num_hashed_path_features = max_patterns
            self.path_feature_columns = [
                f"path_pattern_{index}" for index in range(max_patterns)
            ]
            return
        if vocabulary_mode != "top_train":
            raise ValueError(f"unsupported path vocabulary mode: {vocabulary_mode}")
        self.num_hashed_path_features = 0
        self.select_path_patterns(train, max_patterns)

    def select_path_patterns(self, train, max_patterns):
        if max_patterns <= 0:
            self.selected_path_patterns = []
            self.path_feature_columns = []
            return

        counter = Counter()
        for h, r, t in train[["h", "r", "t"]].itertuples(index=False):
            h = int(h)
            r = int(r)
            t = int(t)
            for r1, targets in self.out_rel_targets[h].items():
                for r2, sources in self.in_rel_sources[t].items():
                    count = len(targets & sources)
                    if count:
                        counter[(r, int(r1), int(r2))] += count
        self.selected_path_patterns = [
            pattern for pattern, _ in counter.most_common(max_patterns)
        ]
        self.path_feature_columns = [
            f"path_pattern_{index}"
            for index in range(len(self.selected_path_patterns))
        ]

    def path_feature_value(self, count):
        if self.path_feature_mode == "raw":
            return float(count)
        if self.path_feature_mode == "binary":
            return float(count > 0)
        if self.path_feature_mode == "log1p":
            return float(np.log1p(count))
        raise ValueError(f"unsupported path feature mode: {self.path_feature_mode}")

    def path_pattern_counts(self, h, r, t):
        if not self.path_feature_columns:
            return []
        counts_by_pair = {}
        for r1, targets in self.out_rel_targets[h].items():
            for r2, sources in self.in_rel_sources[t].items():
                count = len(targets & sources)
                if count:
                    counts_by_pair[(int(r1), int(r2))] = float(count)
        if self.path_vocabulary_mode == "hashed":
            values = np.zeros(self.num_hashed_path_features, dtype=np.float64)
            for (r1, r2), count in counts_by_pair.items():
                index = stable_path_hash(r, r1, r2, self.num_hashed_path_features)
                values[index] += count
            return [self.path_feature_value(value) for value in values]
        values = []
        for target_relation, r1, r2 in self.selected_path_patterns:
            count = (
                counts_by_pair.get((r1, r2), 0.0)
                if int(r) == int(target_relation)
                else 0.0
            )
            values.append(self.path_feature_value(count))
        return values

    def _cluster_entities(self, train, num_entity_clusters, seed):
        signatures = np.zeros(
            (self.num_entities, self.num_relations * 2),
            dtype=np.float64,
        )
        for h, r, t in train[["h", "r", "t"]].itertuples(index=False):
            h = int(h)
            r = int(r)
            t = int(t)
            signatures[h, r] += 1.0
            signatures[t, self.num_relations + r] += 1.0
        row_norms = np.linalg.norm(signatures, axis=1, keepdims=True)
        signatures = np.divide(
            signatures,
            row_norms,
            out=np.zeros_like(signatures),
            where=row_norms > 0,
        )
        return KMeans(
            n_clusters=num_entity_clusters,
            n_init=50,
            random_state=seed,
        ).fit_predict(signatures)

    def cluster_relation_density(self, h, r, t):
        hc = int(self.entity_clusters[h])
        tc = int(self.entity_clusters[t])
        denominator = max(self.cluster_sizes[hc] * self.cluster_sizes[tc], 1.0)
        return self.cluster_relation_counts[(hc, r, tc)] / denominator


def stable_path_hash(target_relation, first_relation, second_relation, num_bins):
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    value = (
        ((int(target_relation) + 1) * 73856093)
        ^ ((int(first_relation) + 1) * 19349663)
        ^ ((int(second_relation) + 1) * 83492791)
    )
    return int(value % num_bins)


def stable_relation_hash(relation, num_bins):
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    return int(((int(relation) + 1) * 2654435761) % num_bins)


def triple_feature_row(h, r, t, stats):
    h = int(h)
    r = int(r)
    t = int(t)
    h_out = stats.out_neighbors[h]
    h_in = stats.in_neighbors[h]
    t_out = stats.out_neighbors[t]
    t_in = stats.in_neighbors[t]
    two_hop = h_out & t_in
    reverse_two_hop = t_out & h_in
    head_relation_targets = stats.out_by_hr.get((h, r), frozenset())
    tail_relation_sources = stats.in_by_rt.get((r, t), frozenset())
    pair_relations = stats.pair_relations.get((h, t), frozenset())
    reverse_pair_relations = stats.pair_relations.get((t, h), frozenset())
    start_r = head_relation_targets & t_in
    end_r = h_out & tail_relation_sources
    pair_other_relations = pair_relations - {r}
    h_cluster = int(stats.entity_clusters[h])
    t_cluster = int(stats.entity_clusters[t])

    return {
        "head_id": float(h),
        "tail_id": float(t),
        "relation_id": float(r),
        "head_total_out": float(len(h_out)),
        "head_total_in": float(len(h_in)),
        "tail_total_out": float(len(t_out)),
        "tail_total_in": float(len(t_in)),
        "head_relation_out_degree": float(len(head_relation_targets)),
        "tail_relation_in_degree": float(len(tail_relation_sources)),
        "relation_frequency": float(stats.relation_count[r]),
        "relation_functionality": float(stats.relation_functionality[r]),
        "relation_inverse_functionality": float(
            stats.relation_inverse_functionality[r]
        ),
        "pair_other_relation_count": float(len(pair_other_relations)),
        "reverse_pair_relation_count": float(len(reverse_pair_relations)),
        "two_hop_count": float(len(two_hop)),
        "two_hop_start_relation_count": float(len(start_r)),
        "two_hop_end_relation_count": float(len(end_r)),
        "reverse_two_hop_count": float(len(reverse_two_hop)),
        "head_cluster": float(h_cluster),
        "tail_cluster": float(t_cluster),
        "same_entity_cluster": float(h_cluster == t_cluster),
        "cluster_relation_density": float(stats.cluster_relation_density(h, r, t)),
        **{
            column: value
            for column, value in zip(
                stats.path_feature_columns,
                stats.path_pattern_counts(h, r, t),
            )
        },
        **stats.extended_feature_values(h, t),
    }


def feature_frame(triples, stats, leave_one_triple_out_positive=False):
    rows = []
    has_labels = "label" in triples.columns
    for row in triples.itertuples(index=False):
        h = int(row.h)
        r = int(row.r)
        t = int(row.t)
        should_exclude = (
            leave_one_triple_out_positive
            and has_labels
            and int(row.label) == 1
        )
        if should_exclude:
            with stats.without_triple(h, r, t):
                feature_row = triple_feature_row(h, r, t, stats)
        else:
            feature_row = triple_feature_row(h, r, t, stats)
        if hasattr(row, "query_direction"):
            feature_row["query_direction"] = float(row.query_direction)
        rows.append(feature_row)
    return pd.DataFrame(rows)


def cross_fitted_feature_frame(
    train,
    context,
    num_entities,
    num_relations,
    num_entity_clusters,
    seed,
    path_feature_mode,
    path_vocabulary_mode,
    num_path_pattern_features,
    num_folds,
    cross_fit_group_column=None,
    num_role_signature_bins=0,
    num_anchors=0,
    anchor_max_distance=3,
):
    if num_folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    if cross_fit_group_column is None:
        split_units = np.arange(len(train), dtype=np.int64)
        group_values = None
    else:
        if cross_fit_group_column not in train:
            raise ValueError(
                f"missing cross-fit group column {cross_fit_group_column!r}"
            )
        group_values = train[cross_fit_group_column].to_numpy(dtype=np.int64)
        split_units = np.unique(group_values)
    if num_folds > len(split_units):
        raise ValueError("cross-fitting cannot use more folds than split units")
    if path_vocabulary_mode != "hashed" and num_path_pattern_features > 0:
        raise ValueError(
            "cross-fitting requires hashed path features so every fold uses the "
            "same target-independent columns"
        )

    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_by_triple = {}
    heldout_indices_by_fold = []
    for fold, (_, heldout_unit_indices) in enumerate(splitter.split(split_units)):
        if group_values is None:
            heldout_indices = heldout_unit_indices
        else:
            heldout_groups = split_units[heldout_unit_indices]
            heldout_indices = np.flatnonzero(np.isin(group_values, heldout_groups))
        heldout_indices_by_fold.append(heldout_indices)
        for row in train.iloc[heldout_indices][["h", "r", "t"]].itertuples(
            index=False
        ):
            fold_by_triple[(int(row.h), int(row.r), int(row.t))] = fold

    fold_ids = np.full(len(context), -1, dtype=np.int64)
    positive_mask = context["label"].to_numpy(dtype=int) == 1
    for index, row in enumerate(context[["h", "r", "t"]].itertuples(index=False)):
        if positive_mask[index]:
            fold_ids[index] = fold_by_triple[(int(row.h), int(row.r), int(row.t))]

    negative_indices = np.flatnonzero(~positive_mask)
    balanced_negative_folds = np.resize(np.arange(num_folds), len(negative_indices))
    rng = np.random.default_rng(seed + 30_000)
    rng.shuffle(balanced_negative_folds)
    fold_ids[negative_indices] = balanced_negative_folds
    if np.any(fold_ids < 0):
        raise RuntimeError("failed to assign every context row to a cross-fit fold")

    feature_parts = []
    fold_metadata = []
    for fold, heldout_indices in enumerate(heldout_indices_by_fold):
        representation_train = train.drop(train.index[heldout_indices]).reset_index(
            drop=True
        )
        fold_stats = KGStats(
            representation_train,
            num_entities,
            num_relations,
            num_entity_clusters,
            seed + fold,
            path_feature_mode,
        )
        fold_stats.configure_path_features(
            representation_train,
            num_path_pattern_features,
            path_vocabulary_mode,
        )
        fold_stats.configure_extended_features(
            num_role_signature_bins=num_role_signature_bins,
            num_anchors=num_anchors,
            anchor_max_distance=anchor_max_distance,
        )
        context_indices = np.flatnonzero(fold_ids == fold)
        fold_features = feature_frame(context.iloc[context_indices], fold_stats)
        fold_features.index = context_indices
        feature_parts.append(fold_features)
        fold_metadata.append(
            {
                "fold": fold,
                "representation_train_triples": int(len(representation_train)),
                "context_rows": int(len(context_indices)),
                "positive_context_rows": int(np.sum(positive_mask[context_indices])),
                "negative_context_rows": int(np.sum(~positive_mask[context_indices])),
                "heldout_split_units": int(
                    len(heldout_indices)
                    if group_values is None
                    else len(np.unique(group_values[heldout_indices]))
                ),
            }
        )
        print(
            f"cross-fit fold={fold + 1}/{num_folds} "
            f"graph={len(representation_train)} context={len(context_indices)}",
            flush=True,
        )

    features = pd.concat(feature_parts).sort_index()
    if not np.array_equal(features.index.to_numpy(), np.arange(len(context))):
        raise RuntimeError("cross-fitted features do not align with context rows")
    return features.reset_index(drop=True), fold_ids, fold_metadata


def cross_fitted_query_feature_frame(
    train,
    context,
    num_entities,
    num_relations,
    num_entity_clusters,
    seed,
    path_feature_mode,
    path_vocabulary_mode,
    num_path_pattern_features,
    num_folds,
    num_role_signature_bins=0,
    num_anchors=0,
    anchor_max_distance=3,
):
    required = {"query_direction", "query_id"}
    missing = required - set(context.columns)
    if missing:
        raise ValueError(f"query cross-fitting is missing columns {sorted(missing)}")
    if num_folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    if path_vocabulary_mode != "hashed" and num_path_pattern_features > 0:
        raise ValueError("query cross-fitting requires hashed path features")

    fold_ids = np.full(len(context), -1, dtype=np.int64)
    feature_parts = []
    fold_metadata = []
    for direction in (0, 1):
        direction_mask = context["query_direction"].to_numpy(dtype=int) == direction
        direction_indices = np.flatnonzero(direction_mask)
        query_ids = np.sort(context.loc[direction_mask, "query_id"].unique())
        if num_folds > len(query_ids):
            raise ValueError("cross-fitting cannot use more folds than queries")
        splitter = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed + direction,
        )
        for fold, (_, heldout_query_indices) in enumerate(splitter.split(query_ids)):
            heldout_query_ids = query_ids[heldout_query_indices]
            context_indices = direction_indices[
                np.isin(
                    context.iloc[direction_indices]["query_id"].to_numpy(),
                    heldout_query_ids,
                )
            ]
            heldout_rows = context.iloc[context_indices]
            if direction == 0:
                heldout_keys = set(
                    heldout_rows[["h", "r"]].itertuples(index=False, name=None)
                )
                remove_mask = np.fromiter(
                    (
                        (int(row.h), int(row.r)) in heldout_keys
                        for row in train[["h", "r"]].itertuples(index=False)
                    ),
                    dtype=bool,
                    count=len(train),
                )
                direction_name = "tail"
            else:
                heldout_keys = set(
                    heldout_rows[["r", "t"]].itertuples(index=False, name=None)
                )
                remove_mask = np.fromiter(
                    (
                        (int(row.r), int(row.t)) in heldout_keys
                        for row in train[["r", "t"]].itertuples(index=False)
                    ),
                    dtype=bool,
                    count=len(train),
                )
                direction_name = "head"
            representation_train = train.loc[~remove_mask].reset_index(drop=True)
            fold_stats = KGStats(
                representation_train,
                num_entities,
                num_relations,
                num_entity_clusters,
                seed + direction * num_folds + fold,
                path_feature_mode,
            )
            fold_stats.configure_path_features(
                representation_train,
                num_path_pattern_features,
                path_vocabulary_mode,
            )
            fold_stats.configure_extended_features(
                num_role_signature_bins=num_role_signature_bins,
                num_anchors=num_anchors,
                anchor_max_distance=anchor_max_distance,
            )
            fold_features = feature_frame(context.iloc[context_indices], fold_stats)
            fold_features.index = context_indices
            feature_parts.append(fold_features)
            encoded_fold = direction * num_folds + fold
            fold_ids[context_indices] = encoded_fold
            labels = context.iloc[context_indices]["label"].to_numpy(dtype=int)
            fold_metadata.append(
                {
                    "direction": direction_name,
                    "fold": fold,
                    "heldout_queries": int(len(heldout_query_ids)),
                    "removed_train_triples": int(np.sum(remove_mask)),
                    "representation_train_triples": int(len(representation_train)),
                    "context_rows": int(len(context_indices)),
                    "positive_context_rows": int(np.sum(labels == 1)),
                    "negative_context_rows": int(np.sum(labels == 0)),
                }
            )
            print(
                f"query cross-fit direction={direction_name} "
                f"fold={fold + 1}/{num_folds} graph={len(representation_train)} "
                f"queries={len(heldout_query_ids)} context={len(context_indices)}",
                flush=True,
            )

    if np.any(fold_ids < 0):
        raise RuntimeError("failed to assign every query-context row to a fold")
    features = pd.concat(feature_parts).sort_index()
    if not np.array_equal(features.index.to_numpy(), np.arange(len(context))):
        raise RuntimeError("query cross-fitted features do not align with context rows")
    return features.reset_index(drop=True), fold_ids, fold_metadata


def all_true_triples(splits):
    triples = set()
    for frame in splits.values():
        triples.update(
            (int(row.h), int(row.r), int(row.t))
            for row in frame[["h", "r", "t"]].itertuples(index=False)
        )
    return triples


def frame_triples(frame):
    return {
        (int(row.h), int(row.r), int(row.t))
        for row in frame[["h", "r", "t"]].itertuples(index=False)
    }


def sample_negative_triples(positive, all_true, num_entities, negative_ratio, seed):
    rng = np.random.default_rng(seed)
    rows = []
    used = set()
    target_count = len(positive) * negative_ratio
    positive_array = positive[["h", "r", "t"]].to_numpy(dtype=int)
    while len(rows) < target_count:
        h, r, t = positive_array[rng.integers(0, len(positive_array))]
        if rng.random() < 0.5:
            h = int(rng.integers(0, num_entities))
        else:
            t = int(rng.integers(0, num_entities))
        candidate = (int(h), int(r), int(t))
        if candidate in all_true or candidate in used:
            continue
        used.add(candidate)
        rows.append(candidate)
    return pd.DataFrame(rows, columns=["h", "r", "t"])


def make_binary_context(train_positive, all_true, num_entities, negative_ratio, seed):
    positive = train_positive[["h", "r", "t"]].copy()
    positive["label"] = 1
    negative = sample_negative_triples(
        train_positive,
        all_true,
        num_entities,
        negative_ratio,
        seed,
    )
    negative["label"] = 0
    context = pd.concat([positive, negative], ignore_index=True)
    return context.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_query_context(train_positive, num_entities, negative_ratio, seed, one_vs_all):
    rng = np.random.default_rng(seed)
    rows = []
    query_id = 0

    tail_targets = defaultdict(set)
    head_targets = defaultdict(set)
    for h, r, t in train_positive[["h", "r", "t"]].itertuples(index=False):
        tail_targets[(int(h), int(r))].add(int(t))
        head_targets[(int(r), int(t))].add(int(h))

    for (h, r), positives in sorted(tail_targets.items()):
        negatives = sorted(set(range(num_entities)) - positives)
        if not one_vs_all:
            count = min(len(negatives), negative_ratio * len(positives))
            negatives = sorted(rng.choice(negatives, size=count, replace=False).tolist())
        for candidate in sorted(positives):
            rows.append((h, r, candidate, 1, 0, query_id))
        for candidate in negatives:
            rows.append((h, r, candidate, 0, 0, query_id))
        query_id += 1

    for (r, t), positives in sorted(head_targets.items()):
        negatives = sorted(set(range(num_entities)) - positives)
        if not one_vs_all:
            count = min(len(negatives), negative_ratio * len(positives))
            negatives = sorted(rng.choice(negatives, size=count, replace=False).tolist())
        for candidate in sorted(positives):
            rows.append((candidate, r, t, 1, 1, query_id))
        for candidate in negatives:
            rows.append((candidate, r, t, 0, 1, query_id))
        query_id += 1

    context = pd.DataFrame(
        rows,
        columns=["h", "r", "t", "label", "query_direction", "query_id"],
    )
    return context.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class ComplExModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)
        scale = 1.0 / math.sqrt(embedding_dim)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -scale, scale)

    def score_hrt(self, h, r, t):
        h_re = self.entity_re(h)
        h_im = self.entity_im(h)
        r_re = self.relation_re(r)
        r_im = self.relation_im(r)
        t_re = self.entity_re(t)
        t_im = self.entity_im(t)
        return torch.sum(
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re,
            dim=-1,
        )

    def score_all_tails(self, h, r):
        h_re = self.entity_re(h)
        h_im = self.entity_im(h)
        r_re = self.relation_re(r)
        r_im = self.relation_im(r)
        tail_re_coeff = h_re * r_re - h_im * r_im
        tail_im_coeff = h_re * r_im + h_im * r_re
        return (
            tail_re_coeff @ self.entity_re.weight.T
            + tail_im_coeff @ self.entity_im.weight.T
        )

    def score_all_heads(self, r, t):
        r_re = self.relation_re(r)
        r_im = self.relation_im(r)
        t_re = self.entity_re(t)
        t_im = self.entity_im(t)
        head_re_coeff = r_re * t_re + r_im * t_im
        head_im_coeff = r_re * t_im - r_im * t_re
        return (
            head_re_coeff @ self.entity_re.weight.T
            + head_im_coeff @ self.entity_im.weight.T
        )


def train_complex_fixed_negative(
    model,
    splits,
    train_true,
    num_entities,
    args,
    device,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.complex_lr,
        weight_decay=args.complex_weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_context = make_binary_context(
        splits["train"],
        train_true,
        num_entities,
        args.complex_negative_ratio,
        args.seed + 10_000,
    )
    triples = torch.as_tensor(
        train_context[["h", "r", "t"]].to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
    )
    labels = torch.as_tensor(
        train_context["label"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    loader = DataLoader(
        TensorDataset(triples, labels),
        batch_size=args.complex_batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.complex_epochs + 1):
        epoch_loss = 0.0
        for batch_triples, batch_labels in loader:
            batch_triples = batch_triples.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_hrt(
                batch_triples[:, 0],
                batch_triples[:, 1],
                batch_triples[:, 2],
            )
            loss = loss_fn(scores, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_labels)
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"complex epoch={epoch} loss={epoch_loss / len(train_context):.6f}",
                flush=True,
            )
    return model


def train_complex_symmetric_lcwa(model, train, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.complex_lr,
        weight_decay=args.complex_weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    triples = torch.as_tensor(
        train[["h", "r", "t"]].to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
    )
    loader = DataLoader(
        TensorDataset(triples),
        batch_size=args.complex_batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.complex_epochs + 1):
        epoch_loss = 0.0
        for (batch_triples,) in loader:
            batch_triples = batch_triples.to(device)
            optimizer.zero_grad(set_to_none=True)
            tail_scores = model.score_all_tails(
                batch_triples[:, 0],
                batch_triples[:, 1],
            )
            head_scores = model.score_all_heads(
                batch_triples[:, 1],
                batch_triples[:, 2],
            )
            loss = 0.5 * (
                loss_fn(tail_scores, batch_triples[:, 2])
                + loss_fn(head_scores, batch_triples[:, 0])
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_triples)
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"complex symmetric_lcwa epoch={epoch} "
                f"loss={epoch_loss / len(train):.6f}",
                flush=True,
            )
    return model


def train_complex_reciprocal_lcwa(model, train, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.complex_lr,
        weight_decay=args.complex_weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    triples = torch.as_tensor(
        train[["h", "r", "t"]].to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
    )
    loader = DataLoader(
        TensorDataset(triples),
        batch_size=args.complex_batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.complex_epochs + 1):
        epoch_loss = 0.0
        for (batch_triples,) in loader:
            batch_triples = batch_triples.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_all_tails(
                batch_triples[:, 0],
                batch_triples[:, 1],
            )
            loss = loss_fn(scores, batch_triples[:, 2])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_triples)
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"complex reciprocal_lcwa epoch={epoch} "
                f"loss={epoch_loss / len(train):.6f}",
                flush=True,
            )
    return model


def build_lcwa_instances(train, num_entities, target):
    grouped_targets = defaultdict(set)
    for h, r, t in train[["h", "r", "t"]].itertuples(index=False):
        if target == "tail":
            grouped_targets[(int(h), int(r))].add(int(t))
        elif target == "head":
            grouped_targets[(int(r), int(t))].add(int(h))
        else:
            raise ValueError(f"unsupported LCWA target: {target}")

    pairs = []
    labels = np.zeros((len(grouped_targets), num_entities), dtype=np.float32)
    for row_index, (pair, entities) in enumerate(sorted(grouped_targets.items())):
        pairs.append(pair)
        labels[row_index, list(entities)] = 1.0 / len(entities)
    return (
        torch.as_tensor(np.asarray(pairs, dtype=np.int64), dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.float32),
    )


def train_complex_multilabel_lcwa(model, train, num_entities, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.complex_lr,
        weight_decay=args.complex_weight_decay,
    )
    tail_pairs, tail_labels = build_lcwa_instances(train, num_entities, "tail")
    head_pairs, head_labels = build_lcwa_instances(train, num_entities, "head")
    tail_loader = DataLoader(
        TensorDataset(tail_pairs, tail_labels),
        batch_size=args.complex_batch_size,
        shuffle=True,
    )
    head_loader = DataLoader(
        TensorDataset(head_pairs, head_labels),
        batch_size=args.complex_batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.complex_epochs + 1):
        epoch_loss = 0.0
        example_count = 0
        for pairs, labels in tail_loader:
            pairs = pairs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_all_tails(pairs[:, 0], pairs[:, 1])
            loss = -(labels * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(pairs)
            example_count += len(pairs)
        for pairs, labels in head_loader:
            pairs = pairs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_all_heads(pairs[:, 0], pairs[:, 1])
            loss = -(labels * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(pairs)
            example_count += len(pairs)
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"complex multilabel_lcwa epoch={epoch} "
                f"loss={epoch_loss / example_count:.6f}",
                flush=True,
            )
    return model


def train_complex(splits, train_true, num_entities, num_relations, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ComplExModel(num_entities, num_relations, args.complex_dim).to(device)
    if args.complex_training_mode == "fixed_negative":
        return train_complex_fixed_negative(
            model,
            splits,
            train_true,
            num_entities,
            args,
            device,
        )
    if args.complex_training_mode == "symmetric_lcwa":
        return train_complex_symmetric_lcwa(
            model,
            splits["train"],
            args,
            device,
        )
    if args.complex_training_mode == "multilabel_lcwa":
        return train_complex_multilabel_lcwa(
            model,
            splits["train"],
            num_entities,
            args,
            device,
        )
    if args.complex_training_mode == "reciprocal_lcwa":
        return train_complex_reciprocal_lcwa(
            model,
            splits["train"],
            args,
            device,
        )
    raise ValueError(f"unsupported ComplEx training mode: {args.complex_training_mode}")


def build_eval_candidates(
    test,
    num_entities,
    all_true,
    inverse_relation_ids=None,
):
    rows = []
    query_index = 0
    for h, r, t in test[["h", "r", "t"]].itertuples(index=False):
        h = int(h)
        r = int(r)
        t = int(t)
        for candidate_tail in range(num_entities):
            triple = (h, r, candidate_tail)
            rows.append(
                {
                    "query_index": query_index,
                    "side": "tail",
                    "query_direction": 0,
                    "h": h,
                    "r": r,
                    "t": candidate_tail,
                    "is_target": int(candidate_tail == t),
                    "is_filtered": int(triple in all_true and candidate_tail != t),
                }
            )
        query_index += 1
        for candidate_head in range(num_entities):
            if inverse_relation_ids is None:
                triple = (candidate_head, r, t)
                candidate_h = candidate_head
                candidate_r = r
                candidate_t = t
            else:
                candidate_h = t
                candidate_r = inverse_relation_ids[r]
                candidate_t = candidate_head
                triple = (candidate_h, candidate_r, candidate_t)
            rows.append(
                {
                    "query_index": query_index,
                    "side": "head",
                    "query_direction": 1,
                    "h": candidate_h,
                    "r": candidate_r,
                    "t": candidate_t,
                    "is_target": int(candidate_head == h),
                    "is_filtered": int(triple in all_true and candidate_head != h),
                }
            )
        query_index += 1
    return pd.DataFrame(rows)


def tied_rank_positions(valid_scores, target_score):
    greater_count = int(np.sum(valid_scores > target_score))
    tied_count = int(
        np.sum(np.isclose(valid_scores, target_score, rtol=1e-12, atol=1e-15))
    )
    if tied_count <= 0:
        tied_count = 1
    return np.arange(
        greater_count + 1,
        greater_count + tied_count + 1,
        dtype=np.float64,
    )


def expected_hits_at_k(tied_positions, k):
    return float(np.mean(tied_positions <= k))


def filtered_ranking_metrics(candidates, scores):
    frame = candidates[["query_index", "is_target", "is_filtered"]].copy()
    frame["score"] = scores
    expected_ranks = []
    reciprocal_ranks = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    for _, group in frame.groupby("query_index", sort=False):
        target_score = float(group.loc[group["is_target"] == 1, "score"].iloc[0])
        valid = group.loc[group["is_filtered"] == 0]
        tied_positions = tied_rank_positions(
            valid["score"].to_numpy(dtype=np.float64),
            target_score,
        )
        expected_ranks.append(float(np.mean(tied_positions)))
        reciprocal_ranks.append(float(np.mean(1.0 / tied_positions)))
        hits_at_1.append(expected_hits_at_k(tied_positions, 1))
        hits_at_3.append(expected_hits_at_k(tied_positions, 3))
        hits_at_10.append(expected_hits_at_k(tied_positions, 10))
    expected_ranks = np.asarray(expected_ranks, dtype=np.float64)
    reciprocal_ranks = np.asarray(reciprocal_ranks, dtype=np.float64)
    return {
        "queries": int(len(expected_ranks)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hits_at_1": float(np.mean(hits_at_1)),
        "hits_at_3": float(np.mean(hits_at_3)),
        "hits_at_10": float(np.mean(hits_at_10)),
        "mean_rank": float(np.mean(expected_ranks)),
    }


def score_complex_candidates(model, candidates, device, batch_size):
    model.eval()
    scores = []
    triples = torch.as_tensor(
        candidates[["h", "r", "t"]].to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
    )
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            batch = triples[start : start + batch_size].to(device)
            batch_scores = model.score_hrt(batch[:, 0], batch[:, 1], batch[:, 2])
            scores.append(batch_scores.detach().cpu().numpy())
    return np.concatenate(scores)


def fit_logistic(train_features, labels, candidate_features):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=0),
    )
    model.fit(train_features, labels)
    return model.predict_proba(candidate_features)[:, 1], model


def fit_tabpfn(train_features, labels, candidate_features, args, categorical_indices):
    from tabpfn import TabPFNClassifier

    categorical_limit = max(
        int(train_features.iloc[:, index].nunique())
        for index in categorical_indices
    ) if categorical_indices else 2
    model_args = {
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "categorical_features_indices": categorical_indices,
        "inference_config": build_tabpfn_inference_config(
            max(categorical_limit, 2)
        ),
    }
    if args.checkpoint_name is not None:
        model_args["model_path"] = Path(args.model_dir) / args.checkpoint_name

    model = TabPFNClassifier(**model_args)
    model.fit(train_features.astype(np.float32), labels)
    probabilities = []
    for start in range(0, len(candidate_features), args.tabpfn_predict_batch_size):
        chunk = candidate_features.iloc[start : start + args.tabpfn_predict_batch_size]
        probabilities.append(
            positive_class_probabilities(model, chunk.astype(np.float32))
        )
    return np.concatenate(probabilities), model


def sampled_binary_metrics(scores_by_model, binary_candidates, labels):
    rows = []
    for model_name, scores in scores_by_model.items():
        if model_name == "complex":
            probabilities = 1.0 / (1.0 + np.exp(-scores))
        else:
            probabilities = scores
        rows.append(
            {
                "model": model_name,
                "binary_roc_auc": float(roc_auc_score(labels, probabilities)),
                "binary_average_precision": float(
                    average_precision_score(labels, probabilities)
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args):
    set_seeds(args.seed)
    if args.data_dir is None:
        args.data_dir = Path("data/kg") / args.dataset
    mapped, entities, relations, entity_to_id, relation_to_id = load_kg_dataset(
        args.data_dir,
        args.download,
        args.dataset,
    )
    original_mapped = mapped
    original_relations = relations
    inverse_relation_ids = None
    if args.reciprocal_augmentation:
        mapped, relations, inverse_relation_ids = augment_splits_with_reciprocals(
            original_mapped,
            original_relations,
        )
        if args.tabular_train_feature_protocol != "kfold_cross_fit":
            raise ValueError(
                "reciprocal augmentation requires kfold_cross_fit so direct and "
                "inverse positives can be excluded together"
            )
        if args.complex_training_mode != "reciprocal_lcwa":
            raise ValueError(
                "reciprocal augmentation requires reciprocal_lcwa for the local "
                "ComplEx comparison"
            )
    if (
        args.tabular_context_mode != "sampled"
        and args.tabular_train_feature_protocol != "kfold_cross_fit"
    ):
        raise ValueError("query-oriented contexts require kfold_cross_fit")
    if (
        args.tabular_train_feature_protocol == "leave_one_triple_out"
        and (args.num_role_signature_bins or args.num_anchors)
    ):
        raise ValueError(
            "extended entity-memory features require kfold_cross_fit so every "
            "derived array is recomputed without the held-out fold"
        )
    # Standard filtered KGC evaluation removes every known true alternative.
    # These triples affect only the metric mask, never model inputs or features.
    evaluation_filter_splits = ("train", "valid", "test")
    all_true = all_true_triples(
        {split: mapped[split] for split in evaluation_filter_splits}
    )
    train_true = frame_triples(mapped["train"])
    num_entities = len(entities)
    num_relations = len(relations)
    stats = KGStats(
        mapped["train"],
        num_entities,
        num_relations,
        args.num_entity_clusters,
        args.seed,
        args.path_feature_mode,
    )
    stats.configure_path_features(
        mapped["train"],
        args.num_path_pattern_features,
        args.path_vocabulary_mode,
    )
    stats.configure_extended_features(
        num_role_signature_bins=args.num_role_signature_bins,
        num_anchors=args.num_anchors,
        anchor_max_distance=args.anchor_max_distance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in mapped.items():
        frame.to_csv(args.output_dir / f"{split}_triples.csv", index=False)
        if args.reciprocal_augmentation:
            original_mapped[split].to_csv(
                args.output_dir / f"original_{split}_triples.csv",
                index=False,
            )
    pd.DataFrame({"entity": entities, "entity_id": range(num_entities)}).to_csv(
        args.output_dir / "entities.csv",
        index=False,
    )
    pd.DataFrame({"relation": relations, "relation_id": range(num_relations)}).to_csv(
        args.output_dir / "relations.csv",
        index=False,
    )

    print(
        f"{args.dataset} entities={num_entities} relations={num_relations} "
        f"train={len(mapped['train'])} valid={len(mapped['valid'])} "
        f"test={len(original_mapped['test'])} "
        f"evaluation={args.evaluation_split} "
        f"reciprocal={args.reciprocal_augmentation}",
        flush=True,
    )

    complex_model = train_complex(
        mapped,
        train_true,
        num_entities,
        num_relations,
        args,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.tabular_context_mode == "sampled":
        binary_context = make_binary_context(
            mapped["train"],
            train_true,
            num_entities,
            args.tabular_negative_ratio,
            args.seed + 20_000,
        )
    else:
        binary_context = make_query_context(
            mapped["train"],
            num_entities,
            args.tabular_negative_ratio,
            args.seed + 20_000,
            one_vs_all=args.tabular_context_mode == "one_vs_all",
        )
    leave_one_triple_out_positive = False
    cross_fit_metadata = []
    if args.tabular_train_feature_protocol == "kfold_cross_fit":
        if args.tabular_context_mode == "sampled":
            binary_features, fold_ids, cross_fit_metadata = cross_fitted_feature_frame(
                mapped["train"],
                binary_context,
                num_entities,
                num_relations,
                args.num_entity_clusters,
                args.seed,
                args.path_feature_mode,
                args.path_vocabulary_mode,
                args.num_path_pattern_features,
                args.cross_fit_folds,
                cross_fit_group_column=(
                    "reciprocal_group" if args.reciprocal_augmentation else None
                ),
                num_role_signature_bins=args.num_role_signature_bins,
                num_anchors=args.num_anchors,
                anchor_max_distance=args.anchor_max_distance,
            )
        else:
            if args.reciprocal_augmentation:
                raise ValueError(
                    "query-oriented contexts and reciprocal graph augmentation "
                    "cannot be combined"
                )
            binary_features, fold_ids, cross_fit_metadata = (
                cross_fitted_query_feature_frame(
                    mapped["train"],
                    binary_context,
                    num_entities,
                    num_relations,
                    args.num_entity_clusters,
                    args.seed,
                    args.path_feature_mode,
                    args.path_vocabulary_mode,
                    args.num_path_pattern_features,
                    args.cross_fit_folds,
                    num_role_signature_bins=args.num_role_signature_bins,
                    num_anchors=args.num_anchors,
                    anchor_max_distance=args.anchor_max_distance,
                )
            )
        binary_context["feature_fold"] = fold_ids
    else:
        leave_one_triple_out_positive = (
            args.tabular_train_feature_protocol == "leave_one_triple_out"
        )
        binary_features = feature_frame(
            binary_context,
            stats,
            leave_one_triple_out_positive=leave_one_triple_out_positive,
        )
    binary_context.to_csv(args.output_dir / "tabular_train_triples.csv", index=False)
    labels = binary_context["label"].to_numpy(dtype=int)

    candidates = build_eval_candidates(
        original_mapped[args.evaluation_split],
        num_entities,
        all_true,
        inverse_relation_ids=inverse_relation_ids,
    )
    candidate_features = feature_frame(candidates, stats)
    candidates.to_csv(args.output_dir / "eval_candidates.csv", index=False)
    if not args.no_save_feature_tables:
        binary_features.to_csv(
            args.output_dir / "tabular_train_features.csv",
            index=False,
        )
        candidate_features.to_csv(
            args.output_dir / "eval_candidate_features.csv",
            index=False,
        )

    feature_groups = parse_feature_groups(args.feature_groups)
    if feature_groups is not None:
        feature_columns = columns_from_feature_groups(feature_groups, stats)
        if not feature_columns:
            raise ValueError(
                f"feature groups {feature_groups} produced no feature columns; "
                "check --num-path-pattern-features if using only path_patterns"
            )
    else:
        feature_columns = columns_from_legacy_feature_set(args.feature_set, stats)
    categorical_indices = [
        feature_columns.index(column)
        for column in CATEGORICAL_FEATURE_COLUMNS
        if column in feature_columns
    ]

    results = []
    predictions = {}

    complex_scores = score_complex_candidates(
        complex_model,
        candidates,
        device,
        args.eval_batch_size,
    )
    predictions["complex"] = complex_scores
    complex_metrics = filtered_ranking_metrics(candidates, complex_scores)
    complex_metrics["model"] = "complex"
    results.append(complex_metrics)

    if not args.skip_logistic:
        logistic_scores, _ = fit_logistic(
            binary_features[feature_columns],
            labels,
            candidate_features[feature_columns],
        )
        predictions["kg_logistic"] = logistic_scores
        logistic_metrics = filtered_ranking_metrics(candidates, logistic_scores)
        logistic_metrics["model"] = "kg_logistic"
        results.append(logistic_metrics)

    tabpfn_scores, tabpfn_model = fit_tabpfn(
        binary_features[feature_columns],
        labels,
        candidate_features[feature_columns],
        args,
        categorical_indices,
    )
    predictions["kg_tabpfn"] = tabpfn_scores
    tabpfn_metrics = filtered_ranking_metrics(candidates, tabpfn_scores)
    tabpfn_metrics["model"] = "kg_tabpfn"
    results.append(tabpfn_metrics)

    metrics = pd.DataFrame(results)
    metrics = metrics[
        ["model", "queries", "mrr", "hits_at_1", "hits_at_3", "hits_at_10", "mean_rank"]
    ]
    metrics.to_csv(args.output_dir / "ranking_metrics.csv", index=False)

    for model_name, scores in predictions.items():
        if args.no_save_candidate_scores:
            continue
        frame = candidates.copy()
        frame["score"] = scores
        frame.to_csv(args.output_dir / f"{model_name}_candidate_scores.csv", index=False)

    delta = {}
    complex_row = metrics.loc[metrics["model"] == "complex"].iloc[0]
    for model_name in ["kg_logistic", "kg_tabpfn"]:
        if model_name not in predictions:
            continue
        row = metrics.loc[metrics["model"] == model_name].iloc[0]
        for metric in ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]:
            delta[f"{model_name}_minus_complex_{metric}"] = float(
                row[metric] - complex_row[metric]
            )
    with (args.output_dir / "delta_vs_complex.json").open("w", encoding="utf-8") as file:
        json.dump(delta, file, indent=2)

    config = {
        "dataset": f"PyKEEN {args.dataset}",
        "evaluation_split": args.evaluation_split,
        "evaluation_triples": int(len(original_mapped[args.evaluation_split])),
        "evaluation_filter_splits": list(evaluation_filter_splits),
        "source_urls": DATASET_URLS[args.dataset],
        "entities": num_entities,
        "relations": num_relations,
        "direct_relations": len(original_relations),
        "reciprocal_augmentation": args.reciprocal_augmentation,
        "reciprocal_grouping": (
            "direct and inverse forms of each training fact are assigned to and "
            "excluded in the same cross-fit fold"
            if args.reciprocal_augmentation
            else None
        ),
        "head_query_protocol": (
            "rewrite (?, r, t) as (t, inverse_r, ?)"
            if args.reciprocal_augmentation
            else "score (?, r, t) directly"
        ),
        "original_train_triples": int(len(original_mapped["train"])),
        "original_valid_triples": int(len(original_mapped["valid"])),
        "original_test_triples": int(len(original_mapped["test"])),
        "train_triples": int(len(mapped["train"])),
        "valid_triples": int(len(mapped["valid"])),
        "test_triples": int(len(mapped["test"])),
        "all_true_triples": int(len(all_true)),
        "train_true_triples": int(len(train_true)),
        "tabular_negative_filter": "train_true_only",
        "tabular_context_mode": args.tabular_context_mode,
        "tabular_context_rows": int(len(binary_context)),
        "tabular_context_positive_rows": int(np.sum(labels == 1)),
        "tabular_context_negative_rows": int(np.sum(labels == 0)),
        "complex_negative_filter": (
            "train_true_only"
            if args.complex_training_mode == "fixed_negative"
            else None
        ),
        "tabular_train_feature_protocol": args.tabular_train_feature_protocol,
        "cross_fit_folds": (
            args.cross_fit_folds
            if args.tabular_train_feature_protocol == "kfold_cross_fit"
            else None
        ),
        "cross_fit_metadata": cross_fit_metadata,
        "cross_fit_scope": (
            "each context row uses statistics, clusters, and hashed paths from "
            "the training graph with its assigned fold removed"
            if args.tabular_train_feature_protocol == "kfold_cross_fit"
            else None
        ),
        "leave_one_triple_out_scope": (
            "positive context rows: graph counts and path values; path vocabulary "
            "and entity cluster assignments fixed from the training graph"
            if leave_one_triple_out_positive
            else None
        ),
        "ranking_tie_policy": "expected_uniform_tie_breaking",
        "feature_columns": feature_columns,
        "num_feature_columns": int(len(feature_columns)),
        "feature_set": args.feature_set,
        "feature_groups": feature_groups,
        "num_path_pattern_features": args.num_path_pattern_features,
        "path_feature_mode": args.path_feature_mode,
        "path_vocabulary_mode": args.path_vocabulary_mode,
        "selected_path_patterns": stats.selected_path_patterns,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "num_entity_clusters": args.num_entity_clusters,
        "num_role_signature_bins": args.num_role_signature_bins,
        "num_anchors": args.num_anchors,
        "anchor_max_distance": args.anchor_max_distance,
        "tabpfn_max_input_samples": int(
            tabpfn_model.inference_config_.MAX_NUMBER_OF_SAMPLES
        ),
        "tabpfn_max_input_features": int(
            tabpfn_model.inference_config_.MAX_NUMBER_OF_FEATURES
        ),
        "tabpfn_features_per_estimator": int(
            max(
                member.X_train.shape[1]
                for member in tabpfn_model.executor_.ensemble_members
            )
        ),
        "num_entities": num_entities,
        "num_relations": num_relations,
        "tabular_negative_ratio": args.tabular_negative_ratio,
        "complex_negative_ratio": args.complex_negative_ratio,
        "complex_training_mode": args.complex_training_mode,
        "complex_dim": args.complex_dim,
        "complex_epochs": args.complex_epochs,
        "complex_lr": args.complex_lr,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "skip_logistic": args.skip_logistic,
        "model_path": (
            str(Path(args.model_dir) / args.checkpoint_name)
            if args.checkpoint_name is not None
            else None
        ),
        "delta_vs_complex": delta,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("RANKING")
    print(metrics)
    print("DELTA")
    print(json.dumps(delta, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare self-contained ComplEx and TabPFN KG-context features."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_URLS), default="umls")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/umls_complex_vs_tabpfn/current"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--evaluation-split",
        choices=("valid", "test"),
        default="test",
        help="Split used only for ranking evaluation; representation uses train.",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-entity-clusters", type=int, default=24)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="kg_context")
    parser.add_argument(
        "--feature-groups",
        help=(
            "Comma-separated representation groups. Overrides --feature-set. "
            f"Valid groups: {', '.join(sorted(FEATURE_GROUPS))}; use all for every group."
        ),
    )
    parser.add_argument("--num-path-pattern-features", type=int, default=0)
    parser.add_argument(
        "--path-vocabulary-mode",
        choices=PATH_VOCABULARY_MODES,
        default="top_train",
        help=(
            "top_train selects frequent relation-conditioned paths from train; "
            "hashed uses deterministic target-independent columns."
        ),
    )
    parser.add_argument(
        "--path-feature-mode",
        choices=PATH_FEATURE_MODES,
        default="raw",
        help="How to encode path-pattern counts before passing them to tabular models.",
    )
    parser.add_argument("--tabular-negative-ratio", type=int, default=4)
    parser.add_argument(
        "--tabular-context-mode",
        choices=TABULAR_CONTEXT_MODES,
        default="sampled",
    )
    parser.add_argument(
        "--tabular-train-feature-protocol",
        choices=TABULAR_TRAIN_FEATURE_PROTOCOLS,
        default="in_graph",
        help=(
            "Use leave_one_triple_out to remove each positive context triple "
            "while computing its graph-derived feature values."
        ),
    )
    parser.add_argument("--cross-fit-folds", type=int, default=10)
    parser.add_argument("--num-role-signature-bins", type=int, default=0)
    parser.add_argument("--num-anchors", type=int, default=0)
    parser.add_argument("--anchor-max-distance", type=int, default=3)
    parser.add_argument(
        "--reciprocal-augmentation",
        action="store_true",
        help=(
            "Add a train-only inverse form of every fact, group direct/inverse "
            "forms in cross-fitting, and rewrite head queries with inverse relations."
        ),
    )
    parser.add_argument("--complex-negative-ratio", type=int, default=16)
    parser.add_argument(
        "--complex-training-mode",
        choices=COMPLEX_TRAINING_MODES,
        default="fixed_negative",
    )
    parser.add_argument("--complex-dim", type=int, default=200)
    parser.add_argument("--complex-epochs", type=int, default=800)
    parser.add_argument("--complex-batch-size", type=int, default=2048)
    parser.add_argument("--complex-lr", type=float, default=0.01)
    parser.add_argument("--complex-weight-decay", type=float, default=1e-7)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--tabpfn-predict-batch-size", type=int, default=20000)
    parser.add_argument("--skip-logistic", action="store_true")
    parser.add_argument("--no-save-feature-tables", action="store_true")
    parser.add_argument("--no-save-candidate-scores", action="store_true")
    parser.add_argument("--model-dir", default="/home/alessandrocagiano/data/models_ckpt/")
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
