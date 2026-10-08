"""Module 3 — Explicit KG structural features (train-graph-only).

Wraps the audited KGStats/feature_frame from the parent project (see
results/audit/sbm_tfm_audit.json). All statistics are computed from train
triples only. Features for train positives are computed leave-one-triple-out
(the triple is removed from the graph statistics while its own features are
computed, then restored — audited + unit-tested in the parent project).

Main feature config ``explicit_v1`` (no raw entity ids):
    relation_id (categorical), head/tail total out/in degree,
    head-relation out degree, tail-relation in degree,
    relation frequency / functionality / inverse functionality,
    pair evidence (other relations on (h,t)), reverse pair evidence,
    two-hop count, relation-conditioned two-hop start/end counts,
    reverse two-hop count, train-only KMeans entity clusters
    (head_cluster, tail_cluster, same_entity_cluster, cluster_relation_density).

Diagnostic-only config ``explicit_ids_diagnostic`` adds raw head_id/tail_id.

Output layout (spec-mandated), per dataset and feature config:
    features/<dataset>/<feature_config>/train_features.parquet
    features/<dataset>/<feature_config>/valid_features.parquet
    features/<dataset>/<feature_config>/test_features.parquet
    features/<dataset>/<feature_config>/feature_names.json
    features/<dataset>/<feature_config>/feature_config.json
    features/<dataset>/<feature_config>/leakage_audit.json

train_features.parquet rows: every train triple (label=1, LOTO protocol) plus
every distinct negative triple referenced by any materialized context of the
dataset (label=0, in-graph protocol). Runs look context rows up by (h, r, t).
valid/test_features.parquet rows: the full filtered-ranking candidate tables
(both query sides, num_entities candidates per query) with candidate metadata.
"""

import json

import numpy as np
import pandas as pd

from .paths import CONTEXTS_DIR, FEATURES_DIR
from .sbm_import import KGStats, feature_frame
from .evaluator import make_candidates

EXPLICIT_FEATURE_COLUMNS = [
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
ID_COLUMNS = ["head_id", "tail_id"]
CATEGORICAL_COLUMNS = ["relation_id", "head_cluster", "tail_cluster", "same_entity_cluster"]

FEATURE_CONFIGS = {
    "explicit_v1": {
        "columns": EXPLICIT_FEATURE_COLUMNS,
        "uses_raw_entity_ids": False,
    },
    "explicit_ids_diagnostic": {
        "columns": ID_COLUMNS + EXPLICIT_FEATURE_COLUMNS,
        "uses_raw_entity_ids": True,
    },
}

NUM_ENTITY_CLUSTERS = 24
STATS_SEED = 7


def build_stats(train_triples, num_entities, num_relations):
    train_frame = pd.DataFrame(
        np.asarray(train_triples, dtype=np.int64), columns=["h", "r", "t"]
    )
    stats = KGStats(
        train_frame,
        num_entities,
        num_relations,
        NUM_ENTITY_CLUSTERS,
        STATS_SEED,
        path_feature_mode="raw",
    )
    return stats


def features_for_frame(frame, stats, loto_positives=False):
    """frame: columns h, r, t (+optional label). Returns feature DataFrame."""
    return feature_frame(frame, stats, leave_one_triple_out_positive=loto_positives)


def _collect_context_negatives(dataset):
    """All distinct negative triples referenced by materialized contexts."""
    negatives = set()
    base = CONTEXTS_DIR / dataset
    if not base.exists():
        return negatives
    for config_path in sorted(base.glob("*/*/context_config.json")):
        ctx_dir = config_path.parent
        triples = np.load(ctx_dir / "context_triples.npy")
        labels = np.load(ctx_dir / "context_labels.npy")
        for triple in triples[labels == 0].tolist():
            negatives.add(tuple(triple))
    return negatives


def export_features(dataset, unified, feature_config="explicit_v1"):
    cfg = FEATURE_CONFIGS[feature_config]
    columns = cfg["columns"]
    num_entities = len(unified["entities"])
    num_relations = len(unified["relations"])
    stats = build_stats(unified["train_triples"], num_entities, num_relations)

    out_dir = FEATURES_DIR / dataset / feature_config
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- train table: LOTO positives + in-graph context negatives ---
    train_frame = pd.DataFrame(unified["train_triples"], columns=["h", "r", "t"])
    train_frame["label"] = 1
    pos_features = features_for_frame(train_frame, stats, loto_positives=True)
    pos_features["feature_protocol"] = "leave_one_triple_out"

    negatives = sorted(_collect_context_negatives(dataset))
    neg_frame = pd.DataFrame(negatives, columns=["h", "r", "t"])
    neg_frame["label"] = 0
    neg_features = (
        features_for_frame(neg_frame, stats, loto_positives=False)
        if len(neg_frame)
        else pd.DataFrame(columns=pos_features.columns)
    )
    neg_features["feature_protocol"] = "in_graph_train_only"

    train_table = pd.concat(
        [
            pd.concat([train_frame.reset_index(drop=True), pos_features[columns + ["feature_protocol"]]], axis=1),
            pd.concat([neg_frame.reset_index(drop=True), neg_features[columns + ["feature_protocol"]]], axis=1),
        ],
        ignore_index=True,
    )
    train_table.to_parquet(out_dir / "train_features.parquet", index=False)

    # --- valid/test candidate tables ---
    for split in ("valid", "test"):
        candidates = make_candidates(
            unified[f"{split}_triples"], num_entities, unified["all_true_triples"]
        )
        cand_features = features_for_frame(candidates, stats, loto_positives=False)
        table = pd.concat(
            [
                candidates.reset_index(drop=True),
                cand_features[[c for c in columns if c not in candidates.columns]].reset_index(drop=True),
            ],
            axis=1,
        )
        table.to_parquet(out_dir / f"{split}_features.parquet", index=False)

    with (out_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"feature_columns": columns, "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in columns]},
            f,
            indent=2,
        )
    with (out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "feature_config": feature_config,
                "num_entity_clusters": NUM_ENTITY_CLUSTERS,
                "stats_seed": STATS_SEED,
                "num_feature_columns": len(columns),
                "train_rows": int(len(train_table)),
                "source": "parent sbm_tfm KGStats/feature_frame (audited); wrapper rfmlink/features.py",
                "train_positive_protocol": "leave_one_triple_out",
                "negative_protocol": "in_graph_train_only",
            },
            f,
            indent=2,
        )
    with (out_dir / "leakage_audit.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_only_features": True,
                "uses_valid_triples": False,
                "uses_test_triples": False,
                "uses_raw_entity_ids": cfg["uses_raw_entity_ids"],
                "context_positive_leave_one_out": True,
                "known_compromises": [
                    "entity cluster assignments (train-only KMeans) are NOT recomputed inside the leave-one-triple-out loop; only count/degree/path statistics are. Recomputing KMeans per left-out triple is prohibitively expensive; the parent project's kfold_cross_fit protocol covers this and is available as a robustness check",
                    "negative context rows use in-graph train statistics (no triple to leave out; they are not in the graph)"
                ],
                "risk_level": "low",
            },
            f,
            indent=2,
        )
    return out_dir


def load_feature_table(dataset, split, feature_config="explicit_v1"):
    out_dir = FEATURES_DIR / dataset / feature_config
    table = pd.read_parquet(out_dir / f"{split}_features.parquet")
    with (out_dir / "feature_names.json").open(encoding="utf-8") as f:
        names = json.load(f)
    return table, names
