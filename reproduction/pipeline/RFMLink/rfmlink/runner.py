"""Module 4 (row builder) + run executor.

A "run" = (dataset, representation, decoder, context_size, seed):
  1. load the context triples (Module 1 output);
  2. look their feature rows up in the canonical train feature table
     (Module 3 output; LOTO for positives) and, for ULTRA representations,
     gather the aligned ULTRA score/representation exports (Module 2 output);
  3. fit the decoder on the context rows only;
  4. score every valid/test filtered-ranking candidate with the same
     representation;
  5. write results/runs/<run_id>/{config,protocol,metrics,runtime,
     leakage_audit}.json + predictions_{valid,test}.parquet.

Representation variants (Module 4):
    explicit_only, ultra_score_only, ultra_repr_only,
    explicit_plus_ultra_score, explicit_plus_ultra_repr,
    explicit_plus_ultra_repr_plus_score
"""

import json
import time

import numpy as np
import pandas as pd

from .contexts import load_context
from .decoders import DECODER_FUNCS
from .evaluator import binary_metrics, ranking_metrics
from .features import CATEGORICAL_COLUMNS, load_feature_table
from .paths import RESULTS_DIR, ULTRA_EXPORTS_DIR
from .query_subset import SUBSET_DATASETS, SUBSET_SEED, SUBSET_SIZE, candidate_row_mask

REPRESENTATIONS = (
    "explicit_only",
    "ultra_score_only",
    "ultra_repr_only",
    "explicit_plus_ultra_score",
    "explicit_plus_ultra_repr",
    "explicit_plus_ultra_repr_plus_score",
)

ULTRA_CONFIG = "ultra_50g_zeroshot_traingraph"


class DatasetAssets:
    """Feature tables + optional ULTRA exports for one dataset, loaded once."""

    def __init__(self, dataset, feature_config="explicit_v1", need_ultra=False):
        self.dataset = dataset
        self.feature_config = feature_config
        self.train_table, self.feature_names = load_feature_table(
            dataset, "train", feature_config
        )
        self.valid_table, _ = load_feature_table(dataset, "valid", feature_config)
        self.test_table, _ = load_feature_table(dataset, "test", feature_config)
        self.explicit_columns = self.feature_names["feature_columns"]
        self.categorical_columns = self.feature_names["categorical_columns"]

        # index for context row lookup: (h, r, t, label) -> train_table row
        key = list(zip(self.train_table.h, self.train_table.r, self.train_table.t,
                       self.train_table.label))
        self.train_row_by_key = {k: i for i, k in enumerate(key)}

        self.ultra = None
        if need_ultra:
            base = ULTRA_EXPORTS_DIR / dataset / ULTRA_CONFIG
            # mmap: repr arrays are up to 3.8GB/split (codex-small); loading
            # them eagerly alongside the concatenated feature matrices caused
            # an OOM kill with parallel workers (see results/STATUS.json)
            self.ultra = {
                split: {
                    "scores": np.load(base / f"{split}_scores.npy"),
                    "repr": np.load(base / f"{split}_repr.npy", mmap_mode="r"),
                }
                for split in ("train", "valid", "test")
            }
            with (base / "ultra_export_config.json").open(encoding="utf-8") as f:
                self.ultra_export_config = json.load(f)
            for split, table in (
                ("train", self.train_table),
                ("valid", self.valid_table),
                ("test", self.test_table),
            ):
                if len(self.ultra[split]["scores"]) != len(table):
                    raise ValueError(
                        f"{self.dataset}/{split}: ULTRA export rows "
                        f"{len(self.ultra[split]['scores'])} != feature rows {len(table)}"
                    )

    def context_row_indices(self, triples, labels):
        indices = []
        for (h, r, t), label in zip(triples.tolist(), labels.tolist()):
            key = (h, r, t, label)
            if key not in self.train_row_by_key:
                raise KeyError(f"context triple {key} missing from train feature table")
            indices.append(self.train_row_by_key[key])
        return np.asarray(indices, dtype=np.int64)

    def representation_matrix(self, representation, split, row_indices=None):
        """Return (X, feature_names, categorical_indices) for rows of a split."""
        table = {"train": self.train_table, "valid": self.valid_table,
                 "test": self.test_table}[split]
        if row_indices is None:
            row_indices = np.arange(len(table))
        parts, names = [], []

        if representation.startswith("explicit"):
            X = table.iloc[row_indices][self.explicit_columns].to_numpy(dtype=np.float32)
            parts.append(X)
            names.extend(self.explicit_columns)

        if "ultra" in representation:
            if self.ultra is None:
                raise RuntimeError("ULTRA exports not loaded for this run")
            exports = self.ultra[split]
            if "repr" in representation:
                parts.append(exports["repr"][row_indices].astype(np.float32))
                names.extend(
                    f"ultra_z{i}" for i in range(exports["repr"].shape[1])
                )
            if "score" in representation:
                parts.append(
                    exports["scores"][row_indices].astype(np.float32).reshape(-1, 1)
                )
                names.append("ultra_score")

        X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        categorical_indices = [
            names.index(c) for c in self.categorical_columns if c in names
        ]
        return X, names, categorical_indices


def run_id_for(dataset, representation, decoder, size_label, seed):
    return f"{dataset}__{representation}__{decoder}__{size_label}__s{seed}"


def execute_run(
    assets,
    representation,
    decoder,
    size_label,
    seed,
    device="cuda",
    save_predictions=True,
    overwrite=False,
):
    dataset = assets.dataset
    run_id = run_id_for(dataset, representation, decoder, size_label, seed)
    out_dir = RESULTS_DIR / "runs" / run_id
    if out_dir.exists() and not overwrite and (out_dir / "metrics.json").exists():
        return run_id, "skipped_existing"
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    triples, labels, context_config = load_context(dataset, seed, size_label)
    row_indices = assets.context_row_indices(triples, labels)
    train_X, feature_names, categorical_indices = assets.representation_matrix(
        representation, "train", row_indices
    )
    # subset datasets: evaluate on the fixed seeded query subset (identical
    # across ALL compared systems; see rfmlink/query_subset.py)
    subset_mode = dataset in SUBSET_DATASETS
    eval_tables = {}
    for split in ("valid", "test"):
        table = {"valid": assets.valid_table, "test": assets.test_table}[split]
        if subset_mode:
            n_queries = table["query_index"].nunique()
            num_entities = len(table) // n_queries
            mask, _ = candidate_row_mask(n_queries // 2, num_entities)
            eval_tables[split] = (
                table.loc[mask].reset_index(drop=True),
                np.flatnonzero(mask),
            )
        else:
            eval_tables[split] = (table, None)

    valid_X, _, _ = assets.representation_matrix(
        representation, "valid", eval_tables["valid"][1]
    )
    test_X, _, _ = assets.representation_matrix(
        representation, "test", eval_tables["test"][1]
    )

    fn = DECODER_FUNCS[decoder]
    kwargs = {"categorical_indices": categorical_indices}
    if decoder == "tabpfn3":
        kwargs["device"] = device
    (valid_scores, test_scores), decoder_info = fn(
        train_X, labels, [valid_X, test_X], seed, **kwargs
    )

    metrics = {}
    for split, scores in (("valid", valid_scores), ("test", test_scores)):
        table = eval_tables[split][0]
        rank = ranking_metrics(table, scores)
        unfiltered = table["is_filtered"].to_numpy() == 0
        binary = binary_metrics(
            table.loc[unfiltered, "is_target"].to_numpy(), scores[unfiltered]
        )
        metrics[split] = {**rank, "binary_on_unfiltered_candidates": binary}
        if subset_mode:
            metrics[split]["evaluation_subset"] = {
                "subset_triples_per_split": SUBSET_SIZE,
                "subset_seed": SUBSET_SEED,
                "note": "fixed seeded query subset, identical across all compared systems",
            }

    if save_predictions:
        for split, scores in (("valid", valid_scores), ("test", test_scores)):
            table = eval_tables[split][0]
            frame = table[["query_index", "side", "h", "r", "t", "is_target",
                           "is_filtered"]].copy()
            # float64: stored predictions must reproduce metrics.json exactly
            # (float32 rounding perturbs ties by ~1e-4 MRR; caught by verify_runs)
            frame["score"] = scores.astype(np.float64)
            frame.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)

    config = {
        "run_id": run_id,
        "dataset": dataset,
        "representation": representation,
        "decoder": decoder,
        "context_size": size_label,
        "seed": seed,
        "feature_config": assets.feature_config,
        "num_features": len(feature_names),
        "feature_names_file": str(
            (RESULTS_DIR / "runs" / run_id / "feature_names.json")
        ),
        "context_config": context_config,
        "ultra_config": ULTRA_CONFIG if "ultra" in representation else None,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with (out_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names,
                   "categorical_indices": categorical_indices}, f, indent=2)
    with (out_dir / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation": "filtered ranking, both query sides, num_entities candidates/query",
                "query_subset": (
                    {"triples_per_split": SUBSET_SIZE, "seed": SUBSET_SEED,
                     "rationale": "documented sampled-query evaluation for expensive dataset; identical subset across all compared systems"}
                    if subset_mode else None
                ),
                "tie_policy": "expected_uniform_tie_breaking",
                "filter_set": "train+valid+test (evaluator only)",
                "decoder_training_rows": int(len(train_X)),
                "decoder_training_positives": int(labels.sum()),
                "same_rows_across_decoders": True,
                "binary_metrics_scope": "unfiltered candidates only (label = is_target)",
            },
            f,
            indent=2,
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "runtime.json").open("w", encoding="utf-8") as f:
        json.dump({**decoder_info, "wall_time": time.time() - wall_start}, f, indent=2)
    with (out_dir / "leakage_audit.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_leakage_audit": f"features/{dataset}/{assets.feature_config}/leakage_audit.json",
                "context_leakage_notes": context_config["leakage_notes"],
                "context_negative_filtering": context_config["negative_filtering_mode"],
                "train_positive_feature_protocol": "leave_one_triple_out",
                "ultra_protocol": (
                    assets.ultra_export_config.get("leakage_protocol")
                    if "ultra" in representation and assets.ultra is not None
                    else None
                ),
                "uses_raw_entity_ids": False,
            },
            f,
            indent=2,
        )
    return run_id, metrics
