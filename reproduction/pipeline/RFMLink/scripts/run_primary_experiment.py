"""PRE-REGISTERED primary experiment (results/summary/primary_experiment_registration.json).

Implements every review fix in one frozen protocol:
  * invariant explicit features (NO raw head_cluster/tail_cluster ids);
  * context positives cross-fitted (10 folds; invariant columns make fold
    alignment moot);
  * DIRECTION-BALANCED contexts: every context triple contributes a direct
    row (ULTRA repr conditioned on (h, r)) and an inverse row (conditioned on
    (t, r+R)); explicit features are orientation-symmetric and shared. The
    direct-only variant runs as the registered ablation;
  * decoders with semantically correct encodings on the same underlying
    matrix: tabpfn_v3 / tabpfn_v2 (categorical indices, fit_with_cache),
    logistic_onehot (one-hot relation_id), hgb (categorical_features),
    xgboost_cat (enable_categorical);
  * 10 seeds (101-110), k = 200+200;
  * head-side and tail-side metrics reported separately plus combined;
  * codex-small on the fixed 500-triple query subset, others full.

NOTE on fit_with_cache: modes differ numerically (max proba diff ~0.017,
measured); all TabPFN runs in THIS experiment use fit_with_cache uniformly,
so within-experiment comparisons are mode-consistent.

Outputs: results/primary_runs/<run_id>/{config,metrics,runtime}.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfmlink.contexts import load_context
from rfmlink.decoders import (
    TABPFN_PREDICT_BATCH,
    fit_predict_hgb,
    fit_predict_tabpfn3,
)
from rfmlink.evaluator import ranking_metrics
from rfmlink.features import EXPLICIT_FEATURE_COLUMNS, NUM_ENTITY_CLUSTERS, build_stats
from rfmlink.data_prep import load_processed
from rfmlink.paths import MODELS_CKPT, RESULTS_DIR, ULTRA_ROOT
from rfmlink.query_subset import SUBSET_DATASETS, candidate_row_mask
from rfmlink.runner import DatasetAssets
from rfmlink.ultra_wrapper import build_train_graph, load_ultra_model, score_triples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from run_umls_complex_vs_tabpfn import cross_fitted_feature_frame  # noqa: E402

INVARIANT_COLUMNS = [c for c in EXPLICIT_FEATURE_COLUMNS
                     if c not in ("head_cluster", "tail_cluster")]
CATEGORICAL = ["relation_id", "same_entity_cluster"]
DATASETS = ["umls", "kinships", "nations", "codex-small"]
SEEDS = list(range(101, 111))
SIZE = "200pos_200neg"
FOLDS = 10
OUT = RESULTS_DIR / "primary_runs"

TABPFN_CKPTS = {
    "tabpfn_v3": MODELS_CKPT / "tabpfn-v3-classifier-v3_default.ckpt",
    "tabpfn_v2": MODELS_CKPT / "tabpfn-v2-classifier-v2_default.ckpt",
}


def logistic_onehot(train_X_df, train_y, eval_X_dfs, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in train_X_df.columns if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
    t0 = time.time()
    model.fit(train_X_df, train_y)
    fit_t = time.time() - t0
    t0 = time.time()
    preds = [model.predict_proba(X)[:, list(model.classes_).index(1)]
             for X in eval_X_dfs]
    return preds, {"model": "logistic_onehot", "fit_time": fit_t,
                   "predict_time": time.time() - t0,
                   "encoding": "one-hot categoricals + standardized numerics"}


def xgboost_cat(train_X_df, train_y, eval_X_dfs, seed, category_domains=None):
    import xgboost as xgb

    def cast(df):
        df = df.copy()
        for c in CATEGORICAL:
            # xgboost requires integer/string category codes, not floats;
            # declare the FULL domain so predict-time categories absent from
            # the small context (e.g. codex relation 39) are legal
            domain = (category_domains or {}).get(c)
            values = df[c].astype(np.int64)
            df[c] = (pd.Categorical(values, categories=domain)
                     if domain is not None else values.astype("category"))
        return df

    t0 = time.time()
    model = xgb.XGBClassifier(random_state=seed, n_estimators=300,
                              eval_metric="logloss", tree_method="hist",
                              enable_categorical=True)
    model.fit(cast(train_X_df), train_y)
    fit_t = time.time() - t0
    t0 = time.time()
    preds = [model.predict_proba(cast(X))[:, 1] for X in eval_X_dfs]
    return preds, {"model": "xgboost_cat", "fit_time": fit_t,
                   "predict_time": time.time() - t0,
                   "encoding": "native categorical (enable_categorical)"}


def side_metrics(table, scores):
    out = {"combined": ranking_metrics(table, scores)}
    for side in ("tail", "head"):
        mask = (table["side"] == side).to_numpy()
        out[side] = ranking_metrics(table.loc[mask].reset_index(drop=True),
                                    scores[mask])
    return out


def per_query_rr(table, scores):
    """Per-query reciprocal ranks (expected tie policy) + sides, in stable
    query order — required for the registered query-level paired bootstrap."""
    from rfmlink.sbm_import import tied_rank_positions

    frame = table[["query_index", "side", "is_target", "is_filtered"]].copy()
    frame["score"] = scores
    rrs, sides, qids = [], [], []
    for qid, group in frame.groupby("query_index", sort=True):
        target_score = float(group.loc[group["is_target"] == 1, "score"].iloc[0])
        valid = group.loc[group["is_filtered"] == 0]
        positions = tied_rank_positions(
            valid["score"].to_numpy(dtype=np.float64), target_score)
        rrs.append(float(np.mean(1.0 / positions)))
        sides.append(group["side"].iloc[0])
        qids.append(qid)
    return (np.asarray(qids), np.asarray(sides, dtype="U4"),
            np.asarray(rrs, dtype=np.float64))


class PrimaryAssets:
    def __init__(self, dataset):
        self.dataset = dataset
        self.unified = load_processed(dataset)
        self.num_entities = len(self.unified["entities"])
        self.num_relations = len(self.unified["relations"])
        self.assets = DatasetAssets(dataset, need_ultra=True)
        self.train_frame = pd.DataFrame(self.unified["train_triples"],
                                        columns=["h", "r", "t"])
        self.stats = build_stats(self.unified["train_triples"],
                                 self.num_entities, self.num_relations)
        self.model = load_ultra_model(ULTRA_ROOT / "ckpts" / "ultra_50g.pth",
                                      device="cuda")
        self.data = build_train_graph(self.unified["train_triples"],
                                      self.num_entities, self.num_relations,
                                      device="cuda")
        # eval matrices: invariant explicit + ultra repr + score (aligned)
        self.eval_tables, self.eval_X = {}, {}
        for split in ("valid", "test"):
            table = {"valid": self.assets.valid_table,
                     "test": self.assets.test_table}[split]
            if dataset in SUBSET_DATASETS:
                n_queries = table["query_index"].nunique()
                mask, _ = candidate_row_mask(n_queries // 2,
                                             len(table) // n_queries)
                rows = np.flatnonzero(mask)
                table = table.loc[mask].reset_index(drop=True)
            else:
                rows = None
            exports = self.assets.ultra[split]
            r = np.arange(len(exports["scores"])) if rows is None else rows
            X = np.concatenate([
                table[INVARIANT_COLUMNS].to_numpy(dtype=np.float32),
                exports["repr"][r].astype(np.float32),
                exports["scores"][r].astype(np.float32).reshape(-1, 1)], axis=1)
            self.eval_tables[split] = table
            self.eval_X[split] = X
        self.feature_names = (INVARIANT_COLUMNS
                              + [f"ultra_z{i}" for i in range(128)]
                              + ["ultra_score"])

    def context_matrix(self, seed, balanced, equal_budget=False):
        """Cross-fitted invariant explicit + direct (and inverse) ULTRA rows.

        equal_budget (review-2 finding 5): 'balanced' doubles rows (each
        triple in both orientations, 800 rows) and so confounds direction
        coverage with row count. balanced_equal keeps the 400-row budget: a
        seeded half of the context triples contribute their DIRECT row, the
        complement their INVERSE row.
        """
        triples, labels, _ = load_context(self.dataset, seed, SIZE)
        return self.context_matrix_from_rows(
            triples, labels, seed, balanced=balanced, equal_budget=equal_budget)

    def context_matrix_from_rows(
        self,
        triples,
        labels,
        seed,
        balanced,
        equal_budget=False,
        return_manifest=False,
    ):
        """Build the primary context representation for explicitly fixed rows.

        ``return_manifest`` is used by causal context interventions to prove
        that only the decoder labels changed. It does not alter the legacy
        return value or the registered primary runner.
        """
        triples = np.asarray(triples, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        if triples.shape != (len(labels), 3):
            raise ValueError("context triples and labels are misaligned")
        context = pd.DataFrame(triples, columns=["h", "r", "t"])
        context["label"] = labels
        features, _, _ = cross_fitted_feature_frame(
            self.train_frame, context, self.num_entities, self.num_relations,
            NUM_ENTITY_CLUSTERS, seed, "raw", "hashed", 0, FOLDS)
        explicit = features[INVARIANT_COLUMNS].to_numpy(dtype=np.float32)

        pos = labels == 1
        def ultra_rows(t_arr, remove_pos, removal=None):
            s = np.empty(len(t_arr), dtype=np.float32)
            z = np.empty((len(t_arr), 128), dtype=np.float32)
            a, b = score_triples(self.model, self.data, t_arr[pos], batch_size=8,
                                 capture_representations=True,
                                 remove_direct_edges=remove_pos,
                                 removal_triples=(removal[pos]
                                                  if removal is not None else None))
            s[pos], z[pos] = a, b
            a, b = score_triples(self.model, self.data, t_arr[~pos], batch_size=64,
                                 capture_representations=True,
                                 remove_direct_edges=False)
            s[~pos], z[~pos] = a, b
            return s, z

        s_dir, z_dir = ultra_rows(triples, True)
        X_dir = np.concatenate([explicit, z_dir, s_dir.reshape(-1, 1)], axis=1)
        if not balanced:
            source_indices = np.arange(len(labels), dtype=np.int64)
            orientations = np.full(len(labels), "direct", dtype="U7")
            result = (X_dir, labels)
            if return_manifest:
                return (*result, source_indices, orientations)
            return result

        inv = np.stack([triples[:, 2],
                        triples[:, 1] + self.num_relations,
                        triples[:, 0]], axis=1)
        # removal_triples must be the DIRECT orientation: passing the
        # inverse form to remove_easy_edges leaves the direct edge in the
        # graph (leakage; external review 2026-07-12, amendment #2)
        s_inv, z_inv = ultra_rows(inv, True, removal=triples)
        X_inv = np.concatenate([explicit, z_inv, s_inv.reshape(-1, 1)], axis=1)

        if equal_budget:
            rng = np.random.default_rng(seed + 55_000)
            direct_half = rng.random(len(labels)) < 0.5
            X = np.concatenate([X_dir[direct_half], X_inv[~direct_half]])
            y = np.concatenate([labels[direct_half], labels[~direct_half]])
            source_indices = np.concatenate([
                np.flatnonzero(direct_half), np.flatnonzero(~direct_half)])
            orientations = np.concatenate([
                np.full(np.sum(direct_half), "direct", dtype="U7"),
                np.full(np.sum(~direct_half), "inverse", dtype="U7"),
            ])
            result = (X, y)
            if return_manifest:
                return (*result, source_indices, orientations)
            return result
        result = (np.concatenate([X_dir, X_inv]), np.concatenate([labels, labels]))
        if return_manifest:
            source_indices = np.concatenate([
                np.arange(len(labels), dtype=np.int64),
                np.arange(len(labels), dtype=np.int64),
            ])
            orientations = np.concatenate([
                np.full(len(labels), "direct", dtype="U7"),
                np.full(len(labels), "inverse", dtype="U7"),
            ])
            return (*result, source_indices, orientations)
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat_idx = [INVARIANT_COLUMNS.index(c) for c in CATEGORICAL]
    for dataset in DATASETS:
        pa = None
        for seed in SEEDS:
            for variant in ("balanced", "direct_only", "balanced_equal"):
                ctx_cache = {}
                for decoder in ("tabpfn_v3", "tabpfn_v2", "logistic_onehot",
                                "hgb", "xgboost_cat"):
                    # the equal-budget ablation targets the direction effect,
                    # not the decoder family (amendment 4): primary decoder +
                    # linear contrast only
                    if (variant == "balanced_equal"
                            and decoder not in ("tabpfn_v3", "logistic_onehot")):
                        continue
                    run_id = f"{dataset}__primary_{variant}__{decoder}__s{seed}"
                    run_dir = OUT / run_id
                    if (run_dir / "metrics.json").exists():
                        continue
                    if pa is None:
                        pa = PrimaryAssets(dataset)
                    if variant not in ctx_cache:
                        ctx_cache[variant] = pa.context_matrix(
                            seed, balanced=variant.startswith("balanced"),
                            equal_budget=(variant == "balanced_equal"))
                    X, y = ctx_cache[variant]
                    eval_Xs = [pa.eval_X["valid"], pa.eval_X["test"]]
                    t0 = time.time()
                    if decoder.startswith("tabpfn"):
                        preds, info = fit_predict_tabpfn3(
                            X, y, eval_Xs, seed, categorical_indices=cat_idx,
                            fit_mode="fit_with_cache",
                            checkpoint_override=str(TABPFN_CKPTS[decoder]))
                    elif decoder == "hgb":
                        preds, info = fit_predict_hgb(
                            X, y, eval_Xs, seed, categorical_indices=cat_idx)
                    else:
                        Xdf = pd.DataFrame(X, columns=pa.feature_names)
                        eval_dfs = [pd.DataFrame(e, columns=pa.feature_names)
                                    for e in eval_Xs]
                        if decoder == "logistic_onehot":
                            preds, info = logistic_onehot(Xdf, y, eval_dfs, seed)
                        else:
                            domains = {"relation_id": list(range(pa.num_relations)),
                                       "same_entity_cluster": [0, 1]}
                            preds, info = xgboost_cat(Xdf, y, eval_dfs, seed,
                                                      category_domains=domains)
                    metrics = {}
                    rr_payload = {}
                    for split, scores in zip(("valid", "test"), preds):
                        scores = np.asarray(scores)
                        metrics[split] = side_metrics(pa.eval_tables[split], scores)
                        qids, sides, rrs = per_query_rr(pa.eval_tables[split], scores)
                        rr_payload[f"{split}_qids"] = qids
                        rr_payload[f"{split}_sides"] = sides
                        rr_payload[f"{split}_rr"] = rrs
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(run_dir / "query_rr.npz", **rr_payload)
                    (run_dir / "config.json").write_text(json.dumps({
                        "run_id": run_id, "experiment": "primary_registered",
                        "registration": "results/summary/primary_experiment_registration.json",
                        "dataset": dataset, "variant": variant,
                        "decoder": decoder, "seed": seed, "context_size": SIZE,
                        "context_rows": int(len(y)),
                        "features": "invariant explicit + ultra repr + score",
                        "context_protocol": "kfold_cross_fit(10) positives, in-graph negatives; ULTRA with target-edge removal for positives (both orientations)",
                    }, indent=2))
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    (run_dir / "runtime.json").write_text(json.dumps(
                        {**info, "wall_time": time.time() - t0}, indent=2))
                    print(f"OK {run_id} test_mrr={metrics['test']['combined']['mrr']:.4f} "
                          f"(head {metrics['test']['head']['mrr']:.4f} / "
                          f"tail {metrics['test']['tail']['mrr']:.4f}) "
                          f"({time.time()-t0:.0f}s)", flush=True)
        del pa
        import torch
        torch.cuda.empty_cache()
    print("PRIMARY EXPERIMENT DONE", flush=True)


if __name__ == "__main__":
    main()
