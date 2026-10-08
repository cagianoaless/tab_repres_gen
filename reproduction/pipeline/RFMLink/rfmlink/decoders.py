"""Module 5 — Decoders.

Every decoder consumes exactly the same rows/features (same DataFrame columns)
and returns P(candidate triple is true). TabPFN 3 requirements from the spec:
  * run in conda env tfms (enforced by asserting sys.executable);
  * never fall back silently — any TabPFN failure raises;
  * log TabPFN version, exact checkpoint path, n_estimators, context size,
    fit time, prediction time.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import TABPFN3_CLASSIFIER_CKPT, TFMS_PYTHON
from .sbm_import import build_tabpfn_inference_config, positive_class_probabilities

DECODERS = ("logistic", "hgb", "xgboost", "tabpfn3")

TABPFN_N_ESTIMATORS = 8
# batching only affects throughput, not predictions; 20k caused pathological
# CPU-bound per-batch preprocessing overhead on the 15M-row codex tables;
# 100k with fit_with_cache + large contexts OOMed an 80GB A100 (KV cache +
# activation peak). 50k balances both.
TABPFN_PREDICT_BATCH = 50000


def _positive_proba(model, features):
    return positive_class_probabilities(model, features)


def fit_predict_logistic(train_X, train_y, eval_Xs, seed, categorical_indices=None):
    t0 = time.time()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=seed),
    )
    model.fit(train_X, train_y)
    fit_time = time.time() - t0
    t0 = time.time()
    preds = [_positive_proba(model, X) for X in eval_Xs]
    predict_time = time.time() - t0
    info = {"model": "logistic", "fit_time": fit_time, "predict_time": predict_time,
            "params": {"max_iter": 5000, "scaler": "standard"}}
    return preds, info


def fit_predict_hgb(train_X, train_y, eval_Xs, seed, categorical_indices=None):
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        random_state=seed,
        categorical_features=categorical_indices if categorical_indices else None,
    )
    model.fit(train_X, train_y)
    fit_time = time.time() - t0
    t0 = time.time()
    preds = [_positive_proba(model, X) for X in eval_Xs]
    predict_time = time.time() - t0
    info = {"model": "hgb", "fit_time": fit_time, "predict_time": predict_time,
            "params": {"defaults": True, "categorical_features": categorical_indices}}
    return preds, info


def fit_predict_xgboost(train_X, train_y, eval_Xs, seed, categorical_indices=None):
    import xgboost as xgb

    t0 = time.time()
    model = xgb.XGBClassifier(
        random_state=seed,
        n_estimators=300,
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(np.asarray(train_X, dtype=np.float32), train_y)
    fit_time = time.time() - t0
    t0 = time.time()
    preds = [
        model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1] for X in eval_Xs
    ]
    predict_time = time.time() - t0
    info = {"model": "xgboost", "version": xgb.__version__,
            "fit_time": fit_time, "predict_time": predict_time,
            "params": {"n_estimators": 300, "tree_method": "hist"}}
    return preds, info


def fit_predict_tabpfn3(train_X, train_y, eval_Xs, seed, categorical_indices=None,
                        device="cuda", n_estimators=TABPFN_N_ESTIMATORS,
                        fit_mode="fit_preprocessors", checkpoint_override=None):
    """fit_mode: the completed main grids used the library default
    'fit_preprocessors'; NEW experiment phases (reranking) pass
    'fit_with_cache' per official guidance (one fit, many predict batches,
    high GPU memory). Do not mix modes within one comparison.

    checkpoint_override: ONLY for the explicit, documented backbone ablation
    (E4) — never a silent fallback; the checkpoint used is always logged."""
    if str(TFMS_PYTHON) not in sys.executable and "tfms" not in sys.executable:
        raise RuntimeError(
            f"TabPFN 3 must run in the tfms conda environment; current interpreter: {sys.executable}"
        )
    import tabpfn
    from tabpfn import TabPFNClassifier

    checkpoint = (Path(checkpoint_override) if checkpoint_override
                  else TABPFN3_CLASSIFIER_CKPT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"TabPFN 3 checkpoint missing: {checkpoint}")

    train_arr = np.asarray(train_X, dtype=np.float32)
    model_args = {
        "model_path": checkpoint,
        "device": device,
        "n_estimators": n_estimators,
        "random_state": seed,
        "fit_mode": fit_mode,
    }
    if len(train_arr) > 10000:
        model_args["ignore_pretraining_limits"] = True
    if categorical_indices:
        max_card = max(
            int(np.unique(np.asarray(train_X)[:, i]).size) for i in categorical_indices
        )
        model_args["categorical_features_indices"] = categorical_indices
        model_args["inference_config"] = build_tabpfn_inference_config(max(max_card, 2))

    t0 = time.time()
    model = TabPFNClassifier(**model_args)
    model.fit(train_arr, np.asarray(train_y))
    fit_time = time.time() - t0

    t0 = time.time()
    preds = []
    for X in eval_Xs:
        X = np.asarray(X, dtype=np.float32)
        chunks = [
            _positive_proba(model, X[s : s + TABPFN_PREDICT_BATCH])
            for s in range(0, len(X), TABPFN_PREDICT_BATCH)
        ]
        preds.append(np.concatenate(chunks))
    predict_time = time.time() - t0

    # free the executor/KV-cache: successive runs in one process otherwise
    # accumulate GPU memory until OOM (observed with fit_with_cache)
    del model
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()

    info = {
        "model": "tabpfn3",
        "conda_environment": "tfms",
        "interpreter": sys.executable,
        "tabpfn_version": tabpfn.__version__,
        "checkpoint_path": str(checkpoint),
        "n_estimators": n_estimators,
        "fit_mode": fit_mode,
        "context_size": int(len(train_arr)),
        "fit_time": fit_time,
        "predict_time": predict_time,
    }
    return preds, info


DECODER_FUNCS = {
    "logistic": fit_predict_logistic,
    "hgb": fit_predict_hgb,
    "xgboost": fit_predict_xgboost,
    "tabpfn3": fit_predict_tabpfn3,
}
