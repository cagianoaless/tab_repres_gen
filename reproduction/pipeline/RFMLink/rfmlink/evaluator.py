"""Filtered ranking evaluation (reused from the audited parent project).

Reused symbols (see results/audit/sbm_tfm_audit.json):
  * build_eval_candidates — head+tail queries, num_entities candidates each,
    is_filtered = (candidate in all_true(train+valid+test)) and not target.
  * filtered_ranking_metrics — MRR / Hits@{1,3,10} / Mean Rank with expected
    uniform tie-breaking.

This module adds:
  * make_candidates: candidates for an arbitrary split from the unified format;
  * binary_metrics: secondary AUC/AP/Brier/LogLoss on candidate scores;
  * run_sanity_tests: the spec-mandated evaluator checks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .sbm_import import (
    build_eval_candidates,
    filtered_ranking_metrics,
    tied_rank_positions,
)

EVALUATOR_CONFIG = {
    "protocol": "filtered ranking, both query sides (tail: (h,r,?), head: (?,r,t))",
    "candidates_per_query": "num_entities",
    "filter_set": "train + valid + test true triples, applied only inside the evaluator",
    "tie_policy": "expected_uniform_tie_breaking (tied_rank_positions)",
    "metrics": ["mrr", "hits_at_1", "hits_at_3", "hits_at_10", "mean_rank"],
    "secondary_binary_metrics": ["roc_auc", "average_precision", "brier", "log_loss"],
    "reused_from_sbm_tfm": True,
    "source_path": "../run_umls_complex_vs_tabpfn.py::{build_eval_candidates,filtered_ranking_metrics,tied_rank_positions,expected_hits_at_k}",
}


def make_candidates(eval_triples, num_entities, all_true):
    frame = pd.DataFrame(np.asarray(eval_triples, dtype=np.int64), columns=["h", "r", "t"])
    return build_eval_candidates(frame, num_entities, all_true)


def ranking_metrics(candidates, scores):
    return filtered_ranking_metrics(candidates, scores)


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def run_sanity_tests():
    """Spec-mandated evaluator sanity checks. Returns a results dict."""
    results = {}

    # Small manual case: 3 entities, 1 relation.
    # train: (0,0,1); valid: (0,0,2); test: (1,0,2)
    all_true = {(0, 0, 1), (0, 0, 2), (1, 0, 2)}
    test = pd.DataFrame([(1, 0, 2)], columns=["h", "r", "t"])
    candidates = build_eval_candidates(test, 3, all_true)

    # candidate count: 2 queries (tail + head) x 3 entities
    results["candidate_count"] = {
        "expected": 6,
        "actual": int(len(candidates)),
        "passed": len(candidates) == 6,
    }

    # known positives filtered: tail query (1,0,?) has no other true tail;
    # head query (?,0,2) has known positive (0,0,2) which must be filtered.
    head_side = candidates[candidates["side"] == "head"]
    filtered_rows = head_side[head_side["is_filtered"] == 1]
    results["known_positive_filtered_correctly"] = {
        "expected_filtered_head_candidates": [(0, 0, 2)],
        "actual": filtered_rows[["h", "r", "t"]].to_numpy().tolist(),
        "target_never_filtered": bool(
            (candidates.loc[candidates["is_target"] == 1, "is_filtered"] == 0).all()
        ),
        "passed": filtered_rows[["h", "r", "t"]].to_numpy().tolist() == [[0, 0, 2]]
        and bool(
            (candidates.loc[candidates["is_target"] == 1, "is_filtered"] == 0).all()
        ),
    }

    # manual metric check: scores such that target ranks 2nd in the tail query
    # (against 2 unfiltered candidates) and 1st in the head query (against 2
    # unfiltered candidates after removing the filtered known positive).
    scores = np.zeros(len(candidates))
    for i, row in candidates.iterrows():
        if row["side"] == "tail":  # query (1,0,?), target t=2
            scores[i] = {0: 0.9, 1: 0.5, 2: 0.7}[row["t"]]
        else:  # query (?,0,2), target h=1, candidate h=0 filtered
            scores[i] = {0: 0.99, 1: 0.8, 2: 0.1}[row["h"]]
    metrics = filtered_ranking_metrics(candidates, scores)
    expected_mrr = (1 / 2 + 1 / 1) / 2
    results["small_manual_case_passed"] = {
        "expected": {"mrr": expected_mrr, "hits_at_1": 0.5, "mean_rank": 1.5},
        "actual": {
            "mrr": metrics["mrr"],
            "hits_at_1": metrics["hits_at_1"],
            "mean_rank": metrics["mean_rank"],
        },
        "passed": bool(
            abs(metrics["mrr"] - expected_mrr) < 1e-12
            and abs(metrics["hits_at_1"] - 0.5) < 1e-12
            and abs(metrics["mean_rank"] - 1.5) < 1e-12
        ),
    }

    # tie handling: all scores equal -> expected rank among k unfiltered
    # candidates is (k+1)/2
    tie_scores = np.ones(len(candidates))
    tie_metrics = filtered_ranking_metrics(candidates, tie_scores)
    results["tie_handling"] = {
        "description": "all-equal scores; tail query 3 unfiltered -> expected rank 2; head query 2 unfiltered -> expected rank 1.5",
        "expected_mean_rank": (2.0 + 1.5) / 2,
        "actual_mean_rank": tie_metrics["mean_rank"],
        "tied_positions_unit": tied_rank_positions(
            np.array([1.0, 1.0, 1.0]), 1.0
        ).tolist(),
        "passed": bool(abs(tie_metrics["mean_rank"] - 1.75) < 1e-12),
    }

    return results


def check_no_test_triples_in_train_graph(train_triples, valid_triples, test_triples):
    train = {tuple(t) for t in np.asarray(train_triples).tolist()}
    valid = {tuple(t) for t in np.asarray(valid_triples).tolist()}
    test = {tuple(t) for t in np.asarray(test_triples).tolist()}
    return {
        "train_valid_overlap": len(train & valid),
        "train_test_overlap": len(train & test),
        "passed": not (train & valid) and not (train & test),
    }
