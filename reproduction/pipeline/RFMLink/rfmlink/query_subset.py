"""Fixed sampled-query evaluation subset for expensive datasets (codex-small).

Full filtered ranking on codex-small requires scoring ~7.4M candidate rows per
split per run, which makes TabPFN runs take 30-90 minutes each. Standard
practice for expensive scorers is to evaluate on a random subset of test
queries; this changes evaluation VARIANCE, not bias, provided the subset is
(a) sampled once with a fixed seed, (b) identical for every system compared,
and (c) documented. This module enforces (a)-(c):

  * subset_triple_indices(dataset, split): deterministic (seed 7) sample of
    SUBSET_SIZE eval triples; every triple contributes its tail query AND its
    head query (both query sides kept, matching the full protocol);
  * candidate_row_mask(...): boolean mask over the canonical candidate table
    rows (query blocks of num_entities rows, in build_eval_candidates order);
  * metrics for ALL systems (including already-completed full-eval runs and
    the ULTRA-alone baseline) are recomputed on this same subset, so paired
    comparisons remain exact.

The full-evaluation metrics of non-TabPFN systems are retained in their
original metrics.json; subset metrics are written under the key
"<split>_subset500" alongside them.
"""

import numpy as np

SUBSET_SIZE = 500
SUBSET_SEED = 7
SUBSET_DATASETS = {"codex-small"}


def subset_triple_indices(n_eval_triples, size=SUBSET_SIZE, seed=SUBSET_SEED):
    if n_eval_triples <= size:
        return np.arange(n_eval_triples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_eval_triples, size=size, replace=False))


def query_indices_for_triples(triple_indices):
    """build_eval_candidates emits queries in order: triple0-tail(q0),
    triple0-head(q1), triple1-tail(q2), ... -> triple i owns queries 2i, 2i+1."""
    triple_indices = np.asarray(triple_indices)
    return np.sort(np.concatenate([2 * triple_indices, 2 * triple_indices + 1]))


def candidate_row_mask(n_eval_triples, num_entities, size=SUBSET_SIZE, seed=SUBSET_SEED):
    """Boolean mask over the canonical candidate-table rows for the subset."""
    triples = subset_triple_indices(n_eval_triples, size, seed)
    queries = query_indices_for_triples(triples)
    mask = np.zeros(2 * n_eval_triples * num_entities, dtype=bool)
    for q in queries:
        mask[q * num_entities : (q + 1) * num_entities] = True
    return mask, queries
