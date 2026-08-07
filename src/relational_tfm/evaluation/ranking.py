from __future__ import annotations

from typing import Any

import numpy as np


def filtered_ranking_metrics(
    scores: np.ndarray,
    filtered: np.ndarray,
    target_entities: np.ndarray,
    side_codes: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    scores = np.asarray(scores, dtype=np.float64)
    filtered = np.asarray(filtered, dtype=np.bool_)
    targets = np.asarray(target_entities, dtype=np.int64)
    sides = np.asarray(side_codes, dtype=np.int8)
    query_count, num_entities = filtered.shape
    if len(scores) != query_count * num_entities:
        raise ValueError("candidate scores do not match the registered query blocks")
    if not (len(targets) == len(sides) == query_count):
        raise ValueError("ranking query metadata is not aligned")

    ranks = np.empty(query_count, dtype=np.float64)
    reciprocal_ranks = np.empty(query_count, dtype=np.float64)
    hits = {cutoff: np.empty(query_count, dtype=np.float64) for cutoff in (1, 3, 10, 50)}
    target_scores = np.empty(query_count, dtype=np.float64)
    for query in range(query_count):
        block = scores[query * num_entities : (query + 1) * num_entities]
        target = int(targets[query])
        if filtered[query, target]:
            raise ValueError("the target entity is masked by the filtered-ranking asset")
        valid_scores = block[~filtered[query]]
        target_score = float(block[target])
        greater = int(np.count_nonzero(valid_scores > target_score))
        tied = int(np.count_nonzero(valid_scores == target_score))
        if tied <= 0:
            raise ValueError("the target score is absent from its candidate block")
        positions = np.arange(greater + 1, greater + tied + 1, dtype=np.float64)
        ranks[query] = float(positions.mean())
        reciprocal_ranks[query] = float(np.mean(1.0 / positions))
        target_scores[query] = target_score
        for cutoff in hits:
            hits[cutoff][query] = float(np.mean(positions <= cutoff))

    def summarize(mask: np.ndarray) -> dict[str, float]:
        return {
            "mrr": float(reciprocal_ranks[mask].mean()),
            "mean_rank": float(ranks[mask].mean()),
            **{
                f"hits_at_{cutoff}": float(values[mask].mean())
                for cutoff, values in hits.items()
            },
        }

    all_queries = np.ones(query_count, dtype=np.bool_)
    direction_metrics = {
        "combined": summarize(all_queries),
        "tail": summarize(sides == 0),
        "head": summarize(sides == 1),
    }
    records: dict[str, Any] = {
        "query_ids": np.arange(query_count, dtype=np.int64),
        "side_codes": sides,
        "target_entities": targets,
        "target_scores": target_scores,
        "ranks": ranks,
        "reciprocal_ranks": reciprocal_ranks,
    }
    records.update({f"hits_at_{cutoff}": values for cutoff, values in hits.items()})
    return direction_metrics["combined"], direction_metrics, records
