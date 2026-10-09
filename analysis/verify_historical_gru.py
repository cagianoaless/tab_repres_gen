"""Verify GRU reference tables without running models or writing outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("experiments/clutrr/results/recurrent-gru")
BUDGETS = (5, 10, 20, 40, 80)
SEEDS = (101, 102, 103)
HASHES = {
    "gru_by_context.csv": "65a39c0af72be4cb38187b3964c348c32df1ee230f04d12681d1d3b3dcbff3ef",
    "gru_by_length.csv": "7e1ea39a98eb11c34b5dd56173b53e783056b32f2cb9d1da627a00482c763430",
}
OOD_MEANS = (
    0.34621817127926913,
    0.39399390352262748,
    0.45476679513201934,
    0.67034349959601525,
    0.7054995165817588,
)


def reconstruct(lengths: pd.DataFrame) -> pd.DataFrame:
    """Equal-length OOD (lengths 4--10), then equal-seed aggregation."""
    keys = ["context_per_relation", "seed", "chain_length"]
    expected = set(itertools.product(BUDGETS, SEEDS, range(2, 11)))
    observed = set(lengths[keys].itertuples(index=False, name=None))
    if len(lengths) != 135 or observed != expected:
        raise ValueError("GRU length table is incomplete or duplicated")
    if not np.isfinite(lengths.accuracy).all() or not lengths.accuracy.between(0, 1).all():
        raise ValueError("invalid GRU accuracy")
    by_seed = lengths.loc[lengths.chain_length >= 4].groupby(keys[:2]).accuracy.mean()
    return by_seed.groupby("context_per_relation").agg(["mean", "min", "max"])


def verify(root: Path = ROOT) -> dict[str, object]:
    for name, digest in HASHES.items():
        if hashlib.sha256((root / RESULTS / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical GRU hash changed: {name}")
    lengths = pd.read_csv(root / RESULTS / "gru_by_length.csv", float_precision="round_trip")
    context = pd.read_csv(root / RESULTS / "gru_by_context.csv", float_precision="round_trip")
    summary = reconstruct(lengths)
    if len(context) != 5 or tuple(context.context_per_relation) != BUDGETS:
        raise ValueError("GRU context table is incomplete or duplicated")
    columns = [f"seed_{stat}_equal_length_ood_accuracy" for stat in ("mean", "min", "max")]
    if not np.allclose(summary.to_numpy(), context[columns], atol=1e-15, rtol=0):
        raise ValueError("GRU context summary disagrees with recovered length table")
    if not np.allclose(summary["mean"], OOD_MEANS, atol=1e-15, rtol=0):
        raise ValueError("historical GRU OOD endpoints changed")
    largest = (
        lengths.loc[lengths.context_per_relation == 80].groupby("chain_length").accuracy.mean()
    )
    endpoints = {4: 0.8982456140350877, 10: 0.5266106442577031}
    for length, expected in endpoints.items():
        if not np.isclose(largest[length], expected, atol=1e-15, rtol=0):
            raise ValueError(f"historical GRU length-{length} endpoint changed")
    return {
        "status": "PASS",
        "boundary": "historical derived tables, not raw-run recovery",
        "length_records": len(lengths),
        "ood_means": summary["mean"].tolist(),
        "largest_budget_length_accuracy": {str(length): largest[length] for length in endpoints},
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
