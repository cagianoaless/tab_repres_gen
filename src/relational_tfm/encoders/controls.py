from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

FROZEN_CONTEXT_BUDGETS = (20, 50, 100, 200, 400)


def frozen_context_increment_tiers(
    pool_size: int,
    *,
    negative_ratio: int = 1,
) -> np.ndarray:
    """Assign rows to nested-budget increments so every registered prefix is preserved."""
    if pool_size <= 0 or negative_ratio <= 0:
        raise ValueError("pool size and negative ratio must be positive")
    boundaries = sorted(
        {
            min(pool_size, budget * negative_ratio)
            for budget in FROZEN_CONTEXT_BUDGETS
        }
        | {pool_size}
    )
    positions = np.arange(1, pool_size + 1)
    tiers = np.empty(pool_size, dtype=object)
    lower = 0
    for boundary in boundaries:
        selected = (positions > lower) & (positions <= boundary)
        tiers[selected] = f"context_increment_to_{boundary}"
        lower = boundary
    return tiers


def _sattolo_cycle(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return one random cycle, hence a derangement whenever size is greater than one."""
    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap = int(rng.integers(0, index))
        permutation[index], permutation[swap] = (
            permutation[swap],
            permutation[index],
        )
    return permutation


def stratified_joint_derangement(
    values: np.ndarray,
    strata: Sequence[tuple[str, ...]],
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Jointly permute complete rows within frozen strata without using labels."""
    values = np.asarray(values)
    if values.ndim != 2 or len(values) != len(strata):
        raise ValueError("values and strata must describe one row-aligned matrix")

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(strata):
        groups[tuple(str(part) for part in key)].append(index)

    controlled = values.copy()
    rng = np.random.default_rng(seed)
    singleton_rows = 0
    deranged_rows = 0
    for key in sorted(groups):
        positions = np.asarray(groups[key], dtype=np.int64)
        if len(positions) == 1:
            singleton_rows += 1
            continue
        source = positions[_sattolo_cycle(len(positions), rng)]
        controlled[positions] = values[source]
        deranged_rows += len(positions)

    return controlled, {
        "stratum_count": len(groups),
        "singleton_rows": singleton_rows,
        "deranged_rows": deranged_rows,
        "fixed_points_in_nonsingleton_strata": 0,
    }
