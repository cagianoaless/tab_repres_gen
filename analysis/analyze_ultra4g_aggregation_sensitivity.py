#!/usr/bin/env python3
"""Report dataset- and provenance-family-weighted Ultra57 headline effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PRACTICAL_BAND = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-metrics",
        type=Path,
        default=Path(
            "experiments/ultra4g/results/q64-encoder-ladder/dataset_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/results/ultra4g-aggregation-sensitivity"),
    )
    args = parser.parse_args()
    source = args.dataset_metrics.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    selected = frame.loc[
        (frame["interface"] == "b2v2_full")
        & (frame["context_per_class"] == 200),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            "combined_mrr",
            "native_combined_mrr",
            "delta_combined_mrr",
        ],
    ].copy()
    if len(selected) != 57 or selected["dataset"].nunique() != 57:
        raise ValueError("expected exactly one full-interface row for each of 57 datasets")
    selected = selected.sort_values("dataset").reset_index(drop=True)
    family = (
        selected.groupby("provenance_family", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            effect=("delta_combined_mrr", "mean"),
            full_mrr=("combined_mrr", "mean"),
            native_mrr=("native_combined_mrr", "mean"),
        )
        .sort_values("provenance_family")
        .reset_index(drop=True)
    )
    if len(family) != 16:
        raise ValueError("expected 16 provenance families")

    effects = selected["delta_combined_mrr"].to_numpy(dtype=float)
    family_effects = family["effect"].to_numpy(dtype=float)
    leave_one_family_out = {
        name: float(family.loc[family["provenance_family"] != name, "effect"].mean())
        for name in family["provenance_family"]
    }
    metafam_names = [
        name for name in family["provenance_family"] if "metafam" in name.lower()
    ]
    if len(metafam_names) != 1:
        raise ValueError(f"expected one MetaFam family, found {metafam_names}")
    metafam = metafam_names[0]
    wins = int(np.sum(effects > PRACTICAL_BAND))
    ties = int(np.sum(np.abs(effects) <= PRACTICAL_BAND))
    losses = int(np.sum(effects < -PRACTICAL_BAND))
    result = {
        "schema_version": 1,
        "status": "POST_HOC_REVIEWER_REQUESTED_FIXED_PANEL_SENSITIVITY",
        "estimand": (
            "b2v2_full minus native ULTRA combined filtered MRR at 200 context "
            "examples per class"
        ),
        "source": {
            "path": str(args.dataset_metrics),
            "sha256": sha256(source),
        },
        "panel": {
            "datasets": 57,
            "provenance_families": 16,
            "practical_tie_band_mrr": PRACTICAL_BAND,
        },
        "equal_dataset": {
            "mean": float(effects.mean()),
            "median": float(np.median(effects)),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "equal_family": {
            "mean": float(family_effects.mean()),
            "median": float(np.median(family_effects)),
            "leave_one_family_out_min": float(min(leave_one_family_out.values())),
            "leave_one_family_out_max": float(max(leave_one_family_out.values())),
            "leave_one_family_out": leave_one_family_out,
            "metafam_family": metafam,
            "excluding_metafam": leave_one_family_out[metafam],
        },
        "interpretation_boundary": (
            "Equal-family is the primary fixed-panel estimand because related benchmark "
            "variants are dependent. Equal-dataset, medians, and leave-one-family-out "
            "results are required sensitivities and are not population estimates."
        ),
    }
    selected.to_csv(output / "full_vs_native_by_dataset.csv", index=False)
    family.to_csv(output / "full_vs_native_by_family.csv", index=False)
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "summary.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
