#!/usr/bin/env python3
"""Summarize the GraphLog CatBoost capacity sweep."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

EXPECTED_PLAN_SHA256 = "a73c22df3f6ca0f80c4723d049be075fcba07edef27447485c86f113502072f0"
EXPECTED_CELLS = 2_394


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cells(run_root: Path) -> pd.DataFrame:
    records = []
    for name in glob.glob(str(run_root / "results" / "shard-*" / "*.metrics.json")):
        path = Path(name)
        payload = json.loads(path.read_text())
        cell = payload["cell"]
        match = re.fullmatch(r"catboost_depth([1-7])", cell["decoder"])
        if not match:
            raise ValueError(f"unexpected decoder: {cell['decoder']}")
        records.append(
            {
                "cell_id": cell["cell_id"],
                "world": cell["world"],
                "seed": int(cell["seed"]),
                "encoder": cell["encoder"],
                "depth": int(match.group(1)),
                "evaluation_balanced_accuracy": float(
                    payload["metrics"]["equal_length_macro_balanced_accuracy"]
                ),
                "context_balanced_accuracy": float(
                    payload["context_fit_metrics"]["balanced_accuracy"]
                ),
                "evaluation_accuracy": float(payload["metrics"]["accuracy"]),
                "context_accuracy": float(payload["context_fit_metrics"]["accuracy"]),
                "evaluation_chance": 1.0 / int(payload["audit"]["evaluation_class_count"]),
                "context_chance": 1.0 / int(payload["audit"]["context_class_count"]),
                "duration_seconds": float(payload["duration_seconds"]),
                "source_path": str(path.relative_to(run_root)),
                "source_sha256": sha256(path),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} cells, found {len(frame)}")
    key = ["world", "seed", "encoder", "depth"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate capacity-sweep cells")
    return frame.sort_values(key).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    if plan["plan_sha256"] != EXPECTED_PLAN_SHA256:
        raise ValueError("capacity-sweep plan hash changed")
    status = json.loads((root / "status.json").read_text())
    if not status.get("complete") or status.get("failures"):
        raise ValueError("capacity sweep is not complete and failure-free")

    cells = load_cells(root)
    metric_columns = [
        "evaluation_balanced_accuracy",
        "context_balanced_accuracy",
        "evaluation_accuracy",
        "context_accuracy",
    ]
    wide = cells.pivot(
        index=["world", "seed", "depth"],
        columns="encoder",
        values=metric_columns,
    )
    wide.columns = [f"{metric}__{encoder}" for metric, encoder in wide.columns]
    wide = wide.reset_index()
    for metric in metric_columns:
        wide[f"shared_minus_absolute__{metric}"] = (
            wide[f"{metric}__relation_multiset"]
            - wide[f"{metric}__ordered_typed_path"]
        )
    chance = cells[["world", "evaluation_chance", "context_chance"]].drop_duplicates()
    if chance["world"].duplicated().any():
        raise ValueError("world-specific chance changed across cells")
    wide = wide.merge(chance, on="world", validate="many_to_one")

    by_world = (
        wide.groupby(["world", "depth"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            evaluation_shared=("evaluation_balanced_accuracy__relation_multiset", "mean"),
            evaluation_absolute=("evaluation_balanced_accuracy__ordered_typed_path", "mean"),
            evaluation_effect=("shared_minus_absolute__evaluation_balanced_accuracy", "mean"),
            context_shared=("context_balanced_accuracy__relation_multiset", "mean"),
            context_absolute=("context_balanced_accuracy__ordered_typed_path", "mean"),
            context_effect=("shared_minus_absolute__context_balanced_accuracy", "mean"),
            evaluation_chance=("evaluation_chance", "first"),
            context_chance=("context_chance", "first"),
        )
        .sort_values(["depth", "world"])
    )
    rows = []
    for depth, group in by_world.groupby("depth", sort=True):
        rows.append(
            {
                "depth": int(depth),
                "worlds": int(group["world"].nunique()),
                "evaluation_shared": float(group["evaluation_shared"].mean()),
                "evaluation_absolute": float(group["evaluation_absolute"].mean()),
                "evaluation_shared_minus_absolute": float(group["evaluation_effect"].mean()),
                "context_shared": float(group["context_shared"].mean()),
                "context_absolute": float(group["context_absolute"].mean()),
                "context_shared_minus_absolute": float(group["context_effect"].mean()),
                "shared_worlds_at_or_below_chance": int(
                    (group["evaluation_shared"] <= group["evaluation_chance"]).sum()
                ),
                "absolute_worlds_at_or_below_chance": int(
                    (group["evaluation_absolute"] <= group["evaluation_chance"]).sum()
                ),
                "effect_world_wins": int((group["evaluation_effect"] > 0).sum()),
                "effect_world_ties": int((group["evaluation_effect"] == 0).sum()),
                "effect_world_losses": int((group["evaluation_effect"] < 0).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    cells.to_csv(output / "cells.csv", index=False)
    wide.to_csv(output / "paired_by_seed.csv", index=False)
    by_world.to_csv(output / "paired_by_world.csv", index=False)
    summary.to_csv(output / "capacity_curve.csv", index=False)
    result = {
        "schema_version": 1,
        "status": "COMPLETE_CAPACITY_SWEEP",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "cells": EXPECTED_CELLS,
        "worlds": 57,
        "seeds": [101, 102, 103],
        "depths": list(range(1, 8)),
        "primary_effect": (
            "relation multiset minus ordered typed path equal-length macro "
            "balanced accuracy"
        ),
        "outcome_conditioning": "none",
        "capacity_curve": summary.to_dict(orient="records"),
    }
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "summary.json")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
