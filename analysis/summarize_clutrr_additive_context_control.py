#!/usr/bin/env python3
"""Audit and summarize the paired additive CLUTRR context intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_rows(root: Path, arm: str, expected_cells: int) -> pd.DataFrame:
    paths = sorted(root.glob("results/shard-*/*.metrics.json"))
    if len(paths) != expected_cells:
        raise ValueError(f"{root}: expected {expected_cells} cells, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload["cell"]
        metadata = payload["task_metadata"]
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete cell: {path}")
        if metadata["context_per_relation"] != cell["context_per_relation"]:
            raise ValueError(f"context-size mismatch: {path}")
        for length_text, metrics in payload["metrics_by_chain_length"].items():
            rows.append(
                {
                    "decoder": str(cell["decoder"]),
                    "encoder": str(cell["encoder"]),
                    "base_context_per_relation": int(
                        cell["base_context_per_relation"]
                    ),
                    "total_context_per_relation": int(
                        cell["context_per_relation"]
                    ),
                    "seed": int(cell["seed"]),
                    "chain_length": int(length_text),
                    "base_context_row_ids_sha256": metadata[
                        "base_context_row_ids_sha256"
                    ],
                    "evaluation_sha256": metadata["asset_sha256"]["test"],
                    arm: float(metrics["accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--length4-root", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    short = load_rows(args.short_root, "added_short", args.expected_cells)
    length4 = load_rows(args.length4_root, "added_length4", args.expected_cells)
    keys = [
        "decoder",
        "encoder",
        "base_context_per_relation",
        "total_context_per_relation",
        "seed",
        "chain_length",
        "base_context_row_ids_sha256",
        "evaluation_sha256",
    ]
    paired = short.merge(length4, on=keys, validate="one_to_one")
    expected_rows = args.expected_cells * 9
    if len(paired) != expected_rows:
        raise ValueError(f"expected {expected_rows} paired length rows, found {len(paired)}")
    paired["length4_minus_short"] = (
        paired["added_length4"] - paired["added_short"]
    )

    scope_rows = []
    for scope, lengths in (
        ("length4", {4}),
        ("length5_to_10", set(range(5, 11))),
        ("ood_length4_to_10", set(range(4, 11))),
    ):
        selected = paired[paired["chain_length"].isin(lengths)]
        grouped = (
            selected.groupby(
                [
                    "decoder",
                    "encoder",
                    "base_context_per_relation",
                    "total_context_per_relation",
                    "seed",
                ],
                as_index=False,
            )[["added_short", "added_length4", "length4_minus_short"]]
            .mean()
            .assign(scope=scope)
        )
        scope_rows.append(grouped)
    by_seed = pd.concat(scope_rows, ignore_index=True)
    summary = (
        by_seed.groupby(
            [
                "decoder",
                "encoder",
                "base_context_per_relation",
                "total_context_per_relation",
                "scope",
            ],
            as_index=False,
        )
        .agg(
            added_short=("added_short", "mean"),
            added_length4=("added_length4", "mean"),
            mean_effect=("length4_minus_short", "mean"),
            min_seed_effect=("length4_minus_short", "min"),
            max_seed_effect=("length4_minus_short", "max"),
        )
        .sort_values(
            ["decoder", "encoder", "scope", "base_context_per_relation"]
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_by_length.csv", index=False)
    by_seed.to_csv(args.output_dir / "paired_by_seed_scope.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    audit = {
        "status": "PASS",
        "expected_cells_per_arm": args.expected_cells,
        "paired_length_rows": len(paired),
        "base_context_hashes_match": True,
        "interpretation": (
            "added length-4 versus added length-2/3 at equal total budget, "
            "with an identical nested length-2/3 base and fixed test population"
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
