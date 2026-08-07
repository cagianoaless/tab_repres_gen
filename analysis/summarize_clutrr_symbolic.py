from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from relational_tfm.plan import atomic_write_json

STUDY_ID = "clutrr_symbolic_cpu_confirmation_v2"
PLAN_SHA256 = "cca4114229dbfac4938d707252c4257817149817becabb4b8e8b43a7bf86df98"
EXPECTED_CELLS = 150
OOD_LENGTHS = tuple(range(4, 11))


def _load_payloads(results_root: Path) -> list[dict[str, Any]]:
    paths = sorted(results_root.glob("shard-*/*.metrics.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(payloads) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} CLUTRR cells, found {len(payloads)}")
    cell_ids = [payload["cell"]["cell_id"] for payload in payloads]
    if len(set(cell_ids)) != EXPECTED_CELLS:
        raise ValueError("CLUTRR result cells are not unique")
    if any(payload.get("status") != "complete" for payload in payloads):
        raise ValueError("CLUTRR result set contains an incomplete cell")
    if any(payload["cell"]["task_family"] != "clutrr_symbolic" for payload in payloads):
        raise ValueError("result set mixes task families")
    if any(payload["cell"]["study_id"] != STUDY_ID for payload in payloads):
        raise ValueError("result set does not match the registered CLUTRR v2 study")
    return payloads


def _frames(
    payloads: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = []
    chains = []
    for payload in payloads:
        cell = payload["cell"]
        common = {
            "cell_id": cell["cell_id"],
            "decoder": cell["decoder"],
            "encoder": cell["encoder"],
            "context_per_relation": int(cell["context_per_relation"]),
            "context_rows": int(payload["context_rows"]),
            "seed": int(cell["seed"]),
        }
        cells.append(
            {
                **common,
                "accuracy": payload["metrics"]["accuracy"],
                "balanced_accuracy": payload["metrics"]["balanced_accuracy"],
                "macro_f1": payload["metrics"]["macro_f1"],
                "log_loss": payload["metrics"]["log_loss"],
                "top3_accuracy": payload["metrics"]["top3_accuracy"],
                "ceiling": payload["metrics"]["collision_conditioned_ceiling"],
                "extraction_efficiency": payload["metrics"]["extraction_efficiency"],
                "duration_seconds": payload["duration_seconds"],
            }
        )
        for chain_length, metrics in payload["metrics_by_chain_length"].items():
            chains.append(
                {
                    **common,
                    "chain_length": int(chain_length),
                    "evaluation_rows": int(metrics["rows"]),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "top3_accuracy": metrics["top3_accuracy"],
                    "ceiling": metrics["collision_conditioned_ceiling"],
                    "extraction_efficiency": metrics["extraction_efficiency"],
                }
            )
    return pd.DataFrame(cells), pd.DataFrame(chains)


def _mean_table(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "top3_accuracy",
        "ceiling",
        "extraction_efficiency",
    ]
    return (
        frame.groupby(groups, dropna=False)[metrics]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def summarize(results_root: Path, output_dir: Path) -> dict[str, Any]:
    payloads = _load_payloads(results_root)
    cells, chains = _frames(payloads)
    expected_decoders = {"logistic", "catboost"}
    expected_encoders = {
        "path_length_only",
        "relation_multiset",
        "ordered_typed_path",
        "within_length_joint_permutation",
        "within_length_independent_permutation",
    }
    if set(cells["decoder"]) != expected_decoders or set(cells["encoder"]) != expected_encoders:
        raise ValueError("CLUTRR decoder or encoder population changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "cell_metrics.csv", index=False)
    chains.to_csv(output_dir / "chain_metrics.csv", index=False)
    _mean_table(
        cells, ["decoder", "encoder", "context_per_relation"]
    ).to_csv(output_dir / "overall_aggregate.csv", index=False)
    _mean_table(
        chains, ["decoder", "encoder", "context_per_relation", "chain_length"]
    ).to_csv(output_dir / "chain_aggregate.csv", index=False)

    ood = (
        chains[chains["chain_length"].isin(OOD_LENGTHS)]
        .groupby(
            ["decoder", "encoder", "context_per_relation", "seed"],
            as_index=False,
        )["accuracy"]
        .mean()
        .rename(columns={"accuracy": "ood_equal_length_accuracy"})
    )
    pivot = ood.pivot(
        index=["decoder", "context_per_relation", "seed"],
        columns="encoder",
        values="ood_equal_length_accuracy",
    ).reset_index()
    pivot["ordered_minus_joint_permutation"] = (
        pivot["ordered_typed_path"] - pivot["within_length_joint_permutation"]
    )
    pivot["ordered_minus_independent_permutation"] = (
        pivot["ordered_typed_path"]
        - pivot["within_length_independent_permutation"]
    )
    pivot["multiset_minus_ordered"] = (
        pivot["relation_multiset"] - pivot["ordered_typed_path"]
    )
    pivot.to_csv(output_dir / "ood_paired_contrasts.csv", index=False)

    contrast_summary = (
        pivot.groupby(["decoder", "context_per_relation"])[
            [
                "ordered_minus_joint_permutation",
                "ordered_minus_independent_permutation",
                "multiset_minus_ordered",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    contrast_summary.to_csv(output_dir / "ood_contrast_summary.csv", index=False)

    ceiling_summary = (
        chains.groupby(["encoder", "chain_length"])["ceiling"]
        .agg(["mean", "min", "max", "nunique"])
        .reset_index()
    )
    ceiling_summary.to_csv(output_dir / "ceiling_by_chain_length.csv", index=False)
    aligned = ceiling_summary[
        ceiling_summary["encoder"].isin(
            {"path_length_only", "relation_multiset", "ordered_typed_path"}
        )
    ]
    if (aligned["nunique"] != 1).any():
        raise ValueError("aligned CLUTRR ceilings changed across decoder, context, or seed")

    headline = {}
    for decoder in sorted(expected_decoders):
        headline[decoder] = {}
        for context in sorted(cells["context_per_relation"].unique()):
            selected = pivot[
                (pivot["decoder"] == decoder)
                & (pivot["context_per_relation"] == context)
            ]
            headline[decoder][str(int(context))] = {
                column: float(selected[column].mean())
                for column in (
                    "ordered_typed_path",
                    "relation_multiset",
                    "within_length_joint_permutation",
                    "within_length_independent_permutation",
                    "ordered_minus_joint_permutation",
                    "ordered_minus_independent_permutation",
                    "multiset_minus_ordered",
                )
            }

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "study_id": STUDY_ID,
        "plan_sha256": PLAN_SHA256,
        "cells": len(cells),
        "chain_metric_rows": len(chains),
        "decoders": sorted(expected_decoders),
        "encoders": sorted(expected_encoders),
        "contexts_per_relation": sorted(
            map(int, cells["context_per_relation"].unique())
        ),
        "seeds": sorted(map(int, cells["seed"].unique())),
        "ood_chain_lengths": list(OOD_LENGTHS),
        "cell_duration_seconds_sum": float(cells["duration_seconds"].sum()),
        "headline_ood_equal_length_accuracy": headline,
        "uncertainty_scope": (
            "descriptive variation over three nested-context seeds on one fixed "
            "official test population; not cross-dataset inference"
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results_root, args.output_dir)
    print(
        f"PASS cells={summary['cells']} "
        f"chain_rows={summary['chain_metric_rows']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
