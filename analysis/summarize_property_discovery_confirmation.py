#!/usr/bin/env python3
"""Audit and summarize the registered property-discovery confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_COUNTS = {
    "core_cpu": 1200,
    "core_tabpfn": 600,
    "ladder_cpu": 1080,
    "ladder_tabpfn": 540,
}
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPLICATES = 20_000
PRIMARY_CONTEXT = 20
PROPERTIES = (
    "discovery_reciprocity",
    "discovery_typed_composition",
    "discovery_structural_roles",
    "discovery_block_affinity",
)


def load_metrics(root: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted(root.glob("results/shard-*/*.metrics.json"))
    if len(paths) != expected:
        raise ValueError(f"{root}: expected {expected} metrics, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("status") != "complete" for payload in payloads):
        raise ValueError(f"{root}: incomplete metric payload")
    return payloads


def representation_kind(payload: dict[str, Any]) -> str:
    control = payload["encoder_provenance"].get("control")
    if control == "permuted":
        return "permuted"
    if control == "dimension_matched_noise":
        return "noise"
    return "aligned"


def core_key(payload: dict[str, Any]) -> tuple[str, int, int, str]:
    cell = payload["cell"]
    return (
        str(cell["property"]),
        int(cell["context_per_class"]),
        int(cell["seed"]),
        representation_kind(payload),
    )


def paired_identity_audit(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_decoder: str,
) -> None:
    selected = {
        core_key(payload): payload
        for payload in left
        if payload["cell"]["decoder"] == left_decoder
    }
    counterpart = {core_key(payload): payload for payload in right}
    if set(selected) != set(counterpart):
        raise ValueError(f"{left_decoder}/TabPFN core cell identities differ")
    for key, first in selected.items():
        second = counterpart[key]
        first_coverage = first["coverage"]
        second_coverage = second["coverage"]
        for name in (
            "context_query_sha256",
            "evaluation_query_sha256",
            "population_query_sha256",
        ):
            if first_coverage[name] != second_coverage[name]:
                raise ValueError(f"{key}: paired {name} differs")
        if first["feature_names"] != second["feature_names"]:
            raise ValueError(f"{key}: paired feature schema differs")


def effect_table(
    payloads: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int, str], float]:
    result = {}
    for payload in payloads:
        cell = payload["cell"]
        key = (
            str(cell["decoder"]),
            str(cell["property"]),
            int(cell["context_per_class"]),
            int(cell["seed"]),
            representation_kind(payload),
        )
        if key in result:
            raise ValueError(f"duplicate core metric key: {key}")
        result[key] = float(payload["metrics"]["roc_auc"])
    return result


def paired_effects(
    values: dict[tuple[str, str, int, int, str], float],
    decoder: str,
    context: int,
    control: str,
) -> dict[tuple[str, int], float]:
    return {
        (property_name, seed): (
            values[(decoder, property_name, context, seed, "aligned")]
            - values[(decoder, property_name, context, seed, control)]
        )
        for property_name in PROPERTIES
        for seed in range(12001, 12011)
    }


def hierarchical_bootstrap(
    values: dict[tuple[str, int], float],
    transform: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = np.arange(12001, 12011)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    property_means = [
        np.mean([values[(property_name, int(seed))] for seed in seeds])
        for property_name in PROPERTIES
    ]
    point = float(np.mean(property_means))
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_properties = rng.choice(PROPERTIES, size=len(PROPERTIES), replace=True)
        sampled_means = []
        for property_name in sampled_properties:
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            sampled_means.append(
                np.mean([values[(str(property_name), int(seed))] for seed in sampled_seeds])
            )
        estimates[index] = np.mean(sampled_means)
    if transform is not None:
        point = transform(point)
        estimates = np.asarray([transform(value) for value in estimates])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": point,
        "ci95": [float(lower), float(upper)],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "property, then graph seed within property",
        "property_means": {
            property_name: float(
                np.mean([values[(property_name, int(seed))] for seed in seeds])
            )
            for property_name in PROPERTIES
        },
        "positive_property_seed_pairs": int(sum(value > 0.0 for value in values.values())),
        "property_seed_pairs": len(values),
    }


def interaction(
    tabpfn: dict[tuple[str, int], float],
    classical: dict[tuple[str, int], float],
) -> dict[tuple[str, int], float]:
    if set(tabpfn) != set(classical):
        raise ValueError("interaction inputs are not paired")
    return {key: tabpfn[key] - classical[key] for key in tabpfn}


def write_core_csv(
    path: Path,
    payloads: list[dict[str, Any]],
) -> None:
    fields = [
        "decoder",
        "property",
        "context_per_class",
        "seed",
        "encoder",
        "representation_kind",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in sorted(
            payloads,
            key=lambda item: (
                item["cell"]["decoder"],
                item["cell"]["property"],
                item["cell"]["context_per_class"],
                item["cell"]["seed"],
                item["cell"]["encoder"],
            ),
        ):
            cell = payload["cell"]
            metrics = payload["metrics"]
            writer.writerow(
                {
                    "decoder": cell["decoder"],
                    "property": cell["property"],
                    "context_per_class": cell["context_per_class"],
                    "seed": cell["seed"],
                    "encoder": cell["encoder"],
                    "representation_kind": representation_kind(payload),
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )


def ladder_summary(
    ladder_payloads: list[dict[str, Any]],
    core_values: dict[tuple[str, str, int, int, str], float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for payload in ladder_payloads:
        cell = payload["cell"]
        grouped[
            (
                str(cell["decoder"]),
                str(cell["property"]),
                int(cell["context_per_class"]),
                str(cell["encoder"]),
            )
        ].append(float(payload["metrics"]["roc_auc"]))
    output = []
    for (decoder, property_name, context, encoder), aucs in sorted(grouped.items()):
        query_only = grouped[(decoder, property_name, context, "query_relation_only")]
        aligned = [
            core_values[(decoder, property_name, context, seed, "aligned")]
            for seed in range(12001, 12011)
        ]
        output.append(
            {
                "decoder": decoder,
                "property": property_name,
                "context_per_class": context,
                "encoder": encoder,
                "mean_roc_auc": float(np.mean(aucs)),
                "std_roc_auc_across_graph_seeds": float(np.std(aucs, ddof=1)),
                "delta_vs_query_relation_only": float(np.mean(aucs) - np.mean(query_only)),
                "gap_to_property_evidence_bank": float(np.mean(aucs) - np.mean(aligned)),
                "graph_seeds": len(aucs),
            }
        )
    return output


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-cpu-root", type=Path, required=True)
    parser.add_argument("--core-tabpfn-root", type=Path, required=True)
    parser.add_argument("--ladder-cpu-root", type=Path, required=True)
    parser.add_argument("--ladder-tabpfn-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    core_cpu = load_metrics(args.core_cpu_root, EXPECTED_COUNTS["core_cpu"])
    core_tabpfn = load_metrics(
        args.core_tabpfn_root, EXPECTED_COUNTS["core_tabpfn"]
    )
    ladder_cpu = load_metrics(args.ladder_cpu_root, EXPECTED_COUNTS["ladder_cpu"])
    ladder_tabpfn = load_metrics(
        args.ladder_tabpfn_root, EXPECTED_COUNTS["ladder_tabpfn"]
    )
    paired_identity_audit(core_cpu, core_tabpfn, "catboost")
    paired_identity_audit(core_cpu, core_tabpfn, "logistic")

    core_payloads = core_cpu + core_tabpfn
    core_values = effect_table(core_payloads)
    estimates: dict[str, Any] = {}
    for context in (20, 50, 100, 200, 400):
        tabpfn_permuted = paired_effects(core_values, "tabpfn", context, "permuted")
        catboost_permuted = paired_effects(core_values, "catboost", context, "permuted")
        logistic_permuted = paired_effects(core_values, "logistic", context, "permuted")
        tabpfn_noise = paired_effects(core_values, "tabpfn", context, "noise")
        estimates[str(context)] = {
            "tabpfn_aligned_minus_row_permutation": hierarchical_bootstrap(
                tabpfn_permuted
            ),
            "tabpfn_aligned_minus_column_permutation": hierarchical_bootstrap(
                tabpfn_noise
            ),
            "tabpfn_vs_catboost_alignment_interaction": hierarchical_bootstrap(
                interaction(tabpfn_permuted, catboost_permuted)
            ),
            "tabpfn_vs_logistic_alignment_interaction": hierarchical_bootstrap(
                interaction(tabpfn_permuted, logistic_permuted)
            ),
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_core_csv(output_dir / "core_cell_metrics.csv", core_payloads)
    ladder_rows = ladder_summary(
        ladder_cpu + ladder_tabpfn,
        core_values,
    )
    write_rows(output_dir / "encoder_ladder_summary.csv", ladder_rows)
    summary = {
        "schema_version": 1,
        "analysis_role": "registered graph-disjoint property discovery confirmation",
        "primary_context_per_class": PRIMARY_CONTEXT,
        "primary_estimand": (
            "(TabPFN aligned - row permutation) - "
            "(CatBoost aligned - row permutation), equal-property macro ROC AUC"
        ),
        "metric": "roc_auc",
        "counts": EXPECTED_COUNTS,
        "properties": list(PROPERTIES),
        "graph_seeds": list(range(12001, 12011)),
        "estimates_by_context": estimates,
        "primary": estimates[str(PRIMARY_CONTEXT)][
            "tabpfn_vs_catboost_alignment_interaction"
        ],
        "prerequisite_structural_signal": estimates[str(PRIMARY_CONTEXT)][
            "tabpfn_aligned_minus_row_permutation"
        ],
        "ladder_rows": len(ladder_rows),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["primary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
