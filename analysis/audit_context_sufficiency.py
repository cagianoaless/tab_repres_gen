#!/usr/bin/env python3
"""Prospective CLUTRR/GraphLog context-to-query composition witness audit.

Use the official study-plan seeds, SHA-verified dataset loaders and complete
length-shift evaluation populations. Infer only *observed* short-path rewrite
productions from labelled context rows; do not use test labels to infer rules.

A witness is conditional on treating these short-path associations as reusable
composition productions. Neither a witness nor its absence proves formal
identifiability of the generative rule system.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "clutrr": (
        "configs/studies/clutrr_symbolic_gpu_tabpfn_confirmation_v2.yaml",
        "edge_types",
    ),
    "graphlog": (
        "configs/studies/graphlog_symbolic_gpu_tabpfn_confirmation_v1.yaml",
        "relation_tokens",
    ),
}


def infer_productions(rows: list[tuple[tuple[str, ...], str]], arity: int):
    """Reject conflicting observed input sequences, retaining no favoured label."""
    observed: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for tokens, target in rows:
        if 2 <= len(tokens) <= arity:
            observed[tokens][target] += 1
    valid = {tokens: next(iter(labels)) for tokens, labels in observed.items() if len(labels) == 1}
    conflicts = {tokens: labels for tokens, labels in observed.items() if len(labels) > 1}
    return valid, conflicts


def possible_roots(tokens: tuple[str, ...], productions: dict, arity: int) -> frozenset[str]:
    """All symbolic results of binary/ternary interval reductions (not an oracle)."""
    n = len(tokens)
    if n == 0:
        return frozenset()
    chart: dict[tuple[int, int], set[str]] = {
        (index, index + 1): {token} for index, token in enumerate(tokens)
    }
    for width in range(2, n + 1):
        for start in range(n - width + 1):
            stop = start + width
            roots: set[str] = set()
            for count in range(2, min(arity, width) + 1):
                for cuts in itertools.combinations(range(start + 1, stop), count - 1):
                    bounds = (start, *cuts, stop)
                    parts = [chart[a, b] for a, b in zip(bounds, bounds[1:])]
                    if not all(parts):
                        continue
                    for symbols in itertools.product(*parts):
                        result = productions.get(symbols)
                        if result is not None:
                            roots.add(result)
            chart[start, stop] = roots
    return frozenset(chart[0, n])


def summarize_task(task: Any, family: str, world: str, seed: int, budget: int):
    """Return per-query evidence and per-length counts for one registered context."""
    input_column = STUDIES[family][1]
    contexts = [
        (tuple(map(str, tokens)), str(label))
        for tokens, label in zip(task.context[input_column], task.context["target_relation"])
    ]
    rules = {arity: infer_productions(contexts, arity) for arity in (2, 3)}
    targets = {target for _, target in contexts}
    tokens_seen = {token for sequence, _ in contexts for token in sequence}
    maximum_length = max(len(sequence) for sequence, _ in contexts)
    length_counts: dict[int, Counter[str]] = defaultdict(Counter)
    queries: list[dict[str, Any]] = []

    for row in task.evaluation.to_dict("records"):
        sequence = tuple(map(str, row[input_column]))
        truth = str(row["target_relation"])
        length = len(sequence)
        counts = length_counts[length]
        counts["rows"] += 1
        counts["target_covered"] += truth in targets
        counts["tokens_covered"] += all(token in tokens_seen for token in sequence)
        counts["absolute_positions_supported"] += length <= maximum_length
        roots = {}
        for arity in (2, 3):
            roots[arity] = possible_roots(sequence, rules[arity][0], arity)
            output = roots[arity]
            prefix = f"arity{arity}_"
            counts[prefix + "has_witness"] += bool(output)
            counts[prefix + "unique"] += len(output) == 1
            counts[prefix + "unique_correct"] += output == {truth}
            counts[prefix + "unique_wrong"] += len(output) == 1 and truth not in output
            counts[prefix + "ambiguous"] += len(output) > 1
            counts[prefix + "includes_truth"] += truth in output
        match = row.get("descriptor_graph_path_exact_match", None)
        if match is not None:
            counts["descriptor_matches_graph_path"] += bool(match)
        queries.append(
            {
                "dataset": family,
                "world": world,
                "seed": seed,
                "context_per_relation": budget,
                "row_id": str(row["row_id"]),
                "length": length,
                "descriptor": ",".join(sequence),
                "truth_posthoc_only": truth,
                "target_covered": int(truth in targets),
                "tokens_covered": int(all(token in tokens_seen for token in sequence)),
                "arity2_roots": ";".join(sorted(roots[2])),
                "arity3_roots": ";".join(sorted(roots[3])),
                "arity2_unique_correct": int(roots[2] == {truth}),
                "arity3_unique_correct": int(roots[3] == {truth}),
                "descriptor_matches_graph_path": "" if match is None else int(bool(match)),
            }
        )

    return (
        {
            "dataset": family,
            "world": world,
            "seed": seed,
            "context_per_relation": budget,
            "context_rows": len(contexts),
            "context_target_classes": len(targets),
            "context_relation_tokens": len(tokens_seen),
            "productions": {
                str(arity): {
                    "unambiguous_patterns": len(rules[arity][0]),
                    "conflicting_patterns_discarded": len(rules[arity][1]),
                }
                for arity in (2, 3)
            },
            "by_length": {str(k): dict(v) for k, v in sorted(length_counts.items())},
        },
        queries,
    )


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Give micro-query and equal-world, equal-seed rates separately."""
    sums: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    per_world: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for cell in cells:
        for length, counts in cell["by_length"].items():
            key = (cell["dataset"], int(length))
            sums[key].update(counts)
            per_world[key].append(
                {name: value / counts["rows"] for name, value in counts.items() if name != "rows"}
            )
    results = []
    for (family, length), counts in sorted(sums.items()):
        fields = sorted(set().union(*(record.keys() for record in per_world[family, length])))
        results.append(
            {
                "dataset": family,
                "length": length,
                "query_instances": counts["rows"],
                "world_seed_cells": len(per_world[family, length]),
                "micro_rates": {
                    name: counts[name] / counts["rows"] for name in fields
                },
                "equal_world_seed_rates": {
                    name: sum(record.get(name, 0.0) for record in per_world[family, length])
                    / len(per_world[family, length])
                    for name in fields
                },
            }
        )
    return {"by_length": results}


def audit_registered(repo_root: Path, budgets: dict[str, tuple[int, ...]],
                     seeds: tuple[int, ...]):
    """Reuse original deterministic expansion + checksum-verifying task loaders."""
    from relational_tfm.config import load_study_config
    from relational_tfm.datasets.clutrr import load_clutrr_task
    from relational_tfm.datasets.graphlog import load_graphlog_task
    from relational_tfm.plan import build_plan

    loaders = {"clutrr": load_clutrr_task, "graphlog": load_graphlog_task}
    cells = []
    queries = []
    plans = []
    for family, (config, _) in STUDIES.items():
        plan = build_plan(load_study_config(repo_root / config))
        plans.append({"dataset": family, "config": config, "plan_sha256": plan["plan_sha256"]})
        selected = {}
        for cell in plan["cells"]:
            if (
                cell["decoder"] != "tabpfn"
                or cell["encoder"] != "ordered_typed_path"
                or cell["context_per_relation"] not in budgets[family]
                or cell["seed"] not in seeds
            ):
                continue
            key = (cell.get("world", "single"), cell["context_per_relation"], cell["seed"])
            if key in selected:
                raise ValueError(f"duplicate registered context: {family} {key}")
            selected[key] = cell
        expected = (57 if family == "graphlog" else 1) * len(budgets[family]) * len(seeds)
        if len(selected) != expected:
            raise ValueError(f"{family}: expected {expected} cells, found {len(selected)}")
        for (world, budget, seed), cell in sorted(selected.items()):
            task = loaders[family](repo_root, cell)
            summary, evidence = summarize_task(task, family, world, seed, budget)
            cells.append(summary)
            queries.extend(evidence)
    return plans, cells, queries


def write_outputs(output: Path, plans: list, cells: list, queries: list, *, emit_queries: bool):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "COMPLETED_ON_CHECKSUM_VERIFIED_DATA",
        "scientific_scope": "conditional short-descriptor rewrite witnesses; not rule identifiability",
        "test_label_policy": "test labels used only for post-hoc agreement, never grammar induction",
        "registered_plans": plans,
        "cells": cells,
        "aggregate": aggregate(cells),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "cell_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "world", "seed", "context_per_relation", "length", "context_rows",
            "query_rows", "target_coverage", "token_coverage", "arity2_witness",
            "arity2_unique_correct", "arity3_witness", "arity3_unique_correct",
            "arity3_unique_wrong", "arity3_ambiguous", "graph_path_match",
        ])
        writer.writeheader()
        for cell in cells:
            for length, counts in cell["by_length"].items():
                n = counts["rows"]
                writer.writerow({
                    "dataset": cell["dataset"], "world": cell["world"], "seed": cell["seed"],
                    "context_per_relation": cell["context_per_relation"], "length": length,
                    "context_rows": cell["context_rows"], "query_rows": n,
                    "target_coverage": counts.get("target_covered", 0) / n,
                    "token_coverage": counts.get("tokens_covered", 0) / n,
                    "arity2_witness": counts.get("arity2_has_witness", 0) / n,
                    "arity2_unique_correct": counts.get("arity2_unique_correct", 0) / n,
                    "arity3_witness": counts.get("arity3_has_witness", 0) / n,
                    "arity3_unique_correct": counts.get("arity3_unique_correct", 0) / n,
                    "arity3_unique_wrong": counts.get("arity3_unique_wrong", 0) / n,
                    "arity3_ambiguous": counts.get("arity3_ambiguous", 0) / n,
                    "graph_path_match": (
                        counts.get("descriptor_matches_graph_path", 0) / n
                        if cell["dataset"] == "graphlog" else ""
                    ),
                })
    if emit_queries:
        with (output / "query_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "dataset", "world", "seed", "context_per_relation", "row_id", "length",
                "descriptor", "truth_posthoc_only", "target_covered", "tokens_covered",
                "arity2_roots", "arity3_roots", "arity2_unique_correct",
                "arity3_unique_correct", "descriptor_matches_graph_path",
            ])
            writer.writeheader()
            writer.writerows(queries)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/context-sufficiency-audit"))
    parser.add_argument("--clutrr-budgets", type=int, nargs="+", default=[80])
    parser.add_argument("--graphlog-budgets", type=int, nargs="+", default=[40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--no-query-csv", action="store_true")
    args = parser.parse_args()
    try:
        plans, cells, queries = audit_registered(
            ROOT,
            {"clutrr": tuple(args.clutrr_budgets), "graphlog": tuple(args.graphlog_budgets)},
            tuple(args.seeds),
        )
    except FileNotFoundError as exc:
        parser.error(f"Missing raw asset: {exc}. Stage datasets per REPRODUCE.md section 8.")
    report = write_outputs(args.output_dir, plans, cells, queries, emit_queries=not args.no_query_csv)
    print(json.dumps({"status": report["status"], "cells": len(cells),
                      "by_length": report["aggregate"]["by_length"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
