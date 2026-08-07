from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json, read_plan

EXPECTED_STREAMS = {
    "cpu": {
        "plan_sha256": "a8f7ceb2f7fa35cb0ab3c3550b8d63158c640facc6094592f2561a8ea70f505b",
        "cells": 24,
    },
    "gpu": {
        "plan_sha256": "69e32a668814177079dcbc54f65c1fb8f91c2c6f3197bc0209ec88c1ceb96efc",
        "cells": 12,
    },
}
OOD_LENGTHS = tuple(range(4, 11))
MATCHED_LENGTHS = (4, 5)


def _metric_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.metrics.json"))
    paths.extend(sorted(root.glob("shard-*/*.metrics.json")))
    paths.extend(sorted(root.glob("shard_*/*.metrics.json")))
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _equal_length_accuracy(payload: dict[str, Any], lengths: tuple[int, ...]) -> float:
    return float(
        np.mean(
            [
                payload["metrics_by_chain_length"][str(length)]["accuracy"]
                for length in lengths
            ]
        )
    )


def _load_new_stream(
    name: str,
    plan_path: Path,
    result_root: Path,
) -> list[dict[str, Any]]:
    expected = EXPECTED_STREAMS[name]
    plan = read_plan(plan_path)
    if (
        plan["plan_sha256"] != expected["plan_sha256"]
        or len(plan["cells"]) != expected["cells"]
    ):
        raise ValueError(f"{name} plan does not match the frozen diagnostic")
    planned = {cell["cell_id"]: cell for cell in plan["cells"]}
    paths = _metric_paths(result_root)
    if len(paths) != expected["cells"]:
        raise ValueError(f"{name} stream is incomplete: {len(paths)}/{expected['cells']}")
    rows = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload["cell"]
        cell_id = cell["cell_id"]
        if (
            cell_id in seen
            or planned.get(cell_id) != cell
            or payload.get("status") != "complete"
            or payload.get("audit", {}).get("status") != "PASS"
        ):
            raise ValueError(f"invalid {name} cell: {cell_id}")
        seen.add(cell_id)
        rows.append(
            {
                "stream": name,
                "cell_id": cell_id,
                "decoder": cell["decoder"],
                "encoder": cell["encoder"],
                "seed": cell["seed"],
                "context_per_relation": cell["context_per_relation"],
                "context_fit_accuracy": payload["context_fit_metrics"]["accuracy"],
                "evaluation_accuracy": payload["metrics"]["accuracy"],
                "ood_equal_length_accuracy": _equal_length_accuracy(
                    payload, OOD_LENGTHS
                ),
                "matched_4_5_equal_length_accuracy": _equal_length_accuracy(
                    payload, MATCHED_LENGTHS
                ),
                "feature_count": payload["feature_count"],
                "decoder_package": payload["decoder_provenance"].get("package", ""),
                "decoder_package_version": payload["decoder_provenance"].get(
                    "package_version", ""
                ),
            }
        )
    if seen != set(planned):
        raise ValueError(f"{name} stream does not cover its complete plan")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["decoder"]), str(row["encoder"]))].append(row)
    output = []
    for (decoder, encoder), members in sorted(grouped.items()):
        output.append(
            {
                "decoder": decoder,
                "encoder": encoder,
                "seeds": len(members),
                "context_fit_accuracy": float(
                    np.mean([row["context_fit_accuracy"] for row in members])
                ),
                "evaluation_accuracy": float(
                    np.mean([row["evaluation_accuracy"] for row in members])
                ),
                "ood_equal_length_accuracy": float(
                    np.mean([row["ood_equal_length_accuracy"] for row in members])
                ),
                "matched_4_5_equal_length_accuracy": float(
                    np.mean(
                        [row["matched_4_5_equal_length_accuracy"] for row in members]
                    )
                ),
            }
        )
    return output


def _matched_clutrr_reference(*roots: Path) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, int], float] = {}
    for root in roots:
        for path in _metric_paths(root):
            payload = json.loads(path.read_text(encoding="utf-8"))
            cell = payload["cell"]
            if (
                cell["context_per_relation"] != 80
                or cell["encoder"] not in {"relation_multiset", "ordered_typed_path"}
            ):
                continue
            key = (cell["decoder"], cell["encoder"], cell["seed"])
            values[key] = _equal_length_accuracy(payload, MATCHED_LENGTHS)
    output = []
    for decoder in ("logistic", "catboost", "tabpfn"):
        multiset = np.mean(
            [values[(decoder, "relation_multiset", seed)] for seed in (101, 102, 103)]
        )
        ordered = np.mean(
            [values[(decoder, "ordered_typed_path", seed)] for seed in (101, 102, 103)]
        )
        output.append(
            {
                "decoder": decoder,
                "lengths": "4,5",
                "relation_multiset_accuracy": float(multiset),
                "ordered_path_accuracy": float(ordered),
                "multiset_minus_ordered": float(multiset - ordered),
            }
        )
    return output


def _representation_audit(path: Path) -> list[dict[str, Any]]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"relation_multiset", "aligned_multiset_with_supported_prefix"}
    output = []
    for row in audit["encoders"]:
        if row["encoder"] in wanted:
            output.append(
                {
                    "encoder": row["encoder"],
                    "accuracy_ceiling": row["overall"]["accuracy_ceiling"],
                    "singleton_fraction": row["overall"]["singleton_fraction"],
                    "collision_coverage": row["overall"]["collision_coverage"],
                }
            )
    if {row["encoder"] for row in output} != wanted:
        raise ValueError("CLUTRR representation audit is incomplete")
    return sorted(output, key=lambda row: row["encoder"])


def _d8_matched_context(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (row["encoder"], row["decoder"], int(row["context_per_class"])): float(
            row["seed_mean_ood_accuracy"]
        )
        for row in rows
    }
    output = []
    for context in (20, 50):
        for decoder in ("logistic", "catboost"):
            shared = values[("coarse_relation_multiset", decoder, context)]
            absolute = values[("ordered_typed_path", decoder, context)]
            combined = values[("ordered_plus_coarse_multiset", decoder, context)]
            permuted = values[
                (
                    "aligned_coarse_multiset_with_ordered_block_permutation",
                    decoder,
                    context,
                )
            ]
            output.append(
                {
                    "context_per_class": context,
                    "decoder": decoder,
                    "shared_minus_absolute": shared - absolute,
                    "permuted_minus_combined": permuted - combined,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-plan", type=Path, required=True)
    parser.add_argument("--cpu-results", type=Path, required=True)
    parser.add_argument("--gpu-plan", type=Path, required=True)
    parser.add_argument("--gpu-results", type=Path, required=True)
    parser.add_argument("--confirmation-cpu", type=Path, required=True)
    parser.add_argument("--confirmation-gpu", type=Path, required=True)
    parser.add_argument("--singleton-audit", type=Path, required=True)
    parser.add_argument("--d8-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cells = _load_new_stream("cpu", args.cpu_plan, args.cpu_results)
    cells.extend(_load_new_stream("gpu", args.gpu_plan, args.gpu_results))
    tables = {
        "cells.csv": cells,
        "aggregate.csv": _aggregate(cells),
        "matched_length_4_5.csv": _matched_clutrr_reference(
            args.confirmation_cpu, args.confirmation_gpu
        ),
        "representation_audit.csv": _representation_audit(args.singleton_audit),
        "dihedral16_matched_context.csv": _d8_matched_context(args.d8_aggregate),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        _write_csv(output / filename, rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "post-hoc shortcut diagnostic with matched-shift and independent-algebra "
            "boundary checks"
        ),
        "source_commit": "artifact-source-revision",
        "streams": EXPECTED_STREAMS,
        "interpretation": (
            "Absolute-prefix interfaces fit the short-chain context almost perfectly "
            "but transfer worse than the coarser multiset. The supported-prefix "
            "interface has a higher finite-population ceiling than the multiset, so the "
            "reversal is decoder extraction rather than information loss. The effect "
            "persists on CLUTRR lengths 4--5 but not on D8 at matched small contexts."
        ),
        "files": {
            filename: {"rows": len(rows), "sha256": file_sha256(output / filename)}
            for filename, rows in tables.items()
        },
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
