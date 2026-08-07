from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.graphlog import load_graphlog_task
from relational_tfm.encoders.graphlog import encode_graphlog_rows
from relational_tfm.evaluation.classification import collision_conditioned_ceiling
from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json, build_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/studies/graphlog_symbolic_cpu_confirmation_v1.yaml"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _representative_cell(
    cells: list[dict[str, Any]], world: str
) -> dict[str, Any]:
    matches = [
        cell
        for cell in cells
        if cell["world"] == world
        and cell["decoder"] == "logistic"
        and cell["encoder"] == "ordered_plus_multiset"
        and cell["context_per_relation"] == 40
        and cell["seed"] == 101
    ]
    if len(matches) != 1:
        raise ValueError(f"representative GraphLog cell is not unique: {world}")
    return matches[0]


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["encoder"]), str(row["evaluation_length"]))].append(row)
    output = []
    for (encoder, length), records in sorted(grouped.items()):
        if len(records) != 57:
            raise ValueError(f"expected 57 worlds for {encoder}/{length}")
        output.append(
            {
                "encoder": encoder,
                "evaluation_length": length,
                "worlds": len(records),
                "feature_count": int(records[0]["feature_count"]),
                "equal_world_accuracy_ceiling": float(
                    np.mean([record["accuracy_ceiling"] for record in records])
                ),
                "equal_world_singleton_fraction": float(
                    np.mean([record["singleton_fraction"] for record in records])
                ),
                "equal_world_collision_coverage": float(
                    np.mean([record["collision_coverage"] for record in records])
                ),
                "equal_world_balanced_accuracy_chance": float(
                    np.mean([record["balanced_accuracy_chance"] for record in records])
                ),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = build_plan(load_study_config(CONFIG))
    encoders = sorted({str(cell["encoder"]) for cell in plan["cells"]})
    rows: list[dict[str, Any]] = []
    for world_index in range(57):
        world = f"rule_{world_index}"
        cell = _representative_cell(plan["cells"], world)
        task = load_graphlog_task(REPO_ROOT, cell)
        for encoder in encoders:
            encoded = encode_graphlog_rows(
                task.evaluation,
                encoder,
                int(cell["encoder_seed"]) + 1,
            )
            for length in (4, 5):
                mask = task.evaluation["chain_length"].to_numpy() == length
                summary = collision_conditioned_ceiling(
                    encoded.values[mask],
                    task.evaluation.loc[mask, "label"].to_numpy(dtype=np.int64),
                )
                class_count = int(task.evaluation.loc[mask, "label"].nunique())
                rows.append(
                    {
                        "world": world,
                        "encoder": encoder,
                        "evaluation_length": length,
                        "evaluation_rows": int(mask.sum()),
                        "evaluation_classes": class_count,
                        "balanced_accuracy_chance": 1.0 / class_count,
                        "feature_count": len(encoded.feature_names),
                        "accuracy_ceiling": float(summary["ceiling"]),
                        "singleton_fraction": float(summary["singleton_fraction"]),
                        "collision_coverage": float(summary["collision_coverage"]),
                    }
                )
    if len(rows) != 57 * 6 * 2:
        raise ValueError(f"expected 684 world/encoder/length rows, received {len(rows)}")
    aggregate = _aggregate(rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "representation_by_world_and_length.csv": rows,
        "representation_by_length_summary.csv": aggregate,
    }
    for filename, table in tables.items():
        _write_csv(output_dir / filename, table)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "exact finite-population GraphLog representation audit by OOD length"
        ),
        "plan_sha256": plan["plan_sha256"],
        "worlds": 57,
        "evaluation_lengths": [4, 5],
        "representative_seed": 101,
        "files": {
            filename: {
                "rows": len(table),
                "sha256": file_sha256(output_dir / filename),
            }
            for filename, table in tables.items()
        },
    }
    atomic_write_json(output_dir / "representation_by_length_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
