from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(comb(trials, index) for index in range(min(wins, losses) + 1))
    return float(min(1.0, 2.0 * tail / (2**trials)))


def _world_representation_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for world_record in audit["world_audits"]:
        world = str(world_record["world"])
        seeds = world_record["seeds"]
        class_counts = {
            int(seed_record["task"]["evaluation_class_count"]) for seed_record in seeds
        }
        evaluation_rows = {
            int(
                next(iter(seed_record["encoders"].values()))["ceiling"]["rows"]
            )
            for seed_record in seeds
        }
        if len(class_counts) != 1 or len(evaluation_rows) != 1:
            raise ValueError(f"GraphLog evaluation population changed across seeds: {world}")
        class_count = next(iter(class_counts))
        for encoder in sorted(seeds[0]["encoders"]):
            summaries = [seed_record["encoders"][encoder] for seed_record in seeds]
            rows.append(
                {
                    "world": world,
                    "encoder": encoder,
                    "evaluation_rows": next(iter(evaluation_rows)),
                    "evaluation_classes": class_count,
                    "balanced_accuracy_chance": 1.0 / class_count,
                    "feature_count": int(summaries[0]["feature_count"]),
                    "accuracy_ceiling": float(
                        np.mean(
                            [record["ceiling"]["ceiling"] for record in summaries]
                        )
                    ),
                    "balanced_accuracy_ceiling": float(
                        np.mean(
                            [
                                record["ceiling"]["balanced_accuracy_ceiling"]
                                for record in summaries
                            ]
                        )
                    ),
                    "singleton_fraction": float(
                        np.mean(
                            [
                                record["ceiling"]["singleton_fraction"]
                                for record in summaries
                            ]
                        )
                    ),
                    "collision_coverage": float(
                        np.mean(
                            [
                                record["ceiling"]["collision_coverage"]
                                for record in summaries
                            ]
                        )
                    ),
                    "active_cell_coverage": float(
                        np.mean(
                            [
                                record["support"]["active_cell_coverage"]
                                for record in summaries
                            ]
                        )
                    ),
                }
            )
    if len(rows) != 57 * 6:
        raise ValueError(f"expected 342 world/encoder rows, received {len(rows)}")
    return rows


def _representation_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["encoder"])].append(row)
    output = []
    for encoder, records in sorted(grouped.items()):
        if len(records) != 57:
            raise ValueError(f"expected 57 worlds for {encoder}")
        output.append(
            {
                "encoder": encoder,
                "worlds": len(records),
                "feature_count": int(records[0]["feature_count"]),
                "equal_world_accuracy_ceiling": float(
                    np.mean([record["accuracy_ceiling"] for record in records])
                ),
                "equal_world_balanced_accuracy_ceiling": float(
                    np.mean(
                        [record["balanced_accuracy_ceiling"] for record in records]
                    )
                ),
                "world_min_accuracy_ceiling": float(
                    np.min([record["accuracy_ceiling"] for record in records])
                ),
                "equal_world_singleton_fraction": float(
                    np.mean([record["singleton_fraction"] for record in records])
                ),
                "world_min_singleton_fraction": float(
                    np.min([record["singleton_fraction"] for record in records])
                ),
                "world_max_singleton_fraction": float(
                    np.max([record["singleton_fraction"] for record in records])
                ),
                "equal_world_collision_coverage": float(
                    np.mean([record["collision_coverage"] for record in records])
                ),
                "equal_world_active_cell_coverage": float(
                    np.mean([record["active_cell_coverage"] for record in records])
                ),
            }
        )
    return output


def _sign_sensitivities(
    contrast_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for row in contrast_rows:
        wins = int(row["world_wins"])
        ties = int(row["world_ties"])
        losses = int(row["world_losses"])
        if wins + ties + losses != 57:
            raise ValueError("GraphLog contrast does not contain 57 worlds")
        output.append(
            {
                "decoder": row["decoder"],
                "contrast": row["contrast"],
                "context_per_relation": int(row["context_per_relation"]),
                "equal_world_mean_delta": float(row["equal_world_mean_delta"]),
                "world_wins": wins,
                "world_ties": ties,
                "world_losses": losses,
                "two_sided_exact_sign_p": exact_two_sided_sign_pvalue(wins, losses),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-audit", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.schedule_audit.read_text())
    if audit.get("status") != "PASS" or audit.get("world_count") != 57:
        raise ValueError("GraphLog schedule audit is incomplete")
    world_rows = _world_representation_rows(audit)
    representation_summary = _representation_summary(world_rows)
    sign_sensitivities = _sign_sensitivities(_read_csv(args.contrasts))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "representation_by_world.csv": world_rows,
        "representation_summary.csv": representation_summary,
        "fixed_panel_sign_sensitivity.csv": sign_sensitivities,
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    class_counts = sorted({int(row["evaluation_classes"]) for row in world_rows})
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "descriptive representation audit and across-world sign sensitivity "
            "for the complete fixed GraphLog v1.1 panel"
        ),
        "worlds": 57,
        "evaluation_class_count_range": [min(class_counts), max(class_counts)],
        "balanced_accuracy_chance_range": [
            1.0 / max(class_counts),
            1.0 / min(class_counts),
        ],
        "sign_test_interpretation": (
            "sensitivity to directional consistency under a 0.5 exchangeable-sign "
            "null; not an interval for a population of logical worlds"
        ),
        "inputs": {
            "schedule_audit": {
                "path": str(args.schedule_audit),
                "sha256": file_sha256(args.schedule_audit),
            },
            "contrasts": {
                "path": str(args.contrasts),
                "sha256": file_sha256(args.contrasts),
            },
        },
        "files": {
            filename: {
                "rows": len(rows),
                "sha256": file_sha256(output_dir / filename),
            }
            for filename, rows in tables.items()
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
