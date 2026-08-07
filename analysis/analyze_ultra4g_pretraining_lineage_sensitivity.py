#!/usr/bin/env python3
"""Summarize ULTRA57 effects outside known ultra_4g source lineages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json

EXCLUDED_FAMILIES = {
    "codex": "CoDEx-Medium is an ultra_4g pretraining graph",
    "fb15k237": "FB15k-237 is an ultra_4g pretraining graph",
    "fbnell": "FBNELL is constructed from FB15k-237 and NELL-995",
    "nell": "NELL-995 is an ultra_4g pretraining graph",
    "wn18rr": "WN18RR is an ultra_4g pretraining graph",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path: Path, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    if (
        frame["provenance_family"].duplicated().any()
        or not set(EXCLUDED_FAMILIES) <= set(frame["provenance_family"])
    ):
        raise ValueError("unexpected provenance-family table")
    retained = frame.loc[
        ~frame["provenance_family"].isin(EXCLUDED_FAMILIES)
    ].copy()
    contrast_columns = [
        column for column in frame.columns if column != "provenance_family"
    ]
    rows = []
    for contrast in contrast_columns:
        values = retained[contrast].to_numpy(dtype=np.float64)
        rows.append(
            {
                "contrast_id": contrast,
                "retained_equal_family_effect": float(values.mean()),
                "retained_family_wins": int((values > 0.0).sum()),
                "retained_family_ties": int((values == 0.0).sum()),
                "retained_family_losses": int((values < 0.0).sum()),
                "retained_family_minimum": float(values.min()),
                "retained_family_maximum": float(values.max()),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "contrast_summary.csv", rows)
    retained.to_csv(
        output / "retained_family_effects.csv",
        index=False,
        lineterminator="\n",
    )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": (
            "descriptive sensitivity excluding provenance families with direct "
            "lineage to the four ultra_4g pretraining graphs"
        ),
        "input_sha256": file_sha256(input_path),
        "excluded_families": EXCLUDED_FAMILIES,
        "retained_families": sorted(retained["provenance_family"].tolist()),
        "retained_family_count": len(retained),
        "contrasts": rows,
        "interpretation_boundary": (
            "This restriction addresses known source lineage only; it is not a "
            "claim that all remaining data are distributionally independent of "
            "ULTRA pretraining. Several retained families share a broad Wikidata "
            "upstream source with CoDEx, which is disclosed separately and is not "
            "treated as direct dataset lineage."
        ),
        "files": {
            name: file_sha256(output / name)
            for name in ("contrast_summary.csv", "retained_family_effects.csv")
        },
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.input.resolve(), args.output.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
