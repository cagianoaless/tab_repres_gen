#!/usr/bin/env python3
"""Audit when binary alignment effects add information beyond raw ROC-AUC."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_panel(
    frame: pd.DataFrame,
    *,
    panel: str,
    group_columns: list[str],
    effect_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    if effect_column not in frame.columns:
        raise ValueError(f"{panel} is missing effect column: {effect_column}")
    frame = frame.rename(columns={effect_column: "alignment_effect"})
    required = {
        "decoder",
        "aligned",
        "conditional_null",
        "alignment_effect",
        *group_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{panel} is missing columns: {missing}")

    frame = frame.copy()
    frame["centered_aligned"] = frame["aligned"] - 0.5
    frame["effect_minus_centered"] = (
        frame["alignment_effect"] - frame["centered_aligned"]
    )
    if not np.allclose(
        frame["alignment_effect"],
        frame["aligned"] - frame["conditional_null"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"{panel} contains an inconsistent alignment effect")

    decoder_rows = []
    for decoder, group in frame.groupby("decoder", sort=True):
        decoder_rows.append(
            {
                "decoder": str(decoder),
                "cells": int(len(group)),
                "aligned_mean": float(group["aligned"].mean()),
                "conditional_null_mean": float(group["conditional_null"].mean()),
                "conditional_null_min": float(group["conditional_null"].min()),
                "conditional_null_max": float(group["conditional_null"].max()),
                "alignment_effect_mean": float(group["alignment_effect"].mean()),
                "effect_minus_centered_mean": float(
                    group["effect_minus_centered"].mean()
                ),
                "effect_minus_centered_max_abs": float(
                    group["effect_minus_centered"].abs().max()
                ),
            }
        )

    grouped = (
        frame.groupby([*group_columns, "decoder"], as_index=False)[
            ["aligned", "conditional_null", "alignment_effect"]
        ]
        .mean()
        .sort_values([*group_columns, "decoder"])
    )
    ranking_rows = []
    ranking_matches = 0
    pairwise_matches = 0
    pairwise_total = 0
    for keys, group in grouped.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        aligned_order = tuple(
            group.sort_values(["aligned", "decoder"], ascending=[False, True])[
                "decoder"
            ]
        )
        effect_order = tuple(
            group.sort_values(
                ["alignment_effect", "decoder"], ascending=[False, True]
            )["decoder"]
        )
        order_matches = aligned_order == effect_order
        ranking_matches += int(order_matches)

        records = {
            str(row.decoder): row
            for row in group.itertuples(index=False)
        }
        local_pairwise_matches = 0
        local_pairwise_total = 0
        for left, right in itertools.combinations(sorted(records), 2):
            aligned_delta = float(records[left].aligned - records[right].aligned)
            effect_delta = float(
                records[left].alignment_effect - records[right].alignment_effect
            )
            aligned_sign = int(np.sign(aligned_delta))
            effect_sign = int(np.sign(effect_delta))
            local_pairwise_matches += int(aligned_sign == effect_sign)
            local_pairwise_total += 1
        pairwise_matches += local_pairwise_matches
        pairwise_total += local_pairwise_total

        ranking_rows.append(
            {
                "panel": panel,
                **{
                    column: value
                    for column, value in zip(group_columns, keys, strict=True)
                },
                "aligned_order": ">".join(aligned_order),
                "alignment_effect_order": ">".join(effect_order),
                "order_matches": bool(order_matches),
                "pairwise_sign_matches": int(local_pairwise_matches),
                "pairwise_comparisons": int(local_pairwise_total),
            }
        )

    ranking_frame = pd.DataFrame(ranking_rows)
    summary = {
        "panel": panel,
        "cells": int(len(frame)),
        "group_columns": group_columns,
        "groups": int(len(ranking_frame)),
        "decoder_summaries": decoder_rows,
        "aggregate_null_mean": float(frame["conditional_null"].mean()),
        "aggregate_null_min": float(frame["conditional_null"].min()),
        "aggregate_null_max": float(frame["conditional_null"].max()),
        "aggregate_null_max_abs_deviation_from_chance": float(
            (frame["conditional_null"] - 0.5).abs().max()
        ),
        "ranking_order_matches": int(ranking_matches),
        "ranking_order_total": int(len(ranking_frame)),
        "pairwise_sign_matches": int(pairwise_matches),
        "pairwise_sign_total": int(pairwise_total),
        "reading": (
            "When aggregate null means remain near 0.5, the alignment effect is "
            "numerically close to centered raw ROC-AUC. Cell-level null variation "
            "still makes it a matched leakage and shortcut audit."
        ),
    }
    return summary, ranking_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-pairs", type=Path, required=True)
    parser.add_argument("--real-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    property_summary, property_rankings = summarize_panel(
        pd.read_csv(args.property_pairs),
        panel="controlled_properties",
        group_columns=["property", "context_per_class"],
        effect_column="conditional_effect",
    )
    real_summary, real_rankings = summarize_panel(
        pd.read_csv(args.real_pairs),
        panel="real_encoder_matrix",
        group_columns=["dataset", "encoder", "context_per_class"],
        effect_column="alignment_effect",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "scope": "binary aligned-versus-conditional-null audit",
                "panels": [property_summary, real_summary],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.concat([property_rankings, real_rankings], ignore_index=True).to_csv(
        args.output_dir / "group_rankings.csv",
        index=False,
    )
    print(
        "PASS "
        f"controlled={property_summary['cells']} "
        f"real={real_summary['cells']}"
    )


if __name__ == "__main__":
    main()
