#!/usr/bin/env python3
"""Multiplicity sensitivity for the selected RFMLink q64 panel contrasts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STUDY_ID = "rfmlink_ultra57_q64_multiplicity_complete57_v1"
SOURCE_STUDY_ID = "rfmlink_ultra57_q64_complete57_v1"

SELECTED_CONTRASTS = (
    (
        "local_relational_vs_entity_tokens_c200_combined",
        "local_relational_vs_entity_tokens",
        200,
        "combined",
    ),
    (
        "ultra_score_given_global_structure_c200_combined",
        "ultra_score_given_global_structure",
        200,
        "combined",
    ),
    (
        "global_structure_given_ultra_c20_combined",
        "global_structure_given_ultra",
        20,
        "combined",
    ),
    (
        "global_structure_given_ultra_c200_combined",
        "global_structure_given_ultra",
        200,
        "combined",
    ),
    (
        "global_structure_given_ultra_c200_head",
        "global_structure_given_ultra",
        200,
        "head",
    ),
    (
        "global_structure_given_ultra_c200_tail",
        "global_structure_given_ultra",
        200,
        "tail",
    ),
    (
        "global_structure_x_ultra_score_c200_combined",
        "global_structure_x_ultra_score",
        200,
        "combined",
    ),
    (
        "full_interface_vs_native_ultra_c200_combined",
        "full_interface_vs_native_ultra",
        200,
        "combined",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_family_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contrast",
        "direction",
        "context_per_class",
        "provenance_family",
        "effect",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")

    columns: dict[str, pd.Series] = {}
    expected_families: tuple[str, ...] | None = None
    for label, contrast, context, direction in SELECTED_CONTRASTS:
        selected = frame.loc[
            (frame["contrast"] == contrast)
            & (frame["context_per_class"] == context)
            & (frame["direction"] == direction),
            ["provenance_family", "effect"],
        ]
        if selected["provenance_family"].duplicated().any():
            raise ValueError(f"duplicate family effect for {label}")
        selected = selected.set_index("provenance_family")["effect"].sort_index()
        families = tuple(selected.index)
        if expected_families is None:
            expected_families = families
        elif families != expected_families:
            raise ValueError(f"family panel differs for {label}")
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite family effect for {label}")
        columns[label] = selected

    matrix = pd.DataFrame(columns)
    matrix.index.name = "provenance_family"
    if len(matrix) < 3:
        raise ValueError("at least three provenance families are required")
    if len(matrix) > 20:
        raise ValueError("exact sign enumeration is limited to at most 20 families")
    return matrix.reset_index()


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    raw_p = np.asarray(raw_p, dtype=float)
    order = np.argsort(raw_p, kind="stable")
    adjusted = np.empty_like(raw_p)
    running = 0.0
    tests = len(raw_p)
    for rank, index in enumerate(order):
        running = max(running, (tests - rank) * raw_p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def studentized_means(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    if (standard_errors <= 0).any():
        raise ValueError("studentized sign-flip test requires nonzero family variation")
    return means / standard_errors


def exact_joint_sign_flip(matrix: pd.DataFrame) -> pd.DataFrame:
    labels = [column for column in matrix.columns if column != "provenance_family"]
    values = matrix[labels].to_numpy(dtype=float)
    family_count = len(values)
    observed_mean = values.mean(axis=0)
    observed_se = values.std(axis=0, ddof=1) / np.sqrt(family_count)
    observed_t = observed_mean / observed_se

    sign_patterns = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=family_count)),
        dtype=float,
    )
    null_t = np.empty((len(sign_patterns), len(labels)), dtype=float)
    for index, signs in enumerate(sign_patterns):
        null_t[index] = studentized_means(values * signs[:, None])

    absolute_null = np.abs(null_t)
    absolute_observed = np.abs(observed_t)
    raw_p = (absolute_null >= absolute_observed[None, :]).mean(axis=0)
    max_t = absolute_null.max(axis=1)
    max_t_adjusted = (max_t[:, None] >= absolute_observed[None, :]).mean(axis=0)
    holm_adjusted = holm_adjust(raw_p)

    return pd.DataFrame(
        {
            "contrast_id": labels,
            "equal_family_effect": observed_mean,
            "family_standard_error": observed_se,
            "studentized_statistic": observed_t,
            "two_sided_sign_flip_p": raw_p,
            "holm_adjusted_p": holm_adjusted,
            "joint_max_t_adjusted_p": max_t_adjusted,
            "family_count": family_count,
            "exact_sign_patterns": len(sign_patterns),
        }
    )


def leave_one_family_out(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [column for column in matrix.columns if column != "provenance_family"]
    records: list[dict[str, Any]] = []
    for omitted_family in matrix["provenance_family"]:
        retained = matrix.loc[matrix["provenance_family"] != omitted_family]
        for contrast_id in labels:
            records.append(
                {
                    "contrast_id": contrast_id,
                    "omitted_family": omitted_family,
                    "leave_one_family_out_effect": float(retained[contrast_id].mean()),
                    "retained_family_count": int(len(retained)),
                }
            )
    estimates = pd.DataFrame(records)

    summaries: list[dict[str, Any]] = []
    for contrast_id in labels:
        full_effect = float(matrix[contrast_id].mean())
        selected = estimates.loc[
            estimates["contrast_id"] == contrast_id,
            ["omitted_family", "leave_one_family_out_effect"],
        ].copy()
        selected["absolute_change"] = (
            selected["leave_one_family_out_effect"] - full_effect
        ).abs()
        most_influential = selected.sort_values(
            ["absolute_change", "omitted_family"],
            ascending=[False, True],
        ).iloc[0]
        summaries.append(
            {
                "contrast_id": contrast_id,
                "full_equal_family_effect": full_effect,
                "minimum_leave_one_family_out_effect": float(
                    selected["leave_one_family_out_effect"].min()
                ),
                "maximum_leave_one_family_out_effect": float(
                    selected["leave_one_family_out_effect"].max()
                ),
                "all_leave_one_family_out_same_sign_as_full": bool(
                    (
                        np.sign(selected["leave_one_family_out_effect"])
                        == np.sign(full_effect)
                    ).all()
                ),
                "most_influential_omitted_family": str(
                    most_influential["omitted_family"]
                ),
                "largest_absolute_change": float(
                    most_influential["absolute_change"]
                ),
            }
        )
    return estimates, pd.DataFrame(summaries)


def render_readme(
    summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    source_path: Path,
    source_sha256: str,
) -> str:
    robust = summary.loc[
        summary["joint_max_t_adjusted_p"] < 0.05,
        ["contrast_id", "equal_family_effect", "joint_max_t_adjusted_p"],
    ]
    robust_lines = "\n".join(
        f"- `{row.contrast_id}`: effect {row.equal_family_effect:+.4f}, "
        f"joint maxT p={row.joint_max_t_adjusted_p:.5f}"
        for row in robust.itertuples(index=False)
    )
    if not robust_lines:
        robust_lines = "- None."
    return f"""# RFMLink q64 multiplicity sensitivity

Status: **POST-HOC FIXED-PANEL SENSITIVITY**.

This audit evaluates the eight selected frozen-ULTRA panel contrasts together.
The unit is the metric-free provenance family. Every exact sign pattern applies
the same family sign to all contrasts, preserving their observed cross-contrast
dependence. Statistics are studentized family means.

The output reports:

- an exact two-sided family sign-flip p-value for each contrast;
- Holm adjustment across the eight reported contrasts;
- a joint max-|t| adjustment from the shared exact sign-flip distribution.
- every leave-one-provenance-family-out estimate and its range.

The sign-flip analysis assumes symmetry of the family-level effects under the
zero null. It is a simultaneous sensitivity for this fixed panel, not uncertainty
over a population of graph families and not a replacement for the unadjusted
effect estimates and fixed-panel bootstrap intervals.
The eight contrasts were selected post hoc as the reported scientific family.
The max-|t| adjustment controls this family only, not every contrast that could
have been computed after inspecting the panel.

## Contrasts surviving joint maxT at 0.05

{robust_lines}

The direction-specific global-feature result and the global-by-score interaction
do not survive the eight-contrast simultaneous sensitivity and must remain
exploratory.

## Leave-one-family-out

{chr(10).join(
    f"- `{row.contrast_id}`: full {row.full_equal_family_effect:+.4f}; "
    f"leave-one-family-out range "
    f"[{row.minimum_leave_one_family_out_effect:+.4f}, "
    f"{row.maximum_leave_one_family_out_effect:+.4f}]; "
    f"same sign throughout: "
    f"{str(row.all_leave_one_family_out_same_sign_as_full).lower()}."
    for row in loo_summary.itertuples(index=False)
)}

## Provenance

- source table: `{source_path.as_posix()}`
- source SHA256: `{source_sha256}`
- source study: `{SOURCE_STUDY_ID}`
- contrast count: {len(summary)}
- family count: {int(summary["family_count"].iloc[0])}
- exact joint sign patterns: {int(summary["exact_sign_patterns"].iloc[0]):,}
"""


def write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    write_json(
        output_dir / "manifest.json",
        {
            "study_id": STUDY_ID,
            "status": "COMPLETE",
            "files": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/ultra57/q64-encoder-ladder/contrast_by_family.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ultra57/multiplicity"),
    )
    args = parser.parse_args()

    source_sha256 = file_sha256(args.input)
    matrix = selected_family_matrix(pd.read_csv(args.input))
    summary = exact_joint_sign_flip(matrix)
    loo_estimates, loo_summary = leave_one_family_out(matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "selected_family_effects.csv", index=False)
    summary.to_csv(args.output_dir / "multiplicity_summary.csv", index=False)
    loo_estimates.to_csv(
        args.output_dir / "leave_one_family_out_estimates.csv",
        index=False,
    )
    loo_summary.to_csv(
        args.output_dir / "leave_one_family_out_summary.csv",
        index=False,
    )
    write_json(
        args.output_dir / "audit.json",
        {
            "study_id": STUDY_ID,
            "status": "COMPLETE",
            "scientific_status": "post_hoc_fixed_panel_multiplicity_sensitivity",
            "source_study_id": SOURCE_STUDY_ID,
            "source_table": args.input.as_posix(),
            "source_sha256": source_sha256,
            "family_count": int(len(matrix)),
            "contrast_count": int(len(summary)),
            "exact_joint_sign_patterns": int(summary["exact_sign_patterns"].iloc[0]),
            "unit": "metric_free_provenance_family",
            "shared_signs_across_contrasts": True,
            "statistics": [
                "two_sided_exact_family_sign_flip",
                "holm_adjusted",
                "joint_max_abs_studentized_statistic",
            ],
            "interpretation": "fixed_panel_sensitivity_not_population_inference",
        },
    )
    (args.output_dir / "README.md").write_text(
        render_readme(summary, loo_summary, args.input, source_sha256),
        encoding="utf-8",
    )
    write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
