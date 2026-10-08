#!/usr/bin/env python3
"""Analyze the complete 57-dataset RFMLink q64 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STUDY_ID = "rfmlink_ultra57_q64_complete57_v1"
SOURCE_RESULT_BRANCH = "results/rfmlink_ultra57_q64_encoder_ladder_v3_remote32"
SOURCE_RESULT_COMMIT = "1aea2543633d7b821e7ca9ac78eb18dd8974dd12"
SOURCE_RELATIVE_PATH = (
    "results/cluster/rfmlink_ultra57_q64_encoder_ladder_v3/remote32/"
    "paper_summary/cell_metrics.csv"
)
INTERFACES = (
    "entity_tokens_only",
    "b0_dyadic_only",
    "b0_paths_only",
    "b0_local_only",
    "ultra_score_only",
    "b2v2_only",
    "b2v2_structure_only",
    "b0_local_ultra",
    "b2v2_full",
)
CONTEXTS = (20, 50, 100, 200)
SEEDS = (101, 102, 103)
DIRECTIONS = ("combined", "head", "tail")
PRACTICAL_TIE_BAND = 0.005
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 100_000

# Metric-free provenance grouping. Variants derived from the same source graph
# receive one family vote in the primary fixed-panel sensitivity analysis.
PROVENANCE_FAMILIES = {
    "aristov4": ("aristov4",),
    "codex": ("codex_small", "codex_medium", "codex_large"),
    "conceptnet100k": ("conceptnet100k",),
    "dbpedia100k": ("dbpedia100k",),
    "fb15k237": (
        "fb15k237",
        "fb15k237_10",
        "fb15k237_20",
        "fb15k237_50",
        "fb15k237_ind_v1",
        "fb15k237_ind_v2",
        "fb15k237_ind_v3",
        "fb15k237_ind_v4",
        "fb_ingram_25",
        "fb_ingram_50",
        "fb_ingram_75",
        "fb_ingram_100",
    ),
    "fbnell": ("fbnell",),
    "hetionet": ("hetionet",),
    "hm": ("hm_1k", "hm_3k", "hm_5k", "hm_indigo"),
    "ilpc2022": ("ilpc2022_small", "ilpc2022_large"),
    "metafam": ("metafam",),
    "nell": (
        "nell23k",
        "nell995",
        "nell_ind_v1",
        "nell_ind_v2",
        "nell_ind_v3",
        "nell_ind_v4",
        "nl_ingram_0",
        "nl_ingram_25",
        "nl_ingram_50",
        "nl_ingram_75",
        "nl_ingram_100",
    ),
    "wdsinger": ("wdsinger",),
    "wiki_topics": (
        "wiki_mt1_tax",
        "wiki_mt1_health",
        "wiki_mt2_org",
        "wiki_mt2_sci",
        "wiki_mt3_art",
        "wiki_mt3_infra",
        "wiki_mt4_sci",
        "wiki_mt4_health",
    ),
    "wk_ingram": ("wk_ingram_25", "wk_ingram_50", "wk_ingram_75", "wk_ingram_100"),
    "wn18rr": (
        "wn18rr",
        "wn18rr_ind_v1",
        "wn18rr_ind_v2",
        "wn18rr_ind_v3",
        "wn18rr_ind_v4",
    ),
    "yago310": ("yago310",),
}

CONTRASTS = {
    "local_relational_vs_entity_tokens": ("b0_local_only", "entity_tokens_only"),
    "ultra_score_given_local": ("b0_local_ultra", "b0_local_only"),
    "ultra_score_given_global_structure": ("b2v2_full", "b2v2_structure_only"),
    "global_structure_given_ultra": ("b2v2_full", "b0_local_ultra"),
    "global_structure_without_ultra": ("b2v2_structure_only", "b0_local_only"),
    "local_plus_ultra_vs_ultra_only": ("b0_local_ultra", "ultra_score_only"),
    "full_interface_vs_ultra_only": ("b2v2_full", "ultra_score_only"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provenance_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for family, members in PROVENANCE_FAMILIES.items():
        for dataset in members:
            if dataset in lookup:
                raise ValueError(f"duplicate provenance assignment for {dataset}")
            lookup[dataset] = family
    return lookup


def validate_complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "transfer_regime",
        "interface",
        "context_per_class",
        "seed",
        "decoder_fit_status",
        "context_representation_constant",
    }
    for direction in DIRECTIONS:
        required.add(f"{direction}_mrr")
        required.add(f"native_{direction}_mrr")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")

    complete = frame.copy()
    lookup = provenance_lookup()
    unknown = sorted(set(complete["dataset"]) - set(lookup))
    absent = sorted(set(lookup) - set(complete["dataset"]))
    if unknown or absent:
        raise ValueError(f"provenance coverage mismatch: unknown={unknown}, absent={absent}")
    complete["provenance_family"] = complete["dataset"].map(lookup)

    if set(complete["interface"]) != set(INTERFACES):
        raise ValueError("interface panel differs from the frozen nine-interface design")
    if set(complete["context_per_class"]) != set(CONTEXTS):
        raise ValueError("context panel differs from the frozen four-budget design")
    if set(complete["seed"]) != set(SEEDS):
        raise ValueError("seed panel differs from the frozen three-seed design")
    if complete.duplicated(["dataset", "interface", "context_per_class", "seed"]).any():
        raise ValueError("duplicate dataset/interface/context/seed cells")

    expected_cells = len(INTERFACES) * len(CONTEXTS) * len(SEEDS)
    counts = complete.groupby("dataset", observed=True).size()
    if len(counts) != 57 or not (counts == expected_cells).all():
        raise ValueError(
            f"complete-case imbalance: datasets={len(counts)}, "
            f"counts={counts.value_counts().to_dict()}"
        )
    if len(complete) != 57 * expected_cells:
        raise ValueError("complete-case row count differs from 6,156")
    if complete[[f"{direction}_mrr" for direction in DIRECTIONS]].isna().any().any():
        raise ValueError("non-finite MRR in complete case")
    return complete


def dataset_metrics(complete: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in complete.columns
        if any(
            column.startswith(prefix)
            for prefix in (
                "combined_",
                "head_",
                "tail_",
                "native_combined_",
                "native_head_",
                "native_tail_",
                "delta_combined_",
                "delta_head_",
                "delta_tail_",
            )
        )
    ]
    keys = [
        "dataset",
        "provenance_family",
        "transfer_regime",
        "interface",
        "context_per_class",
    ]
    return (
        complete.groupby(keys, as_index=False, observed=True)[metric_columns]
        .mean()
        .sort_values(keys, ignore_index=True)
    )


def paired_effects(
    dataset: pd.DataFrame,
    left: str,
    right: str,
    direction: str,
    context: int,
) -> pd.DataFrame:
    subset = dataset.loc[
        (dataset["context_per_class"] == context) & dataset["interface"].isin((left, right)),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            "interface",
            f"{direction}_mrr",
        ],
    ]
    pivot = subset.pivot(
        index=["dataset", "provenance_family", "transfer_regime"],
        columns="interface",
        values=f"{direction}_mrr",
    )
    if set(pivot.columns) != {left, right} or len(pivot) != 57:
        raise ValueError(f"incomplete pair for {left} - {right}, {direction}, c={context}")
    output = pivot.reset_index()
    output["effect"] = output[left] - output[right]
    return output


def native_effects(
    dataset: pd.DataFrame, interface: str, direction: str, context: int
) -> pd.DataFrame:
    subset = dataset.loc[
        (dataset["context_per_class"] == context) & (dataset["interface"] == interface),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            f"{direction}_mrr",
            f"native_{direction}_mrr",
        ],
    ].copy()
    if len(subset) != 57:
        raise ValueError(f"incomplete native comparison for {interface}, c={context}")
    subset["effect"] = subset[f"{direction}_mrr"] - subset[f"native_{direction}_mrr"]
    return subset


def interaction_effects(dataset: pd.DataFrame, direction: str, context: int) -> pd.DataFrame:
    with_ultra = paired_effects(dataset, "b2v2_full", "b0_local_ultra", direction, context).rename(
        columns={"effect": "with_ultra"}
    )
    without_ultra = paired_effects(
        dataset, "b2v2_structure_only", "b0_local_only", direction, context
    ).rename(columns={"effect": "without_ultra"})
    keys = ["dataset", "provenance_family", "transfer_regime"]
    merged = with_ultra[keys + ["with_ultra"]].merge(
        without_ultra[keys + ["without_ultra"]], on=keys, validate="one_to_one"
    )
    merged["effect"] = merged["with_ultra"] - merged["without_ultra"]
    return merged


def context_change_effects(dataset: pd.DataFrame, interface: str, direction: str) -> pd.DataFrame:
    subset = dataset.loc[
        (dataset["interface"] == interface) & dataset["context_per_class"].isin((20, 200)),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            "context_per_class",
            f"{direction}_mrr",
        ],
    ]
    pivot = subset.pivot(
        index=["dataset", "provenance_family", "transfer_regime"],
        columns="context_per_class",
        values=f"{direction}_mrr",
    )
    if set(pivot.columns) != {20, 200} or len(pivot) != 57:
        raise ValueError(f"incomplete context change for {interface}/{direction}")
    output = pivot.reset_index()
    output["effect"] = output[200] - output[20]
    return output


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_effect(
    effects: pd.DataFrame,
    *,
    contrast: str,
    direction: str,
    context: str | int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    family = (
        effects.groupby("provenance_family", as_index=False, observed=True)["effect"]
        .mean()
        .sort_values("provenance_family", ignore_index=True)
    )
    values = family["effect"].to_numpy()
    low, high = bootstrap_mean_interval(values)
    leave_one_out = np.asarray([np.delete(values, index).mean() for index in range(len(values))])
    dataset_values = effects["effect"].to_numpy()
    row = {
        "contrast": contrast,
        "direction": direction,
        "context_per_class": context,
        "equal_family_effect": float(values.mean()),
        "fixed_panel_bootstrap_low": low,
        "fixed_panel_bootstrap_high": high,
        "family_count": int(len(family)),
        "positive_families": int((values > 0).sum()),
        "negative_families": int((values < 0).sum()),
        "leave_one_family_out_min": float(leave_one_out.min()),
        "leave_one_family_out_max": float(leave_one_out.max()),
        "equal_dataset_effect": float(dataset_values.mean()),
        "median_dataset_effect": float(np.median(dataset_values)),
        "dataset_count": int(len(dataset_values)),
        "dataset_wins": int((dataset_values > PRACTICAL_TIE_BAND).sum()),
        "dataset_ties": int((np.abs(dataset_values) <= PRACTICAL_TIE_BAND).sum()),
        "dataset_losses": int((dataset_values < -PRACTICAL_TIE_BAND).sum()),
    }
    family.insert(0, "context_per_class", context)
    family.insert(0, "direction", direction)
    family.insert(0, "contrast", contrast)
    return row, family


def absolute_mrr_tables(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = dataset[
        [
            "dataset",
            "provenance_family",
            "interface",
            "context_per_class",
            "combined_mrr",
        ]
    ]
    equal_dataset = (
        subset.groupby(["interface", "context_per_class"], as_index=False, observed=True)[
            "combined_mrr"
        ]
        .mean()
        .rename(columns={"combined_mrr": "equal_dataset_mrr"})
    )
    family = subset.groupby(
        ["provenance_family", "interface", "context_per_class"],
        as_index=False,
        observed=True,
    )["combined_mrr"].mean()
    equal_family = (
        family.groupby(["interface", "context_per_class"], as_index=False, observed=True)[
            "combined_mrr"
        ]
        .mean()
        .rename(columns={"combined_mrr": "equal_family_mrr"})
    )
    return equal_family, equal_dataset


def summarize_contrasts(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    family_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    dataset_rows: list[pd.DataFrame] = []
    for context in CONTEXTS:
        for direction in DIRECTIONS:
            effect_sets = {
                name: paired_effects(dataset, left, right, direction, context)
                for name, (left, right) in CONTRASTS.items()
            }
            effect_sets["global_structure_x_ultra_score"] = interaction_effects(
                dataset, direction, context
            )
            effect_sets["full_interface_vs_native_ultra"] = native_effects(
                dataset, "b2v2_full", direction, context
            )
            for name, effects in effect_sets.items():
                row, family = summarize_effect(
                    effects, contrast=name, direction=direction, context=context
                )
                summary_rows.append(row)
                family_rows.append(family)
                labeled_effects = effects[
                    ["dataset", "provenance_family", "transfer_regime", "effect"]
                ].copy()
                labeled_effects.insert(0, "context_per_class", context)
                labeled_effects.insert(0, "direction", direction)
                labeled_effects.insert(0, "contrast", name)
                dataset_rows.append(labeled_effects)
                for regime, members in effects.groupby("transfer_regime", observed=True):
                    regime_family = members.groupby("provenance_family", observed=True)[
                        "effect"
                    ].mean()
                    low, high = bootstrap_mean_interval(regime_family.to_numpy())
                    regime_rows.append(
                        {
                            "contrast": name,
                            "direction": direction,
                            "context_per_class": context,
                            "transfer_regime": regime,
                            "equal_family_within_regime_effect": float(regime_family.mean()),
                            "fixed_panel_bootstrap_low": low,
                            "fixed_panel_bootstrap_high": high,
                            "families_in_regime": int(len(regime_family)),
                            "datasets_in_regime": int(len(members)),
                            "equal_dataset_within_regime_effect": float(members["effect"].mean()),
                        }
                    )
    return (
        pd.DataFrame(summary_rows),
        pd.concat(family_rows, ignore_index=True),
        pd.DataFrame(regime_rows),
        pd.concat(dataset_rows, ignore_index=True),
    )


def summarize_context_scaling(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    family_rows: list[pd.DataFrame] = []
    for interface in INTERFACES:
        for direction in DIRECTIONS:
            effects = context_change_effects(dataset, interface, direction)
            row, family = summarize_effect(
                effects,
                contrast=f"context_200_minus_20::{interface}",
                direction=direction,
                context="200_minus_20",
            )
            row["interface"] = interface
            rows.append(row)
            family["interface"] = interface
            family_rows.append(family)
    return pd.DataFrame(rows), pd.concat(family_rows, ignore_index=True)


def summarize_constant_fallbacks(complete: pd.DataFrame) -> pd.DataFrame:
    keys = ["interface", "context_per_class", "transfer_regime"]
    rows = []
    for group, members in complete.groupby(keys, observed=True):
        fallback_mask = members["context_representation_constant"].astype(bool)
        rows.append(
            {
                **dict(zip(keys, group, strict=True)),
                "cells": int(len(members)),
                "fallback_cells": int(fallback_mask.sum()),
                "affected_datasets": int(members.loc[fallback_mask, "dataset"].nunique()),
            }
        )
    fallback = pd.DataFrame(rows)
    fallback["fallback_rate"] = fallback["fallback_cells"] / fallback["cells"]
    return fallback


def selected_value(
    table: pd.DataFrame,
    contrast: str,
    context: int,
    direction: str = "combined",
) -> dict[str, Any]:
    match = table.loc[
        (table["contrast"] == contrast)
        & (table["context_per_class"] == context)
        & (table["direction"] == direction)
    ]
    if len(match) != 1:
        raise ValueError(f"missing selected finding: {contrast}, {context}, {direction}")
    return match.iloc[0].to_dict()


def selected_regime_value(
    table: pd.DataFrame,
    contrast: str,
    context: int,
    regime: str,
    direction: str = "combined",
) -> dict[str, Any]:
    match = table.loc[
        (table["contrast"] == contrast)
        & (table["context_per_class"] == context)
        & (table["direction"] == direction)
        & (table["transfer_regime"] == regime)
    ]
    if len(match) != 1:
        raise ValueError(f"missing regime finding: {contrast}, {context}, {direction}, {regime}")
    return match.iloc[0].to_dict()


def native_ultra_sensitivity(contrast_family: pd.DataFrame) -> pd.DataFrame:
    subset = contrast_family.loc[contrast_family["contrast"] == "full_interface_vs_native_ultra"]
    rows = []
    for (context, direction), members in subset.groupby(
        ["context_per_class", "direction"], observed=True
    ):
        without_metafam = members.loc[members["provenance_family"] != "metafam", "effect"]
        rows.append(
            {
                "context_per_class": context,
                "direction": direction,
                "equal_family_effect": float(members["effect"].mean()),
                "metafam_effect": float(
                    members.loc[members["provenance_family"] == "metafam", "effect"].iloc[0]
                ),
                "equal_family_effect_without_metafam": float(without_metafam.mean()),
                "positive_families": int((members["effect"] > 0).sum()),
                "negative_families": int((members["effect"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def render_readme(
    contrasts: pd.DataFrame,
    contrast_regime: pd.DataFrame,
    scaling: pd.DataFrame,
    complete: pd.DataFrame,
    native_sensitivity: pd.DataFrame,
    source_sha256: str,
) -> str:
    local = selected_value(contrasts, "local_relational_vs_entity_tokens", 200)
    ultra = selected_value(contrasts, "ultra_score_given_global_structure", 200)
    b2_low = selected_value(contrasts, "global_structure_given_ultra", 20)
    b2_combined = selected_value(contrasts, "global_structure_given_ultra", 200)
    b2_tail = selected_value(contrasts, "global_structure_given_ultra", 200, direction="tail")
    b2_head = selected_value(contrasts, "global_structure_given_ultra", 200, direction="head")
    interaction = selected_value(contrasts, "global_structure_x_ultra_score", 200)
    native = selected_value(contrasts, "full_interface_vs_native_ultra", 200)
    inductive_entities = selected_regime_value(
        contrast_regime,
        "global_structure_given_ultra",
        200,
        "inductive_entities",
    )
    inductive_entities_relations = selected_regime_value(
        contrast_regime,
        "global_structure_given_ultra",
        200,
        "inductive_entities_relations",
    )
    transductive = selected_regime_value(
        contrast_regime,
        "global_structure_given_ultra",
        200,
        "transductive",
    )
    native_without_metafam = native_sensitivity.loc[
        (native_sensitivity["context_per_class"] == 200)
        & (native_sensitivity["direction"] == "combined"),
        "equal_family_effect_without_metafam",
    ].iloc[0]
    local_scaling = scaling.loc[
        (scaling["interface"] == "b0_local_only") & (scaling["direction"] == "combined")
    ].iloc[0]
    full_scaling = scaling.loc[
        (scaling["interface"] == "b2v2_full") & (scaling["direction"] == "combined")
    ].iloc[0]
    tokens_scaling = scaling.loc[
        (scaling["interface"] == "entity_tokens_only") & (scaling["direction"] == "combined")
    ].iloc[0]
    constant_cells = int(complete["context_representation_constant"].sum())

    def interval(row: dict[str, Any] | pd.Series) -> str:
        return (
            f"{row['equal_family_effect']:+.4f} "
            f"[{row['fixed_panel_bootstrap_low']:+.4f}, "
            f"{row['fixed_panel_bootstrap_high']:+.4f}]"
        )

    def regime_interval(row: dict[str, Any]) -> str:
        return (
            f"{row['equal_family_within_regime_effect']:+.4f} "
            f"[{row['fixed_panel_bootstrap_low']:+.4f}, "
            f"{row['fixed_panel_bootstrap_high']:+.4f}]"
        )

    return f"""# RFMLink q64 complete-panel findings

Status: **POST-HOC COMPLETE-PANEL ANALYSIS OF A FROZEN SWEEP**.

This analysis treats all 57 datasets in the immutable execution plan as the
fixed completed panel. The primary sensitivity gives equal weight to each of
16 metric-free provenance families;
equal-dataset summaries are secondary. Intervals are fixed-panel
source-family bootstrap sensitivity intervals, not population confidence
intervals.

## Audited design

- 57 datasets, 16 provenance families, and 6,156 complete TabPFN cells;
- nine frozen interfaces, four nested contexts per class, and three seeds;
- relation-stratified q64 filtered ranking in both head and tail directions;
- native zero-shot ULTRA computed on the same candidate rows;
- source result commit: `{SOURCE_RESULT_COMMIT}`;
- source `cell_metrics.csv` SHA256: `{source_sha256}`;
- explicit practical tie band for descriptive wins/ties/losses: +/-0.005 MRR.

## Findings

1. **Stable relational coordinates matter more than opaque entity tokens.**
   At 200 examples per class, B0 local features exceed token-only rows by
   {interval(local)} MRR; all 16 source families have a positive raw effect.

2. **Context is an extraction resource, not merely extra rows.** From 20 to
   200 examples per class, equal-family MRR changes by
   {tokens_scaling["equal_family_effect"]:+.4f} for opaque entity tokens,
   {local_scaling["equal_family_effect"]:+.4f} for the 18 local relational
   coordinates, and {full_scaling["equal_family_effect"]:+.4f} for the full
   32-column interface. The richer reusable interfaces benefit much more from
   context.

3. **The frozen KGFM score is the dominant broadly reusable signal.** Adding
   ULTRA to local+global structure yields {interval(ultra)} MRR at the largest
   context and is positive in all 16 families. This is an interface analysis,
   not evidence that the handcrafted table replaces the KGFM.

4. **The 13 global descriptors have a context- and direction-dependent
   incremental effect.** Conditional on ULTRA they are harmful at 20 examples
   per class ({interval(b2_low)} combined MRR). At 200 examples per class the
   tail effect is {interval(b2_tail)}, whereas the head effect is
   {interval(b2_head)}. The combined endpoint remains small and its interval
   includes zero. Its equal-dataset estimate is
   {b2_combined["equal_dataset_effect"]:+.4f}; provenance weighting therefore
   materially attenuates the apparent gain.

5. **Global descriptors complement the score more than they work alone.**
   The difference-in-differences between their effect with and without ULTRA is
   {interval(interaction)} at 200 examples per class. This post-hoc mechanism
   result is consistent with score conditioning/calibration, not standalone
   graph reasoning.

6. **The global-feature effect is regime-heterogeneous.** At 200 examples per
   class it is {regime_interval(inductive_entities_relations)} in
   entity+relation-inductive graphs, {regime_interval(inductive_entities)} in
   entity-inductive graphs, and {regime_interval(transductive)} in transductive
   graphs. These are descriptive fixed-panel strata, not a causal regime
   interaction.

7. **The direct table readout does not generally beat native ULTRA.** At 200
   examples per class, the equal-family full-interface contrast is
   {interval(native)}, but it is positive on only
   {int(native["positive_families"])}/16 families and
   {int(native["dataset_wins"])}/57 datasets outside the practical tie band.
   The positive equal-family mean is driven by MetaFam; excluding that family
   gives {native_without_metafam:+.4f} MRR. Native ULTRA remains the appropriate
   system baseline.

8. **Sparse handcrafted interfaces expose a concrete coverage failure.**
   {constant_cells} cells use the registered constant-input fallback. These
   occur only in dyadic/path interfaces and concentrate in entity+relation
   inductive graphs; their frequency falls sharply as context grows.

## Reading rule

The robust paper-level evidence is the representation--context pattern:
reusable relational coordinates and the frozen score support extraction, while
opaque tokens, constant interfaces, and insufficient context do not. The tail
specificity and B2v2-by-ULTRA interaction are mechanism evidence. The study does
not establish universal TabPFN superiority, native-ULTRA improvement, or a
population-level law over knowledge graphs.

## Files

- `audit.json`: frozen panel and lineage checks;
- `dataset_metrics.csv`: seed-averaged dataset cells;
- `absolute_mrr_equal_family.csv` and `absolute_mrr_equal_dataset.csv`;
- `contrast_summary.csv`: all registered-style post-hoc contrasts;
- `contrast_by_dataset.csv`, `contrast_by_family.csv`, and
  `contrast_by_regime.csv`;
- `context_scaling_summary.csv` and `context_scaling_by_family.csv`;
- `constant_fallback_summary.csv`;
- `native_ultra_sensitivity.csv`;
- `finding_summary.json`: compact machine-readable headline values.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.cell_metrics.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_sha256 = file_sha256(source)
    raw = pd.read_csv(source)
    complete = validate_complete_case(raw)
    dataset = dataset_metrics(complete)

    equal_family, equal_dataset = absolute_mrr_tables(dataset)
    contrasts, contrast_family, contrast_regime, contrast_dataset = summarize_contrasts(dataset)
    scaling, scaling_family = summarize_context_scaling(dataset)
    fallbacks = summarize_constant_fallbacks(complete)
    native_sensitivity = native_ultra_sensitivity(contrast_family)

    provenance_rows = [
        {"dataset": dataset_name, "provenance_family": family}
        for family, members in PROVENANCE_FAMILIES.items()
        for dataset_name in members
    ]
    pd.DataFrame(provenance_rows).sort_values("dataset").to_csv(
        output / "provenance_map.csv", index=False
    )
    dataset.to_csv(output / "dataset_metrics.csv", index=False)
    equal_family.to_csv(output / "absolute_mrr_equal_family.csv", index=False)
    equal_dataset.to_csv(output / "absolute_mrr_equal_dataset.csv", index=False)
    contrasts.to_csv(output / "contrast_summary.csv", index=False)
    contrast_dataset.to_csv(output / "contrast_by_dataset.csv", index=False)
    contrast_family.to_csv(output / "contrast_by_family.csv", index=False)
    contrast_regime.to_csv(output / "contrast_by_regime.csv", index=False)
    scaling.to_csv(output / "context_scaling_summary.csv", index=False)
    scaling_family.to_csv(output / "context_scaling_by_family.csv", index=False)
    fallbacks.to_csv(output / "constant_fallback_summary.csv", index=False)
    native_sensitivity.to_csv(output / "native_ultra_sensitivity.csv", index=False)

    audit = {
        "status": "COMPLETE_PANEL_57_AUDITED",
        "study_id": STUDY_ID,
        "scientific_status": "post_hoc_complete_panel_analysis",
        "source_result_branch": SOURCE_RESULT_BRANCH,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_cell_metrics": SOURCE_RELATIVE_PATH,
        "source_cell_metrics_sha256": source_sha256,
        "excluded_datasets": [],
        "complete_datasets": sorted(complete["dataset"].unique().tolist()),
        "complete_dataset_count": int(complete["dataset"].nunique()),
        "provenance_families": len(PROVENANCE_FAMILIES),
        "cells": int(len(complete)),
        "interfaces": list(INTERFACES),
        "contexts_per_class": list(CONTEXTS),
        "seeds": list(SEEDS),
        "directions": list(DIRECTIONS),
        "bootstrap": {
            "unit": "metric_free_provenance_family",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interpretation": "fixed_panel_sensitivity_interval",
        },
        "practical_tie_band_mrr": PRACTICAL_TIE_BAND,
    }
    write_json(output / "audit.json", audit)

    finding_keys = (
        ("local_relational_vs_entity_tokens", 200, "combined"),
        ("ultra_score_given_global_structure", 200, "combined"),
        ("global_structure_given_ultra", 20, "combined"),
        ("global_structure_given_ultra", 200, "combined"),
        ("global_structure_given_ultra", 200, "head"),
        ("global_structure_given_ultra", 200, "tail"),
        ("global_structure_x_ultra_score", 200, "combined"),
        ("full_interface_vs_native_ultra", 200, "combined"),
    )
    findings = {
        f"{name}__c{context}__{direction}": selected_value(contrasts, name, context, direction)
        for name, context, direction in finding_keys
    }
    findings["context_scaling_20_to_200"] = {
        interface: scaling.loc[
            (scaling["interface"] == interface) & (scaling["direction"] == "combined")
        ]
        .iloc[0]
        .to_dict()
        for interface in INTERFACES
    }
    findings["global_structure_given_ultra_c200_by_regime"] = {
        regime: selected_regime_value(
            contrast_regime,
            "global_structure_given_ultra",
            200,
            regime,
        )
        for regime in sorted(complete["transfer_regime"].unique())
    }
    findings["native_ultra_sensitivity"] = native_sensitivity.to_dict(orient="records")
    write_json(output / "finding_summary.json", findings)
    (output / "README.md").write_text(
        render_readme(
            contrasts,
            contrast_regime,
            scaling,
            complete,
            native_sensitivity,
            source_sha256,
        ),
        encoding="utf-8",
    )

    artifact_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    write_json(
        output / "manifest.json",
        {
            "study_id": STUDY_ID,
            "status": "COMPLETE_PANEL_57_AUDITED",
            "files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in artifact_files
            ],
        },
    )


if __name__ == "__main__":
    main()
