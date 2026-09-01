#!/usr/bin/env python3
"""Validate and summarize the query-relative and matched-filter KG controls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIRECTIONS = ("combined", "head", "tail")
TIE_BAND = 0.005
BOOTSTRAP_DRAWS = 10_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def provenance_lookup(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["dataset", "provenance_family"]:
        raise ValueError("provenance map schema changed")
    if len(frame) != 57 or frame["dataset"].duplicated().any():
        raise ValueError("provenance map is not a 57-dataset bijection")
    return dict(zip(frame.dataset, frame.provenance_family, strict=True))


def load_run(
    root: Path,
    *,
    profile: str,
    provenance: dict[str, str],
    require_complete: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registration = read_json(root / "registration.json")
    plan = read_json(root / "plan.json")
    if registration["profile"] != profile:
        raise ValueError(f"{root}: expected profile {profile!r}")
    if registration["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError(f"{root}: registration/plan hash mismatch")
    planned = {cell["cell_id"]: cell for cell in plan["cells"]}
    if len(planned) != int(registration["cells"]):
        raise ValueError(f"{root}: registration has the wrong cell count")

    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for shard in sorted((root / "results").glob("shard-*")):
        manifest_path = shard / "shard_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError(f"{shard}: plan hash mismatch")
        for cell_id in manifest["cell_ids"]:
            if cell_id in observed or cell_id not in planned:
                raise ValueError(f"{shard}: invalid or duplicate cell {cell_id}")
            payload = read_json(shard / f"{cell_id}.metrics.json")
            if payload["status"] != "complete" or payload["cell"] != planned[cell_id]:
                raise ValueError(f"{cell_id}: result differs from frozen plan")
            cell = payload["cell"]
            row: dict[str, Any] = {
                "cell_id": cell_id,
                "dataset": cell["dataset"],
                "provenance_family": provenance[cell["dataset"]],
                "decoder": cell["decoder"],
                "interface": cell["interface"],
                "context_mode": cell["context_mode"],
                "query_transform": cell.get("query_transform", "raw"),
                "seed": int(cell["seed"]),
                "prediction_path": str(shard / payload["predictions"]["path"]),
            }
            for direction in DIRECTIONS:
                row[f"{direction}_mrr"] = float(
                    payload["direction_metrics"][direction]["mrr"]
                )
                row[f"native_{direction}_mrr"] = float(
                    payload["native_ultra_direction_metrics"][direction]["mrr"]
                )
            rows.append(row)
            observed.add(cell_id)

    if require_complete and observed != set(planned):
        raise ValueError(f"{root}: missing {len(set(planned) - observed)} cells")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, registration
    if frame.dataset.nunique() != 57 and require_complete:
        raise ValueError(f"{root}: expected all 57 datasets")
    return frame, registration


def dataset_means(cells: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "provenance_family",
        "decoder",
        "interface",
        "context_mode",
        "query_transform",
    ]
    values = [
        f"{prefix}{direction}_mrr"
        for direction in DIRECTIONS
        for prefix in ("", "native_")
    ]
    return cells.groupby(keys, as_index=False, observed=True)[values].mean()


def absolute_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    keys = ["decoder", "interface", "context_mode", "query_transform"]
    rows: list[dict[str, Any]] = []
    for key, group in dataset.groupby(keys, observed=True):
        family = group.groupby("provenance_family", observed=True)[
            ["combined_mrr", "native_combined_mrr"]
        ].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "equal_dataset_mrr": float(group.combined_mrr.mean()),
                "equal_family_mrr": float(family.combined_mrr.mean()),
                "equal_dataset_delta_vs_native": float(
                    (group.combined_mrr - group.native_combined_mrr).mean()
                ),
                "equal_family_delta_vs_native": float(
                    (family.combined_mrr - family.native_combined_mrr).mean()
                ),
                "datasets": int(group.dataset.nunique()),
                "families": int(group.provenance_family.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def paired_summary(
    dataset: pd.DataFrame,
    *,
    axis: str,
    reference: str,
    alternatives: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["decoder", "interface"]
    nuisance = "context_mode" if axis == "query_transform" else "query_transform"
    dataset_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key, group in dataset.groupby(keys, observed=True):
        if group[nuisance].nunique() != 1:
            raise ValueError(f"uncontrolled nuisance axis {nuisance}")
        pivot = group.pivot(
            index=["dataset", "provenance_family"],
            columns=axis,
            values="combined_mrr",
        )
        for alternative in alternatives:
            paired = pivot[[reference, alternative]].dropna()
            effect = paired[alternative] - paired[reference]
            for index, value in effect.items():
                dataset_rows.append(
                    {
                        "decoder": key[0],
                        "interface": key[1],
                        "axis": axis,
                        "reference": reference,
                        "alternative": alternative,
                        "dataset": index[0],
                        "provenance_family": index[1],
                        "effect": float(value),
                    }
                )
            family = effect.groupby(level="provenance_family").mean()
            loo = [float(family.drop(index=name).mean()) for name in family.index]
            contrast_name = "|".join((*key, axis, reference, alternative))
            bootstrap_seed = int.from_bytes(
                hashlib.sha256(contrast_name.encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            rng = np.random.default_rng(bootstrap_seed)
            family_values = family.to_numpy(dtype=float)
            draws = family_values[
                rng.integers(
                    0,
                    len(family_values),
                    size=(BOOTSTRAP_DRAWS, len(family_values)),
                )
            ].mean(axis=1)
            bootstrap_low, bootstrap_high = np.quantile(draws, [0.025, 0.975])
            summary_rows.append(
                {
                    "decoder": key[0],
                    "interface": key[1],
                    "axis": axis,
                    "reference": reference,
                    "alternative": alternative,
                    "equal_dataset_effect": float(effect.mean()),
                    "equal_family_effect": float(family.mean()),
                    "wins": int((effect > TIE_BAND).sum()),
                    "ties": int((effect.abs() <= TIE_BAND).sum()),
                    "losses": int((effect < -TIE_BAND).sum()),
                    "leave_one_family_out_min": min(loo),
                    "leave_one_family_out_max": max(loo),
                    "bootstrap_sensitivity_low": float(bootstrap_low),
                    "bootstrap_sensitivity_high": float(bootstrap_high),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": bootstrap_seed,
                    "datasets": int(len(effect)),
                    "families": int(len(family)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(dataset_rows)


def _t_statistic(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    statistic = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors != 0.0,
    )
    constant = (standard_errors == 0.0) & (means != 0.0)
    statistic[constant] = np.sign(means[constant]) * np.inf
    return statistic


def exact_joint_max_t(dataset_effects: pd.DataFrame) -> pd.DataFrame:
    keys = ["decoder", "interface", "axis", "reference", "alternative"]
    family = dataset_effects.groupby(
        keys + ["provenance_family"], as_index=False, observed=True
    ).effect.mean()
    family["contrast_id"] = family[keys].astype(str).agg("|".join, axis=1)
    complete_ids = (
        family.groupby("contrast_id", observed=True).provenance_family.nunique()
    )
    complete_ids = complete_ids.loc[complete_ids == 16].index
    selected = family.loc[family.contrast_id.isin(complete_ids)]
    if selected.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "observed_t",
                "exact_two_sided_p",
                "exact_max_t_adjusted_p",
                "sign_patterns",
                "joint_contrasts",
            ]
        )
    matrix = selected.pivot(
        index="provenance_family", columns="contrast_id", values="effect"
    ).sort_index(axis=1)
    if matrix.shape[0] != 16 or matrix.isna().any().any():
        raise ValueError("complete family-effect matrix is malformed")
    base = matrix.to_numpy(dtype=float)
    observed = _t_statistic(base)
    raw_counts = np.zeros(base.shape[1], dtype=np.int64)
    adjusted_counts = np.zeros(base.shape[1], dtype=np.int64)
    patterns = 0
    for signs in itertools.product((-1.0, 1.0), repeat=base.shape[0]):
        statistic = np.abs(
            _t_statistic(base * np.asarray(signs, dtype=float)[:, None])
        )
        maximum = statistic.max()
        raw_counts += statistic >= np.abs(observed) - 1e-12
        adjusted_counts += maximum >= np.abs(observed) - 1e-12
        patterns += 1
    result = pd.DataFrame(
        {
            "contrast_id": matrix.columns,
            "observed_t": observed,
            "exact_two_sided_p": raw_counts / patterns,
            "exact_max_t_adjusted_p": adjusted_counts / patterns,
            "sign_patterns": patterns,
            "joint_contrasts": matrix.shape[1],
        }
    )
    definitions = selected[["contrast_id", *keys]].drop_duplicates()
    return definitions.merge(result, on="contrast_id", validate="one_to_one").drop(
        columns="contrast_id"
    )


def score_rank_checks(cells: pd.DataFrame) -> pd.DataFrame:
    selected = cells.loc[cells.interface == "ultra_score_only"]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["decoder", "query_transform"], observed=True):
        matches = 0
        total = 0
        exact_cells = 0
        for path in group.prediction_path:
            with np.load(path) as predictions:
                equal = np.array_equal(
                    predictions["ranks"], predictions["native_ultra_ranks"]
                )
                exact_cells += int(equal)
                matches += int(
                    np.equal(
                        predictions["ranks"], predictions["native_ultra_ranks"]
                    ).sum()
                )
                total += int(predictions["ranks"].size)
        rows.append(
            {
                "decoder": key[0],
                "query_transform": key[1],
                "exact_rank_cells": exact_cells,
                "cells": len(group),
                "rank_match_fraction": matches / total,
                "queries": total,
            }
        )
    return pd.DataFrame(rows)


def contamination_summary(asset_manifest: Path) -> dict[str, Any]:
    manifest = read_json(asset_manifest)
    rows = []
    for dataset, metadata in manifest["datasets"].items():
        for seed, audit in metadata["seed_audits"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "known_positive_negatives": int(audit["known_positive_negatives"]),
                    "negative_rows": int(audit["negative_rows"]),
                    "fraction": float(audit["known_positive_fraction"]),
                }
            )
    frame = pd.DataFrame(rows)
    maximum = frame.loc[frame.fraction.idxmax()]
    return {
        "known_positive_negatives": int(frame.known_positive_negatives.sum()),
        "negative_rows": int(frame.negative_rows.sum()),
        "pooled_fraction": float(
            frame.known_positive_negatives.sum() / frame.negative_rows.sum()
        ),
        "dataset_seed_cells_with_contamination": int(
            (frame.known_positive_negatives > 0).sum()
        ),
        "dataset_seed_cells": int(len(frame)),
        "maximum_fraction": float(maximum.fraction),
        "maximum_dataset": str(maximum.dataset),
        "maximum_seed": int(maximum.seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cpu-root", type=Path, required=True)
    parser.add_argument("--query-tabpfn-root", type=Path)
    parser.add_argument("--matched-tabpfn-root", type=Path)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--provenance-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-gpu", action="store_true")
    args = parser.parse_args()

    provenance = provenance_lookup(args.provenance_map.resolve())
    query_cpu, cpu_registration = load_run(
        args.query_cpu_root.resolve(),
        profile="query-relative-cpu",
        provenance=provenance,
        require_complete=True,
    )
    query_parts = [query_cpu]
    registrations: dict[str, Any] = {"query_cpu": cpu_registration}
    if args.query_tabpfn_root:
        query_tabpfn, registration = load_run(
            args.query_tabpfn_root.resolve(),
            profile="query-relative-tabpfn",
            provenance=provenance,
            require_complete=not args.allow_incomplete_gpu,
        )
        query_parts.append(query_tabpfn)
        registrations["query_tabpfn"] = registration
    query_cells = pd.concat(query_parts, ignore_index=True)
    query_dataset = dataset_means(query_cells)
    query_summary, query_effects = paired_summary(
        query_dataset,
        axis="query_transform",
        reference="raw",
        alternatives=("within_query_zscore", "within_query_percentile"),
    )
    contrast_keys = ["decoder", "interface", "axis", "reference", "alternative"]
    query_summary = query_summary.merge(
        exact_joint_max_t(query_effects),
        on=contrast_keys,
        how="left",
        validate="one_to_one",
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    absolute_summary(query_dataset).to_csv(output / "query_absolute_mrr.csv", index=False)
    query_summary.to_csv(output / "query_transform_contrasts.csv", index=False)
    query_effects.to_csv(output / "query_transform_dataset_effects.csv", index=False)
    score_rank_checks(query_cells).to_csv(output / "score_rank_checks.csv", index=False)

    matched_complete = False
    if args.matched_tabpfn_root:
        matched_cells, registration = load_run(
            args.matched_tabpfn_root.resolve(),
            profile="matched-filter-tabpfn",
            provenance=provenance,
            require_complete=not args.allow_incomplete_gpu,
        )
        registrations["matched_tabpfn"] = registration
        if not matched_cells.empty:
            matched_dataset = dataset_means(matched_cells)
            matched_summary, matched_effects = paired_summary(
                matched_dataset,
                axis="context_mode",
                reference="unfiltered_uniform",
                alternatives=("filtered_uniform",),
            )
            matched_summary = matched_summary.merge(
                exact_joint_max_t(matched_effects),
                on=contrast_keys,
                how="left",
                validate="one_to_one",
            )
            absolute_summary(matched_dataset).to_csv(
                output / "matched_filter_absolute_mrr.csv", index=False
            )
            matched_summary.to_csv(
                output / "matched_filter_contrasts.csv", index=False
            )
            matched_effects.to_csv(
                output / "matched_filter_dataset_effects.csv", index=False
            )
            matched_complete = len(matched_cells) == int(registration["cells"])

    summary = {
        "query_cells": int(len(query_cells)),
        "query_datasets": int(query_cells.dataset.nunique()),
        "query_decoders": sorted(query_cells.decoder.unique()),
        "query_complete_cpu": len(query_cpu) == int(cpu_registration["cells"]),
        "matched_complete": matched_complete,
        "contamination": contamination_summary(args.asset_manifest.resolve()),
        "registrations": {
            name: {
                key: value
                for key, value in registration.items()
                if key in {"profile", "cells", "shards", "plan_sha256", "asset_manifest_sha256"}
            }
            for name, registration in registrations.items()
        },
        "tie_band": TIE_BAND,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
