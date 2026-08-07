#!/usr/bin/env python3
"""Analyze the 57-KG filtered-negative and nominal-decoder control."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ultra4g_q64_complete_panel import provenance_lookup

from relational_tfm.runner import verify_shard

SOURCE_COMMIT = "artifact-source-revision"
ASSET_ID = "rfmlink_ultra4g_negative_robustness_v1"
INTERFACES = (
    "entity_tokens_only",
    "b0_local_only",
    "ultra_score_only",
    "b0_local_ultra",
)
CONTEXT_MODES = (
    "relation_matched_filtered_uniform",
    "relation_matched_role_degree",
)
DECODERS = ("logistic_nominal", "catboost_nominal", "tabpfn")
DIRECTIONS = ("combined", "head", "tail")
SEEDS = (101, 102, 103)
TIE_BAND = 0.005
EXPECTED_DATASETS = 57
EXPECTED_FAMILIES = 16
EXPECTED_CELLS = 4_104
BOOTSTRAP_REPLICATES = 20_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_run(
    run_root: Path,
    asset_manifest: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    registration = read_json(run_root / "registration.json")
    plan = read_json(run_root / "plan.json")
    if registration["source_commit"] != SOURCE_COMMIT:
        raise ValueError(f"{run_root} was produced by the wrong source commit")
    if registration["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError(f"{run_root} registration and plan differ")
    if registration["asset_manifest_sha256"] != file_sha256(asset_manifest):
        raise ValueError(f"{run_root} uses the wrong asset manifest")
    planned = {cell["cell_id"]: cell for cell in plan["cells"]}
    if len(planned) != int(registration["cells"]):
        raise ValueError(f"{run_root} plan cell IDs are incomplete")

    shard_dirs = sorted((run_root / "results").glob("shard-*"))
    if len(shard_dirs) != int(registration["shards"]):
        raise ValueError(f"{run_root} has {len(shard_dirs)}/{registration['shards']} shards")
    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    query_rows: list[pd.DataFrame] = []
    shard_hashes: dict[str, str] = {}
    for shard_dir in shard_dirs:
        manifest = verify_shard(shard_dir)
        if manifest["plan_sha256"] != registration["plan_sha256"]:
            raise ValueError(f"{shard_dir} has the wrong plan hash")
        shard_hashes[shard_dir.name] = file_sha256(shard_dir / "shard_manifest.json")
        for cell_id in manifest["cell_ids"]:
            if cell_id in observed:
                raise ValueError(f"duplicate cell {cell_id}")
            metric = read_json(shard_dir / f"{cell_id}.metrics.json")
            cell = metric["cell"]
            if cell != planned.get(cell_id) or metric.get("status") != "complete":
                raise ValueError(f"invalid metric payload {cell_id}")
            observed.add(cell_id)
            row: dict[str, Any] = {
                "cell_id": cell_id,
                "dataset": cell["dataset"],
                "transfer_regime": cell["transfer_regime"],
                "decoder": cell["decoder"],
                "interface": cell["interface"],
                "context_mode": cell["context_mode"],
                "context_per_class": int(cell["context_per_class"]),
                "seed": int(cell["seed"]),
                "feature_count": int(metric["feature_count"]),
                "duration_seconds": float(metric["duration_seconds"]),
            }
            for direction in DIRECTIONS:
                row[f"{direction}_mrr"] = float(metric["direction_metrics"][direction]["mrr"])
                row[f"native_{direction}_mrr"] = float(
                    metric["native_ultra_direction_metrics"][direction]["mrr"]
                )
            rows.append(row)
            with np.load(shard_dir / f"{cell_id}.predictions.npz") as predictions:
                query_rows.append(
                    pd.DataFrame(
                        {
                            "dataset": cell["dataset"],
                            "decoder": cell["decoder"],
                            "interface": cell["interface"],
                            "context_mode": cell["context_mode"],
                            "seed": int(cell["seed"]),
                            "query_id": predictions["query_ids"].astype(np.int64),
                            "side_code": predictions["side_codes"].astype(np.int8),
                            "reciprocal_rank": predictions["reciprocal_ranks"].astype(np.float64),
                        }
                    )
                )
    if observed != set(planned):
        raise ValueError(f"{run_root} result cells do not match the plan")
    return (
        pd.DataFrame(rows),
        pd.concat(query_rows, ignore_index=True),
        {
            "root": str(run_root.resolve()),
            "profile": registration["profile"],
            "plan_sha256": registration["plan_sha256"],
            "cells": len(rows),
            "shards": len(shard_dirs),
            "shard_manifest_sha256": shard_hashes,
        },
    )


def validate_cells(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = provenance_lookup()
    frame["provenance_family"] = frame["dataset"].map(lookup)
    if frame["provenance_family"].isna().any():
        raise ValueError("one or more datasets lack a provenance family")
    expected = {
        "dataset": EXPECTED_DATASETS,
        "provenance_family": EXPECTED_FAMILIES,
        "decoder": len(DECODERS),
        "interface": len(INTERFACES),
        "context_mode": len(CONTEXT_MODES),
        "seed": len(SEEDS),
    }
    observed = {column: int(frame[column].nunique()) for column in expected}
    if len(frame) != EXPECTED_CELLS or observed != expected:
        raise ValueError(f"combined cell matrix drift: rows={len(frame)}, {observed}")
    if set(frame["decoder"]) != set(DECODERS):
        raise ValueError("decoder matrix drift")
    if set(frame["interface"]) != set(INTERFACES):
        raise ValueError("interface matrix drift")
    if set(frame["context_mode"]) != set(CONTEXT_MODES):
        raise ValueError("context matrix drift")
    if set(map(int, frame["seed"])) != set(SEEDS):
        raise ValueError("seed matrix drift")
    grouping = ["dataset", "decoder", "interface", "context_mode"]
    if not (frame.groupby(grouping, observed=True).size() == len(SEEDS)).all():
        raise ValueError("one or more cells lack the three registered seeds")
    return frame.sort_values(grouping + ["seed"], ignore_index=True)


def average_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "provenance_family",
        "transfer_regime",
        "decoder",
        "interface",
        "context_mode",
    ]
    metrics = [f"{direction}_mrr" for direction in DIRECTIONS]
    metrics += [f"native_{direction}_mrr" for direction in DIRECTIONS]
    return frame.groupby(keys, as_index=False, observed=True)[metrics].mean()


def paired_interface_effect(
    dataset: pd.DataFrame,
    first: str,
    second: str,
    *,
    decoder: str,
    context_mode: str,
    direction: str,
) -> pd.DataFrame:
    selected = dataset.loc[
        (dataset["decoder"] == decoder)
        & (dataset["context_mode"] == context_mode)
        & dataset["interface"].isin((first, second)),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            "interface",
            f"{direction}_mrr",
        ],
    ]
    pivot = selected.pivot(
        index=["dataset", "provenance_family", "transfer_regime"],
        columns="interface",
        values=f"{direction}_mrr",
    ).reset_index()
    if len(pivot) != EXPECTED_DATASETS or pivot[[first, second]].isna().any().any():
        raise ValueError(f"incomplete interface pair {first}-{second}")
    pivot["effect"] = pivot[first] - pivot[second]
    return pivot


def paired_context_effect(
    dataset: pd.DataFrame,
    *,
    decoder: str,
    interface: str,
    direction: str,
) -> pd.DataFrame:
    selected = dataset.loc[
        (dataset["decoder"] == decoder) & (dataset["interface"] == interface),
        [
            "dataset",
            "provenance_family",
            "transfer_regime",
            "context_mode",
            f"{direction}_mrr",
        ],
    ]
    pivot = selected.pivot(
        index=["dataset", "provenance_family", "transfer_regime"],
        columns="context_mode",
        values=f"{direction}_mrr",
    ).reset_index()
    if len(pivot) != EXPECTED_DATASETS:
        raise ValueError(f"incomplete context pair {decoder}/{interface}")
    pivot["effect"] = pivot[CONTEXT_MODES[1]] - pivot[CONTEXT_MODES[0]]
    return pivot


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def summarize_contrast(
    effects: pd.DataFrame,
    metadata: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    family = (
        effects.groupby("provenance_family", as_index=False, observed=True)["effect"]
        .mean()
        .sort_values("provenance_family", ignore_index=True)
    )
    values = family["effect"].to_numpy(dtype=float)
    if len(values) != EXPECTED_FAMILIES:
        raise ValueError("contrast lacks one or more provenance families")
    low, high = bootstrap_interval(values, seed)
    leave_one_out = np.asarray([np.delete(values, index).mean() for index in range(len(values))])
    dataset_values = effects["effect"].to_numpy(dtype=float)
    summary = {
        **metadata,
        "equal_family_effect": float(values.mean()),
        "family_bootstrap_low": low,
        "family_bootstrap_high": high,
        "equal_dataset_effect": float(dataset_values.mean()),
        "dataset_wins": int((dataset_values > TIE_BAND).sum()),
        "dataset_ties": int((np.abs(dataset_values) <= TIE_BAND).sum()),
        "dataset_losses": int((dataset_values < -TIE_BAND).sum()),
        "leave_one_family_out_min": float(leave_one_out.min()),
        "leave_one_family_out_max": float(leave_one_out.max()),
    }
    for key, value in metadata.items():
        family[key] = value
    return summary, family


def build_contrasts(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interface_pairs = (
        ("local_minus_tokens", "b0_local_only", "entity_tokens_only"),
        (
            "local_plus_score_minus_score",
            "b0_local_ultra",
            "ultra_score_only",
        ),
        ("score_given_local", "b0_local_ultra", "b0_local_only"),
    )
    summaries: list[dict[str, Any]] = []
    families: list[pd.DataFrame] = []
    datasets: list[pd.DataFrame] = []
    contrast_index = 0
    for decoder in DECODERS:
        for context_mode in CONTEXT_MODES:
            for direction in DIRECTIONS:
                for contrast, first, second in interface_pairs:
                    effects = paired_interface_effect(
                        dataset,
                        first,
                        second,
                        decoder=decoder,
                        context_mode=context_mode,
                        direction=direction,
                    )
                    contrast_id = ":".join((contrast, decoder, context_mode, direction))
                    metadata = {
                        "contrast_id": contrast_id,
                        "contrast": contrast,
                        "decoder": decoder,
                        "interface": f"{first}-{second}",
                        "context_mode": context_mode,
                        "direction": direction,
                    }
                    summary, family = summarize_contrast(
                        effects,
                        metadata,
                        71_000 + contrast_index,
                    )
                    effects = effects.assign(**metadata)
                    summaries.append(summary)
                    families.append(family)
                    datasets.append(effects)
                    contrast_index += 1
        for interface in INTERFACES:
            for direction in DIRECTIONS:
                effects = paired_context_effect(
                    dataset,
                    decoder=decoder,
                    interface=interface,
                    direction=direction,
                )
                contrast = "structured_minus_filtered_uniform"
                contrast_id = ":".join((contrast, decoder, interface, direction))
                metadata = {
                    "contrast_id": contrast_id,
                    "contrast": contrast,
                    "decoder": decoder,
                    "interface": interface,
                    "context_mode": f"{CONTEXT_MODES[1]}-{CONTEXT_MODES[0]}",
                    "direction": direction,
                }
                summary, family = summarize_contrast(
                    effects,
                    metadata,
                    71_000 + contrast_index,
                )
                effects = effects.assign(**metadata)
                summaries.append(summary)
                families.append(family)
                datasets.append(effects)
                contrast_index += 1
    return (
        pd.DataFrame(summaries),
        pd.concat(families, ignore_index=True),
        pd.concat(datasets, ignore_index=True),
    )


def _t_statistic(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    statistic = np.divide(
        means,
        errors,
        out=np.zeros_like(means),
        where=errors > 0,
    )
    constant = (errors == 0) & (means != 0)
    statistic[constant] = np.sign(means[constant]) * np.inf
    return statistic


def exact_joint_max_t(family: pd.DataFrame) -> pd.DataFrame:
    combined = family.loc[family["direction"] == "combined"]
    matrix = combined.pivot(
        index="provenance_family",
        columns="contrast_id",
        values="effect",
    ).sort_index(axis=1)
    if matrix.shape[0] != EXPECTED_FAMILIES or matrix.isna().any().any():
        raise ValueError("family-effect matrix is incomplete")
    base = matrix.to_numpy(dtype=float)
    observed = _t_statistic(base)
    raw_counts = np.zeros(base.shape[1], dtype=np.int64)
    adjusted_counts = np.zeros(base.shape[1], dtype=np.int64)
    patterns = 0
    for signs in itertools.product((-1.0, 1.0), repeat=base.shape[0]):
        statistic = np.abs(_t_statistic(base * np.asarray(signs, dtype=float)[:, None]))
        maximum = statistic.max()
        raw_counts += statistic >= np.abs(observed) - 1e-12
        adjusted_counts += maximum >= np.abs(observed) - 1e-12
        patterns += 1
    return pd.DataFrame(
        {
            "contrast_id": matrix.columns,
            "observed_t": observed,
            "exact_two_sided_p": raw_counts / patterns,
            "exact_max_t_adjusted_p": adjusted_counts / patterns,
            "sign_patterns": patterns,
            "joint_contrasts": matrix.shape[1],
        }
    )


def configuration_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    keys = ["decoder", "interface", "context_mode"]
    metric_columns = [f"{direction}_mrr" for direction in DIRECTIONS]
    family = dataset.groupby(
        keys + ["provenance_family"],
        as_index=False,
        observed=True,
    )[metric_columns].mean()
    rows: list[dict[str, Any]] = []
    for key, group in dataset.groupby(keys, observed=True):
        selector = (family[keys] == pd.Series(key, index=keys)).all(axis=1)
        family_group = family.loc[selector]
        row = dict(zip(keys, key, strict=True))
        for direction in DIRECTIONS:
            row[f"equal_dataset_{direction}_mrr"] = float(group[f"{direction}_mrr"].mean())
            row[f"equal_family_{direction}_mrr"] = float(family_group[f"{direction}_mrr"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def context_audit(asset_manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = provenance_lookup()
    for dataset, record in asset_manifest["datasets"].items():
        for seed, audit in record["harder_context_audits"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "provenance_family": lookup[dataset],
                    "transfer_regime": record["transfer_regime"],
                    "seed": int(seed),
                    **audit,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "seed"], ignore_index=True)


def query_evidence_summary(
    query: pd.DataFrame,
    properties_path: Path,
) -> pd.DataFrame:
    properties = pd.read_csv(
        properties_path,
        usecols=[
            "dataset",
            "query_id",
            "provenance_family",
            "pair_other_relation_count",
            "reverse_pair_relation_count",
        ],
    )
    direct = properties["pair_other_relation_count"].gt(0)
    reverse = properties["reverse_pair_relation_count"].gt(0)
    properties["dyadic_stratum"] = np.select(
        [direct & reverse, direct, reverse],
        ["both", "direct_only", "reverse_only"],
        default="none",
    )
    averaged = query.groupby(
        ["dataset", "decoder", "interface", "context_mode", "query_id"],
        as_index=False,
        observed=True,
    )["reciprocal_rank"].mean()
    pairs = (
        ("local_minus_tokens", "b0_local_only", "entity_tokens_only"),
        (
            "local_plus_score_minus_score",
            "b0_local_ultra",
            "ultra_score_only",
        ),
    )
    rows: list[dict[str, Any]] = []
    for decoder in DECODERS:
        for context_mode in CONTEXT_MODES:
            for contrast, first, second in pairs:
                selected = averaged.loc[
                    (averaged["decoder"] == decoder)
                    & (averaged["context_mode"] == context_mode)
                    & averaged["interface"].isin((first, second))
                ]
                pivot = selected.pivot(
                    index=["dataset", "query_id"],
                    columns="interface",
                    values="reciprocal_rank",
                ).reset_index()
                if len(pivot) != EXPECTED_DATASETS * 128:
                    raise ValueError(f"incomplete query pair {contrast}/{decoder}")
                pivot["effect"] = pivot[first] - pivot[second]
                pivot = pivot.merge(
                    properties[
                        [
                            "dataset",
                            "query_id",
                            "provenance_family",
                            "dyadic_stratum",
                        ]
                    ],
                    on=["dataset", "query_id"],
                    how="left",
                    validate="one_to_one",
                )
                if pivot["dyadic_stratum"].isna().any():
                    raise ValueError("query properties do not match predictions")
                dataset_stratum = pivot.groupby(
                    ["provenance_family", "dataset", "dyadic_stratum"],
                    as_index=False,
                    observed=True,
                )["effect"].mean()
                family = dataset_stratum.groupby(
                    ["provenance_family", "dyadic_stratum"],
                    as_index=False,
                    observed=True,
                )["effect"].mean()
                pooled = pivot.groupby("dyadic_stratum", observed=True).size()
                for stratum, group in family.groupby(
                    "dyadic_stratum",
                    observed=True,
                ):
                    values = group["effect"].to_numpy(dtype=float)
                    low, high = bootstrap_interval(
                        values,
                        91_000
                        + 101 * DECODERS.index(decoder)
                        + 17 * CONTEXT_MODES.index(context_mode)
                        + 3 * [item[0] for item in pairs].index(contrast)
                        + len(rows),
                    )
                    rows.append(
                        {
                            "contrast": contrast,
                            "decoder": decoder,
                            "context_mode": context_mode,
                            "dyadic_stratum": stratum,
                            "equal_family_effect": float(values.mean()),
                            "family_bootstrap_low": low,
                            "family_bootstrap_high": high,
                            "families": len(values),
                            "query_rows": int(pooled.loc[stratum]),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["contrast", "decoder", "context_mode", "dyadic_stratum"],
        ignore_index=True,
    )


def write_report(summary: pd.DataFrame, output: Path) -> None:
    primary = summary.loc[summary["direction"] == "combined"].copy()
    lines = [
        "# Filtered Negative and Nominal Decoder Control",
        "",
        "Effects are paired by dataset after averaging the three registered context seeds. "
        "Positive context rows, q64 evaluation queries, candidates, masks, and feature rows "
        "are fixed across context modes. Intervals resample the 16 fixed provenance-family "
        "effects and are sensitivity summaries rather than superpopulation intervals.",
        "",
        (
            "| Contrast | Decoder | Context/interface | Equal-family effect | "
            "95% interval | W/T/L | LOFO | max-|t| p |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values(["contrast", "decoder", "context_mode", "interface"]).itertuples(
        index=False
    ):
        label = (
            row.context_mode
            if row.contrast != "structured_minus_filtered_uniform"
            else row.interface
        )
        lines.append(
            f"| {row.contrast} | {row.decoder} | {label} | "
            f"{row.equal_family_effect:+.4f} | "
            f"[{row.family_bootstrap_low:+.4f}, {row.family_bootstrap_high:+.4f}] | "
            f"{row.dataset_wins}/{row.dataset_ties}/{row.dataset_losses} | "
            f"[{row.leave_one_family_out_min:+.4f}, {row.leave_one_family_out_max:+.4f}] | "
            f"{row.exact_max_t_adjusted_p:.5f} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run-root", type=Path, required=True)
    parser.add_argument("--tabpfn-run-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--query-properties", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    asset = read_json(args.asset_manifest)
    if asset.get("asset_id") != ASSET_ID:
        raise ValueError("wrong negative-robustness asset")
    cpu, cpu_query, cpu_audit = collect_run(
        args.cpu_run_root,
        args.asset_manifest,
    )
    tabpfn, tabpfn_query, tabpfn_audit = collect_run(
        args.tabpfn_run_root,
        args.asset_manifest,
    )
    cells = validate_cells(pd.concat((cpu, tabpfn), ignore_index=True))
    dataset = average_seeds(cells)
    summary, family, dataset_effects = build_contrasts(dataset)
    exact = exact_joint_max_t(family)
    summary = summary.merge(exact, on="contrast_id", how="left", validate="one_to_one")
    configurations = configuration_summary(dataset)
    contexts = context_audit(asset)
    query_strata = query_evidence_summary(
        pd.concat((cpu_query, tabpfn_query), ignore_index=True),
        args.query_properties,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "cell_metrics.csv.gz", index=False, compression="gzip")
    dataset.to_csv(args.output / "dataset_metrics.csv", index=False)
    configurations.to_csv(args.output / "configuration_summary.csv", index=False)
    summary.to_csv(args.output / "contrast_summary.csv", index=False)
    family.to_csv(args.output / "family_effects.csv", index=False)
    dataset_effects.to_csv(args.output / "dataset_effects.csv", index=False)
    exact.to_csv(args.output / "exact_joint_max_t.csv", index=False)
    contexts.to_csv(args.output / "context_audit.csv", index=False)
    query_strata.to_csv(args.output / "query_evidence_strata.csv", index=False)
    write_report(summary, args.output)
    write_json(
        args.output / "audit.json",
        {
            "analysis_id": "rfmlink_ultra4g_negative_robustness_v1",
            "source_commit": SOURCE_COMMIT,
            "asset_manifest_sha256": file_sha256(args.asset_manifest),
            "runs": [cpu_audit, tabpfn_audit],
            "cells": len(cells),
            "datasets": int(cells["dataset"].nunique()),
            "provenance_families": int(cells["provenance_family"].nunique()),
            "decoders": sorted(cells["decoder"].unique()),
            "interfaces": sorted(cells["interface"].unique()),
            "context_modes": sorted(cells["context_mode"].unique()),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "tie_band": TIE_BAND,
            "max_t_joint_combined_contrasts": int(len(exact)),
            "status": "PASS",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
