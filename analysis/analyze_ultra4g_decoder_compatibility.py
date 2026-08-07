#!/usr/bin/env python3
"""Analyze the complete ultra_4g encoder-by-decoder compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_CPU_CELLS = 12_312
EXPECTED_CPU_SHARDS = 57
EXPECTED_DATASETS = 57
EXPECTED_ASSET_SHA256 = "6c3977f6e14e89a636415d96a78ad638f0cc550df5b04b960a8b423a9bebb317"
EXPECTED_PLAN_SHA256 = "1efdbf85f1bf4c3db9997b6b5d78e86d6ed14415b5fffda1272dea303357fac0"
DECODERS = ("tabpfn", "logistic", "catboost")
INTERFACES = (
    "entity_tokens_only",
    "b0_dyadic_only",
    "b0_paths_only",
    "b2v2_only",
    "b0_local_only",
    "b2v2_structure_only",
    "ultra_score_only",
    "b0_local_ultra",
    "b2v2_full",
)
CONTEXTS = (20, 50, 100, 200)
SEEDS = (101, 102, 103)
DIRECTIONS = ("combined", "head", "tail")
TIE_BAND = 0.005
CONTRASTS = {
    "local_relational_vs_entity_tokens": ("b0_local_only", "entity_tokens_only"),
    "ultra_score_given_local": ("b0_local_ultra", "b0_local_only"),
    "global_structure_given_ultra": ("b2v2_full", "b0_local_ultra"),
    "local_plus_ultra_vs_ultra_only": ("b0_local_ultra", "ultra_score_only"),
    "full_interface_vs_ultra_only": ("b2v2_full", "ultra_score_only"),
}


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


def load_cpu_cells(run_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    registration = read_json(run_root / "registration.json")
    plan = read_json(run_root / "plan.json")
    for key, expected in (
        ("cells", EXPECTED_CPU_CELLS),
        ("shards", EXPECTED_CPU_SHARDS),
        ("asset_manifest_sha256", EXPECTED_ASSET_SHA256),
        ("plan_sha256", EXPECTED_PLAN_SHA256),
    ):
        if registration.get(key) != expected:
            raise ValueError(
                f"CPU registration mismatch for {key}: "
                f"expected {expected}, found {registration.get(key)}"
            )
    if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("CPU plan hash differs from the frozen registration")
    planned = {cell["cell_id"]: cell for cell in plan["cells"]}
    if len(planned) != EXPECTED_CPU_CELLS:
        raise ValueError("CPU plan has the wrong number of unique cells")

    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for shard_index in range(EXPECTED_CPU_SHARDS):
        shard = run_root / "results" / f"shard-{shard_index:04d}"
        manifest = read_json(shard / "shard_manifest.json")
        if manifest["plan_sha256"] != EXPECTED_PLAN_SHA256:
            raise ValueError(f"{shard.name}: plan hash mismatch")
        for entry in manifest["files"]:
            path = shard / entry["path"]
            if path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"{path}: byte count mismatch")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"{path}: file hash mismatch")
        for cell_id in manifest["cell_ids"]:
            if cell_id in observed or cell_id not in planned:
                raise ValueError(f"{shard.name}: invalid cell ID {cell_id}")
            payload = read_json(shard / f"{cell_id}.metrics.json")
            cell = payload["cell"]
            if cell != planned[cell_id] or payload.get("status") != "complete":
                raise ValueError(f"{cell_id}: result differs from the plan")
            row = {
                "dataset": cell["dataset"],
                "transfer_regime": cell["transfer_regime"],
                "interface": cell["interface"],
                "decoder": cell["decoder"],
                "context_per_class": int(cell["context_per_class"]),
                "seed": int(cell["seed"]),
            }
            for direction in DIRECTIONS:
                row[f"{direction}_mrr"] = float(payload["direction_metrics"][direction]["mrr"])
                row[f"native_{direction}_mrr"] = float(
                    payload["native_ultra_direction_metrics"][direction]["mrr"]
                )
            rows.append(row)
            observed.add(cell_id)
    if observed != set(planned):
        raise ValueError(f"CPU matrix is missing {len(set(planned) - observed)} cells")
    frame = pd.DataFrame(rows)
    if set(frame["decoder"]) != {"logistic", "catboost"}:
        raise ValueError("CPU matrix decoder set changed")
    return frame, registration


def load_provenance(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["dataset", "provenance_family"]:
        raise ValueError("provenance map schema changed")
    if frame["dataset"].duplicated().any() or len(frame) != EXPECTED_DATASETS:
        raise ValueError("provenance map is not a 57-dataset bijection")
    return dict(zip(frame["dataset"], frame["provenance_family"], strict=True))


def dataset_metrics(
    cpu: pd.DataFrame,
    tabpfn_path: Path,
    provenance: dict[str, str],
) -> pd.DataFrame:
    value_columns = [
        f"{prefix}{direction}_mrr" for direction in DIRECTIONS for prefix in ("", "native_")
    ]
    cpu_dataset = (
        cpu.groupby(
            [
                "dataset",
                "transfer_regime",
                "interface",
                "decoder",
                "context_per_class",
            ],
            as_index=False,
            observed=True,
        )[value_columns]
        .mean()
        .sort_values(
            [
                "dataset",
                "decoder",
                "interface",
                "context_per_class",
            ],
            ignore_index=True,
        )
    )
    tabpfn = pd.read_csv(tabpfn_path)
    required = {
        "dataset",
        "transfer_regime",
        "interface",
        "context_per_class",
        *value_columns,
    }
    if not required <= set(tabpfn.columns):
        raise ValueError("TabPFN dataset metrics schema changed")
    tabpfn = tabpfn.loc[:, sorted(required)].copy()
    tabpfn["decoder"] = "tabpfn"
    complete = pd.concat([cpu_dataset, tabpfn], ignore_index=True)
    complete["provenance_family"] = complete["dataset"].map(provenance)
    if complete["provenance_family"].isna().any():
        raise ValueError("dataset missing from the provenance map")
    expected = EXPECTED_DATASETS * len(INTERFACES) * len(CONTEXTS) * len(DECODERS)
    if len(complete) != expected:
        raise ValueError(f"combined matrix has {len(complete)} rows, expected {expected}")
    axes = {
        "dataset": EXPECTED_DATASETS,
        "interface": set(INTERFACES),
        "decoder": set(DECODERS),
        "context_per_class": set(CONTEXTS),
    }
    for column, expected_value in axes.items():
        observed = (
            complete[column].nunique() if isinstance(expected_value, int) else set(complete[column])
        )
        if observed != expected_value:
            raise ValueError(f"combined matrix axis mismatch for {column}")
    keys = ["dataset", "decoder", "interface", "context_per_class"]
    if complete.duplicated(keys).any():
        raise ValueError("combined matrix contains duplicate cells")
    return complete


def equal_family_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    values = [f"{direction}_mrr" for direction in DIRECTIONS]
    family = dataset.groupby(
        [
            "transfer_regime",
            "provenance_family",
            "decoder",
            "interface",
            "context_per_class",
        ],
        as_index=False,
        observed=True,
    )[values].mean()
    summary = family.groupby(
        ["transfer_regime", "decoder", "interface", "context_per_class"],
        as_index=False,
        observed=True,
    )[values].mean()
    counts = (
        family.groupby(
            ["transfer_regime", "decoder", "interface", "context_per_class"],
            as_index=False,
            observed=True,
        )["provenance_family"]
        .nunique()
        .rename(columns={"provenance_family": "family_count"})
    )
    return summary.merge(
        counts,
        on=["transfer_regime", "decoder", "interface", "context_per_class"],
        validate="one_to_one",
    )


def encoder_contrasts(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for contrast, (left, right) in CONTRASTS.items():
        for direction in DIRECTIONS:
            value = f"{direction}_mrr"
            subset = dataset.loc[
                dataset["interface"].isin((left, right)),
                [
                    "dataset",
                    "provenance_family",
                    "transfer_regime",
                    "decoder",
                    "context_per_class",
                    "interface",
                    value,
                ],
            ]
            pivot = subset.pivot(
                index=[
                    "dataset",
                    "provenance_family",
                    "transfer_regime",
                    "decoder",
                    "context_per_class",
                ],
                columns="interface",
                values=value,
            )
            if set(pivot.columns) != {left, right}:
                raise ValueError(f"incomplete interface pair for {contrast}")
            for item in pivot.reset_index().to_dict("records"):
                rows.append(
                    {
                        **{
                            key: item[key]
                            for key in (
                                "dataset",
                                "provenance_family",
                                "transfer_regime",
                                "decoder",
                                "context_per_class",
                            )
                        },
                        "contrast": contrast,
                        "direction": direction,
                        "effect": float(item[left] - item[right]),
                    }
                )
    effects = pd.DataFrame(rows)
    family = effects.groupby(
        [
            "contrast",
            "direction",
            "context_per_class",
            "transfer_regime",
            "decoder",
            "provenance_family",
        ],
        as_index=False,
        observed=True,
    )["effect"].mean()
    summary_rows: list[dict[str, Any]] = []
    group_keys = [
        "contrast",
        "direction",
        "context_per_class",
        "transfer_regime",
        "decoder",
    ]
    for key, members in effects.groupby(group_keys, observed=True):
        selector = dict(zip(group_keys, key, strict=True))
        family_members = family
        for column, value in selector.items():
            family_members = family_members.loc[family_members[column] == value]
        family_values = family_members["effect"]
        dataset_values = members["effect"]
        loo = [
            float(family_values.drop(index=index).mean())
            for index in family_values.index
            if len(family_values) > 1
        ]
        summary_rows.append(
            {
                **selector,
                "equal_family_effect": float(family_values.mean()),
                "equal_dataset_effect": float(dataset_values.mean()),
                "family_count": len(family_values),
                "dataset_count": len(dataset_values),
                "dataset_wins": int((dataset_values > TIE_BAND).sum()),
                "dataset_ties": int((dataset_values.abs() <= TIE_BAND).sum()),
                "dataset_losses": int((dataset_values < -TIE_BAND).sum()),
                "leave_one_family_out_min": min(loo) if loo else float("nan"),
                "leave_one_family_out_max": max(loo) if loo else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return effects, summary


def decoder_interactions(effects: pd.DataFrame) -> pd.DataFrame:
    pivot = effects.pivot(
        index=[
            "dataset",
            "provenance_family",
            "transfer_regime",
            "contrast",
            "direction",
            "context_per_class",
        ],
        columns="decoder",
        values="effect",
    )
    if set(pivot.columns) != set(DECODERS):
        raise ValueError("decoder interaction matrix is incomplete")
    rows: list[dict[str, Any]] = []
    for left, right in (
        ("tabpfn", "logistic"),
        ("tabpfn", "catboost"),
        ("catboost", "logistic"),
    ):
        paired = pivot.reset_index()
        paired["interaction"] = paired[left] - paired[right]
        paired["decoder_pair"] = f"{left}_minus_{right}"
        rows.extend(paired.to_dict("records"))
    interactions = pd.DataFrame(rows)
    family = interactions.groupby(
        [
            "contrast",
            "direction",
            "context_per_class",
            "transfer_regime",
            "decoder_pair",
            "provenance_family",
        ],
        as_index=False,
        observed=True,
    )["interaction"].mean()
    return (
        family.groupby(
            [
                "contrast",
                "direction",
                "context_per_class",
                "transfer_regime",
                "decoder_pair",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            equal_family_interaction=("interaction", "mean"),
            family_count=("provenance_family", "nunique"),
        )
        .sort_values(
            [
                "contrast",
                "direction",
                "context_per_class",
                "transfer_regime",
                "decoder_pair",
            ],
            ignore_index=True,
        )
    )


def best_interfaces(equal_family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in DIRECTIONS:
        value = f"{direction}_mrr"
        keys = ["transfer_regime", "decoder", "context_per_class"]
        for key, members in equal_family.groupby(keys, observed=True):
            ordered = members.sort_values(value, ascending=False)
            winner = ordered.iloc[0]
            runner_up = ordered.iloc[1]
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "direction": direction,
                    "best_interface": winner["interface"],
                    "best_equal_family_mrr": float(winner[value]),
                    "runner_up_interface": runner_up["interface"],
                    "margin_over_runner_up": float(winner[value] - runner_up[value]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run-root", type=Path, required=True)
    parser.add_argument("--tabpfn-dataset-metrics", type=Path, required=True)
    parser.add_argument("--provenance-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu, registration = load_cpu_cells(args.cpu_run_root.resolve())
    provenance = load_provenance(args.provenance_map.resolve())
    dataset = dataset_metrics(
        cpu,
        args.tabpfn_dataset_metrics.resolve(),
        provenance,
    )
    equal_family = equal_family_summary(dataset)
    effects, contrast_summary = encoder_contrasts(dataset)
    interactions = decoder_interactions(effects)
    winners = best_interfaces(equal_family)

    artifacts = {
        "dataset_metrics.csv": dataset,
        "equal_family_absolute_mrr.csv": equal_family,
        "encoder_contrast_by_dataset.csv": effects,
        "encoder_contrast_summary.csv": contrast_summary,
        "decoder_interaction_summary.csv": interactions,
        "best_interface_by_regime.csv": winners,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False)

    summary = {
        "schema_version": 1,
        "status": "COMPLETE_AND_AUDITED",
        "scientific_role": (
            "post-hoc fixed-panel encoder-by-decoder compatibility characterization"
        ),
        "cpu_registration": registration,
        "tabpfn_dataset_metrics_sha256": file_sha256(args.tabpfn_dataset_metrics.resolve()),
        "provenance_map_sha256": file_sha256(args.provenance_map.resolve()),
        "datasets": EXPECTED_DATASETS,
        "decoders": list(DECODERS),
        "interfaces": list(INTERFACES),
        "contexts_per_class": list(CONTEXTS),
        "directions": list(DIRECTIONS),
        "selected_contrasts": CONTRASTS,
        "tie_band": TIE_BAND,
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
