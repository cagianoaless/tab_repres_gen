#!/usr/bin/env python3
"""Extract paired query-level effects from completed RFMLink result matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

B0_NAMES = (
    "relation_id",
    "head_total_out",
    "head_total_in",
    "tail_total_out",
    "tail_total_in",
    "head_relation_out_degree",
    "tail_relation_in_degree",
    "relation_frequency",
    "relation_functionality",
    "relation_inverse_functionality",
    "pair_other_relation_count",
    "reverse_pair_relation_count",
    "two_hop_count",
    "two_hop_start_relation_count",
    "two_hop_end_relation_count",
    "reverse_two_hop_count",
    "same_entity_cluster",
    "cluster_relation_density",
    "ultra_score",
)
B2V2_NAMES = (
    "head_pagerank",
    "tail_pagerank",
    "same_connected_component",
    "head_component_log_fraction",
    "tail_component_log_fraction",
    "heat_similarity_t0p1",
    "heat_similarity_t1p0",
    "heat_similarity_t10p0",
    "heat_similarity_t100p0",
    "diffusion_distance_t0p1",
    "diffusion_distance_t1p0",
    "diffusion_distance_t10p0",
    "diffusion_distance_t100p0",
)
SEEDS = (101, 102, 103)
QUERY_COUNT = 128


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_regimes(dataset_metrics: Path) -> dict[str, str]:
    frame = pd.read_csv(dataset_metrics, usecols=["dataset", "transfer_regime"])
    frame = frame.drop_duplicates()
    if len(frame) != 57 or frame["dataset"].duplicated().any():
        raise ValueError("dataset-to-regime map must contain 57 unique datasets")
    return dict(zip(frame["dataset"], frame["transfer_regime"], strict=True))


def _provenance_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["dataset", "provenance_family"]:
        raise ValueError("unexpected provenance-map schema")
    if len(frame) != 57 or frame["dataset"].duplicated().any():
        raise ValueError("provenance map must contain 57 unique datasets")
    return dict(zip(frame["dataset"], frame["provenance_family"], strict=True))


def query_properties(
    asset_root: Path,
    provenance: dict[str, str],
    regimes: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    manifest_path = asset_root / "manifest.json"
    manifest = read_json(manifest_path)
    if tuple(manifest["b0_feature_names"]) != B0_NAMES:
        raise ValueError("B0 feature schema changed")
    if tuple(manifest["b2v2_feature_names"]) != B2V2_NAMES:
        raise ValueError("B2v2 feature schema changed")
    if set(manifest["datasets"]) != set(provenance):
        raise ValueError("asset and provenance dataset panels differ")

    frames: list[pd.DataFrame] = []
    identity: dict[str, dict[str, np.ndarray]] = {}
    asset_hashes: dict[str, str] = {}
    for dataset in sorted(provenance):
        path = asset_root / f"{dataset}.npz"
        expected_hash = manifest["datasets"][dataset]["archive_sha256"]
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"{dataset}: frozen query asset checksum changed")
        asset_hashes[dataset] = observed_hash
        with np.load(path, allow_pickle=False) as arrays:
            targets = np.asarray(arrays["target_entities"], dtype=np.int64)
            sides = np.asarray(arrays["side_codes"], dtype=np.int8)
            anchors = np.asarray(arrays["anchor_entities"], dtype=np.int64)
            relations = np.asarray(arrays["direction_relations"], dtype=np.int64)
            triples = np.asarray(arrays["selected_triples"], dtype=np.int64)
            num_entities = int(arrays["num_entities"])
            num_relations = int(arrays["num_relations"])
            if not all(
                len(values) == QUERY_COUNT for values in (targets, sides, anchors, relations)
            ):
                raise ValueError(f"{dataset}: directional query count changed")
            if not np.array_equal(sides, np.tile([0, 1], QUERY_COUNT // 2)):
                raise ValueError(f"{dataset}: directional query order changed")
            if triples.shape != (QUERY_COUNT // 2, 3):
                raise ValueError(f"{dataset}: source triple count changed")
            row_indices = np.arange(QUERY_COUNT, dtype=np.int64) * num_entities + targets
            b0 = np.asarray(arrays["candidate_b0"][row_indices], dtype=np.float64)
            b2v2 = np.asarray(arrays["candidate_b2v2"][row_indices], dtype=np.float64)
            valid_candidates = np.count_nonzero(
                ~np.asarray(arrays["filtered"], dtype=np.bool_), axis=1
            )

        source_ids = np.arange(QUERY_COUNT, dtype=np.int64) // 2
        source = triples[source_ids]
        frame = pd.DataFrame(
            {
                "dataset": dataset,
                "provenance_family": provenance[dataset],
                "transfer_regime": regimes[dataset],
                "query_id": np.arange(QUERY_COUNT, dtype=np.int64),
                "source_triple_id": source_ids,
                "side_code": sides,
                "side": np.where(sides == 0, "tail", "head"),
                "source_head": source[:, 0],
                "relation_id": relations,
                "source_tail": source[:, 2],
                "anchor_entity": anchors,
                "target_entity": targets,
                "num_entities": num_entities,
                "num_relations": num_relations,
                "valid_candidates": valid_candidates,
            }
        )
        for index, name in enumerate(B0_NAMES):
            if name != "relation_id":
                frame[name] = b0[:, index]
        for index, name in enumerate(B2V2_NAMES):
            frame[name] = b2v2[:, index]

        head_degree = b0[:, 1] + b0[:, 2]
        tail_degree = b0[:, 3] + b0[:, 4]
        frame["anchor_total_degree"] = np.where(sides == 0, head_degree, tail_degree)
        frame["target_total_degree"] = np.where(sides == 0, tail_degree, head_degree)
        frame["anchor_relation_degree"] = np.where(sides == 0, b0[:, 5], b0[:, 6])
        frame["target_relation_degree"] = np.where(sides == 0, b0[:, 6], b0[:, 5])
        frame["log_relation_frequency"] = np.log1p(b0[:, 7])
        frame["log_anchor_total_degree"] = np.log1p(frame["anchor_total_degree"])
        frame["log_target_total_degree"] = np.log1p(frame["target_total_degree"])
        frame["log_anchor_relation_degree"] = np.log1p(frame["anchor_relation_degree"])
        frame["log_target_relation_degree"] = np.log1p(frame["target_relation_degree"])
        frame["degree_log_imbalance"] = np.abs(
            frame["log_anchor_total_degree"] - frame["log_target_total_degree"]
        )
        frame["log_dyadic_evidence"] = np.log1p(b0[:, 10] + b0[:, 11])
        path_total = b0[:, 12:16].sum(axis=1)
        frame["log_path_evidence"] = np.log1p(path_total)
        frame["path_available"] = (path_total > 0).astype(np.int8)
        frame["log_valid_candidates"] = np.log1p(valid_candidates)
        frames.append(frame)
        identity[dataset] = {
            "side_codes": sides,
            "target_entities": targets,
            "native_ultra_scores": b0[:, 18],
        }
    output = pd.concat(frames, ignore_index=True)
    return (
        output,
        identity,
        {
            "asset_manifest_sha256": file_sha256(manifest_path),
            "dataset_archive_sha256": asset_hashes,
        },
    )


def reused_query_properties(
    path: Path,
    provenance: dict[str, str],
    regimes: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Reuse the previously checksum-audited 57-dataset query table.

    The public repository intentionally retains only the three-dataset raw-asset
    pilot. Reusing the immutable query table avoids reconstructing removed bulk
    assets while prediction checksums are still revalidated below.
    """
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "query_id",
        "side_code",
        "target_entity",
        "ultra_score",
        "provenance_family",
        "transfer_regime",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reused query-property table lacks columns: {sorted(missing)}")
    if len(frame) != 57 * QUERY_COUNT or set(frame["dataset"]) != set(provenance):
        raise ValueError("reused query-property panel changed")
    observed_provenance = frame.groupby("dataset")["provenance_family"].first().to_dict()
    observed_regimes = frame.groupby("dataset")["transfer_regime"].first().to_dict()
    if observed_provenance != provenance or observed_regimes != regimes:
        raise ValueError("reused query-property metadata changed")
    identity: dict[str, dict[str, np.ndarray]] = {}
    for dataset in sorted(provenance):
        part = frame.loc[frame["dataset"] == dataset].sort_values("query_id")
        if not np.array_equal(part["query_id"].to_numpy(), np.arange(QUERY_COUNT)):
            raise ValueError(f"{dataset}: reused query order changed")
        identity[dataset] = {
            "side_codes": part["side_code"].to_numpy(dtype=np.int8),
            "target_entities": part["target_entity"].to_numpy(dtype=np.int64),
            "native_ultra_scores": part["ultra_score"].to_numpy(dtype=np.float64),
        }
    native_columns = [
        "native_ultra_reciprocal_rank",
        "native_ultra_rank",
        "native_ultra_target_score",
        "log_native_ultra_rank",
    ]
    return (
        frame.drop(columns=[name for name in native_columns if name in frame.columns]),
        identity,
        {"reused_query_properties_sha256": file_sha256(path)},
    )


def _result_paths(root: Path) -> list[Path]:
    paths = sorted((root / "results").glob("shard-*/*.metrics.json"))
    if not paths:
        raise ValueError(f"no metric files found under {root}")
    return paths


def collect_seed_averaged_outcomes(
    roots: list[Path],
    expected_cells: int,
    identity: dict[str, dict[str, np.ndarray]],
    stream: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    observed_cells: set[str] = set()
    registrations = []
    native_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    for root in roots:
        registration_path = root / "registration.json"
        registration = read_json(registration_path)
        registrations.append(
            {
                "path": str(root),
                "registration_sha256": file_sha256(registration_path),
                "registered_cells": registration.get("cells"),
                "plan_sha256": registration.get("plan_sha256"),
                "source_commit": registration.get("source_commit"),
            }
        )
        for metric_path in _result_paths(root):
            payload = read_json(metric_path)
            cell = payload["cell"]
            cell_id = str(cell["cell_id"])
            if cell_id in observed_cells:
                raise ValueError(f"duplicate cell ID: {cell_id}")
            observed_cells.add(cell_id)
            prediction_path = metric_path.parent / payload["predictions"]["path"]
            if file_sha256(prediction_path) != payload["predictions"]["sha256"]:
                raise ValueError(f"prediction checksum changed: {prediction_path}")
            with np.load(prediction_path, allow_pickle=False) as arrays:
                query_ids = np.asarray(arrays["query_ids"], dtype=np.int64)
                side_codes = np.asarray(arrays["side_codes"], dtype=np.int8)
                target_entities = np.asarray(arrays["target_entities"], dtype=np.int64)
                rr = np.asarray(arrays["reciprocal_ranks"], dtype=np.float64)
                ranks = np.asarray(arrays["ranks"], dtype=np.float64)
                scores = np.asarray(arrays["target_scores"], dtype=np.float64)
                native_rr = np.asarray(arrays["native_ultra_reciprocal_ranks"], dtype=np.float64)
                native_ranks = np.asarray(arrays["native_ultra_ranks"], dtype=np.float64)
                native_scores = np.asarray(arrays["native_ultra_target_scores"], dtype=np.float64)
            dataset = str(cell["dataset"])
            if not np.array_equal(query_ids, np.arange(QUERY_COUNT)):
                raise ValueError(f"{cell_id}: query ordering changed")
            if not np.array_equal(side_codes, identity[dataset]["side_codes"]):
                raise ValueError(f"{cell_id}: side ordering differs from frozen asset")
            if not np.array_equal(target_entities, identity[dataset]["target_entities"]):
                raise ValueError(f"{cell_id}: targets differ from frozen asset")
            expected_native_scores = identity[dataset]["native_ultra_scores"]
            if not np.allclose(native_scores, expected_native_scores, atol=1e-6, rtol=1e-6):
                raise ValueError(f"{cell_id}: native target scores differ from frozen asset")
            native = native_by_dataset.setdefault(
                dataset,
                {"rr": native_rr, "ranks": native_ranks, "scores": native_scores},
            )
            for key, values in (
                ("rr", native_rr),
                ("ranks", native_ranks),
                ("scores", native_scores),
            ):
                if not np.array_equal(native[key], values):
                    raise ValueError(f"{cell_id}: native {key} changed across cells")

            count = len(query_ids)
            metadata = {
                "stream": stream,
                "dataset": dataset,
                "decoder": str(cell["decoder"]),
                "interface": str(cell["interface"]),
                "context_per_class": int(cell["context_per_class"]),
                "context_mode": str(cell.get("context_mode", "original_uniform")),
                "seed": int(cell["seed"]),
            }
            for name, value in metadata.items():
                columns[name].append(np.repeat(value, count))
            columns["query_id"].append(query_ids)
            columns["reciprocal_rank"].append(rr)
            columns["rank"].append(ranks)
            columns["hit_at_10"].append((ranks <= 10.0).astype(np.float64))
            columns["target_score"].append(scores)
            columns["native_ultra_reciprocal_rank"].append(native_rr)
            columns["native_ultra_rank"].append(native_ranks)
            columns["native_ultra_target_score"].append(native_scores)
    if len(observed_cells) != expected_cells:
        raise ValueError(f"{stream}: expected {expected_cells} cells, found {len(observed_cells)}")
    long = pd.DataFrame({name: np.concatenate(values) for name, values in columns.items()})
    keys = [
        "stream",
        "dataset",
        "decoder",
        "interface",
        "context_per_class",
        "context_mode",
        "query_id",
    ]
    means = (
        long.groupby(keys, as_index=False, observed=True)[
            [
                "reciprocal_rank",
                "rank",
                "hit_at_10",
                "target_score",
                "native_ultra_reciprocal_rank",
                "native_ultra_rank",
                "native_ultra_target_score",
            ]
        ]
        .mean()
        .sort_values(keys, ignore_index=True)
    )
    counts = long.groupby(keys, observed=True)["seed"].nunique()
    if not (counts == len(SEEDS)).all():
        raise ValueError(f"{stream}: at least one query outcome lacks three context seeds")
    return means, {
        "stream": stream,
        "cells": len(observed_cells),
        "seed_averaged_rows": len(means),
        "registrations": registrations,
    }


def _append_effect(
    frames: list[pd.DataFrame],
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    stream: str,
    effect_family: str,
    contrast: str,
    metadata: dict[str, Any],
) -> None:
    keys = ["dataset", "query_id"]
    merged = left[keys + ["reciprocal_rank"]].merge(
        right[keys + ["reciprocal_rank"]],
        on=keys,
        suffixes=("_treatment", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != 57 * QUERY_COUNT and stream != "graphlet":
        raise ValueError(f"{stream}/{effect_family}/{contrast}: incomplete query pairing")
    merged["stream"] = stream
    merged["effect_family"] = effect_family
    merged["contrast"] = contrast
    for name, value in metadata.items():
        merged[name] = value
    merged["delta_reciprocal_rank"] = (
        merged["reciprocal_rank_treatment"] - merged["reciprocal_rank_reference"]
    )
    frames.append(merged)


def ladder_effects(
    outcomes: pd.DataFrame,
    plan: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    original = outcomes.loc[outcomes["context_mode"] == "original_uniform"]
    for decoder in sorted(original["decoder"].unique()):
        current = original.loc[original["decoder"] == decoder]
        at_200 = current.loc[current["context_per_class"] == 200]
        for contrast, (treatment, reference) in plan["encoder_contrasts"].items():
            _append_effect(
                frames,
                at_200.loc[at_200["interface"] == treatment],
                at_200.loc[at_200["interface"] == reference],
                stream="encoder_ladder",
                effect_family="encoder",
                contrast=contrast,
                metadata={
                    "decoder": decoder,
                    "interface": "",
                    "context_per_class": 200,
                    "context_mode": "original_uniform",
                },
            )
        for interface in sorted(current["interface"].unique()):
            _append_effect(
                frames,
                current.loc[
                    (current["interface"] == interface) & (current["context_per_class"] == 200)
                ],
                current.loc[
                    (current["interface"] == interface) & (current["context_per_class"] == 20)
                ],
                stream="encoder_ladder",
                effect_family="context_gain",
                contrast="context_200_minus_20",
                metadata={
                    "decoder": decoder,
                    "interface": interface,
                    "context_per_class": 200,
                    "context_mode": "original_uniform",
                },
            )
    at_200 = original.loc[original["context_per_class"] == 200]
    for interface in plan["decoder_interfaces"]:
        current = at_200.loc[at_200["interface"] == interface]
        for contrast, (treatment, reference) in plan["decoder_contrasts"].items():
            _append_effect(
                frames,
                current.loc[current["decoder"] == treatment],
                current.loc[current["decoder"] == reference],
                stream="encoder_ladder",
                effect_family="decoder",
                contrast=contrast,
                metadata={
                    "decoder": "",
                    "interface": interface,
                    "context_per_class": 200,
                    "context_mode": "original_uniform",
                },
            )
    return pd.concat(frames, ignore_index=True)


def hard_context_effects(
    outcomes: pd.DataFrame,
    plan: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for decoder in sorted(outcomes["decoder"].unique()):
        for interface in sorted(outcomes["interface"].unique()):
            current = outcomes.loc[
                (outcomes["decoder"] == decoder) & (outcomes["interface"] == interface)
            ]
            _append_effect(
                frames,
                current.loc[current["context_mode"] == "ultra_hard_top1"],
                current.loc[current["context_mode"] == "filtered_uniform"],
                stream="hard_context",
                effect_family="hard_context",
                contrast="hard_minus_uniform",
                metadata={
                    "decoder": decoder,
                    "interface": interface,
                    "context_per_class": 200,
                    "context_mode": "hard_minus_uniform",
                },
            )
    for mode in sorted(outcomes["context_mode"].unique()):
        for interface in sorted(outcomes["interface"].unique()):
            current = outcomes.loc[
                (outcomes["context_mode"] == mode) & (outcomes["interface"] == interface)
            ]
            for contrast, (treatment, reference) in plan["decoder_contrasts"].items():
                _append_effect(
                    frames,
                    current.loc[current["decoder"] == treatment],
                    current.loc[current["decoder"] == reference],
                    stream="hard_context",
                    effect_family="decoder",
                    contrast=contrast,
                    metadata={
                        "decoder": "",
                        "interface": interface,
                        "context_per_class": 200,
                        "context_mode": mode,
                    },
                )
    return pd.concat(frames, ignore_index=True)


def graphlet_effects(path: Path, plan: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    keys = ["dataset", "decoder", "interface", "context_mode", "query_ids"]
    mean = (
        raw.groupby(keys, as_index=False, observed=True)["reciprocal_ranks"]
        .mean()
        .rename(columns={"query_ids": "query_id", "reciprocal_ranks": "reciprocal_rank"})
    )
    frames: list[pd.DataFrame] = []
    for decoder in sorted(mean["decoder"].unique()):
        for mode in sorted(mean["context_mode"].unique()):
            current = mean.loc[(mean["decoder"] == decoder) & (mean["context_mode"] == mode)]
            for contrast, (treatment, reference) in plan["graphlet_contrasts"].items():
                _append_effect(
                    frames,
                    current.loc[current["interface"] == treatment],
                    current.loc[current["interface"] == reference],
                    stream="graphlet",
                    effect_family="encoder",
                    contrast=contrast,
                    metadata={
                        "decoder": decoder,
                        "interface": "",
                        "context_per_class": 200,
                        "context_mode": mode,
                    },
                )
        for interface in sorted(mean["interface"].unique()):
            current = mean.loc[(mean["decoder"] == decoder) & (mean["interface"] == interface)]
            for contrast, (treatment, reference) in plan["graphlet_sampling_contrasts"].items():
                _append_effect(
                    frames,
                    current.loc[current["context_mode"] == treatment],
                    current.loc[current["context_mode"] == reference],
                    stream="graphlet",
                    effect_family="sampling",
                    contrast=contrast,
                    metadata={
                        "decoder": decoder,
                        "interface": interface,
                        "context_per_class": 200,
                        "context_mode": contrast,
                    },
                )
    return pd.concat(frames, ignore_index=True)


def graphlet_properties(asset_root: Path) -> pd.DataFrame:
    manifest = read_json(asset_root / "manifest.json")
    if len(manifest["graphlet_feature_names"]) != 18:
        raise ValueError("graphlet feature schema changed")
    frames = []
    for dataset in ("fb15k237", "hetionet", "wn18rr"):
        with np.load(asset_root / f"{dataset}.npz", allow_pickle=False) as arrays:
            num_entities = int(arrays["num_entities"])
            targets = np.asarray(arrays["target_entities"], dtype=np.int64)
            rows = np.arange(QUERY_COUNT, dtype=np.int64) * num_entities + targets
            values = np.asarray(arrays["candidate_graphlet"][rows], dtype=np.float64)
        count_values = values[:, :9]
        cosine_values = values[:, 9:]
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "query_id": np.arange(QUERY_COUNT),
                    "graphlet_nonzero_count": np.count_nonzero(values, axis=1),
                    "graphlet_any": (np.count_nonzero(values, axis=1) > 0).astype(int),
                    "graphlet_count_l1": np.abs(count_values).sum(axis=1),
                    "graphlet_cosine_l1": np.abs(cosine_values).sum(axis=1),
                }
            )
        )
    output = pd.concat(frames, ignore_index=True)
    output.attrs["manifest_sha256"] = file_sha256(asset_root / "manifest.json")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--provenance-map", type=Path, required=True)
    parser.add_argument("--dataset-metrics", type=Path, required=True)
    parser.add_argument("--tabpfn-root", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--hard-tabpfn-root", type=Path, required=True)
    parser.add_argument("--hard-cpu-root", type=Path, required=True)
    parser.add_argument("--graphlet-query-metrics", type=Path, required=True)
    parser.add_argument("--graphlet-asset-root", type=Path, required=True)
    parser.add_argument("--reuse-query-properties", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    provenance = _provenance_map(args.provenance_map)
    regimes = _dataset_regimes(args.dataset_metrics)
    if args.reuse_query_properties:
        properties, identity, asset_audit = reused_query_properties(
            args.reuse_query_properties.resolve(), provenance, regimes
        )
    else:
        properties, identity, asset_audit = query_properties(
            args.asset_root.resolve(), provenance, regimes
        )
    ladder_outcomes, ladder_audit = collect_seed_averaged_outcomes(
        [args.tabpfn_root.resolve(), args.cpu_root.resolve()],
        int(plan["panels"]["encoder_ladder"]["expected_cells"]),
        identity,
        "encoder_ladder",
    )
    hard_outcomes, hard_audit = collect_seed_averaged_outcomes(
        [args.hard_tabpfn_root.resolve(), args.hard_cpu_root.resolve()],
        int(plan["panels"]["hard_context"]["expected_cells"]),
        identity,
        "hard_context",
    )

    native = ladder_outcomes.loc[
        (ladder_outcomes["decoder"] == "tabpfn")
        & (ladder_outcomes["interface"] == "ultra_score_only")
        & (ladder_outcomes["context_per_class"] == 200),
        [
            "dataset",
            "query_id",
            "native_ultra_reciprocal_rank",
            "native_ultra_rank",
            "native_ultra_target_score",
        ],
    ]
    if len(native) != 57 * QUERY_COUNT:
        raise ValueError("native query diagnostics are incomplete")
    properties = properties.merge(native, on=["dataset", "query_id"], validate="one_to_one")
    properties["log_native_ultra_rank"] = np.log1p(properties["native_ultra_rank"])

    effects = pd.concat(
        [
            ladder_effects(ladder_outcomes, plan),
            hard_context_effects(hard_outcomes, plan),
            graphlet_effects(args.graphlet_query_metrics.resolve(), plan),
        ],
        ignore_index=True,
    )
    effects = effects.merge(
        properties[
            [
                "dataset",
                "query_id",
                "source_triple_id",
                "side",
                "provenance_family",
                "transfer_regime",
            ]
        ],
        on=["dataset", "query_id"],
        how="left",
        validate="many_to_one",
    )
    if effects["provenance_family"].isna().any():
        raise ValueError("at least one effect lacks query provenance")
    graphlet = graphlet_properties(args.graphlet_asset_root.resolve())

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    properties.to_csv(output / "query_properties.csv.gz", index=False, compression="gzip")
    effects.to_csv(output / "query_effects.csv.gz", index=False, compression="gzip")
    graphlet.to_csv(output / "graphlet_query_properties.csv.gz", index=False, compression="gzip")
    write_json(
        output / "extraction_audit.json",
        {
            "analysis_id": plan["analysis_id"],
            "analysis_specification_recorded_at_utc": plan[
                "analysis_specification_recorded_at_utc"
            ],
            "query_property_rows": len(properties),
            "effect_rows": len(effects),
            "graphlet_property_rows": len(graphlet),
            "asset_audit": asset_audit,
            "graphlet_asset_manifest_sha256": graphlet.attrs["manifest_sha256"],
            "ladder_audit": ladder_audit,
            "hard_context_audit": hard_audit,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "extraction_audit.json"
            },
            "status": "PASS",
        },
    )
    print(
        json.dumps(
            {
                "query_property_rows": len(properties),
                "effect_rows": len(effects),
                "graphlet_property_rows": len(graphlet),
                "status": "PASS",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
