#!/usr/bin/env python3
"""Prospective positive-context/identity sensitivity; never generates model results.

This stricter fixed-negative-row contract and its 12/24-test scopes do NOT
reproduce the historical resampled-negative, 60-test analysis. Historical derived
tables are shipped separately in relation-context-historical; their verifier is
analysis.verify_historical_relation_context. Raw historical assets remain absent.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.analyze_ultra4g_controls import exact_joint_max_t, paired_summary, provenance_lookup
from relational_tfm.datasets.rfmlink import _nested_context_indices, encode_feature_schema
from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.plan import read_plan
from relational_tfm.protocol import FROZEN_KG_RELATION_CONTEXT_ASSET_ID
from relational_tfm.runner import verify_shard

ENCODERS = (
    "b0_local_only",
    "b0_local_ultra",
    "b0_local_no_relation_id",
    "b0_local_ultra_no_relation_id",
)
IDENTITY_PAIRS = tuple(zip(ENCODERS[:2], ENCODERS[2:], strict=True))
KEYS = ["dataset", "provenance_family", "decoder", "interface", "context_mode", "seed"]
DECODER_FAMILIES = {
    "logistic": "logistic",
    "logistic_nominal": "logistic",
    "catboost": "catboost",
    "catboost_nominal": "catboost",
    "tabpfn": "tabpfn",
}


def decoder_pairing_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    """Compare learning settings/checkpoint hashes, not schema indices or GPU IDs.

    Categorical indices are separately checked against the encoder schema. Model
    file paths are deployment locations; their stored hashes remain in identity.
    """
    deployment_or_schema = {"categorical_features_indices", "cat_features", "resolved_devices"}

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in deployment_or_schema
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    identity = normalize(provenance)
    if "checkpoints" in identity:
        identity["checkpoints"] = [
            {key: value for key, value in checkpoint.items() if key != "path"}
            for checkpoint in identity["checkpoints"]
        ]
    return identity


def relation_coverage(evaluation: np.ndarray, positives: np.ndarray) -> dict[str, float]:
    """Type/mass coverage and TV on the union of raw nominal relation IDs."""
    for values in (evaluation, positives):
        if values.ndim != 1 or not len(values):
            raise ValueError("relation IDs must be nonempty vectors")
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values != values.astype(int))
        ):
            raise ValueError("relation IDs must be finite nonnegative integers")
    vocabulary = np.union1d(evaluation, positives)
    evaluation_mass = np.array([(evaluation == r).mean() for r in vocabulary])
    positive_mass = np.array([(positives == r).mean() for r in vocabulary])
    covered = positive_mass > 0
    return {
        "relation_type_coverage": float(covered[evaluation_mass > 0].mean()),
        "relation_mass_coverage": float(evaluation_mass[covered].sum()),
        "total_variation_distance": float(0.5 * np.abs(evaluation_mass - positive_mass).sum()),
    }


def validate_matrix(
    cells: pd.DataFrame,
    modes: tuple[str, str],
    *,
    datasets: int = 57,
    families: int = 16,
) -> None:
    """Reject partial/unpaired panels, rather than silently dropping observations."""
    if cells.empty or cells.duplicated(KEYS).any():
        raise ValueError("empty or duplicate result matrix")
    if cells.dataset.nunique() != datasets or cells.provenance_family.nunique() != families:
        raise ValueError("incomplete dataset/family panel")
    if (cells.groupby("dataset").provenance_family.nunique() != 1).any():
        raise ValueError("a dataset has multiple provenance families")
    if set(cells.interface) != set(ENCODERS) or set(cells.context_mode) != set(modes):
        raise ValueError("encoder/context axes differ from the declared control")
    if (
        cells.decoder.nunique() != 3
        or {DECODER_FAMILIES.get(name) for name in cells.decoder}
        != {"logistic", "catboost", "tabpfn"}
        or cells.seed.nunique() != 3
    ):
        raise ValueError("control requires three decoders and three paired seeds")
    expected = set(
        itertools.product(
            cells.dataset.unique(), cells.decoder.unique(), ENCODERS, modes, cells.seed.unique()
        )
    )
    observed = set(
        cells[["dataset", "decoder", "interface", "context_mode", "seed"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected:
        raise ValueError("incomplete Cartesian result matrix")
    if not np.isfinite(cells.combined_mrr).all() or not cells.combined_mrr.between(0, 1).all():
        raise ValueError("invalid MRR")
    if cells.context_per_class.nunique() != 1 or cells.context_per_class.iloc[0] <= 0:
        raise ValueError("context size is not fixed")
    # An archive fixes all candidate rows, queries, scores and filtering masks.
    for dataset, group in cells.groupby("dataset"):
        if group.archive_sha256.nunique() != 1:
            raise ValueError(f"{dataset}: evaluation archive changed between cells")
        for _, paired in group.groupby(["seed", "context_mode"]):
            if paired.context_indices_sha256.nunique() != 1:
                raise ValueError(f"{dataset}: encoders/decoders do not share context rows")


def contrast_tables(
    cells: pd.DataFrame, modes: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed means, then dataset effects; family weighting occurs in shared helper."""
    dataset = cells.groupby(KEYS[:-1], as_index=False).combined_mrr.mean()
    dataset["query_transform"] = "raw"
    summary, effects = paired_summary(
        dataset, axis="context_mode", reference=modes[0], alternatives=(modes[1],)
    )
    summaries, contrasts = [summary], [effects]
    for mode in modes:
        for reference, alternative in IDENTITY_PAIRS:
            selected = dataset.loc[
                (dataset.context_mode == mode) & dataset.interface.isin((reference, alternative))
            ].copy()
            # Preserve schema IDs in reference/alternative; context mode is the
            # fixed stratum encoded in the interface field of the contrast table.
            selected["encoder_variant"] = selected["interface"]
            selected["interface"] = mode
            part, delta = paired_summary(
                selected, axis="encoder_variant", reference=reference, alternatives=(alternative,)
            )
            summaries.append(part)
            contrasts.append(delta)
    return pd.concat(summaries, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def load_results(roots: list[Path], asset_root: Path, provenance: dict[str, str]) -> pd.DataFrame:
    manifest_path = asset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["asset_id"] != FROZEN_KG_RELATION_CONTEXT_ASSET_ID:
        raise ValueError("wrong frozen relation-context asset")
    manifest_hash = file_sha256(manifest_path)
    seen: set[str] = set()
    paired_protocols: dict[tuple[str, str, int], str] = {}
    evaluation_records: dict[str, tuple[np.ndarray, ...]] = {}
    native_metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for root in roots:
        plan = read_plan(root / "plan.json")
        planned = {cell["cell_id"]: cell for cell in plan["cells"]}
        observed: set[str] = set()
        for shard in sorted((root / "results").glob("shard-*")):
            checked = verify_shard(shard)
            if checked["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("shard/plan hash mismatch")
            for cell_id in checked["cell_ids"]:
                if cell_id in seen or cell_id not in planned:
                    raise ValueError("duplicate or unplanned result")
                payload = json.loads((shard / f"{cell_id}.metrics.json").read_text())
                cell = payload["cell"]
                if cell != planned[cell_id]:
                    raise ValueError("cell differs from frozen plan")
                if (
                    cell["asset_id"] != manifest["asset_id"]
                    or cell["asset_manifest_sha256"] != manifest_hash
                ):
                    raise ValueError("result uses a different asset manifest")
                if cell.get("query_transform", "raw") != "raw" or cell["query_triples"] != 64:
                    raise ValueError("control is not raw q64 filtered all-entity evaluation")
                record = manifest["datasets"][cell["dataset"]]
                if cell["archive_sha256"] != record["archive_sha256"]:
                    raise ValueError("result/archive checksum mismatch")
                metadata = payload["task_metadata"]
                for name in ("archive_sha256", "asset_manifest_sha256", "context_mode"):
                    if metadata[name] != cell[name]:
                        raise ValueError(f"task metadata/cell mismatch: {name}")
                budget = int(cell["context_per_class"])
                if (
                    payload["context_rows"] != 2 * budget
                    or payload["context_positive_rows"] != budget
                ):
                    raise ValueError("context is not fixed-size balanced binary classification")
                _, names, categorical = encode_feature_schema(
                    np.zeros((1, 19), dtype=np.float32),
                    np.zeros((1, 13), dtype=np.float32),
                    cell["interface"],
                    tuple(manifest["b0_feature_names"]),
                    tuple(manifest["b2v2_feature_names"]),
                )
                if (
                    payload["feature_names"] != list(names)
                    or payload["feature_count"] != len(names)
                    or payload["categorical_feature_indices"] != list(categorical)
                ):
                    raise ValueError("result encoder schema differs from loader")
                protocol_key = (cell["dataset"], cell["decoder"], int(cell["seed"]))
                protocol = {
                    key: value
                    for key, value in cell.items()
                    if key not in {"index", "cell_id", "study_id", "interface", "context_mode"}
                }
                decoder = payload["decoder_provenance"]
                if "categorical_features_indices" in decoder and decoder[
                    "categorical_features_indices"
                ] != list(categorical):
                    raise ValueError("decoder categorical schema differs from encoder")
                protocol["decoder_provenance"] = decoder_pairing_identity(decoder)
                signature = object_sha256(protocol)
                if paired_protocols.setdefault(protocol_key, signature) != signature:
                    raise ValueError("uncontrolled protocol/model change between paired cells")
                mrr = float(payload["direction_metrics"]["combined"]["mrr"])
                with np.load(shard / payload["predictions"]["path"], allow_pickle=False) as arrays:
                    ranks = arrays["ranks"]
                    if len(ranks) != 128 or not np.allclose(
                        arrays["reciprocal_ranks"], 1 / ranks, atol=1e-12, rtol=0
                    ):
                        raise ValueError("invalid q64 reciprocal ranking records")
                    if not np.isclose(mrr, arrays["reciprocal_ranks"].mean(), atol=1e-12, rtol=0):
                        raise ValueError("MRR differs from stored query ranks")
                    frozen_queries = tuple(
                        arrays[name].copy()
                        for name in (
                            "query_ids",
                            "side_codes",
                            "target_entities",
                            "native_ultra_ranks",
                        )
                    )
                    reference_queries = evaluation_records.setdefault(
                        cell["dataset"], frozen_queries
                    )
                    if any(
                        not np.array_equal(left, right)
                        for left, right in zip(reference_queries, frozen_queries, strict=True)
                    ):
                        raise ValueError("query identity/native ranking changed between cells")
                    native = float(arrays["native_ultra_reciprocal_ranks"].mean())
                    if not np.allclose(
                        arrays["native_ultra_reciprocal_ranks"],
                        1 / arrays["native_ultra_ranks"],
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError("native reciprocal ranks are inconsistent")
                    if not np.isclose(
                        native,
                        payload["native_ultra_direction_metrics"]["combined"]["mrr"],
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError("native MRR differs from query ranks")
                    if native_metrics.setdefault(cell["dataset"], native) != native:
                        raise ValueError("native ranking baseline changed between cells")
                rows.append(
                    {
                        **{key: cell[key] for key in KEYS if key != "provenance_family"},
                        "provenance_family": provenance[cell["dataset"]],
                        "combined_mrr": mrr,
                        "context_per_class": budget,
                        "archive_sha256": cell["archive_sha256"],
                        "context_indices_sha256": metadata["context_indices_sha256"],
                    }
                )
                seen.add(cell_id)
                observed.add(cell_id)
        if observed != set(planned):
            raise ValueError(f"{root}: incomplete run")
    return pd.DataFrame(rows)


def coverage_rows(cells: pd.DataFrame, asset_root: Path, modes: tuple[str, str]) -> pd.DataFrame:
    manifest = json.loads((asset_root / "manifest.json").read_text())
    if len(manifest["b0_feature_names"]) != 19 or manifest["b0_feature_names"][0] != "relation_id":
        raise ValueError("coverage requires raw b0 relation identity at column zero")
    records: list[dict[str, Any]] = []
    for dataset, group in cells.groupby("dataset", sort=True):
        if Path(dataset).name != dataset:
            raise ValueError("unsafe dataset archive name")
        archive = asset_root / f"{dataset}.npz"
        if file_sha256(archive) != group.archive_sha256.iloc[0]:
            raise ValueError(f"{dataset}: archive bytes changed")
        with np.load(archive, allow_pickle=False) as arrays:
            entities = int(arrays["num_entities"])
            candidate = arrays["candidate_b0"]
            blocks = candidate.reshape(-1, entities, 19)
            relations = blocks[:, 0, 0]
            if len(blocks) != 128 or not np.all(blocks[:, :, 0] == relations[:, None]):
                raise ValueError("expected 64 head/tail queries with query-constant relation IDs")
            if arrays["filtered"].shape != (128, entities):
                raise ValueError("evaluation does not use all entities")
            # All triples receive equal head/tail weight; require equal relation mass.
            sides = arrays["side_codes"]
            if (sides == 0).sum() != 64 or (sides == 1).sum() != 64:
                raise ValueError("unbalanced evaluation directions")
            head = dict(zip(*np.unique(relations[sides == 1], return_counts=True), strict=True))
            tail = dict(zip(*np.unique(relations[sides == 0], return_counts=True), strict=True))
            if head != tail:
                raise ValueError("head/tail relation masses differ")
            for seed in sorted(group.seed.unique()):
                negative_reference: tuple[np.ndarray, ...] | None = None
                for mode in modes:
                    prefix = (
                        f"context_{seed}"
                        if mode == "original_uniform"
                        else f"context_{mode}_{seed}"
                    )
                    labels = arrays[f"{prefix}_y"]
                    x = arrays[f"{prefix}_X"]
                    global_x = arrays[f"{prefix}_b2v2"]
                    triples = arrays[f"{prefix}_triples"]
                    if (
                        not set(np.unique(labels)) <= {0, 1}
                        or x.shape != (len(labels), 19)
                        or global_x.shape != (len(labels), 13)
                        or triples.shape != (len(labels), 3)
                    ):
                        raise ValueError("context archive schema changed")
                    if not np.array_equal(triples[:, 1], x[:, 0]):
                        raise ValueError("stored context relation IDs disagree with triples")
                    budget = int(group.context_per_class.iloc[0])
                    selected = _nested_context_indices(labels, int(seed), budget)
                    cell = group.loc[(group.seed == seed) & (group.context_mode == mode)].iloc[0]
                    if object_sha256(selected.tolist()) != cell.context_indices_sha256:
                        raise ValueError("analysis selection differs from decoder context")
                    negatives = selected[labels[selected] == 0]
                    negative_rows = (triples[negatives], x[negatives], global_x[negatives])
                    if negative_reference is not None and any(
                        not np.array_equal(left, right)
                        for left, right in zip(negative_reference, negative_rows, strict=True)
                    ):
                        raise ValueError("negative context changed: not a positive-only control")
                    negative_reference = negative_rows
                    positives = x[selected[labels[selected] == 1], 0]
                    records.append(
                        {
                            "dataset": dataset,
                            "provenance_family": cell.provenance_family,
                            "seed": int(seed),
                            "context_mode": mode,
                            "positive_rows": budget,
                            **relation_coverage(relations, positives),
                        }
                    )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reference-mode", required=True)
    parser.add_argument("--matched-mode", required=True)
    parser.add_argument(
        "--correction-scope",
        choices=("joint", "separate"),
        required=True,
        help="Prospective scope: separate 12+12 or joint 24; neither is historical 60",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = (args.reference_mode, args.matched_mode)
    if modes[0] == modes[1]:
        raise ValueError("context modes must differ")
    provenance = provenance_lookup(args.provenance)
    cells = load_results(args.run_root, args.asset_root, provenance)
    if set(cells.dataset) != set(provenance):
        raise ValueError("provenance map and observed dataset panel differ")
    validate_matrix(cells, modes)
    coverage = coverage_rows(cells, args.asset_root, modes)
    summary, effects = contrast_tables(cells, modes)
    if len(summary) != 24:
        raise ValueError("expected 12 positive-context and 12 relation-identity contrasts")
    correction = (
        exact_joint_max_t(effects)
        if args.correction_scope == "joint"
        else pd.concat(
            [exact_joint_max_t(group) for _, group in effects.groupby("axis")], ignore_index=True
        )
    )
    if len(correction) != len(summary):
        raise ValueError("incomplete correction family")
    summary = summary.merge(
        correction,
        on=["decoder", "interface", "axis", "reference", "alternative"],
        validate="one_to_one",
    )
    metrics = ["relation_type_coverage", "relation_mass_coverage", "total_variation_distance"]
    dataset_coverage = coverage.groupby(
        ["dataset", "provenance_family", "context_mode"], as_index=False
    )[metrics].mean()
    coverage_summary = (
        dataset_coverage.groupby("context_mode")[metrics].mean().add_prefix("equal_dataset_")
    )
    family_coverage = dataset_coverage.groupby(["provenance_family", "context_mode"])[
        metrics
    ].mean()
    coverage_summary = coverage_summary.join(
        family_coverage.groupby("context_mode").mean().add_prefix("equal_family_")
    )
    audit = {
        "status": "PASS",
        "asset_id": FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
        "asset_manifest_sha256": file_sha256(args.asset_root / "manifest.json"),
        "provenance_sha256": file_sha256(args.provenance),
        "plan_sha256": [read_plan(root / "plan.json")["plan_sha256"] for root in args.run_root],
        "analysis_sha256": file_sha256(Path(__file__)),
        "cells": len(cells),
        "datasets": 57,
        "provenance_families": 16,
        "context_per_class": int(cells.context_per_class.iloc[0]),
        "context_modes": modes,
        "correction_scope": args.correction_scope,
        "positive_context_contrasts": 12,
        "relation_identity_contrasts": 12,
        "claim_boundary": (
            "prospective fixed-negative sensitivity; not historical 60-test reproduction"
        ),
    }
    # Validate everything before creating outputs; never overwrite historical artifacts.
    names = (
        "coverage_by_dataset_seed.csv",
        "coverage_summary.csv",
        "dataset_contrasts.csv",
        "contrast_summary.csv",
        "audit.json",
    )
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("output already contains analysis results; select a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output / names[0], index=False)
    coverage_summary.to_csv(args.output / names[1])
    effects.to_csv(args.output / names[2], index=False)
    summary.to_csv(args.output / names[3], index=False)
    (args.output / names[4]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
