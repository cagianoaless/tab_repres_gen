from __future__ import annotations

import itertools
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from relational_tfm.config import StudyConfig
from relational_tfm.hashing import object_sha256

PLAN_SCHEMA_VERSION = 1


def expand_cells(config: StudyConfig) -> list[dict[str, Any]]:
    if config.schema_version == 8:
        return _expand_graphlog_cells(config)
    if config.schema_version == 7:
        return _expand_compositional_shift_cells(config)
    if config.schema_version == 6:
        return _expand_clutrr_cells(config)
    if config.schema_version == 5:
        return _expand_rule_induction_cells(config)
    if config.schema_version == 4:
        return _expand_frozen_kg_cells(config)
    if config.schema_version == 3:
        return _expand_real_cells(config)
    if config.schema_version == 2:
        return _expand_property_cells(config)
    axes = itertools.product(
        config.properties,
        config.encoders,
        config.decoders,
        config.contexts_per_class,
        config.graph_sizes,
        config.noise_regimes,
        config.seeds,
    )
    cells: list[dict[str, Any]] = []
    for index, values in enumerate(axes):
        property_name, encoder, decoder, context, graph_size, noise, seed = values
        scientific = {
            "study_id": config.study_id,
            "task_family": config.task_family,
            "property": property_name,
            "encoder": encoder,
            "decoder": decoder,
            "context_per_class": context,
            "graph_size": graph_size,
            "noise_regime": noise,
            "seed": seed,
        }
        cells.append(
            {
                "index": index,
                "cell_id": object_sha256(scientific)[:20],
                **scientific,
            }
        )
    return cells


def _expand_graphlog_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for world in config.properties:
        axes = itertools.product(
            config.encoders,
            config.decoders,
            config.contexts_per_class,
            config.seeds,
        )
        for encoder, decoder, context, seed in axes:
            scientific = {
                "study_id": config.study_id,
                "task_family": config.task_family,
                "dataset": config.raw["dataset"],
                "asset": config.raw["asset"],
                "world": world,
                "context_inner_split": config.raw["context_inner_split"],
                "evaluation_inner_split": config.raw["evaluation_inner_split"],
                "train_chain_lengths": list(config.raw["train_chain_lengths"]),
                "evaluation_chain_lengths": list(
                    config.raw["evaluation_chain_lengths"]
                ),
                "max_path_length": int(config.raw["max_path_length"]),
                "max_context_per_relation": int(
                    config.raw["max_context_per_relation"]
                ),
                "encoder": encoder,
                "decoder": decoder,
                "context_per_relation": context,
                "seed": seed,
                **_seed_streams(seed),
            }
            if config.model_identity is not None:
                scientific["model_identity"] = config.model_identity
            cells.append(
                {
                    "index": len(cells),
                    "cell_id": object_sha256(scientific)[:20],
                    **scientific,
                }
            )
    return cells


def _expand_compositional_shift_cells(
    config: StudyConfig,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    axes = itertools.product(
        config.encoders,
        config.decoders,
        config.contexts_per_class,
        config.seeds,
    )
    for encoder, decoder, context, seed in axes:
        scientific = {
            "study_id": config.study_id,
            "task_family": config.task_family,
            "algebra": config.raw["algebra"],
            "train_chain_lengths": list(config.raw["train_chain_lengths"]),
            "evaluation_chain_lengths": list(
                config.raw["evaluation_chain_lengths"]
            ),
            "max_path_length": int(config.raw["max_path_length"]),
            "evaluation_per_class_per_length": int(
                config.raw["evaluation_per_class_per_length"]
            ),
            "max_context_per_class": int(config.raw["max_context_per_class"]),
            "encoder": encoder,
            "decoder": decoder,
            "context_per_class": context,
            "seed": seed,
            **_seed_streams(seed),
        }
        if config.model_identity is not None:
            scientific["model_identity"] = config.model_identity
        cells.append(
            {
                "index": len(cells),
                "cell_id": object_sha256(scientific)[:20],
                **scientific,
            }
        )
    return cells


def _expand_clutrr_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    axes = itertools.product(
        config.encoders,
        config.decoders,
        config.contexts_per_class,
        config.seeds,
    )
    for encoder, decoder, context, seed in axes:
        context_sampling = config.raw.get("context_sampling")
        scientific = {
            "study_id": config.study_id,
            "task_family": config.task_family,
            "dataset": "CLUTRR/v1",
            "dataset_config": config.raw["dataset_config"],
            "asset_files": config.raw["asset_files"],
            "train_chain_lengths": list(config.raw["train_chain_lengths"]),
            "evaluation_chain_lengths": list(config.raw["evaluation_chain_lengths"]),
            "max_path_length": int(config.raw["max_path_length"]),
            "encoder": encoder,
            "decoder": decoder,
            "context_per_relation": context,
            "max_context_per_relation": int(config.raw["max_context_per_relation"]),
            "seed": seed,
            **_seed_streams(seed),
        }
        if context_sampling is not None:
            scientific.update(
                {
                    "context_per_relation": context
                    + int(context_sampling["added_per_relation"]),
                    "base_context_per_relation": context,
                    "max_base_context_per_relation": max(
                        config.contexts_per_class
                    ),
                }
            )
            scientific["context_sampling"] = context_sampling
        if config.model_identity is not None:
            scientific["model_identity"] = config.model_identity
        cells.append(
            {
                "index": len(cells),
                "cell_id": object_sha256(scientific)[:20],
                **scientific,
            }
        )
    return cells


def _expand_frozen_kg_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in config.frozen_kg_dataset_specs:
        axes = itertools.product(
            spec.interfaces,
            config.decoders,
            config.raw.get("context_modes", ["original_uniform"]),
            config.contexts_per_class,
            config.seeds,
        )
        for interface, decoder, context_mode, context, seed in axes:
            scientific = {
                "study_id": config.study_id,
                "task_family": config.task_family,
                "dataset": spec.dataset_name,
                "asset_id": config.raw["asset_id"],
                "asset_manifest_sha256": config.raw["asset_manifest_sha256"],
                "archive_sha256": spec.archive_sha256,
                "transfer_regime": spec.transfer_regime,
                "interface": interface,
                "decoder": decoder,
                "context_per_class": context,
                "query_triples": int(config.raw["query_triples"]),
                "seed": seed,
                **_seed_streams(seed),
            }
            if "context_modes" in config.raw:
                scientific["context_mode"] = context_mode
            if config.model_identity is not None:
                scientific["model_identity"] = config.model_identity
            cells.append(
                {
                    "index": len(cells),
                    "cell_id": object_sha256(scientific)[:20],
                    **scientific,
                }
            )
    return cells


def _expand_real_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in config.real_dataset_specs:
        axes = itertools.product(
            spec.encoders,
            config.decoders,
            config.contexts_per_class,
            config.context_negative_ratios,
            config.seeds,
        )
        for encoder, decoder, context, ratio, seed in axes:
            scientific = {
                "study_id": config.study_id,
                "task_family": config.task_family,
                "dataset": spec.dataset_name,
                "asset_id": spec.asset_id,
                "macro_graph_family": spec.macro_graph_family,
                "aligned_encoder": spec.aligned_encoder,
                "encoder": encoder,
                "decoder": decoder,
                "context_per_class": context,
                "context_negative_ratio": ratio,
                "max_context_per_class": config.max_context_per_class,
                "max_context_positive": max(config.contexts_per_class),
                "evaluation_per_class": config.evaluation_per_class,
                "evaluation_negative_ratio": config.evaluation_negative_ratio,
                "seed": seed,
                **_seed_streams(seed),
            }
            if config.model_identity is not None:
                scientific["model_identity"] = config.model_identity
            cells.append(
                {
                    "index": len(cells),
                    "cell_id": object_sha256(scientific)[:20],
                    **scientific,
                }
            )
    return cells


def _expand_rule_induction_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in config.rule_specs:
        axes = itertools.product(
            spec.encoders,
            config.decoders,
            config.graph_sizes,
            config.contexts_per_class,
            config.seeds,
        )
        for encoder, decoder, graph_size, context, seed in axes:
            scientific = {
                "study_id": config.study_id,
                "task_family": config.task_family,
                "rule": spec.rule_name,
                "encoder": encoder,
                "decoder": decoder,
                "graph_size": graph_size,
                "context_per_class": context,
                "max_context_per_class": config.max_context_per_class,
                "evaluation_per_class": config.evaluation_per_class,
                "seed": seed,
                **_seed_streams(seed),
            }
            if config.model_identity is not None:
                scientific["model_identity"] = config.model_identity
            cells.append(
                {
                    "index": len(cells),
                    "cell_id": object_sha256(scientific)[:20],
                    **scientific,
                }
            )
    return cells


def _seed_streams(seed: int) -> dict[str, int]:
    return {
        "graph_seed": seed,
        "relabel_seed": seed + 100_000,
        "split_seed": seed + 200_000,
        "noise_seed": seed + 300_000,
        "encoder_seed": seed + 400_000,
        "decoder_seed": seed + 500_000,
    }


def _expand_property_cells(config: StudyConfig) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in config.property_specs:
        axes = itertools.product(
            spec.encoders,
            config.decoders,
            config.contexts_per_class,
            config.context_negative_ratios,
            spec.graph_sizes,
            spec.noise_regimes,
            config.seeds,
        )
        for encoder, decoder, context, negative_ratio, graph_size, noise, seed in axes:
            scientific = {
                "study_id": config.study_id,
                "task_family": config.task_family,
                "property": spec.property_name,
                "encoder": encoder,
                "decoder": decoder,
                "context_per_class": context,
                "max_context_per_class": config.max_context_per_class,
                "evaluation_per_class": config.evaluation_per_class,
                "graph_size": graph_size,
                "noise_regime": noise,
                "seed": seed,
                **_seed_streams(seed),
                "ceiling_seeds": list(config.ceiling_seeds),
            }
            if "context_negative_ratios" in config.raw:
                scientific["context_negative_ratio"] = negative_ratio
            if "evaluation_negative_ratio" in config.raw:
                scientific["evaluation_negative_ratio"] = config.evaluation_negative_ratio
            if spec.macro_graph_family is not None:
                scientific["macro_graph_family"] = spec.macro_graph_family
            if config.model_identity is not None:
                scientific["model_identity"] = config.model_identity
            cells.append(
                {
                    "index": len(cells),
                    "cell_id": object_sha256(scientific)[:20],
                    **scientific,
                }
            )
    return cells


def build_plan(config: StudyConfig) -> dict[str, Any]:
    cells = expand_cells(config)
    scientific_body = {
        "schema_version": config.schema_version,
        "study_id": config.study_id,
        "status": config.status,
        "task_family": config.task_family,
        "config": config.raw,
        "cells": cells,
    }
    return {
        **scientific_body,
        "plan_sha256": object_sha256(scientific_body),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def verify_plan(plan: dict[str, Any]) -> None:
    expected = plan.get("plan_sha256")
    if not isinstance(expected, str):
        raise ValueError("plan_sha256 is missing")
    scientific_body = {
        key: value
        for key, value in plan.items()
        if key not in {"plan_sha256", "created_at_utc"}
    }
    actual = object_sha256(scientific_body)
    if actual != expected:
        raise ValueError(f"plan hash mismatch: expected {expected}, computed {actual}")

    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("plan must contain at least one cell")
    cell_ids = [cell.get("cell_id") for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("plan contains duplicate cell IDs")
    if [cell.get("index") for cell in cells] != list(range(len(cells))):
        raise ValueError("cell indices must be contiguous and zero-based")
    config = StudyConfig.from_mapping(plan.get("config"))
    if plan.get("schema_version") != config.schema_version:
        raise ValueError("plan/config schema versions differ")
    expected_cells = expand_cells(config)
    if cells != expected_cells:
        raise ValueError("plan cells do not match deterministic config expansion")
    if plan.get("study_id") != config.study_id:
        raise ValueError("plan/config study IDs differ")
    if plan.get("status") != config.status or plan.get("task_family") != config.task_family:
        raise ValueError("plan/config scientific identity differs")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    verify_plan(plan)
    return plan
