from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.datasets.clutrr import CLUTRR_RELATIONS, load_clutrr_task
from relational_tfm.datasets.clutrr_audit import audit_clutrr_task
from relational_tfm.datasets.compositional_shift import (
    COMPOSITION_CLASS_NAMES,
    generate_compositional_shift_task,
)
from relational_tfm.datasets.compositional_shift_audit import (
    audit_compositional_shift_task,
)
from relational_tfm.datasets.graphlog import load_graphlog_task
from relational_tfm.datasets.graphlog_audit import audit_graphlog_task
from relational_tfm.datasets.real import generate_real_task
from relational_tfm.datasets.rfmlink import load_frozen_kg_task
from relational_tfm.datasets.rule_induction import generate_rule_induction_task
from relational_tfm.datasets.rule_induction_audit import audit_rule_induction_task
from relational_tfm.datasets.synthetic import generate_synthetic_task
from relational_tfm.decoders.classification import (
    build_classifier,
    class_probabilities,
    decoder_provenance,
    positive_decision_orientation,
    positive_probability,
    positive_ranking_score,
)
from relational_tfm.encoders.clutrr import encode_clutrr_rows
from relational_tfm.encoders.compositional_shift import (
    encode_compositional_shift_rows,
)
from relational_tfm.encoders.graphlog import (
    GRAPHLOG_RELATIONS,
    encode_graphlog_rows,
)
from relational_tfm.encoders.real import encode_real_rows
from relational_tfm.encoders.rule_induction import encode_rule_induction_rows
from relational_tfm.encoders.synthetic import encode_synthetic_rows
from relational_tfm.evaluation.classification import (
    classification_metrics,
    collision_conditioned_ceiling,
    extraction_efficiency,
    multiclass_classification_metrics,
)
from relational_tfm.evaluation.ranking import filtered_ranking_metrics
from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.plan import atomic_write_json, read_plan
from relational_tfm.provenance import peak_resources, reset_peak_resources, runtime_state

GPU_DECODERS = {"tabpfn", "limix2m", "tabm", "tabicl"}
GPU_80GB_THRESHOLD_BYTES = 70_000_000_000
GPU_WORKERS_40GB = 4
GPU_WORKERS_80GB = 8


# Task generation and encoding depend only on (property/dataset, encoder, graph
# size, context/evaluation sizes, noise regime, and seeds) -- never on the
# decoder. Frozen property-matrix schedules commonly reuse the same generation
# parameters across multiple decoder-specific waves (measured on one such
# schedule: exactly 4x redundancy, i.e. every unique generation configuration
# recomputed once per decoder-specific wave that shares it), so this step is
# worth sharing across those waves. Caching is strictly opt-in
# (RTFM_ENCODING_CACHE_DIR unset means identical behavior to before this
# existed) and produces byte-identical results to an uncached run: the cache
# key is exactly the deterministic input to task generation, computed the same
# way for every wave that shares a configuration.
def _encoding_cache_dir() -> Path | None:
    raw = os.environ.get("RTFM_ENCODING_CACHE_DIR", "").strip()
    return Path(raw) if raw else None


def _synthetic_encoding_cache_key(cell: dict[str, Any]) -> str:
    fields = {
        "task_family": "synthetic_relational",
        "property": cell["property"],
        "encoder": cell["encoder"],
        "graph_size": int(cell["graph_size"]),
        "context_per_class": int(cell["context_per_class"]),
        "max_context_per_class": cell.get("max_context_per_class"),
        "evaluation_per_class": cell.get("evaluation_per_class"),
        "noise_regime": cell["noise_regime"],
        "seed": int(cell["seed"]),
        "graph_seed": cell.get("graph_seed"),
        "relabel_seed": cell.get("relabel_seed"),
        "split_seed": cell.get("split_seed"),
        "noise_seed": cell.get("noise_seed"),
        "encoder_seed": int(cell.get("encoder_seed", cell["seed"])),
        "context_negative_ratio": int(cell.get("context_negative_ratio", 1)),
        "evaluation_negative_ratio": int(cell.get("evaluation_negative_ratio", 1)),
    }
    return object_sha256(fields)


def _real_encoding_cache_key(cell: dict[str, Any]) -> str:
    fields = {
        "task_family": "real_graph",
        "dataset": cell["dataset"],
        "encoder": cell["encoder"],
        "context_per_class": int(cell["context_per_class"]),
        "context_negative_ratio": int(cell["context_negative_ratio"]),
        "max_context_positive": int(cell["max_context_positive"]),
        "max_context_per_class": int(cell["max_context_per_class"]),
        "evaluation_per_class": int(cell["evaluation_per_class"]),
        "evaluation_negative_ratio": int(cell["evaluation_negative_ratio"]),
        "split_seed": int(cell["split_seed"]),
        "encoder_seed": int(cell["encoder_seed"]),
        "aligned_encoder": cell["aligned_encoder"],
    }
    return object_sha256(fields)


# Certifying that a representation carries no information about the labels is a claim
# about the representation, not about one sampled context: a small context can be
# constant while the population it was drawn from is not (a 3-row sample of a varying
# column is constant with high probability). Deciding this from the context alone would
# confuse sparse context coverage with representational insufficiency, and would do so
# precisely along the context-scaling axis this study measures. Every matrix derived
# from the same encoding is therefore checked together, so a true result means the
# encoded population is constant and the collision-conditioned ceiling equals the
# majority prior by construction.
#
# This is a legitimate outcome for the weaker rungs of an encoder ladder:
# `dyadic_invariants` on `typed_composition_l2` has all three of its features
# identically zero over the whole population, because candidate pairs are distinct
# entities and a composed relation is never a direct support edge.
def _representation_is_constant(*matrices: np.ndarray) -> bool:
    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    arrays = [array for array in arrays if array.size and array.shape[0]]
    if not arrays:
        return True
    return bool(np.all(np.ptp(np.vstack(arrays), axis=0) == 0.0))


def _context_prior_probabilities(context_labels: np.ndarray, evaluation_size: int) -> np.ndarray:
    prior = float(np.asarray(context_labels, dtype=np.float64).mean())
    return np.full(int(evaluation_size), prior, dtype=np.float64)


# Only a decoder rejecting constant input may be absorbed into a recorded result. An
# out-of-memory condition, a CUDA fault or a library bug must still fail the cell rather
# than be published as a complete measurement at the prior, so anything whose message is
# not a recognised constant-input rejection is re-raised. An unrecognised message
# therefore fails loudly instead of silently fabricating a result.
_CONSTANT_INPUT_REJECTIONS = (
    "all features are either constant or ignored",  # CatBoost
    "all features are constant",  # TabPFN
)


def _is_constant_input_rejection(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _CONSTANT_INPUT_REJECTIONS)


def _fit_on_constant_representation(
    model: Any, values: np.ndarray, labels: np.ndarray, decoder: str
) -> str:
    """Attempt the fit so each decoder still performs its own frozen-identity checks.

    CatBoost and TabPFN reject constant input from `fit`; LimiX-2M fits (verifying its
    frozen checkpoint hash, frozen inference config and single-device requirement in the
    process) and would only fail later in `predict`, which is never reached because the
    prediction is replaced. Logistic regression completes normally. TabICL cannot build
    its feature-shuffle ensemble for a constant context, so its fit is skipped after the
    checkpoint and frozen configuration have already been validated during construction.
    """
    if decoder == "tabicl":
        return "skipped_constant_input:context_prior_fallback"
    try:
        model.fit(values, labels)
    except Exception as error:
        if not _is_constant_input_rejection(error):
            raise
        return f"rejected_constant_input:{type(error).__name__}"
    _synchronize_gpu_decoder(decoder)
    return "succeeded"


def _cache_load(cache_dir: Path, key: str) -> Any | None:
    path = cache_dir / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 -- our own cache, not untrusted input
    except Exception:  # noqa: BLE001 -- a corrupt/partial cache entry must not fail the cell
        return None


def _cache_store(cache_dir: Path, key: str, value: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".pkl", dir=cache_dir)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(cache_dir / f"{key}.pkl")
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
CellRunner = Callable[[dict[str, Any], Path, Path], dict[str, Any]]


def _atomic_write_predictions(
    path: Path, labels: np.ndarray, probabilities: np.ndarray, query_ids: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                labels=np.asarray(labels, dtype=np.int8),
                probabilities=np.asarray(probabilities, dtype=np.float64),
                query_ids=np.asarray(query_ids, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_ranking_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _generate_task(cell: dict[str, Any]):
    return generate_synthetic_task(
        property_name=cell["property"],
        graph_size=int(cell["graph_size"]),
        context_per_class=int(cell["context_per_class"]),
        noise_regime=cell["noise_regime"],
        seed=int(cell["seed"]),
        max_context_per_class=cell.get("max_context_per_class"),
        evaluation_per_class=cell.get("evaluation_per_class"),
        graph_seed=cell.get("graph_seed"),
        relabel_seed=cell.get("relabel_seed"),
        split_seed=cell.get("split_seed"),
        noise_seed=cell.get("noise_seed"),
        context_negative_ratio=int(cell.get("context_negative_ratio", 1)),
        evaluation_negative_ratio=int(cell.get("evaluation_negative_ratio", 1)),
    )


def _encode(task: Any, rows: Any, cell: dict[str, Any]):
    return encode_synthetic_rows(
        task.support_graph,
        rows,
        cell["encoder"],
        int(cell.get("encoder_seed", cell["seed"])),
        property_name=cell["property"],
        control_population=task.population,
    )


def _select_encoded_rows(
    values: np.ndarray,
    population_query_ids: np.ndarray,
    selected_query_ids: np.ndarray,
) -> np.ndarray:
    positions = {
        int(query_id): index
        for index, query_id in enumerate(population_query_ids)
    }
    try:
        selected_positions = [positions[int(query_id)] for query_id in selected_query_ids]
    except KeyError as error:
        raise ValueError("selected query is absent from the encoded population") from error
    return values[np.asarray(selected_positions, dtype=np.int64)]


@lru_cache(maxsize=256)
def _cross_graph_ceiling(
    property_name: str,
    graph_size: int,
    encoder: str,
    noise_regime: str,
    max_context_per_class: int,
    evaluation_per_class: int,
    evaluation_negative_ratio: int,
    ceiling_seeds: tuple[int, ...],
) -> dict[str, float | int | list[int] | str]:
    feature_blocks = []
    label_blocks = []
    feature_names: tuple[str, ...] | None = None
    for seed in ceiling_seeds:
        task = generate_synthetic_task(
            property_name=property_name,
            graph_size=graph_size,
            context_per_class=max_context_per_class,
            noise_regime=noise_regime,
            seed=seed,
            max_context_per_class=max_context_per_class,
            evaluation_per_class=evaluation_per_class,
            graph_seed=seed,
            relabel_seed=seed + 100_000,
            split_seed=seed + 200_000,
            noise_seed=seed + 300_000,
            evaluation_negative_ratio=evaluation_negative_ratio,
        )
        encoded = encode_synthetic_rows(
            task.support_graph,
            task.population,
            encoder,
            seed + 400_000,
            property_name=property_name,
            control_population=task.population,
        )
        if feature_names is not None and encoded.feature_names != feature_names:
            raise ValueError("cross-graph ceiling feature schemas differ")
        feature_names = encoded.feature_names
        feature_blocks.append(encoded.values)
        label_blocks.append(task.population["label"].to_numpy(dtype=np.int64))
    summary = collision_conditioned_ceiling(
        np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks)
    )
    return {
        **summary,
        "scope": "exact finite-population ceiling across independently relabelled graphs",
        "graph_count": len(ceiling_seeds),
        "ceiling_seeds": list(ceiling_seeds),
    }


def _coverage_metadata(task: Any) -> dict[str, Any]:
    context_entities = set(task.context["head"]) | set(task.context["tail"])
    evaluation_entities = set(task.evaluation["head"]) | set(task.evaluation["tail"])
    population_entities = set(task.population["head"]) | set(task.population["tail"])
    context_relations = sorted(str(value) for value in task.context["relation"].unique())
    evaluation_relations = sorted(str(value) for value in task.evaluation["relation"].unique())
    context_label_counts = {
        str(int(label)): int(count)
        for label, count in task.context["label"].value_counts().sort_index().items()
    }
    evaluation_label_counts = {
        str(int(label)): int(count)
        for label, count in task.evaluation["label"].value_counts().sort_index().items()
    }
    return {
        "context_entities": len(context_entities),
        "evaluation_entities": len(evaluation_entities),
        "population_entities": len(population_entities),
        "evaluation_entities_seen_in_context": len(context_entities & evaluation_entities),
        "evaluation_entity_coverage": float(
            len(context_entities & evaluation_entities) / len(evaluation_entities)
        ),
        "context_relations": context_relations,
        "evaluation_relations": evaluation_relations,
        "context_label_counts": context_label_counts,
        "evaluation_label_counts": evaluation_label_counts,
        "context_has_both_labels": set(context_label_counts) == {"0", "1"},
        "evaluation_has_both_labels": set(evaluation_label_counts) == {"0", "1"},
        "context_relation_coverage": float(
            len(set(context_relations) & set(evaluation_relations)) / len(evaluation_relations)
        ),
        "context_query_sha256": task.metadata["context_query_sha256"],
        "evaluation_query_sha256": task.metadata["evaluation_query_sha256"],
        "population_query_sha256": task.metadata["population_query_sha256"],
    }


def _validate_decoder_identity(cell: dict[str, Any], provenance: dict[str, Any]) -> None:
    expected = cell.get("model_identity")
    if expected is None:
        return
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"decoder identity mismatch for {key}")
    expected_hash = expected.get("checkpoint_sha256")
    if expected_hash is None:
        return
    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("decoder checkpoint provenance is missing")
    if any(checkpoint.get("sha256") != expected_hash for checkpoint in checkpoints):
        raise ValueError("decoder checkpoint hash does not match the frozen identity")


def _synchronize_gpu_decoder(decoder: str) -> None:
    if decoder not in GPU_DECODERS:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_synthetic_cell(cell: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = _encoding_cache_dir()
    cache_key = _synthetic_encoding_cache_key(cell) if cache_dir else None
    cached = _cache_load(cache_dir, cache_key) if cache_dir else None
    if cached is not None:
        task, population_encoded = cached
    else:
        task = _generate_task(cell)
        population_encoded = _encode(task, task.population, cell)
        if cache_dir:
            _cache_store(cache_dir, cache_key, (task, population_encoded))
    population_query_ids = task.population["query_id"].to_numpy(dtype=np.int64)
    context_values = _select_encoded_rows(
        population_encoded.values,
        population_query_ids,
        task.context["query_id"].to_numpy(dtype=np.int64),
    )
    evaluation_values = _select_encoded_rows(
        population_encoded.values,
        population_query_ids,
        task.evaluation["query_id"].to_numpy(dtype=np.int64),
    )

    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    model = build_classifier(cell["decoder"], int(cell.get("decoder_seed", cell["seed"])))
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started

    representation_constant = _representation_is_constant(population_encoded.values)
    decoder_fit_status = "not_attempted"
    decoder_prediction_status = "not_attempted"
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _context_prior_probabilities(context_labels, len(evaluation_values))
        prediction_seconds = time.perf_counter() - prediction_started
    else:
        fit_started = time.perf_counter()
        model.fit(context_values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started

        prediction_started = time.perf_counter()
        probabilities = positive_probability(model, evaluation_values)
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started

    metrics = classification_metrics(evaluation_labels, probabilities)
    ceiling_summary = collision_conditioned_ceiling(
        evaluation_values,
        evaluation_labels,
    )
    ceiling = float(ceiling_summary["ceiling"])
    metrics["within_graph_empirical_collision_ceiling"] = ceiling
    metrics["within_graph_empirical_roc_auc_ceiling"] = float(
        ceiling_summary["roc_auc_ceiling"]
    )
    ceiling_seeds = tuple(int(seed) for seed in cell.get("ceiling_seeds", []))
    cross_graph_ceiling = None
    if ceiling_seeds:
        cross_graph_ceiling = _cross_graph_ceiling(
            cell["property"],
            int(cell["graph_size"]),
            cell["encoder"],
            cell["noise_regime"],
            int(cell["max_context_per_class"]),
            int(cell["evaluation_per_class"]),
            int(cell.get("evaluation_negative_ratio", 1)),
            ceiling_seeds,
        )
        efficiency = extraction_efficiency(
            metrics["accuracy"],
            float(cross_graph_ceiling["ceiling"]),
            float(cross_graph_ceiling["majority_prior"]),
        )
        metrics["cross_graph_collision_ceiling"] = float(cross_graph_ceiling["ceiling"])
        metrics["cross_graph_roc_auc_ceiling"] = float(
            cross_graph_ceiling["roc_auc_ceiling"]
        )
        metrics["extraction_efficiency"] = efficiency

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    query_ids = task.evaluation["query_id"].to_numpy(dtype=np.int64)
    _atomic_write_predictions(prediction_path, evaluation_labels, probabilities, query_ids)

    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 2,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "task_generation_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(population_encoded.feature_names),
        "feature_count": len(population_encoded.feature_names),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "context_positive_rows": int(context_labels.sum()),
        "evaluation_positive_rows": int(evaluation_labels.sum()),
        "task_metadata": task.metadata,
        "coverage": _coverage_metadata(task),
        "encoder_provenance": population_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "cell_peak_resources": peak_resources(),
        "ceiling_scope": (
            "diagnostic only; not a cross-graph representation bound because entity identities "
            "can be unique within one graph"
        ),
        "cross_graph_representation_ceiling": cross_graph_ceiling,
        "metrics": metrics,
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
        },
    }
    metric_path = output_dir / f"{cell['cell_id']}.metrics.json"
    atomic_write_json(metric_path, payload)
    return payload


def run_real_cell(cell: dict[str, Any], output_dir: Path, repo_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = _encoding_cache_dir()
    cache_key = _real_encoding_cache_key(cell) if cache_dir else None
    cached = _cache_load(cache_dir, cache_key) if cache_dir else None
    if cached is not None:
        task, context_encoded, evaluation_encoded = cached
    else:
        task = generate_real_task(
            repo_root,
            cell["dataset"],
            int(cell["context_per_class"]),
            int(cell["context_negative_ratio"]),
            int(cell["max_context_positive"]),
            int(cell["max_context_per_class"]),
            int(cell["evaluation_per_class"]),
            int(cell["evaluation_negative_ratio"]),
            int(cell["split_seed"]),
        )
        context_encoded = encode_real_rows(
            task.support_graph,
            task.context,
            cell["encoder"],
            int(cell["encoder_seed"]),
            aligned_encoder=cell["aligned_encoder"],
            control_population=task.population,
        )
        evaluation_encoded = encode_real_rows(
            task.support_graph,
            task.evaluation,
            cell["encoder"],
            int(cell["encoder_seed"]),
            aligned_encoder=cell["aligned_encoder"],
            control_population=task.population,
        )
        if cache_dir:
            _cache_store(cache_dir, cache_key, (task, context_encoded, evaluation_encoded))
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started
    model_started = time.perf_counter()
    model = build_classifier(cell["decoder"], int(cell["decoder_seed"]))
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    representation_constant = _representation_is_constant(
        context_encoded.values, evaluation_encoded.values
    )
    decoder_fit_status = "not_attempted"
    decoder_prediction_status = "not_attempted"
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_encoded.values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _context_prior_probabilities(
            context_labels, len(evaluation_encoded.values)
        )
        prediction_seconds = time.perf_counter() - prediction_started
    else:
        fit_started = time.perf_counter()
        model.fit(context_encoded.values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = positive_probability(model, evaluation_encoded.values)
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started
    metrics = classification_metrics(evaluation_labels, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_predictions(
        prediction_path,
        evaluation_labels,
        probabilities,
        task.evaluation["query_id"].to_numpy(dtype=np.int64),
    )
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 2,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "task_generation_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(context_encoded.feature_names),
        "feature_count": len(context_encoded.feature_names),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "context_positive_rows": int(context_labels.sum()),
        "evaluation_positive_rows": int(evaluation_labels.sum()),
        "task_metadata": task.metadata,
        "coverage": _coverage_metadata(task),
        "encoder_provenance": context_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "cell_peak_resources": peak_resources(),
        "metrics": metrics,
        "predictions": {"path": prediction_path.name, "sha256": file_sha256(prediction_path)},
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def run_rule_induction_cell(
    cell: dict[str, Any], output_dir: Path, _repo_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    task = generate_rule_induction_task(
        cell["rule"],
        int(cell["graph_size"]),
        int(cell["context_per_class"]),
        max_context_per_class=int(cell["max_context_per_class"]),
        evaluation_per_class=int(cell["evaluation_per_class"]),
        graph_seed=int(cell["graph_seed"]),
        split_seed=int(cell["split_seed"]),
        relabel_seed=int(cell["relabel_seed"]),
    )
    audit = audit_rule_induction_task(task)
    context_encoded = encode_rule_induction_rows(
        task.support_graph, task.context, cell["encoder"], int(cell["encoder_seed"])
    )
    evaluation_encoded = encode_rule_induction_rows(
        task.support_graph, task.evaluation, cell["encoder"], int(cell["encoder_seed"])
    )
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started
    model_started = time.perf_counter()
    model = build_classifier(cell["decoder"], int(cell["decoder_seed"]))
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    representation_constant = _representation_is_constant(
        context_encoded.values, evaluation_encoded.values
    )
    decoder_fit_status = "not_attempted"
    decoder_prediction_status = "not_attempted"
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_encoded.values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _context_prior_probabilities(
            context_labels, len(evaluation_encoded.values)
        )
        prediction_seconds = time.perf_counter() - prediction_started
    else:
        fit_started = time.perf_counter()
        model.fit(context_encoded.values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = positive_probability(model, evaluation_encoded.values)
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started
    metrics = classification_metrics(evaluation_labels, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_predictions(
        prediction_path,
        evaluation_labels,
        probabilities,
        task.evaluation["query_id"].to_numpy(dtype=np.int64),
    )
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 2,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "task_generation_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(context_encoded.feature_names),
        "feature_count": len(context_encoded.feature_names),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "context_positive_rows": int(context_labels.sum()),
        "evaluation_positive_rows": int(evaluation_labels.sum()),
        "task_metadata": task.metadata,
        "audit": audit,
        "encoder_provenance": context_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "cell_peak_resources": peak_resources(),
        "metrics": metrics,
        "predictions": {"path": prediction_path.name, "sha256": file_sha256(prediction_path)},
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def run_clutrr_cell(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    task = load_clutrr_task(repo_root, cell)
    audit = audit_clutrr_task(task)
    context_encoded = encode_clutrr_rows(
        task.context, cell["encoder"], int(cell["encoder_seed"])
    )
    evaluation_encoded = encode_clutrr_rows(
        task.evaluation, cell["encoder"], int(cell["encoder_seed"]) + 1
    )
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    model = build_classifier(
        cell["decoder"],
        int(cell["decoder_seed"]),
        multiclass=True,
    )
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    representation_constant = _representation_is_constant(
        context_encoded.values, evaluation_encoded.values
    )
    decoder_fit_status = "not_attempted"
    decoder_prediction_status = "not_attempted"
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_encoded.values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        classes = np.arange(len(CLUTRR_RELATIONS), dtype=np.int64)
        prior = np.bincount(
            context_labels, minlength=len(CLUTRR_RELATIONS)
        ).astype(np.float64)
        prior /= prior.sum()
        probabilities = np.tile(prior, (len(evaluation_labels), 1))
        context_probabilities = np.tile(prior, (len(context_labels), 1))
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_prediction_status = "context_prior_fallback"
    else:
        fit_started = time.perf_counter()
        model.fit(context_encoded.values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities, classes = class_probabilities(
            model, evaluation_encoded.values
        )
        context_probabilities, context_classes = class_probabilities(
            model, context_encoded.values
        )
        if not np.array_equal(classes, context_classes):
            raise ValueError("context and evaluation class order changed")
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"

    metrics = multiclass_classification_metrics(
        evaluation_labels, probabilities, classes
    )
    context_fit_metrics = multiclass_classification_metrics(
        context_labels, context_probabilities, classes
    )
    ceiling = collision_conditioned_ceiling(
        evaluation_encoded.values, evaluation_labels
    )
    metrics["collision_conditioned_ceiling"] = float(ceiling["ceiling"])
    metrics["extraction_efficiency"] = extraction_efficiency(
        metrics["accuracy"],
        float(ceiling["ceiling"]),
        float(ceiling["majority_prior"]),
    )
    per_chain_length = {}
    ceiling_by_chain_length = {}
    chain_lengths = task.evaluation["chain_length"].to_numpy(dtype=np.int64)
    for chain_length in sorted(map(int, np.unique(chain_lengths))):
        selected = chain_lengths == chain_length
        key = str(chain_length)
        per_chain_length[key] = multiclass_classification_metrics(
            evaluation_labels[selected],
            probabilities[selected],
            classes,
        )
        selected_ceiling = collision_conditioned_ceiling(
            evaluation_encoded.values[selected], evaluation_labels[selected]
        )
        ceiling_by_chain_length[key] = selected_ceiling
        per_chain_length[key]["rows"] = int(selected.sum())
        per_chain_length[key]["collision_conditioned_ceiling"] = float(
            selected_ceiling["ceiling"]
        )
        per_chain_length[key]["extraction_efficiency"] = extraction_efficiency(
            per_chain_length[key]["accuracy"],
            float(selected_ceiling["ceiling"]),
            float(selected_ceiling["majority_prior"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_ranking_predictions(
        prediction_path,
        {
            "labels": evaluation_labels.astype(np.int8),
            "probabilities": probabilities.astype(np.float64),
            "classes": classes.astype(np.int8),
            "query_ids": task.evaluation["query_id"].to_numpy(dtype=np.int64),
            "chain_lengths": chain_lengths.astype(np.int8),
        },
    )
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 1,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "task_loading_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(context_encoded.feature_names),
        "feature_count": len(context_encoded.feature_names),
        "class_names": list(CLUTRR_RELATIONS),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "task_metadata": task.metadata,
        "audit": audit,
        "encoder_provenance": context_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "collision_conditioned_ceiling": ceiling,
        "collision_conditioned_ceiling_by_chain_length": ceiling_by_chain_length,
        "metrics": metrics,
        "context_fit_metrics": context_fit_metrics,
        "metrics_by_chain_length": per_chain_length,
        "cell_peak_resources": peak_resources(),
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
            "schema": "multiclass_classification_v1",
        },
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def run_compositional_shift_cell(
    cell: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    task = generate_compositional_shift_task(cell)
    audit = audit_compositional_shift_task(task)
    context_encoded = encode_compositional_shift_rows(
        task.context, cell["encoder"], int(cell["encoder_seed"])
    )
    evaluation_encoded = encode_compositional_shift_rows(
        task.evaluation, cell["encoder"], int(cell["encoder_seed"]) + 1
    )
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    model = build_classifier(
        cell["decoder"],
        int(cell["decoder_seed"]),
        multiclass=True,
    )
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    representation_constant = _representation_is_constant(
        context_encoded.values, evaluation_encoded.values
    )
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_encoded.values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        classes = np.arange(len(COMPOSITION_CLASS_NAMES), dtype=np.int64)
        prior = np.bincount(
            context_labels, minlength=len(COMPOSITION_CLASS_NAMES)
        ).astype(np.float64)
        prior /= prior.sum()
        probabilities = np.tile(prior, (len(evaluation_labels), 1))
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_prediction_status = "context_prior_fallback"
    else:
        fit_started = time.perf_counter()
        model.fit(context_encoded.values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities, classes = class_probabilities(
            model, evaluation_encoded.values
        )
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"

    metrics = multiclass_classification_metrics(
        evaluation_labels, probabilities, classes
    )
    ceiling = collision_conditioned_ceiling(
        evaluation_encoded.values, evaluation_labels
    )
    metrics["collision_conditioned_ceiling"] = float(ceiling["ceiling"])
    metrics["extraction_efficiency"] = extraction_efficiency(
        metrics["accuracy"],
        float(ceiling["ceiling"]),
        float(ceiling["majority_prior"]),
    )
    per_chain_length: dict[str, dict[str, Any]] = {}
    ceiling_by_chain_length: dict[str, dict[str, Any]] = {}
    chain_lengths = task.evaluation["chain_length"].to_numpy(dtype=np.int64)
    for chain_length in sorted(map(int, np.unique(chain_lengths))):
        selected = chain_lengths == chain_length
        key = str(chain_length)
        per_chain_length[key] = multiclass_classification_metrics(
            evaluation_labels[selected],
            probabilities[selected],
            classes,
        )
        selected_ceiling = collision_conditioned_ceiling(
            evaluation_encoded.values[selected], evaluation_labels[selected]
        )
        ceiling_by_chain_length[key] = selected_ceiling
        per_chain_length[key]["rows"] = int(selected.sum())
        per_chain_length[key]["collision_conditioned_ceiling"] = float(
            selected_ceiling["ceiling"]
        )
        per_chain_length[key]["extraction_efficiency"] = extraction_efficiency(
            per_chain_length[key]["accuracy"],
            float(selected_ceiling["ceiling"]),
            float(selected_ceiling["majority_prior"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_ranking_predictions(
        prediction_path,
        {
            "labels": evaluation_labels.astype(np.int8),
            "probabilities": probabilities.astype(np.float64),
            "classes": classes.astype(np.int8),
            "query_ids": task.evaluation["query_id"].to_numpy(dtype=np.int64),
            "chain_lengths": chain_lengths.astype(np.int8),
        },
    )
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 1,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "task_generation_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(context_encoded.feature_names),
        "feature_count": len(context_encoded.feature_names),
        "class_names": list(COMPOSITION_CLASS_NAMES),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "task_metadata": task.metadata,
        "audit": audit,
        "encoder_provenance": context_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "collision_conditioned_ceiling": ceiling,
        "collision_conditioned_ceiling_by_chain_length": ceiling_by_chain_length,
        "metrics": metrics,
        "metrics_by_chain_length": per_chain_length,
        "cell_peak_resources": peak_resources(),
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
            "schema": "multiclass_classification_v1",
        },
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def run_graphlog_cell(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    task = load_graphlog_task(repo_root, cell)
    audit = audit_graphlog_task(task)
    context_encoded = encode_graphlog_rows(
        task.context, cell["encoder"], int(cell["encoder_seed"])
    )
    evaluation_encoded = encode_graphlog_rows(
        task.evaluation, cell["encoder"], int(cell["encoder_seed"]) + 1
    )
    context_labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)
    preparation_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    model = build_classifier(
        cell["decoder"],
        int(cell["decoder_seed"]),
        multiclass=True,
    )
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    representation_constant = _representation_is_constant(
        context_encoded.values, evaluation_encoded.values
    )
    if representation_constant:
        fit_started = time.perf_counter()
        decoder_fit_status = _fit_on_constant_representation(
            model, context_encoded.values, context_labels, cell["decoder"]
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        classes = np.unique(context_labels)
        prior = np.asarray(
            [np.sum(context_labels == label) for label in classes],
            dtype=np.float64,
        )
        prior /= prior.sum()
        probabilities = np.tile(prior, (len(evaluation_labels), 1))
        context_probabilities = np.tile(prior, (len(context_labels), 1))
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_prediction_status = "context_prior_fallback"
    else:
        fit_started = time.perf_counter()
        model.fit(context_encoded.values, context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities, classes = class_probabilities(
            model, evaluation_encoded.values
        )
        context_probabilities, context_classes = class_probabilities(
            model, context_encoded.values
        )
        if not np.array_equal(classes, context_classes):
            raise RuntimeError("context and evaluation class ordering changed")
        _synchronize_gpu_decoder(cell["decoder"])
        prediction_seconds = time.perf_counter() - prediction_started
        decoder_fit_status = "succeeded"
        decoder_prediction_status = "succeeded"

    metrics = multiclass_classification_metrics(
        evaluation_labels, probabilities, classes
    )
    context_fit_metrics = multiclass_classification_metrics(
        context_labels, context_probabilities, classes
    )
    ceiling = collision_conditioned_ceiling(
        evaluation_encoded.values, evaluation_labels
    )
    metrics["collision_conditioned_ceiling"] = float(ceiling["ceiling"])
    metrics["extraction_efficiency"] = extraction_efficiency(
        metrics["accuracy"],
        float(ceiling["ceiling"]),
        float(ceiling["majority_prior"]),
    )
    per_chain_length: dict[str, dict[str, Any]] = {}
    ceiling_by_chain_length: dict[str, dict[str, Any]] = {}
    chain_lengths = task.evaluation["chain_length"].to_numpy(dtype=np.int64)
    for chain_length in sorted(map(int, np.unique(chain_lengths))):
        selected = chain_lengths == chain_length
        key = str(chain_length)
        per_chain_length[key] = multiclass_classification_metrics(
            evaluation_labels[selected],
            probabilities[selected],
            classes,
        )
        selected_ceiling = collision_conditioned_ceiling(
            evaluation_encoded.values[selected], evaluation_labels[selected]
        )
        ceiling_by_chain_length[key] = selected_ceiling
        per_chain_length[key]["rows"] = int(selected.sum())
        per_chain_length[key]["collision_conditioned_ceiling"] = float(
            selected_ceiling["ceiling"]
        )
        per_chain_length[key]["extraction_efficiency"] = extraction_efficiency(
            per_chain_length[key]["accuracy"],
            float(selected_ceiling["ceiling"]),
            float(selected_ceiling["majority_prior"]),
        )
    metrics["equal_length_macro_balanced_accuracy"] = float(
        np.mean(
            [
                per_chain_length[str(length)]["balanced_accuracy"]
                for length in map(int, np.unique(chain_lengths))
            ]
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_ranking_predictions(
        prediction_path,
        {
            "labels": evaluation_labels.astype(np.int8),
            "probabilities": probabilities.astype(np.float64),
            "classes": classes.astype(np.int8),
            "query_ids": task.evaluation["query_id"].to_numpy(dtype=np.int64),
            "chain_lengths": chain_lengths.astype(np.int8),
        },
    )
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 1,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "asset_loading_and_encoding": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(context_encoded.feature_names),
        "feature_count": len(context_encoded.feature_names),
        "class_names": list(GRAPHLOG_RELATIONS),
        "context_rows": len(task.context),
        "evaluation_rows": len(task.evaluation),
        "task_metadata": task.metadata,
        "audit": audit,
        "encoder_provenance": context_encoded.provenance,
        "decoder_provenance": decoder_identity,
        "representation_constant": representation_constant,
        "decoder_fit_attempted": True,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": (
            "context_prior_fallback" if representation_constant else "decoder"
        ),
        "collision_conditioned_ceiling": ceiling,
        "collision_conditioned_ceiling_by_chain_length": ceiling_by_chain_length,
        "metrics": metrics,
        "context_fit_metrics": context_fit_metrics,
        "metrics_by_chain_length": per_chain_length,
        "cell_peak_resources": peak_resources(),
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
            "schema": "multiclass_classification_v1",
        },
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def run_frozen_kg_cell(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    task = load_frozen_kg_task(repo_root, cell)
    preparation_seconds = time.perf_counter() - started
    model_started = time.perf_counter()
    model = build_classifier(
        cell["decoder"],
        int(cell["decoder_seed"]),
        task.categorical_feature_indices,
    )
    _synchronize_gpu_decoder(cell["decoder"])
    model_build_seconds = time.perf_counter() - model_started
    context_representation_constant = _representation_is_constant(task.context_values)
    representation_constant = _representation_is_constant(
        task.context_values, task.candidate_values
    )
    fit_started = time.perf_counter()
    if context_representation_constant:
        decoder_fit_status = _fit_on_constant_representation(
            model,
            task.context_values,
            task.context_labels,
            str(cell["decoder"]),
        )
    else:
        model.fit(task.context_values, task.context_labels)
        _synchronize_gpu_decoder(cell["decoder"])
        decoder_fit_status = "succeeded"
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    ranking_score_orientation = "not_applicable"
    if context_representation_constant:
        probabilities = _context_prior_probabilities(
            task.context_labels, len(task.candidate_values)
        )
        prediction_policy = "context_prior_fallback"
        decoder_prediction_status = "not_attempted"
    elif cell["decoder"] == "limix2m":
        num_entities = int(task.metadata["num_entities"])
        probabilities = np.concatenate(
            [
                positive_probability(
                    model,
                    task.candidate_values[start : start + num_entities],
                )
                for start in range(0, len(task.candidate_values), num_entities)
            ]
        )
        prediction_policy = "one complete candidate set per query"
        decoder_prediction_status = "succeeded"
    elif cell["interface"] == "ultra_score_only" and cell["decoder"] in {
        "logistic",
        "logistic_tuned",
    }:
        orientation = positive_decision_orientation(model, task.context_values)
        ranking_score_orientation = {
            -1: "decreasing",
            0: "constant",
            1: "increasing",
        }[orientation]
        if orientation:
            probabilities = orientation * np.asarray(
                task.native_ultra_scores,
                dtype=np.float64,
            )
            prediction_policy = (
                "native upstream score ordered by fitted logistic orientation; "
                "preserves upstream ties exactly"
            )
        else:
            probabilities = positive_ranking_score(model, task.candidate_values)
            prediction_policy = "constant fitted logistic decision score"
        decoder_prediction_status = "succeeded"
    else:
        probabilities = positive_ranking_score(model, task.candidate_values)
        prediction_policy = (
            "least-compressed positive-class ranking score over row-aligned query "
            "blocks; decision_function when available, probability otherwise"
        )
        decoder_prediction_status = "succeeded"
    _synchronize_gpu_decoder(cell["decoder"])
    prediction_seconds = time.perf_counter() - prediction_started

    metrics, direction_metrics, records = filtered_ranking_metrics(
        probabilities,
        task.filtered,
        task.target_entities,
        task.side_codes,
    )
    native_metrics, native_direction_metrics, native_records = filtered_ranking_metrics(
        task.native_ultra_scores,
        task.filtered,
        task.target_entities,
        task.side_codes,
    )
    metrics = {
        **metrics,
        **{f"native_ultra_{key}": value for key, value in native_metrics.items()},
        "mrr_delta_vs_native_ultra": metrics["mrr"] - native_metrics["mrr"],
    }
    ranking_arrays = {
        **records,
        **{
            f"native_ultra_{key}": value
            for key, value in native_records.items()
            if key not in {"query_ids", "side_codes", "target_entities"}
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{cell['cell_id']}.predictions.npz"
    _atomic_write_ranking_predictions(prediction_path, ranking_arrays)
    decoder_identity = decoder_provenance(model)
    _validate_decoder_identity(cell, decoder_identity)
    payload = {
        "schema_version": 2,
        "cell": cell,
        "cell_payload_sha256": object_sha256(cell),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "phase_durations_seconds": {
            "asset_load_and_interface": preparation_seconds,
            "model_build": model_build_seconds,
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "feature_names": list(task.feature_names),
        "feature_count": len(task.feature_names),
        "categorical_feature_indices": list(task.categorical_feature_indices),
        "context_rows": len(task.context_labels),
        "evaluation_rows": len(task.filtered),
        "context_positive_rows": int(task.context_labels.sum()),
        "evaluation_positive_rows": len(task.filtered),
        "task_metadata": task.metadata,
        "decoder_provenance": decoder_identity,
        "context_representation_constant": context_representation_constant,
        "representation_constant": representation_constant,
        "decoder_fit_status": decoder_fit_status,
        "decoder_prediction_status": decoder_prediction_status,
        "prediction_policy": prediction_policy,
        "ranking_score_orientation": ranking_score_orientation,
        "cell_peak_resources": peak_resources(),
        "metrics": metrics,
        "direction_metrics": direction_metrics,
        "native_ultra_direction_metrics": native_direction_metrics,
        "predictions": {
            "path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
            "schema": "filtered_ranking_query_metrics_v1",
        },
    }
    atomic_write_json(output_dir / f"{cell['cell_id']}.metrics.json", payload)
    return payload


def _run_synthetic_registered(
    cell: dict[str, Any], output_dir: Path, _repo_root: Path
) -> dict[str, Any]:
    return run_synthetic_cell(cell, output_dir)


def _run_real_registered(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    return run_real_cell(cell, output_dir, repo_root)


def _run_frozen_kg_registered(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    return run_frozen_kg_cell(cell, output_dir, repo_root)


def _run_rule_induction_registered(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    return run_rule_induction_cell(cell, output_dir, repo_root)


def _run_clutrr_registered(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    return run_clutrr_cell(cell, output_dir, repo_root)


def _run_compositional_shift_registered(
    cell: dict[str, Any], output_dir: Path, _repo_root: Path
) -> dict[str, Any]:
    return run_compositional_shift_cell(cell, output_dir)


def _run_graphlog_registered(
    cell: dict[str, Any], output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    return run_graphlog_cell(cell, output_dir, repo_root)


TASK_RUNNERS: dict[str, CellRunner] = {
    "synthetic_relational": _run_synthetic_registered,
    "real_graph": _run_real_registered,
    "frozen_kg_interface": _run_frozen_kg_registered,
    "rule_induction": _run_rule_induction_registered,
    "clutrr_symbolic": _run_clutrr_registered,
    "compositional_length_shift": _run_compositional_shift_registered,
    "graphlog_symbolic": _run_graphlog_registered,
}

# The payload schema each task family currently writes. Resuming a shard must not
# silently keep a result produced under an older schema: a schema-1 artifact predates
# the representation_constant/prediction_policy fields, so a schema-2 consumer would
# read its probabilities as ordinary decoder predictions with no way to tell otherwise.
CELL_PAYLOAD_SCHEMA_VERSIONS: dict[str, int] = {
    "synthetic_relational": 2,
    "real_graph": 2,
    "rule_induction": 2,
    "frozen_kg_interface": 2,
    "clutrr_symbolic": 1,
    "compositional_length_shift": 1,
    "graphlog_symbolic": 1,
}


def _existing_result_is_current(existing: dict[str, Any], cell: dict[str, Any]) -> bool:
    """Whether an on-disk result may satisfy this plan cell without recomputation.

    The cell comparison is full equality against the plan cell (strictly stronger than
    the truncated cell_id hash), and the schema comparison pins the payload to what the
    current runner would itself produce for this task family.
    """
    return (
        existing.get("cell") == cell
        and existing.get("status") == "complete"
        and existing.get("schema_version")
        == CELL_PAYLOAD_SCHEMA_VERSIONS[str(cell["task_family"])]
    )


def run_cell(
    cell: dict[str, Any], output_dir: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    try:
        task_runner = TASK_RUNNERS[str(cell["task_family"])]
    except KeyError as error:
        raise ValueError(f"task family is not implemented: {cell['task_family']}") from error
    return task_runner(cell, output_dir, repo_root or Path.cwd())


def adaptive_gpu_cell_workers(total_memory_bytes: int, attempt: int) -> int:
    if total_memory_bytes <= 0:
        raise ValueError("GPU memory must be positive")
    if attempt <= 0:
        raise ValueError("result attempt must be positive")
    if total_memory_bytes < 40_000_000_000:
        initial = 1
    else:
        initial = (
            GPU_WORKERS_80GB
            if total_memory_bytes >= GPU_80GB_THRESHOLD_BYTES
            else GPU_WORKERS_40GB
        )
    if attempt == 1:
        return initial
    if attempt == 2:
        return 2
    return 1


def _shard_cell_workers(cells: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    decoders = {str(cell["decoder"]) for cell in cells}
    if not decoders <= GPU_DECODERS:
        return 1, {"mode": "serial-cpu", "cell_workers": 1}
    if len(decoders) != 1:
        raise ValueError("a GPU shard must contain exactly one decoder")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("parallel GPU execution requires exactly one visible CUDA device")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    attempt = int(os.environ.get("RTFM_RESULT_ATTEMPT", "1"))
    override = os.environ.get("RTFM_GPU_CELL_WORKERS")
    workers = (
        int(override)
        if override is not None
        else adaptive_gpu_cell_workers(total_memory, attempt)
    )
    if workers <= 0:
        raise ValueError("RTFM_GPU_CELL_WORKERS must be positive")
    workers = min(workers, len(cells))
    return workers, {
        "mode": "spawned-independent-gpu-cells",
        "cell_workers": workers,
        "device_total_memory_bytes": total_memory,
        "result_attempt": attempt,
        "worker_policy": "1@<40GB, 4@40GB, 8@80GB; retry with 2 then 1",
    }


def _run_cell_worker(item: tuple[dict[str, Any], Path, Path]) -> None:
    cell, output_dir, repo_root = item
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    run_cell(cell, output_dir, repo_root)


def shard_bounds(cell_count: int, shard_index: int, cells_per_shard: int) -> tuple[int, int]:
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    if cells_per_shard <= 0:
        raise ValueError("cells_per_shard must be positive")
    shard_count = math.ceil(cell_count / cells_per_shard)
    if shard_index >= shard_count:
        raise ValueError(f"shard_index {shard_index} is outside [0, {shard_count})")
    start = shard_index * cells_per_shard
    return start, min(start + cells_per_shard, cell_count)


def build_shard_manifest(
    plan: dict[str, Any], shard_index: int, start: int, end: int, output_dir: Path
) -> dict[str, Any]:
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "shard_manifest.json"
    ]
    scientific_body = {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "plan_sha256": plan["plan_sha256"],
        "shard_index": shard_index,
        "cell_indices": list(range(start, end)),
        "cell_ids": [plan["cells"][index]["cell_id"] for index in range(start, end)],
        "files": files,
    }
    return {
        **scientific_body,
        "manifest_sha256": object_sha256(scientific_body),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def run_shard(
    plan_path: Path,
    shard_index: int,
    cells_per_shard: int,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    reset_peak_resources()
    plan = read_plan(plan_path)
    start, end = shard_bounds(len(plan["cells"]), shard_index, cells_per_shard)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_cells = []
    for index in range(start, end):
        cell = plan["cells"][index]
        metrics_path = output_dir / f"{cell['cell_id']}.metrics.json"
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("cell", {}).get("cell_id") != cell["cell_id"]:
                raise ValueError(f"invalid existing result: {metrics_path}")
            if _existing_result_is_current(existing, cell):
                continue
            # Same cell, but written by an older runner (different payload schema or
            # cell spec): recompute under the current schema rather than reuse it.
        pending_cells.append(cell)
    worker_count, execution = _shard_cell_workers(plan["cells"][start:end])
    if worker_count == 1:
        for cell in pending_cells:
            run_cell(cell, output_dir, repo_root)
    elif pending_cells:
        root = repo_root or Path.cwd()
        work = [(cell, output_dir, root) for cell in pending_cells]
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
        ) as executor:
            list(executor.map(_run_cell_worker, work, chunksize=1))

    try:
        provenance = {
            "schema_version": 1,
            "study_id": plan["study_id"],
            "plan_sha256": plan["plan_sha256"],
            "shard_index": shard_index,
            "cell_indices": list(range(start, end)),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "execution": execution,
            "runtime": runtime_state(repo_root or Path.cwd(), plan_path),
            "peak_resources": peak_resources(),
        }
        provenance["provenance_sha256"] = object_sha256(provenance)
        atomic_write_json(output_dir / "runtime_provenance.json", provenance)
    except Exception as error:  # Provenance is diagnostic and must not discard completed cells.
        atomic_write_json(
            output_dir / "runtime_provenance_warning.json",
            {
                "schema_version": 1,
                "warning": "runtime provenance collection failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    manifest = build_shard_manifest(plan, shard_index, start, end, output_dir)
    atomic_write_json(output_dir / "shard_manifest.json", manifest)
    return manifest


def verify_shard(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.get("manifest_sha256")
    scientific_body = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "completed_at_utc"}
    }
    actual_manifest_hash = object_sha256(scientific_body)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"shard manifest hash mismatch: expected {expected_manifest_hash}, "
            f"computed {actual_manifest_hash}"
        )

    registered_files = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "shard_manifest.json"
    }
    if actual_files != set(registered_files):
        raise ValueError(
            f"shard file set mismatch: registered={sorted(registered_files)}, "
            f"actual={sorted(actual_files)}"
        )
    for name, entry in registered_files.items():
        path = output_dir / name
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"file size mismatch: {name}")
        actual_hash = file_sha256(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"file checksum mismatch: {name}")

    metric_paths = sorted(output_dir.glob("*.metrics.json"))
    prediction_paths = {path.name: path for path in output_dir.glob("*.predictions.npz")}
    observed_cell_ids = []
    for metric_path in metric_paths:
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete cell result: {metric_path.name}")
        cell_id = payload.get("cell", {}).get("cell_id")
        if not isinstance(cell_id, str):
            raise ValueError(f"missing cell ID: {metric_path.name}")
        if payload.get("cell_payload_sha256") != object_sha256(payload["cell"]):
            raise ValueError(f"cell hash mismatch: {cell_id}")
        prediction = payload.get("predictions", {})
        prediction_path = prediction_paths.get(prediction.get("path"))
        if prediction_path is None:
            raise ValueError(f"missing predictions for cell {cell_id}")
        if file_sha256(prediction_path) != prediction.get("sha256"):
            raise ValueError(f"prediction hash mismatch for cell {cell_id}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            prediction_arrays = {name: np.asarray(arrays[name]) for name in arrays.files}
        expected_rows = payload.get("evaluation_rows")
        if prediction.get("schema") == "filtered_ranking_query_metrics_v1":
            expected_names = {
                "query_ids",
                "side_codes",
                "target_entities",
                "target_scores",
                "ranks",
                "reciprocal_ranks",
                "hits_at_1",
                "hits_at_3",
                "hits_at_10",
                "hits_at_50",
                "native_ultra_target_scores",
                "native_ultra_ranks",
                "native_ultra_reciprocal_ranks",
                "native_ultra_hits_at_1",
                "native_ultra_hits_at_3",
                "native_ultra_hits_at_10",
                "native_ultra_hits_at_50",
            }
            if set(prediction_arrays) != expected_names:
                raise ValueError(f"ranking prediction schema mismatch for cell {cell_id}")
            if not all(len(values) == expected_rows for values in prediction_arrays.values()):
                raise ValueError(f"ranking prediction row count mismatch for cell {cell_id}")
            query_ids = prediction_arrays["query_ids"]
            if query_ids.ndim != 1 or len(np.unique(query_ids)) != len(query_ids):
                raise ValueError(f"ranking query IDs are invalid for cell {cell_id}")
            if not set(np.unique(prediction_arrays["side_codes"])) <= {0, 1}:
                raise ValueError(f"ranking side codes are invalid for cell {cell_id}")
            if any(
                values.ndim != 1 or not np.isfinite(values).all()
                for values in prediction_arrays.values()
            ):
                raise ValueError(f"ranking predictions are invalid for cell {cell_id}")
            if np.any(prediction_arrays["ranks"] < 1.0) or np.any(
                prediction_arrays["native_ultra_ranks"] < 1.0
            ):
                raise ValueError(f"ranking positions are invalid for cell {cell_id}")
            metric_values = payload.get("metrics")
            if not isinstance(metric_values, dict) or not metric_values:
                raise ValueError(f"metrics are missing for cell {cell_id}")
            if any(
                value is not None
                and (not isinstance(value, int | float) or not np.isfinite(value))
                for value in metric_values.values()
            ):
                raise ValueError(f"ranking metrics are invalid for cell {cell_id}")
            observed_cell_ids.append(cell_id)
            continue
        if prediction.get("schema") == "multiclass_classification_v1":
            expected_names = {
                "labels",
                "probabilities",
                "classes",
                "query_ids",
                "chain_lengths",
            }
            if set(prediction_arrays) != expected_names:
                raise ValueError(
                    f"multiclass prediction schema mismatch for cell {cell_id}"
                )
            labels = prediction_arrays["labels"]
            probabilities = prediction_arrays["probabilities"]
            classes = prediction_arrays["classes"]
            query_ids = prediction_arrays["query_ids"]
            chain_lengths = prediction_arrays["chain_lengths"]
            if not all(
                len(values) == expected_rows
                for values in (labels, probabilities, query_ids, chain_lengths)
            ):
                raise ValueError(
                    f"multiclass prediction row count mismatch for cell {cell_id}"
                )
            if (
                labels.ndim != 1
                or classes.ndim != 1
                or probabilities.shape != (expected_rows, len(classes))
                or not set(map(int, np.unique(labels))) <= set(map(int, classes))
            ):
                raise ValueError(f"multiclass labels are invalid for cell {cell_id}")
            if (
                not np.isfinite(probabilities).all()
                or np.any((probabilities < 0.0) | (probabilities > 1.0))
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            ):
                raise ValueError(
                    f"multiclass probabilities are invalid for cell {cell_id}"
                )
            if query_ids.ndim != 1 or len(np.unique(query_ids)) != len(query_ids):
                raise ValueError(f"multiclass query IDs are invalid for cell {cell_id}")
            if chain_lengths.ndim != 1 or np.any(
                (chain_lengths < 2) | (chain_lengths > 10)
            ):
                raise ValueError(f"CLUTRR chain lengths are invalid for cell {cell_id}")
            metric_values = payload.get("metrics")
            if not isinstance(metric_values, dict) or not metric_values:
                raise ValueError(f"metrics are missing for cell {cell_id}")
            if any(
                value is not None
                and (not isinstance(value, int | float) or not np.isfinite(value))
                for value in metric_values.values()
            ):
                raise ValueError(f"multiclass metrics are invalid for cell {cell_id}")
            observed_cell_ids.append(cell_id)
            continue
        if set(prediction_arrays) != {"labels", "probabilities", "query_ids"}:
            raise ValueError(f"prediction schema mismatch for cell {cell_id}")
        labels = prediction_arrays["labels"]
        probabilities = prediction_arrays["probabilities"]
        query_ids = prediction_arrays["query_ids"]
        if not all(len(values) == expected_rows for values in (labels, probabilities, query_ids)):
            raise ValueError(f"prediction row count mismatch for cell {cell_id}")
        if labels.ndim != 1 or not set(np.unique(labels)) <= {0, 1}:
            raise ValueError(f"prediction labels are invalid for cell {cell_id}")
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
            raise ValueError(f"prediction probabilities are invalid for cell {cell_id}")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"prediction probabilities are out of range for cell {cell_id}")
        if query_ids.ndim != 1 or len(np.unique(query_ids)) != len(query_ids):
            raise ValueError(f"prediction query IDs are invalid for cell {cell_id}")
        if int(labels.sum()) != payload.get("evaluation_positive_rows"):
            raise ValueError(f"prediction labels disagree with metrics for cell {cell_id}")
        metric_values = payload.get("metrics")
        if not isinstance(metric_values, dict) or not metric_values:
            raise ValueError(f"metrics are missing for cell {cell_id}")
        for metric_name, value in metric_values.items():
            invalid = not isinstance(value, int | float) or not np.isfinite(value)
            if value is not None and invalid:
                raise ValueError(f"metric {metric_name} is invalid for cell {cell_id}")
        observed_cell_ids.append(cell_id)
    if set(observed_cell_ids) != set(manifest.get("cell_ids", [])):
        raise ValueError("metric cell IDs do not match the shard manifest")
    return manifest
