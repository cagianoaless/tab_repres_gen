#!/usr/bin/env python3
"""Audit the unexpectedly low TabPFN CLUTRR length-only endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.datasets.clutrr import CLUTRR_RELATIONS, load_clutrr_task
from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json


def _load_cells(result_root: Path) -> list[tuple[dict[str, Any], Path]]:
    cells = []
    for metric_path in sorted(result_root.glob("results/shard-*/*.metrics.json")):
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        cell = payload["cell"]
        if cell["decoder"] == "tabpfn" and cell["encoder"] == "path_length_only":
            cells.append((payload, metric_path))
    if len(cells) != 15:
        raise ValueError(f"expected 15 TabPFN length-only cells, found {len(cells)}")
    return cells


def _distribution(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLUTRR_RELATIONS)).astype(np.float64)
    return counts / counts.sum()


def _support_audit(
    payload: dict[str, Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    chain_lengths: np.ndarray,
    repo_root: Path,
) -> dict[str, Any]:
    task = load_clutrr_task(repo_root, payload["cell"])
    records = {}
    for chain_length in (2, 3):
        context_labels = task.context.loc[
            task.context["chain_length"] == chain_length, "label"
        ].to_numpy(dtype=np.int64)
        selected = chain_lengths == chain_length
        evaluation_labels = labels[selected]
        context_distribution = _distribution(context_labels)
        evaluation_distribution = _distribution(evaluation_labels)
        unique_probability_rows = np.unique(
            np.round(probabilities[selected], decimals=12), axis=0
        )
        transferred_mode = int(np.argmax(context_distribution))
        prediction_classes = np.unique(predicted[selected])
        records[str(chain_length)] = {
            "context_rows": int(len(context_labels)),
            "evaluation_rows": int(len(evaluation_labels)),
            "context_label_distribution": context_distribution.tolist(),
            "evaluation_label_distribution": evaluation_distribution.tolist(),
            "total_variation_context_vs_evaluation": float(
                0.5 * np.abs(context_distribution - evaluation_distribution).sum()
            ),
            "evaluation_oracle_accuracy_given_length": float(
                evaluation_distribution.max()
            ),
            "transferred_context_mode_class": transferred_mode,
            "transferred_context_mode_relation": CLUTRR_RELATIONS[transferred_mode],
            "transferred_context_mode_accuracy": float(
                np.mean(evaluation_labels == transferred_mode)
            ),
            "tabpfn_accuracy": float(
                np.mean(predicted[selected] == evaluation_labels)
            ),
            "tabpfn_predicted_classes": prediction_classes.tolist(),
            "tabpfn_predicted_relations": [
                CLUTRR_RELATIONS[int(value)] for value in prediction_classes
            ],
            "unique_probability_rows": int(len(unique_probability_rows)),
        }
    return records


def _audit_cell(
    payload: dict[str, Any], metric_path: Path, repo_root: Path | None
) -> dict[str, Any]:
    prediction_path = metric_path.parent / payload["predictions"]["path"]
    if file_sha256(prediction_path) != payload["predictions"]["sha256"]:
        raise ValueError(f"prediction checksum mismatch: {prediction_path}")
    predictions = np.load(prediction_path)
    labels = np.asarray(predictions["labels"], dtype=np.int64)
    probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
    classes = np.asarray(predictions["classes"], dtype=np.int64)
    chain_lengths = np.asarray(predictions["chain_lengths"], dtype=np.int64)
    expected_classes = np.arange(len(CLUTRR_RELATIONS), dtype=np.int64)
    if not np.array_equal(classes, expected_classes):
        raise ValueError("prediction class order does not match the canonical CLUTRR mapping")
    if payload["class_names"] != list(CLUTRR_RELATIONS):
        raise ValueError("metric class names do not match the canonical CLUTRR mapping")
    if probabilities.shape != (len(labels), len(classes)):
        raise ValueError("probability matrix has the wrong shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows do not sum to one")

    context_count = int(payload["cell"]["context_per_relation"])
    context_counts = payload["task_metadata"]["context_class_counts"]
    if context_counts != {relation: context_count for relation in CLUTRR_RELATIONS}:
        raise ValueError("context does not contain every class at the registered count")

    predicted = classes[np.argmax(probabilities, axis=1)]
    accuracy = float(np.mean(predicted == labels))
    if not np.isclose(accuracy, payload["metrics"]["accuracy"], atol=1e-15):
        raise ValueError("recomputed accuracy does not match the recorded metric")
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    label_counts = np.bincount(labels, minlength=len(classes))
    predicted_counts = np.bincount(predicted, minlength=len(classes))
    by_length = {}
    for chain_length in sorted(map(int, np.unique(chain_lengths))):
        selected = chain_lengths == chain_length
        by_length[str(chain_length)] = {
            "rows": int(selected.sum()),
            "accuracy": float(np.mean(predicted[selected] == labels[selected])),
            "predicted_class_counts": np.bincount(
                predicted[selected], minlength=len(classes)
            ).tolist(),
        }
    result = {
        "seed": int(payload["cell"]["seed"]),
        "context_per_relation": context_count,
        "context_rows": int(payload["context_rows"]),
        "evaluation_rows": int(payload["evaluation_rows"]),
        "accuracy": accuracy,
        "label_counts": label_counts.tolist(),
        "predicted_class_counts": predicted_counts.tolist(),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "majority_class_accuracy": float(label_counts.max() / label_counts.sum()),
        "uniform_random_expected_accuracy": float(1.0 / len(classes)),
        "empirical_prior_random_expected_accuracy": float(
            np.sum(np.square(label_counts / label_counts.sum()))
        ),
        "by_chain_length": by_length,
    }
    if repo_root is not None:
        result["seen_length_support_audit"] = _support_audit(
            payload,
            labels,
            probabilities,
            predicted,
            chain_lengths,
            repo_root,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="reconstruct official contexts and audit label support at seen lengths",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root is not None else None
    loaded_cells = _load_cells(args.result_root)
    maximum_context = max(
        int(payload["cell"]["context_per_relation"])
        for payload, _ in loaded_cells
    )
    records = [
        _audit_cell(
            payload,
            path,
            repo_root
            if int(payload["cell"]["context_per_relation"]) == maximum_context
            else None,
        )
        for payload, path in loaded_cells
    ]
    endpoint = [record for record in records if record["context_per_relation"] == maximum_context]
    aggregate_confusion = np.sum(
        [
            np.asarray(
                record["confusion_matrix_rows_true_columns_predicted"],
                dtype=np.int64,
            )
            for record in endpoint
        ],
        axis=0,
    )
    seen_length_summary = None
    if repo_root is not None:
        endpoint_seen = [
            record["seen_length_support_audit"]
            for record in endpoint
        ]
        seen_length_summary = {
            "chain_lengths": [2, 3],
            "mean_tabpfn_accuracy": float(
                np.mean(
                    [
                        seed[str(length)]["tabpfn_accuracy"]
                        for seed in endpoint_seen
                        for length in (2, 3)
                    ]
                )
            ),
            "mean_transferred_context_mode_accuracy": float(
                np.mean(
                    [
                        seed[str(length)]["transferred_context_mode_accuracy"]
                        for seed in endpoint_seen
                        for length in (2, 3)
                    ]
                )
            ),
            "mean_total_variation_context_vs_evaluation": float(
                np.mean(
                    [
                        seed[str(length)][
                            "total_variation_context_vs_evaluation"
                        ]
                        for seed in endpoint_seen
                        for length in (2, 3)
                    ]
                )
            ),
            "all_rows_within_length_have_identical_probabilities": bool(
                all(
                    seed[str(length)]["unique_probability_rows"] == 1
                    for seed in endpoint_seen
                    for length in (2, 3)
                )
            ),
        }
    output = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_role": "class-mapping and prediction-collapse audit",
        "result_root": str(args.result_root.resolve()),
        "canonical_class_names": list(CLUTRR_RELATIONS),
        "cells_audited": len(records),
        "all_contexts_cover_all_classes": True,
        "all_probability_rows_normalized": True,
        "all_recorded_accuracies_reproduced": True,
        "max_context_summary": {
            "context_per_relation": maximum_context,
            "seeds": sorted(record["seed"] for record in endpoint),
            "mean_accuracy": float(np.mean([record["accuracy"] for record in endpoint])),
            "majority_class_accuracy": float(
                np.mean([record["majority_class_accuracy"] for record in endpoint])
            ),
            "uniform_random_expected_accuracy": float(1.0 / len(CLUTRR_RELATIONS)),
            "aggregate_confusion_matrix_rows_true_columns_predicted": (
                aggregate_confusion.tolist()
            ),
        },
        "seen_length_support_summary": seen_length_summary,
        "cells": records,
        "interpretation_boundary": (
            "The audit rules out missing classes, class-index drift, checksum failure, "
            "metric-recomputation error, and malformed probabilities. When official assets "
            "are supplied, it also shows whether identical length-only rows face conditional "
            "label-distribution shift. It does not replace an independent model re-invocation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                key: value
                for key, value in output["max_context_summary"].items()
                if key != "aggregate_confusion_matrix_rows_true_columns_predicted"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
