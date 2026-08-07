from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.datasets.synthetic import SyntheticTask, generate_synthetic_task
from relational_tfm.decoders.classification import build_classifier, positive_probability
from relational_tfm.encoders.synthetic import EncodedRows, encode_synthetic_rows
from relational_tfm.evaluation.classification import classification_metrics
from relational_tfm.hashing import file_sha256, object_sha256
from relational_tfm.plan import atomic_write_json


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _task_signature(task: SyntheticTask) -> str:
    payload = {
        "context_query_ids": task.context["query_id"].astype(int).tolist(),
        "context_labels": task.context["label"].astype(int).tolist(),
        "evaluation_query_ids": task.evaluation["query_id"].astype(int).tolist(),
        "evaluation_labels": task.evaluation["label"].astype(int).tolist(),
        "context_relations": task.context["relation"].astype(str).tolist(),
        "evaluation_relations": task.evaluation["relation"].astype(str).tolist(),
    }
    return object_sha256(payload)


def _generate_task(config: dict[str, Any], spec: dict[str, Any], relabel_seed: int):
    return generate_synthetic_task(
        property_name=str(spec["property"]),
        graph_size=int(spec["graph_size"]),
        context_per_class=int(config["context_per_class"]),
        noise_regime="none",
        seed=int(config["base_seed"]),
        max_context_per_class=int(config["context_per_class"]),
        evaluation_per_class=int(config["evaluation_per_class"]),
        graph_seed=int(config["graph_seed"]),
        relabel_seed=int(relabel_seed),
        split_seed=int(config["split_seed"]),
        noise_seed=int(config["noise_seed"]),
        context_negative_ratio=1,
        evaluation_negative_ratio=1,
    )


def _encode(
    task: SyntheticTask,
    rows: Any,
    encoder: str,
    config: dict[str, Any],
) -> EncodedRows:
    return encode_synthetic_rows(
        task.support_graph,
        rows,
        encoder,
        int(config["encoder_seed"]),
        property_name=task.property_name,
        control_population=task.population,
    )


def build_plan(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    identifier_encoders = tuple(str(value) for value in config["identifier_encoders"])
    cells: list[dict[str, Any]] = []
    for spec in config["property_specs"]:
        encoders = (*identifier_encoders, str(spec["aligned_encoder"]))
        for encoder in encoders:
            for decoder in config["decoders"]:
                for relabel_seed in config["relabel_seeds"]:
                    scientific = {
                        "study_id": config["study_id"],
                        "property": spec["property"],
                        "graph_size": int(spec["graph_size"]),
                        "aligned_encoder": spec["aligned_encoder"],
                        "encoder": encoder,
                        "decoder": decoder,
                        "context_per_class": int(config["context_per_class"]),
                        "evaluation_per_class": int(config["evaluation_per_class"]),
                        "base_seed": int(config["base_seed"]),
                        "graph_seed": int(config["graph_seed"]),
                        "split_seed": int(config["split_seed"]),
                        "noise_seed": int(config["noise_seed"]),
                        "encoder_seed": int(config["encoder_seed"]),
                        "decoder_seed": int(config["decoder_seed"]),
                        "relabel_seed": int(relabel_seed),
                    }
                    cells.append(
                        {
                            "index": len(cells),
                            "cell_id": object_sha256(scientific)[:20],
                            **scientific,
                        }
                    )
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "design_role": config["design_role"],
        "config_sha256": config_sha256,
        "cells": cells,
    }
    payload["plan_sha256"] = object_sha256(payload)
    return payload


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    property_records: list[dict[str, Any]] = []
    for spec in config["property_specs"]:
        tasks = [
            _generate_task(config, spec, int(seed)) for seed in config["relabel_seeds"]
        ]
        task_signatures = {_task_signature(task) for task in tasks}
        if len(task_signatures) != 1:
            raise ValueError(f"{spec['property']} semantics changed across entity relabelings")
        support_hashes = {
            str(task.metadata["clean_support_sha256"]) for task in tasks
        }
        if len(support_hashes) != len(tasks):
            raise ValueError(f"{spec['property']} did not receive distinct relabelings")

        aligned_hashes = {
            _array_sha256(
                _encode(task, task.evaluation, str(spec["aligned_encoder"]), config).values
            )
            for task in tasks
        }
        if len(aligned_hashes) != 1:
            raise ValueError(
                f"{spec['property']} aligned representation is not relabel-invariant"
            )
        artifact_hashes = {
            _array_sha256(_encode(task, task.evaluation, "artifact_ids", config).values)
            for task in tasks
        }
        if len(artifact_hashes) != len(tasks):
            raise ValueError(
                f"{spec['property']} arbitrary identifier control did not change"
            )
        property_records.append(
            {
                "property": spec["property"],
                "graph_size": spec["graph_size"],
                "task_signature_sha256": next(iter(task_signatures)),
                "distinct_support_relabelings": len(support_hashes),
                "aligned_encoder": spec["aligned_encoder"],
                "aligned_evaluation_hashes": len(aligned_hashes),
                "artifact_evaluation_hashes": len(artifact_hashes),
            }
        )
    return {
        "status": "PASS",
        "interpretation": (
            "graph, split, labels, and row identities are fixed; only the entity "
            "bijection changes"
        ),
        "properties": property_records,
    }


def _run_cell(
    cell: dict[str, Any],
    config: dict[str, Any],
    output_root: str,
) -> dict[str, Any]:
    os.environ["RTFM_CPU_THREADS"] = "1"
    output_dir = Path(output_root) / "cells"
    output_path = output_dir / f"{cell['cell_id']}.json"
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("cell_sha256") == object_sha256(cell)
        ):
            return existing

    spec = next(
        item for item in config["property_specs"] if item["property"] == cell["property"]
    )
    task = _generate_task(config, spec, int(cell["relabel_seed"]))
    context = _encode(task, task.context, str(cell["encoder"]), config)
    evaluation = _encode(task, task.evaluation, str(cell["encoder"]), config)
    if context.feature_names != evaluation.feature_names:
        raise ValueError("context and evaluation feature schemas differ")
    labels = task.context["label"].to_numpy(dtype=np.int64)
    evaluation_labels = task.evaluation["label"].to_numpy(dtype=np.int64)

    model = build_classifier(str(cell["decoder"]), int(cell["decoder_seed"]))
    model.fit(context.values, labels)
    probabilities = positive_probability(model, evaluation.values)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "cell": cell,
        "cell_sha256": object_sha256(cell),
        "task_signature_sha256": _task_signature(task),
        "support_sha256": task.metadata["clean_support_sha256"],
        "context_feature_sha256": _array_sha256(context.values),
        "evaluation_feature_sha256": _array_sha256(evaluation.values),
        "feature_count": int(context.values.shape[1]),
        "metrics": classification_metrics(evaluation_labels, probabilities),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    return payload


def summarize(
    results: list[dict[str, Any]],
    plan: dict[str, Any],
    preflight_result: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        cell = result["cell"]
        grouped[(cell["property"], cell["encoder"], cell["decoder"])].append(result)

    summaries = []
    for (property_name, encoder, decoder), group in sorted(grouped.items()):
        metric_summary = {}
        for metric in sorted(group[0]["metrics"]):
            values = np.asarray(
                [float(result["metrics"][metric]) for result in group], dtype=np.float64
            )
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        summaries.append(
            {
                "property": property_name,
                "encoder": encoder,
                "decoder": decoder,
                "relabelings": len(group),
                "distinct_evaluation_feature_hashes": len(
                    {result["evaluation_feature_sha256"] for result in group}
                ),
                "metrics": metric_summary,
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "study_id": plan["study_id"],
        "plan_sha256": plan["plan_sha256"],
        "cells": len(results),
        "preflight": preflight_result,
        "groups": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(config, file_sha256(config_path))
    plan_path = output_root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("registered relabeling audit plan changed")
    else:
        atomic_write_json(plan_path, plan)
    print(
        f"study_id={plan['study_id']} cells={len(plan['cells'])} "
        f"plan_sha256={plan['plan_sha256']}",
        flush=True,
    )
    if args.plan_only:
        return

    preflight_result = preflight(config)
    atomic_write_json(output_root / "preflight.json", preflight_result)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_run_cell, cell, config, str(output_root))
            for cell in plan["cells"]
        ]
        results = []
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed={completed}/{len(futures)}", flush=True)
    report = summarize(results, plan, preflight_result)
    atomic_write_json(output_root / "summary.json", report)
    print(f"PASS cells={len(results)} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
