from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from relational_tfm.config import load_study_config
from relational_tfm.datasets.graphlog import load_graphlog_task
from relational_tfm.datasets.graphlog_audit import audit_graphlog_task
from relational_tfm.encoders.graphlog import (
    GRAPHLOG_MAX_PATH_LENGTH,
    GRAPHLOG_RELATIONS,
    encode_graphlog_rows,
)
from relational_tfm.hashing import file_sha256
from relational_tfm.plan import build_plan
from relational_tfm.runner import run_graphlog_cell

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_row(tokens: list[str], target: str, offset: int) -> dict[str, object]:
    path = list(range(offset, offset + len(tokens) + 1))
    edges = [
        [path[index], path[index + 1], token]
        for index, token in enumerate(tokens)
    ]
    return {
        "edges": edges,
        "query": [path[0], path[-1], target],
        "target": target,
        "descriptor": ",".join(tokens),
        "rules_used": [],
        "rules_used_pos": [],
        "resolution_path": path,
    }


def _fixture_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "graphlog_fixture.zip"
    train_rows = [
        _raw_row(["R_0_+", "R_1_+"], "R_0_+", 10),
        _raw_row(["R_1_+", "R_0_+"], "R_0_+", 20),
        _raw_row(["R_2_+", "R_3_+", "R_4_+"], "R_1_+", 30),
        _raw_row(["R_4_+", "R_3_+", "R_2_+"], "R_1_+", 40),
    ]
    test_rows = [
        _raw_row(["R_0_+", "R_1_+", "R_2_+", "R_3_+"], "R_0_+", 50),
        _raw_row(
            ["R_3_+", "R_2_+", "R_1_+", "R_0_+", "R_4_+"],
            "R_1_+",
            60,
        ),
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(57):
            outer = "train" if index < 51 else ("valid" if index < 54 else "test")
            prefix = f"graphlog_v1.1/{outer}/rule_{index}"
            for split, rows in (
                ("train", train_rows),
                ("valid", test_rows),
                ("test", test_rows),
            ):
                archive.writestr(
                    f"{prefix}/{split}.jsonl",
                    "".join(json.dumps(row) + "\n" for row in rows),
                )
    return archive_path


def _fixture_cell(tmp_path: Path, context: int = 1) -> dict[str, object]:
    archive = _fixture_archive(tmp_path)
    return {
        "index": 0,
        "cell_id": "graphlog-test-cell",
        "study_id": "graphlog-test",
        "task_family": "graphlog_symbolic",
        "dataset": "GraphLog/v1.1",
        "asset": {
            "path": archive.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(archive),
        },
        "world": "rule_0",
        "context_inner_split": "train",
        "evaluation_inner_split": "test",
        "train_chain_lengths": [2, 3],
        "evaluation_chain_lengths": [4, 5],
        "max_path_length": 5,
        "max_context_per_relation": 2,
        "encoder": "relation_multiset",
        "decoder": "logistic",
        "context_per_relation": context,
        "seed": 101,
        "graph_seed": 101,
        "relabel_seed": 100_101,
        "split_seed": 200_101,
        "noise_seed": 300_101,
        "encoder_seed": 400_101,
        "decoder_seed": 500_101,
    }


def test_graphlog_plans_have_frozen_cell_counts_and_hashes() -> None:
    cpu = build_plan(
        load_study_config(
            REPO_ROOT / "configs/studies/graphlog_symbolic_cpu_confirmation_v1.yaml"
        )
    )
    gpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/graphlog_symbolic_gpu_tabpfn_confirmation_v1.yaml"
        )
    )
    assert len(cpu["cells"]) == 8_208
    assert (
        cpu["plan_sha256"]
        == "dc6872565a8726a715a9437cf5029db6c733c469b08e15fc097b5fd9dc8e7006"
    )
    assert len(gpu["cells"]) == 4_104
    assert (
        gpu["plan_sha256"]
        == "decac0f768513abfd742bbd3e84bf3934aefa22043043e35a940ebf14406b75d"
    )

    tuned = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/graphlog_symbolic_cpu_catboost_tuned_endpoint_v1.yaml"
        )
    )
    assert len(tuned["cells"]) == 684
    assert (
        tuned["plan_sha256"]
        == "4a84be6406befdf810dd174a8abfa1d307d04b6c65e1c1b1d65c565ec0ac8b2b"
    )

    depth1 = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/graphlog_symbolic_cpu_catboost_depth1_endpoint_v1.yaml"
        )
    )
    assert len(depth1["cells"]) == 684
    assert (
        depth1["plan_sha256"]
        == "37230cb4f827567de0c14428661efe39dfddee212fbec044c32d3c7c85761487"
    )


def test_graphlog_blockwise_null_preserves_shared_block() -> None:
    rows = pd.DataFrame(
        {
            "chain_length": [4, 4, 5, 5],
            "relation_tokens": [
                ("R_0_+", "R_1_+", "R_2_+", "R_3_+"),
                ("R_3_+", "R_2_+", "R_1_+", "R_0_+"),
                ("R_0_-", "R_1_-", "R_2_-", "R_3_-", "R_4_-"),
                ("R_4_-", "R_3_-", "R_2_-", "R_1_-", "R_0_-"),
            ],
        }
    )
    aligned = encode_graphlog_rows(rows, "ordered_plus_multiset", 3)
    null = encode_graphlog_rows(
        rows,
        "aligned_multiset_with_ordered_block_permutation",
        3,
    )
    ordered_width = 1 + GRAPHLOG_MAX_PATH_LENGTH * len(GRAPHLOG_RELATIONS)
    assert aligned.values.shape == null.values.shape
    np.testing.assert_array_equal(
        aligned.values[:, ordered_width:],
        null.values[:, ordered_width:],
    )
    assert not np.array_equal(
        aligned.values[:, 1:ordered_width],
        null.values[:, 1:ordered_width],
    )


def test_graphlog_task_is_nested_disjoint_and_audited(tmp_path: Path) -> None:
    small_cell = _fixture_cell(tmp_path, context=1)
    large_cell = {**small_cell, "context_per_relation": 2}
    small = load_graphlog_task(tmp_path, small_cell)
    large = load_graphlog_task(tmp_path, large_cell)
    assert audit_graphlog_task(small)["status"] == "PASS"
    assert set(small.context["row_id"]) < set(large.context["row_id"])
    assert set(small.context["row_id"]).isdisjoint(small.evaluation["row_id"])
    assert set(small.context["chain_length"]) == {2, 3}
    assert set(small.evaluation["chain_length"]) == {4, 5}


def test_graphlog_logistic_cell_runs_end_to_end(tmp_path: Path) -> None:
    cell = _fixture_cell(tmp_path)
    output = tmp_path / "output"
    payload = run_graphlog_cell(cell, output, tmp_path)
    assert payload["status"] == "complete"
    assert payload["audit"]["status"] == "PASS"
    assert payload["context_rows"] == 2
    assert payload["evaluation_rows"] == 2
    assert set(payload["metrics_by_chain_length"]) == {"4", "5"}
    assert 0.0 <= payload["metrics"]["equal_length_macro_balanced_accuracy"] <= 1.0
    assert (output / "graphlog-test-cell.predictions.npz").is_file()
