from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from relational_tfm.config import load_study_config
from relational_tfm.datasets.clutrr import (
    CLUTRR_EDGE_RELATIONS,
    CLUTRR_RELATIONS,
    _paired_additive_context,
    load_clutrr_task,
)
from relational_tfm.datasets.clutrr_audit import audit_clutrr_task
from relational_tfm.decoders.classification import _ordered_path_tokens
from relational_tfm.encoders.clutrr import encode_clutrr_rows
from relational_tfm.hashing import file_sha256
from relational_tfm.plan import atomic_write_json, build_plan
from relational_tfm.runner import build_shard_manifest, run_clutrr_cell, verify_shard
from analysis.audit_clutrr_coordinate_coverage import coordinate_support

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "id",
    "story",
    "query",
    "target",
    "target_text",
    "clean_story",
    "proof_state",
    "f_comb",
    "task_name",
    "story_edges",
    "edge_types",
    "query_edge",
    "genders",
    "task_split",
)


def _write_fixture(path: Path, split: str, repeats: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for label, relation in enumerate(CLUTRR_RELATIONS):
            for repeat in range(repeats):
                writer.writerow(
                    {
                        "id": f"{split}-{label}-{repeat}",
                        "story": "",
                        "query": "('a', 'c')",
                        "target": label,
                        "target_text": relation,
                        "clean_story": "",
                        "proof_state": "",
                        "f_comb": "mother-father",
                        "task_name": "task_1.2",
                        "story_edges": "[(0, 1), (1, 2)]",
                        "edge_types": (
                            f"['{('mother', 'father', 'sister')[label % 3]}', "
                            f"'{('son', 'daughter')[label % 2]}']"
                        ),
                        "query_edge": "(0, 2)",
                        "genders": "",
                        "task_split": split,
                    }
                )


def _fixture_cell(root: Path) -> dict[str, object]:
    paths = {
        split: root / "data/raw/clutrr/v1/fixture" / f"{split}.csv"
        for split in ("train", "validation", "test")
    }
    _write_fixture(paths["train"], "train", 2)
    _write_fixture(paths["validation"], "validation", 1)
    _write_fixture(paths["test"], "test", 1)
    return {
        "index": 0,
        "cell_id": "clutrr-test-cell",
        "study_id": "clutrr-test",
        "task_family": "clutrr_symbolic",
        "dataset": "CLUTRR/v1",
        "dataset_config": "gen_train23_test2to10",
        "asset_files": {
            split: {
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
            }
            for split, path in paths.items()
        },
        "train_chain_lengths": [2, 3],
        "evaluation_chain_lengths": list(range(2, 11)),
        "max_path_length": 10,
        "encoder": "ordered_typed_path",
        "decoder": "logistic",
        "context_per_relation": 1,
        "max_context_per_relation": 2,
        "seed": 101,
        "graph_seed": 101,
        "relabel_seed": 100_101,
        "split_seed": 200_101,
        "noise_seed": 300_101,
        "encoder_seed": 400_101,
        "decoder_seed": 500_101,
    }


def test_clutrr_task_is_balanced_relabelled_and_audited(tmp_path: Path) -> None:
    cell = _fixture_cell(tmp_path)
    task = load_clutrr_task(tmp_path, cell)
    audit = audit_clutrr_task(task)
    assert audit["status"] == "PASS"
    assert len(task.context) == len(CLUTRR_RELATIONS)
    assert len(task.evaluation) == len(CLUTRR_RELATIONS)
    minimum_node = min(
        node
        for edges in task.context["story_edges"]
        for edge in edges
        for node in edge
    )
    assert minimum_node >= 2**40


def test_clutrr_encoders_are_label_free_and_relabel_invariant(tmp_path: Path) -> None:
    cell = _fixture_cell(tmp_path)
    task = load_clutrr_task(tmp_path, cell)
    encoded = encode_clutrr_rows(task.context, "ordered_typed_path", 7)
    changed = task.context.copy()
    changed["label"] = np.roll(changed["label"].to_numpy(), 1)
    changed["story_edges"] = [
        tuple((head + 123_456_789, tail + 123_456_789) for head, tail in edges)
        for edges in changed["story_edges"]
    ]
    changed["query_edge"] = [
        (head + 123_456_789, tail + 123_456_789)
        for head, tail in changed["query_edge"]
    ]
    reencoded = encode_clutrr_rows(changed, "ordered_typed_path", 7)
    np.testing.assert_array_equal(encoded.values, reencoded.values)
    assert encoded.values.shape == (len(CLUTRR_RELATIONS), 141)
    assert encoded.provenance["raw_entity_ids_exposed"] is False
    assert encoded.provenance["target_relation_exposed"] is False


def test_clutrr_controls_break_alignment_without_changing_shape(tmp_path: Path) -> None:
    cell = _fixture_cell(tmp_path)
    task = load_clutrr_task(tmp_path, cell)
    aligned = encode_clutrr_rows(task.context, "ordered_typed_path", 11)
    permuted = encode_clutrr_rows(task.context, "aligned_permutation", 11)
    noise = encode_clutrr_rows(task.context, "dimension_matched_noise", 11)
    assert aligned.values.shape == permuted.values.shape == noise.values.shape
    assert not np.array_equal(aligned.values, permuted.values)
    assert not np.array_equal(aligned.values, noise.values)
    np.testing.assert_allclose(
        np.sort(aligned.values, axis=0),
        np.sort(permuted.values, axis=0),
    )
    np.testing.assert_allclose(
        np.sort(aligned.values, axis=0),
        np.sort(noise.values, axis=0),
    )


def test_clutrr_v2_controls_preserve_length_and_stratify_permutations() -> None:
    rows = pd.DataFrame(
        {
            "chain_length": [2, 2, 3, 3],
            "edge_types": [
                ("mother", "father"),
                ("brother", "sister"),
                ("son", "daughter", "uncle"),
                ("wife", "husband", "aunt"),
            ],
            "story_edges": [
                ((0, 1), (1, 2)),
                ((3, 4), (4, 5)),
                ((6, 7), (7, 8), (8, 9)),
                ((10, 11), (11, 12), (12, 13)),
            ],
            "query_edge": [(0, 2), (3, 5), (6, 9), (10, 13)],
        }
    )
    aligned = encode_clutrr_rows(rows, "ordered_typed_path", 17)
    joint = encode_clutrr_rows(rows, "within_length_joint_permutation", 17)
    independent = encode_clutrr_rows(
        rows, "within_length_independent_permutation", 17
    )

    np.testing.assert_array_equal(aligned.values[:, 0], joint.values[:, 0])
    np.testing.assert_array_equal(aligned.values[:, 0], independent.values[:, 0])
    for chain_length in (2, 3):
        selected = rows["chain_length"].to_numpy() == chain_length
        aligned_block = aligned.values[selected, 1:]
        joint_block = joint.values[selected, 1:]
        np.testing.assert_array_equal(
            np.sort(aligned_block, axis=0),
            np.sort(joint_block, axis=0),
        )
        np.testing.assert_array_equal(
            np.sort(aligned_block, axis=0),
            np.sort(independent.values[selected, 1:], axis=0),
        )
        assert {tuple(row) for row in joint_block} == {
            tuple(row) for row in aligned_block
        }
    assert joint.provenance["chain_length_alignment_preserved"] is True
    assert joint.provenance["permutation_stratified_by_chain_length"] is True


def test_clutrr_shared_coordinate_encoders_separate_order_from_position() -> None:
    rows = pd.DataFrame(
        {
            "chain_length": [2, 2, 3, 3],
            "edge_types": [
                ("mother", "father"),
                ("father", "mother"),
                ("mother", "father", "sister"),
                ("father", "mother", "sister"),
            ],
            "story_edges": [
                ((0, 1), (1, 2)),
                ((3, 4), (4, 5)),
                ((6, 7), (7, 8), (8, 9)),
                ((10, 11), (11, 12), (12, 13)),
            ],
            "query_edge": [(0, 2), (3, 5), (6, 9), (10, 13)],
        }
    )
    multiset = encode_clutrr_rows(rows, "relation_multiset", 19)
    transitions = encode_clutrr_rows(rows, "relation_transition_counts", 19)
    ordered_plus = encode_clutrr_rows(rows, "ordered_plus_multiset", 19)
    transition_null = encode_clutrr_rows(
        rows, "within_length_transition_joint_permutation", 19
    )
    ordered_plus_null = encode_clutrr_rows(
        rows,
        "within_length_ordered_plus_multiset_joint_permutation",
        19,
    )
    aligned_multiset_permuted_ordered = encode_clutrr_rows(
        rows,
        "aligned_multiset_with_ordered_block_permutation",
        19,
    )
    supported_prefix = encode_clutrr_rows(
        rows,
        "aligned_multiset_with_supported_prefix",
        19,
    )

    np.testing.assert_array_equal(multiset.values[0], multiset.values[1])
    assert not np.array_equal(transitions.values[0], transitions.values[1])
    assert transitions.values.shape[1] == 197
    assert ordered_plus.values.shape[1] == 155
    assert transitions.provenance["position_shared_coordinates"] is True
    assert transitions.provenance["position_specific_coordinates"] is False
    assert ordered_plus.provenance["position_shared_coordinates"] is True
    assert ordered_plus.provenance["position_specific_coordinates"] is True
    assert (
        aligned_multiset_permuted_ordered.provenance[
            "aligned_multiset_block_preserved"
        ]
        is True
    )
    assert (
        aligned_multiset_permuted_ordered.provenance[
            "ordered_block_alignment_preserved"
        ]
        is False
    )
    ordered_width = 1 + 10 * len(CLUTRR_EDGE_RELATIONS)
    np.testing.assert_array_equal(
        aligned_multiset_permuted_ordered.values[:, ordered_width:],
        ordered_plus.values[:, ordered_width:],
    )
    assert not np.array_equal(
        aligned_multiset_permuted_ordered.values[:, 1:ordered_width],
        ordered_plus.values[:, 1:ordered_width],
    )
    first_unsupported = 1 + 3 * len(CLUTRR_EDGE_RELATIONS)
    np.testing.assert_array_equal(
        supported_prefix.values[:, :first_unsupported],
        ordered_plus.values[:, :first_unsupported],
    )
    np.testing.assert_array_equal(
        supported_prefix.values[:, ordered_width:],
        ordered_plus.values[:, ordered_width:],
    )
    assert np.count_nonzero(
        supported_prefix.values[:, first_unsupported:ordered_width]
    ) == 0
    assert supported_prefix.provenance["candidate_alignment_preserved"] is True
    assert supported_prefix.provenance["supported_absolute_positions"] == 3
    for chain_length in (2, 3):
        selected = rows["chain_length"].to_numpy() == chain_length
        assert {
            tuple(row)
            for row in aligned_multiset_permuted_ordered.values[
                selected, 1:ordered_width
            ]
        } == {
            tuple(row)
            for row in ordered_plus.values[selected, 1:ordered_width]
        }

    for aligned, null in (
        (transitions, transition_null),
        (ordered_plus, ordered_plus_null),
    ):
        np.testing.assert_array_equal(aligned.values[:, 0], null.values[:, 0])
        for chain_length in (2, 3):
            selected = rows["chain_length"].to_numpy() == chain_length
            assert {tuple(row) for row in aligned.values[selected, 1:]} == {
                tuple(row) for row in null.values[selected, 1:]
            }


def test_paired_additive_context_keeps_base_and_budget_matched() -> None:
    rows = []
    for label in range(len(CLUTRR_RELATIONS)):
        for index in range(8):
            rows.append(
                {
                    "row_id": f"short-{label}-{index}",
                    "label": label,
                    "chain_length": 2 + index % 2,
                }
            )
        for index in range(3):
            rows.append(
                {
                    "row_id": f"length4-{label}-{index}",
                    "label": label,
                    "chain_length": 4,
                }
            )
    training = pd.DataFrame.from_records(rows)
    common = {
        "training": training,
        "base_context_per_relation": 2,
        "max_base_context_per_relation": 4,
        "added_per_relation": 2,
        "base_chain_lengths": (2, 3),
        "split_seed": 73,
    }
    short = _paired_additive_context(
        **common,
        added_chain_lengths=(2, 3),
    )
    length4 = _paired_additive_context(
        **common,
        added_chain_lengths=(4,),
    )

    assert len(short) == len(length4) == len(CLUTRR_RELATIONS) * 4
    short_base = set(short.loc[short["context_role"] == "base", "row_id"])
    length4_base = set(length4.loc[length4["context_role"] == "base", "row_id"])
    assert short_base == length4_base
    assert len(short_base) == len(CLUTRR_RELATIONS) * 2
    assert set(short["chain_length"]) <= {2, 3}
    assert (
        length4[length4["chain_length"] == 4].groupby("label").size().eq(2).all()
    )

    larger = _paired_additive_context(
        **{**common, "base_context_per_relation": 4},
        added_chain_lengths=(4,),
    )
    assert set(length4["row_id"]) <= set(larger["row_id"])


def test_coordinate_support_counts_unseen_evaluation_columns() -> None:
    context = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    evaluation = np.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    result = coordinate_support(context, evaluation)
    assert result["context_active_columns"] == 2
    assert result["evaluation_active_columns"] == 3
    assert result["shared_active_columns"] == 2
    assert result["active_coordinate_recall"] == 2 / 3
    assert result["active_cell_coverage"] == 3 / 4
    assert result["fully_supported_row_fraction"] == 0.5


def test_registered_clutrr_plans_have_expected_cells() -> None:
    cpu = load_study_config(
        REPO_ROOT / "configs/studies/clutrr_symbolic_cpu_confirmation_v2.yaml"
    )
    gpu = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_gpu_tabpfn_confirmation_v2.yaml"
    )
    cpu_plan = build_plan(cpu)
    gpu_plan = build_plan(gpu)
    assert len(cpu_plan["cells"]) == 150
    assert (
        cpu_plan["plan_sha256"]
        == "cca4114229dbfac4938d707252c4257817149817becabb4b8e8b43a7bf86df98"
    )
    assert len(gpu_plan["cells"]) == 75
    assert (
        gpu_plan["plan_sha256"]
        == "d118695450af93907a347aab5760dfb6bb3067ed30ffe3255771722bfd348a4b"
    )


def test_clutrr_shared_coordinate_plans_have_expected_cells() -> None:
    cpu = load_study_config(
        REPO_ROOT / "configs/studies/clutrr_symbolic_cpu_shared_coordinates_v1.yaml"
    )
    gpu = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_gpu_tabpfn_shared_coordinates_v1.yaml"
    )
    cpu_plan = build_plan(cpu)
    gpu_plan = build_plan(gpu)
    assert len(cpu_plan["cells"]) == 300
    assert cpu_plan["plan_sha256"] == (
        "106789bc7c224b5ef6ac0c9d00fbc2b3ec64ac3c7741429f0e70866401dcd7f5"
    )
    assert len(gpu_plan["cells"]) == 60
    assert gpu_plan["plan_sha256"] == (
        "f911307adcd680acaeb25db68429fd8357c1b190544f6f318fa06b59f92229d5"
    )


def test_clutrr_ordered_interference_plans_are_frozen() -> None:
    cpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_cpu_ordered_interference_v1.yaml"
        )
    )
    gpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_gpu_tabpfn_ordered_interference_v1.yaml"
        )
    )
    assert len(cpu["cells"]) == 75
    assert cpu["plan_sha256"] == (
        "2a11bd24c71462a3c3de7b6589c896cd59a0e5d47df55f9bf6544bf77a79d059"
    )
    assert len(gpu["cells"]) == 15
    assert gpu["plan_sha256"] == (
        "2e63e3f9386c720542218f483b99b3e343715cf470486f2d77b4c382b5c398d4"
    )


def test_same_split_context23_plan_is_paired_to_train234_assets() -> None:
    context23 = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_cpu_same_split_context23_v1.yaml"
    )
    train234 = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_cpu_train234_sensitivity_v2.yaml"
    )
    context23_plan = build_plan(context23)
    train234_plan = build_plan(train234)
    assert len(context23_plan["cells"]) == len(train234_plan["cells"]) == 150
    assert (
        context23_plan["plan_sha256"]
        == "5a50eb1725852e08c8b08d35cebf6ebbd0542d14b62072ca28941376bd3fc402"
    )
    for context23_cell, train234_cell in zip(
        context23_plan["cells"], train234_plan["cells"], strict=True
    ):
        assert context23_cell["asset_files"] == train234_cell["asset_files"]
        assert context23_cell["evaluation_chain_lengths"] == train234_cell[
            "evaluation_chain_lengths"
        ]
        assert context23_cell["encoder"] == train234_cell["encoder"]
        assert context23_cell["decoder"] == train234_cell["decoder"]
        assert context23_cell["context_per_relation"] == train234_cell[
            "context_per_relation"
        ]
        assert context23_cell["seed"] == train234_cell["seed"]
        assert context23_cell["train_chain_lengths"] == [2, 3]
        assert train234_cell["train_chain_lengths"] == [2, 3, 4]


def test_same_split_tabpfn_plan_is_paired_to_train234_gpu_plan() -> None:
    context23 = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_gpu_tabpfn_same_split_context23_v1.yaml"
    )
    train234 = load_study_config(
        REPO_ROOT
        / "configs/studies/clutrr_symbolic_gpu_tabpfn_train234_sensitivity_v2.yaml"
    )
    context23_plan = build_plan(context23)
    train234_plan = build_plan(train234)
    assert len(context23_plan["cells"]) == len(train234_plan["cells"]) == 75
    assert (
        context23_plan["plan_sha256"]
        == "6a1a85e945989e9a9663f237ebd7960b9ed6cf8175d262c37e5b1a1c73944b29"
    )
    for context23_cell, train234_cell in zip(
        context23_plan["cells"], train234_plan["cells"], strict=True
    ):
        assert context23_cell["asset_files"] == train234_cell["asset_files"]
        assert context23_cell["model_identity"] == train234_cell["model_identity"]
        assert context23_cell["evaluation_chain_lengths"] == train234_cell[
            "evaluation_chain_lengths"
        ]
        assert context23_cell["encoder"] == train234_cell["encoder"]
        assert context23_cell["decoder"] == train234_cell["decoder"] == "tabpfn"
        assert context23_cell["context_per_relation"] == train234_cell[
            "context_per_relation"
        ]
        assert context23_cell["seed"] == train234_cell["seed"]
        assert context23_cell["train_chain_lengths"] == [2, 3]
        assert train234_cell["train_chain_lengths"] == [2, 3, 4]


def test_additive_context_plans_are_equal_budget_and_paired() -> None:
    configs = {
        "cpu_short": "clutrr_symbolic_cpu_additive_short_control_v1.yaml",
        "cpu_length4": "clutrr_symbolic_cpu_additive_length4_v1.yaml",
        "gpu_short": "clutrr_symbolic_gpu_tabpfn_additive_short_control_v1.yaml",
        "gpu_length4": "clutrr_symbolic_gpu_tabpfn_additive_length4_v1.yaml",
    }
    plans = {
        key: build_plan(load_study_config(REPO_ROOT / "configs/studies" / filename))
        for key, filename in configs.items()
    }
    assert len(plans["cpu_short"]["cells"]) == len(plans["cpu_length4"]["cells"]) == 60
    assert len(plans["gpu_short"]["cells"]) == len(plans["gpu_length4"]["cells"]) == 30
    assert plans["cpu_short"]["plan_sha256"] == (
        "73743334bcd52105aa6d76473bd349d6bcb15f2fb7925dcfb50ad58410dd27a4"
    )
    assert plans["cpu_length4"]["plan_sha256"] == (
        "4bc6ca6dddc8bdd3cc82b226b8f7484541e5d97a746a0819c665ceb90584daf1"
    )
    assert plans["gpu_short"]["plan_sha256"] == (
        "d81775aed237aa4727ff85f506b0bce1fc4d64eb2913d33a158b199627ce1df9"
    )
    assert plans["gpu_length4"]["plan_sha256"] == (
        "1ccb954df4de03414107508c961b22681a69ddbcdbfac3d7476078fa6af85088"
    )

    for resource in ("cpu", "gpu"):
        short_cells = plans[f"{resource}_short"]["cells"]
        length4_cells = plans[f"{resource}_length4"]["cells"]
        for short, length4 in zip(short_cells, length4_cells, strict=True):
            for key in (
                "asset_files",
                "encoder",
                "decoder",
                "seed",
                "base_context_per_relation",
                "context_per_relation",
                "max_base_context_per_relation",
                "max_context_per_relation",
            ):
                assert short[key] == length4[key]
            assert short["context_sampling"]["added_chain_lengths"] == [2, 3]
            assert length4["context_sampling"]["added_chain_lengths"] == [4]
            assert short["context_per_relation"] == (
                short["base_context_per_relation"] + 20
            )


def test_clutrr_logistic_cell_runs_end_to_end(tmp_path: Path) -> None:
    cell = _fixture_cell(tmp_path)
    output_dir = tmp_path / "output"
    payload = run_clutrr_cell(cell, output_dir, tmp_path)
    assert payload["status"] == "complete"
    assert payload["context_rows"] == 18
    assert payload["evaluation_rows"] == 18
    assert 0.0 <= payload["metrics"]["accuracy"] <= 1.0
    assert "collision_conditioned_ceiling" in payload["metrics"]
    assert "extraction_efficiency" in payload["metrics"]
    assert set(payload["metrics_by_chain_length"]) == {"2"}
    assert set(payload["collision_conditioned_ceiling_by_chain_length"]) == {"2"}
    assert (output_dir / "clutrr-test-cell.predictions.npz").is_file()
    plan = {
        "study_id": "clutrr-test",
        "plan_sha256": "fixture-plan",
        "cells": [cell],
    }
    manifest = build_shard_manifest(plan, 0, 0, 1, output_dir)
    atomic_write_json(output_dir / "shard_manifest.json", manifest)
    verification = verify_shard(output_dir)
    assert verification["cell_ids"] == ["clutrr-test-cell"]


def test_recursive_gru_plan_and_ordered_token_parser() -> None:
    config = load_study_config(
        REPO_ROOT / "configs/studies/clutrr_symbolic_cpu_recursive_gru_v1.yaml"
    )
    plan = build_plan(config)
    assert len(plan["cells"]) == 15
    assert (
        plan["plan_sha256"]
        == "bded7fc607c0fff7036f617599f721d3954d18f0d24df506dd94a2683bf686f5"
    )
    assert {cell["decoder"] for cell in plan["cells"]} == {"gru_sequence"}
    assert {cell["encoder"] for cell in plan["cells"]} == {"ordered_typed_path"}

    rows = pd.DataFrame(
        {
            "chain_length": [2, 3],
            "edge_types": [
                ("mother", "father"),
                ("son", "daughter", "sister"),
            ],
            "story_edges": [
                ((0, 1), (1, 2)),
                ((0, 1), (1, 2), (2, 3)),
            ],
            "query_edge": [(0, 2), (0, 3)],
        }
    )
    encoded = encode_clutrr_rows(rows, "ordered_typed_path", 101)
    tokens, lengths, relation_count = _ordered_path_tokens(encoded.values)
    assert lengths.tolist() == [2, 3]
    assert relation_count == 14
    assert np.all(tokens[0, 2:] == 0)
    assert np.all(tokens[1, 3:] == 0)


def test_supported_prefix_plans_have_frozen_counts_and_hashes() -> None:
    cpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_cpu_supported_prefix_v1.yaml"
        )
    )
    gpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_gpu_tabpfn_supported_prefix_v1.yaml"
        )
    )
    assert len(cpu["cells"]) == 30
    assert (
        cpu["plan_sha256"]
        == "b79695cce1226249e4e76b6173603744d05d40ddc9ad77f9544181d63eff219a"
    )
    assert len(gpu["cells"]) == 15
    assert (
        gpu["plan_sha256"]
        == "d79d377854011288e0a54e03d631391d886b0e421fbe9a7cc3a1608d27ef4c75"
    )

    shortcut_cpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_cpu_shortcut_diagnostic_v1.yaml"
        )
    )
    shortcut_gpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/clutrr_symbolic_gpu_tabpfn_shortcut_diagnostic_v1.yaml"
        )
    )
    assert len(shortcut_cpu["cells"]) == 24
    assert (
        shortcut_cpu["plan_sha256"]
        == "a8f7ceb2f7fa35cb0ab3c3550b8d63158c640facc6094592f2561a8ea70f505b"
    )
    assert len(shortcut_gpu["cells"]) == 12
    assert (
        shortcut_gpu["plan_sha256"]
        == "69e32a668814177079dcbc54f65c1fb8f91c2c6f3197bc0209ec88c1ceb96efc"
    )
