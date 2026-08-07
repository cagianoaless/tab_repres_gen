from __future__ import annotations

from pathlib import Path

import numpy as np

from relational_tfm.config import load_study_config
from relational_tfm.datasets.compositional_shift import (
    COMPOSITION_CLASS_COUNT,
    compose_tokens,
    generate_compositional_shift_task,
    inverse_element,
    multiply_elements,
)
from relational_tfm.datasets.compositional_shift_audit import (
    audit_compositional_shift_task,
)
from relational_tfm.encoders.compositional_shift import (
    COMPOSITION_CLASS_COUNT as ENCODER_CLASS_COUNT,
)
from relational_tfm.encoders.compositional_shift import (
    encode_compositional_shift_rows,
)
from relational_tfm.plan import build_plan
from relational_tfm.runner import run_compositional_shift_cell

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cell(
    context: int = 20, encoder: str = "coarse_relation_multiset"
) -> dict[str, object]:
    return {
        "index": 0,
        "cell_id": "d8-test-cell",
        "study_id": "d8-test",
        "task_family": "compositional_length_shift",
        "algebra": "dihedral_d8",
        "train_chain_lengths": [2, 3],
        "evaluation_chain_lengths": list(range(4, 11)),
        "max_path_length": 10,
        "evaluation_per_class_per_length": 4,
        "max_context_per_class": 40,
        "encoder": encoder,
        "decoder": "logistic",
        "context_per_class": context,
        "seed": 701,
        "graph_seed": 701,
        "relabel_seed": 100_701,
        "split_seed": 200_701,
        "noise_seed": 300_701,
        "encoder_seed": 400_701,
        "decoder_seed": 500_701,
    }


def test_dihedral_group_oracle_is_closed_associative_and_invertible() -> None:
    assert ENCODER_CLASS_COUNT == COMPOSITION_CLASS_COUNT
    for left in range(COMPOSITION_CLASS_COUNT):
        assert multiply_elements(0, left) == left
        assert multiply_elements(left, 0) == left
        inverse = inverse_element(left)
        assert multiply_elements(left, inverse) == 0
        assert multiply_elements(inverse, left) == 0
        for right in range(COMPOSITION_CLASS_COUNT):
            assert 0 <= multiply_elements(left, right) < COMPOSITION_CLASS_COUNT
            for third in range(COMPOSITION_CLASS_COUNT):
                assert multiply_elements(
                    multiply_elements(left, right), third
                ) == multiply_elements(left, multiply_elements(right, third))


def test_generated_task_is_balanced_nested_disjoint_and_oracle_valid() -> None:
    small = generate_compositional_shift_task(_cell(context=20))
    large = generate_compositional_shift_task(_cell(context=40))
    assert audit_compositional_shift_task(small)["status"] == "PASS"
    assert audit_compositional_shift_task(large)["status"] == "PASS"
    assert set(small.context["row_id"]) < set(large.context["row_id"])
    assert set(small.context["chain_length"]) == {2, 3}
    assert set(small.evaluation["chain_length"]) == set(range(4, 11))
    assert len(small.context) == 20 * COMPOSITION_CLASS_COUNT
    assert len(small.evaluation) == 4 * COMPOSITION_CLASS_COUNT * 7
    for row in small.evaluation.itertuples(index=False):
        assert compose_tokens(tuple(row.relation_tokens)) == int(row.label)


def test_blockwise_null_preserves_shared_block_and_dimensions() -> None:
    task = generate_compositional_shift_task(_cell())
    aligned = encode_compositional_shift_rows(
        task.evaluation, "ordered_plus_coarse_multiset", 17
    )
    null = encode_compositional_shift_rows(
        task.evaluation,
        "aligned_coarse_multiset_with_ordered_block_permutation",
        17,
    )
    assert aligned.values.shape == null.values.shape
    ordered_width = 1 + 10 * COMPOSITION_CLASS_COUNT
    np.testing.assert_array_equal(
        aligned.values[:, ordered_width:],
        null.values[:, ordered_width:],
    )
    assert not np.array_equal(
        aligned.values[:, 1:ordered_width],
        null.values[:, 1:ordered_width],
    )
    for chain_length in range(4, 11):
        selected = task.evaluation["chain_length"].to_numpy() == chain_length
        assert {
            tuple(row) for row in aligned.values[selected, 1:ordered_width]
        } == {tuple(row) for row in null.values[selected, 1:ordered_width]}


def test_encoder_is_invariant_to_entity_relabeling_and_labels() -> None:
    task = generate_compositional_shift_task(_cell())
    encoded = encode_compositional_shift_rows(
        task.evaluation, "ordered_typed_path", 23
    )
    changed = task.evaluation.copy()
    changed["label"] = np.roll(changed["label"].to_numpy(), 1)
    changed["story_edges"] = [
        tuple((head + 9_999_991, tail + 9_999_991) for head, tail in edges)
        for edges in changed["story_edges"]
    ]
    changed["query_edge"] = [
        (head + 9_999_991, tail + 9_999_991)
        for head, tail in changed["query_edge"]
    ]
    reencoded = encode_compositional_shift_rows(
        changed, "ordered_typed_path", 23
    )
    np.testing.assert_array_equal(encoded.values, reencoded.values)


def test_compositional_plans_have_frozen_cell_counts() -> None:
    cpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/dihedral16_cpu_confirmation_v1.yaml"
        )
    )
    gpu = build_plan(
        load_study_config(
            REPO_ROOT
            / "configs/studies/dihedral16_gpu_tabpfn_confirmation_v1.yaml"
        )
    )
    assert len(cpu["cells"]) == 600
    assert cpu["plan_sha256"] == (
        "91558079d61c7006ced18d1885d074fd2de31ac221bfdcd2c96c1591137c0d55"
    )
    assert len(gpu["cells"]) == 300
    assert gpu["plan_sha256"] == (
        "cec127de036bca92f4f710e53b429d72431c1c1c83f3509edfb436d9547085d1"
    )


def test_logistic_cell_runs_end_to_end(tmp_path: Path) -> None:
    payload = run_compositional_shift_cell(_cell(), tmp_path)
    assert payload["status"] == "complete"
    assert payload["audit"]["status"] == "PASS"
    assert payload["context_rows"] == 20 * COMPOSITION_CLASS_COUNT
    assert payload["evaluation_rows"] == 4 * COMPOSITION_CLASS_COUNT * 7
    assert set(payload["metrics_by_chain_length"]) == {
        str(length) for length in range(4, 11)
    }
