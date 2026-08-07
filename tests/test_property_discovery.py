from pathlib import Path

import numpy as np
import pytest

from relational_tfm.config import load_study_config
from relational_tfm.datasets.synthetic import (
    discovery_latent_labels,
    generate_synthetic_task,
)
from relational_tfm.datasets.synthetic_audit import audit_synthetic_task
from relational_tfm.encoders.synthetic import encode_synthetic_rows
from relational_tfm.plan import build_plan

CASES = (
    ("discovery_reciprocity", "reciprocity_evidence_bank", 256),
    ("discovery_typed_composition", "composition_evidence_bank", 288),
    ("discovery_structural_roles", "role_evidence_bank", 256),
    ("discovery_block_affinity", "topology_evidence_bank", 256),
)

LADDER_ENCODERS = {
    "discovery_reciprocity": (
        "query_relation_only",
        "categorical_ordinal",
        "role_onehot",
        "shared_embedding_directed",
        "query_dyadic_invariants",
        "query_local_roles",
    ),
    "discovery_typed_composition": (
        "query_relation_only",
        "categorical_ordinal",
        "role_onehot",
        "shared_embedding_directed",
        "query_dyadic_invariants",
        "query_local_roles",
        "query_graph_reachability",
        "query_typed_path_k2",
        "query_typed_path_k3",
    ),
    "discovery_structural_roles": (
        "query_relation_only",
        "categorical_ordinal",
        "role_onehot",
        "shared_embedding_directed",
        "query_dyadic_invariants",
        "query_local_roles",
    ),
    "discovery_block_affinity": (
        "query_relation_only",
        "categorical_ordinal",
        "unordered_onehot",
        "shared_embedding_symmetric",
        "query_dyadic_invariants",
        "query_local_roles",
    ),
}


def test_discovery_latent_labels_recover_pre_flip_partition() -> None:
    query_ids = np.asarray([0, 7, 799, 800, 801, 1599], dtype=np.int64)
    np.testing.assert_array_equal(
        discovery_latent_labels(query_ids, positive_required=800),
        np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8),
    )


@pytest.mark.parametrize(("property_name", "encoder", "graph_size"), CASES)
def test_discovery_tasks_pass_independent_audit_without_target_columns(
    property_name: str, encoder: str, graph_size: int
) -> None:
    task = generate_synthetic_task(
        property_name,
        graph_size,
        20,
        "none",
        9101,
        max_context_per_class=400,
        evaluation_per_class=400,
    )
    audit = audit_synthetic_task(task)
    assert audit["oracle_mismatches"] == 0
    assert task.context["label"].value_counts().to_dict() == {0: 20, 1: 20}
    assert task.evaluation["label"].value_counts().to_dict() == {0: 400, 1: 400}
    assert set(task.context["relation"]) == {"q0", "q1", "q2", "q3"}

    encoded = encode_synthetic_rows(
        task.support_graph,
        task.population,
        encoder,
        9101,
        property_name=property_name,
        control_population=task.population,
    )
    labels = task.population["label"].to_numpy(dtype=np.float64)
    assert encoded.values.shape[0] == len(labels)
    assert not any(
        np.array_equal(encoded.values[:, column], labels)
        or np.array_equal(encoded.values[:, column], 1.0 - labels)
        for column in range(encoded.values.shape[1])
    )
    assert "latent" not in " ".join(encoded.feature_names)
    assert "community" not in " ".join(encoded.feature_names)


@pytest.mark.parametrize(("property_name", "encoder", "graph_size"), CASES)
def test_discovery_controls_are_dimension_matched_and_deterministic(
    property_name: str, encoder: str, graph_size: int
) -> None:
    task = generate_synthetic_task(
        property_name,
        graph_size,
        20,
        "none",
        9101,
        max_context_per_class=400,
        evaluation_per_class=400,
    )
    aligned = encode_synthetic_rows(
        task.support_graph,
        task.population,
        encoder,
        9101,
        property_name=property_name,
        control_population=task.population,
    )
    controls = []
    for control in ("aligned_permutation", "dimension_matched_noise"):
        first = encode_synthetic_rows(
            task.support_graph,
            task.population,
            control,
            9101,
            property_name=property_name,
            control_population=task.population,
        )
        second = encode_synthetic_rows(
            task.support_graph,
            task.population,
            control,
            9101,
            property_name=property_name,
            control_population=task.population,
        )
        assert first.values.shape == aligned.values.shape
        np.testing.assert_array_equal(first.values, second.values)
        controls.append(first.values)
    assert not np.array_equal(controls[0], aligned.values)
    assert not np.array_equal(controls[1], aligned.values)


@pytest.mark.parametrize(
    ("property_name", "encoder", "graph_size"),
    [
        (property_name, encoder, graph_size)
        for property_name, _, graph_size in CASES
        for encoder in LADDER_ENCODERS[property_name]
    ],
)
def test_discovery_ladder_encoders_are_non_oracle_and_deterministic(
    property_name: str, encoder: str, graph_size: int
) -> None:
    task = generate_synthetic_task(
        property_name,
        graph_size,
        20,
        "none",
        12001,
        max_context_per_class=400,
        evaluation_per_class=400,
    )
    first = encode_synthetic_rows(
        task.support_graph,
        task.population,
        encoder,
        412001,
        property_name=property_name,
        control_population=task.population,
    )
    second = encode_synthetic_rows(
        task.support_graph,
        task.population,
        encoder,
        412001,
        property_name=property_name,
        control_population=task.population,
    )
    labels = task.population["label"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.values.shape[0] == len(labels)
    assert not any(
        np.array_equal(first.values[:, column], labels)
        or np.array_equal(first.values[:, column], 1.0 - labels)
        for column in range(first.values.shape[1])
    )
    assert "latent" not in " ".join(first.feature_names)
    assert "community" not in " ".join(first.feature_names)


def _paired_identity(cell: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        cell[key]
        for key in (
            "property",
            "encoder",
            "context_per_class",
            "graph_size",
            "noise_regime",
            "seed",
            "graph_seed",
            "relabel_seed",
            "split_seed",
            "noise_seed",
            "encoder_seed",
        )
    )


@pytest.mark.parametrize(
    ("cpu_name", "gpu_name", "cpu_cells", "gpu_cells", "cpu_hash", "gpu_hash"),
    [
        (
            "property_discovery_cpu_confirmation.yaml",
            "property_discovery_gpu_tabpfn_confirmation.yaml",
            1200,
            600,
            "65ee7370cb78eb9d1520517701d4fdd58a33d596d4b5661bfbb15698bc8bc322",
            "b7300253ac197a9ac25e3025d1b79b81b8c22b685db868e7e3830b19f5ac0bc4",
        ),
        (
            "property_discovery_encoder_ladder_cpu_confirmation.yaml",
            "property_discovery_encoder_ladder_gpu_tabpfn_confirmation.yaml",
            1080,
            540,
            "8292b28fed602b53803c8b41405aa06f3adfa14b80ed1711ef1bbe297463b520",
            "40ffdc076cedda3ea643d04213ad4329c60def1681834746a9405974ca6236ba",
        ),
    ],
)
def test_property_discovery_confirmations_are_frozen_decoder_substitutions(
    cpu_name: str,
    gpu_name: str,
    cpu_cells: int,
    gpu_cells: int,
    cpu_hash: str,
    gpu_hash: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs/studies"
    cpu = build_plan(load_study_config(config_root / cpu_name))
    gpu = build_plan(load_study_config(config_root / gpu_name))
    assert len(cpu["cells"]) == cpu_cells
    assert len(gpu["cells"]) == gpu_cells
    assert cpu["plan_sha256"] == cpu_hash
    assert gpu["plan_sha256"] == gpu_hash
    assert {_paired_identity(cell) for cell in cpu["cells"]} == {
        _paired_identity(cell) for cell in gpu["cells"]
    }
    assert {cell["decoder"] for cell in cpu["cells"]} == {"logistic", "catboost"}
    assert {cell["decoder"] for cell in gpu["cells"]} == {"tabpfn"}
