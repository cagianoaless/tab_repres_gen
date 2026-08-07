from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

TASK_FAMILY_SCHEMAS: Final = {
    "synthetic_relational": frozenset({1, 2}),
    "real_graph": frozenset({3}),
    "frozen_kg_interface": frozenset({4}),
    "rule_induction": frozenset({5}),
    "clutrr_symbolic": frozenset({6}),
    "compositional_length_shift": frozenset({7}),
    "graphlog_symbolic": frozenset({8}),
}
SUPPORTED_TASK_FAMILIES: Final = frozenset(TASK_FAMILY_SCHEMAS)

CONTROL_ENCODERS: Final = frozenset(
    {
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
        "matched_noise",
    }
)

POINTWISE_ENCODERS: Final = (
    "categorical_ordinal",
    "role_onehot",
    "unordered_onehot",
    "shared_embedding_symmetric",
    "shared_embedding_directed",
    "embedding_diff",
)

PROPERTY_ALIGNED_ENCODER: Final = {
    "reflexivity": "dyadic_invariants",
    "symmetry": "dyadic_invariants",
    "antisymmetry": "local_roles",
    "transitivity": "typed_paths",
    "typed_composition_l2": "typed_paths",
    "typed_composition_l3": "typed_paths",
    "local_structural_roles": "local_roles",
    "block_affinity_homophilic": "community_global",
    "block_affinity_heterophilic": "community_global",
    "discovery_reciprocity": "reciprocity_evidence_bank",
    "discovery_typed_composition": "composition_evidence_bank",
    "discovery_structural_roles": "role_evidence_bank",
    "discovery_block_affinity": "topology_evidence_bank",
}

DISCOVERY_COMMON_LADDER: Final = (
    "query_relation_only",
    "categorical_ordinal",
    "query_dyadic_invariants",
    "query_local_roles",
)

PROPERTY_ENCODERS: Final = {
    "reflexivity": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "symmetry": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "antisymmetry": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "transitivity": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "graph_reachability",
        "typed_paths",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "typed_composition_l2": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "graph_reachability",
        "typed_path_k2",
        "typed_path_k3",
        "typed_paths",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "typed_composition_l3": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "graph_reachability",
        "typed_path_k2",
        "typed_path_k3",
        "typed_paths",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "local_structural_roles": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "block_affinity_homophilic": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "community_global",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "block_affinity_heterophilic": (
        "artifact_ids",
        *POINTWISE_ENCODERS,
        "dyadic_invariants",
        "local_roles",
        "community_global",
        "aligned_permutation",
        "dimension_matched_noise",
    ),
    "discovery_reciprocity": (
        *DISCOVERY_COMMON_LADDER,
        "role_onehot",
        "shared_embedding_directed",
        "reciprocity_evidence_bank",
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
    ),
    "discovery_typed_composition": (
        *DISCOVERY_COMMON_LADDER,
        "role_onehot",
        "shared_embedding_directed",
        "query_graph_reachability",
        "query_typed_path_k2",
        "query_typed_path_k3",
        "composition_evidence_bank",
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
    ),
    "discovery_structural_roles": (
        *DISCOVERY_COMMON_LADDER,
        "role_onehot",
        "shared_embedding_directed",
        "role_evidence_bank",
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
    ),
    "discovery_block_affinity": (
        *DISCOVERY_COMMON_LADDER,
        "unordered_onehot",
        "shared_embedding_symmetric",
        "topology_evidence_bank",
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
    ),
}

SUPPORTED_PROPERTIES: Final = frozenset(PROPERTY_ENCODERS)
PROPERTY_MACRO_GRAPH_FAMILY: Final = {
    "reflexivity": "identity_control",
    "symmetry": "directed_reciprocal",
    "antisymmetry": "directed_bipartite",
    "transitivity": "directed_chain",
    "typed_composition_l2": "directed_typed_layered",
    "typed_composition_l3": "directed_typed_layered",
    "local_structural_roles": "directed_role_structured",
    "block_affinity_homophilic": "undirected_reciprocal_block",
    "block_affinity_heterophilic": "undirected_reciprocal_block",
    "discovery_reciprocity": "directed_multi_relation",
    "discovery_typed_composition": "directed_typed_layered",
    "discovery_structural_roles": "directed_role_structured",
    "discovery_block_affinity": "undirected_reciprocal_block",
}
SUPPORTED_ENCODERS: Final = frozenset(
    encoder for encoders in PROPERTY_ENCODERS.values() for encoder in encoders
)
SUPPORTED_DECODERS: Final = frozenset(
    {
        "logistic",
        "logistic_nominal",
        "logistic_tuned",
        "catboost",
        "catboost_nominal",
        "catboost_depth1",
        "catboost_tuned",
        "mlp",
        "gru_sequence",
        "tabpfn",
        "limix2m",
    }
)
SUPPORTED_REAL_DECODERS: Final = SUPPORTED_DECODERS | {"tabm"}
SUPPORTED_FROZEN_KG_DECODERS: Final = SUPPORTED_DECODERS | {"tabicl"}
REAL_BASE_ENCODERS: Final = frozenset({"artifact_ids", "local_roles", "typed_paths"})
REAL_ALIGNED_ENCODERS: Final = frozenset({"local_roles", "typed_paths"})
REAL_CONTROL_ENCODERS: Final = frozenset(
    {
        "aligned_permutation",
        "split_relation_permutation",
        "dimension_matched_noise",
    }
)
SUPPORTED_REAL_ENCODERS: Final = REAL_BASE_ENCODERS | REAL_CONTROL_ENCODERS
SUPPORTED_NOISE_REGIMES: Final = frozenset(
    {"none", "disconnected", "endpoint_attached", "support_corrupting"}
)

FROZEN_KG_ASSET_ID: Final = "rfmlink_frozen_interface_q64_v1"
FROZEN_KG_ASSET_ROOT: Final = "data/frozen/rfmlink_frozen_interface_q64_v1"
FROZEN_KG_MANIFEST_SHA256: Final = (
    "14066c1b2e8cd2db90e8fceec3d0522cedd05c2f125e7c07afd7bb02bff325aa"
)
FROZEN_KG_DATASETS: Final = {
    "codex_small": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "08f853419c08024819ee7353b2c24185c391d70a19ee78b6275397c62fe2ad23"
        ),
    },
    "nell_ind_v1": {
        "transfer_regime": "inductive_entities",
        "archive_sha256": (
            "5c7fa806015d84da3c4002df3e299558095f86e0291c5fec9594b4cfb47eb9b8"
        ),
    },
    "metafam": {
        "transfer_regime": "inductive_entities_relations",
        "archive_sha256": (
            "da36f36478d570b6c617fb9fed9e29622f02cd417c83bc12ae2ab92e6eed0568"
        ),
    },
}
FROZEN_KG_SOURCE_PANEL_ASSET_ID: Final = "rfmlink_source_panel_q64_v1"
FROZEN_KG_SOURCE_PANEL_ASSET_ROOT: Final = (
    "data/runtime/rfmlink_source_panel_q64_v1"
)
FROZEN_KG_SOURCE_PANEL_MANIFEST_SHA256: Final = (
    "912e59a9b8ccbef2678cb761f4d563731246087540f2d8799e6a86498368d23b"
)
FROZEN_KG_SOURCE_PANEL_DATASETS: Final = {
    "aristov4": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "4afd756203535e857fed5b2e6234b8434f15eadcbb1ff11380d7580c9f49d0f9"
        ),
    },
    "conceptnet100k": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "dbf55b477b5120833cb23a1913a5c25780f1b00806865f8f608a2cfb201b7ad3"
        ),
    },
    "dbpedia100k": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "63f308e0819b771ba302e892b8c0e71c541bbf1c9829843335def147aed0f2b1"
        ),
    },
    "fb15k237_ind_v4": {
        "transfer_regime": "inductive_entities",
        "archive_sha256": (
            "b0fc31be642816be25d1343888bca24d20385924bda926b51655f93fd1745a77"
        ),
    },
    "fbnell": {
        "transfer_regime": "inductive_entities_relations",
        "archive_sha256": (
            "2506031b1c406e503ca8f050b7c48393564855d66a1101cc433d6878fcfa5cb4"
        ),
    },
    "hetionet": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "7ab3e93bb6571ed2c6718fea1f0b6df2b9558404b7af6717cd8e042643cf65d7"
        ),
    },
    "hm_3k": {
        "transfer_regime": "inductive_entities",
        "archive_sha256": (
            "4314ec9014a95147226705755a820919533910f67d95db15815e775cb6c9c68c"
        ),
    },
    "ilpc2022_large": {
        "transfer_regime": "inductive_entities",
        "archive_sha256": (
            "3f1dc41038023358a9bfb59c0132e0be243847a9d00966d31868cd439c16d783"
        ),
    },
    "wiki_mt1_tax": {
        "transfer_regime": "inductive_entities_relations",
        "archive_sha256": (
            "cf3f64ed39564e53c4ccdcfb388227709ad8f8fa643b6e85141f429ea7469ed7"
        ),
    },
    "wk_ingram_50": {
        "transfer_regime": "inductive_entities_relations",
        "archive_sha256": (
            "2eed8437c5e02cd5d83e9cb1a279aeb492489174a57c9cfe90110d98063c2d3d"
        ),
    },
    "wn18rr_ind_v3": {
        "transfer_regime": "inductive_entities",
        "archive_sha256": (
            "358c318f84487ae57358743e66ca139dbe5803b37f79894db7a61f3d4ad4a697"
        ),
    },
    "yago310": {
        "transfer_regime": "transductive",
        "archive_sha256": (
            "01ced5cf2a9eef80621c658a0b86801e05f5464179752bd8b05a9056cb1a4cd2"
        ),
    },
}
FROZEN_KG_INTERFACES: Final = (
    "ultra_score_only",
    "b0_local_only",
    "b0_local_no_relation_id",
    "b0_local_ultra",
    "b0_local_ultra_no_relation_id",
    "b2v2_structure_only",
    "b2v2_full",
    "b2v2_permuted_full",
    # Information-ladder rungs mirroring the rule_induction encoder ladder on the
    # KG interface: an identity control (random per-entity scalar tokens, the
    # random_entity_tokens analog -- requires archive schema 2 for the per-row
    # entity identities), an adjacency-only rung (dyadic_premises analog), a
    # path-evidence-only rung (bounded_path_evidence analog), and the spectral
    # features alone. ultra_score_only already serves as the trust-the-teacher
    # top rung, so no separate oracle interface exists.
    "entity_tokens_only",
    "b0_dyadic_only",
    "b0_paths_only",
    "b2v2_only",
    "graphlet_only",
    "b0_local_graphlet",
    "b0_local_graphlet_ultra",
)
FROZEN_KG_ULTRA57_ASSET_ID: Final = "rfmlink_frozen_interface_ultra57_v2"
FROZEN_KG_ULTRA57_HARD_CONTEXT_ASSET_ID: Final = (
    "rfmlink_frozen_interface_ultra57_hard_context_v1"
)
FROZEN_KG_GRAPHLET_CONTEXT_ASSET_ID: Final = "rfmlink_graphlet_context_pilot_v1"
FROZEN_KG_RELATION_CONTEXT_ASSET_ID: Final = "rfmlink_ultra57_relation_context_v1"
FROZEN_KG_NEGATIVE_ROBUSTNESS_ASSET_ID: Final = (
    "rfmlink_ultra4g_negative_robustness_v1"
)
FROZEN_KG_ASSETS: Final = {
    FROZEN_KG_ASSET_ID: {
        "asset_root": FROZEN_KG_ASSET_ROOT,
        "manifest_sha256": FROZEN_KG_MANIFEST_SHA256,
        "datasets": FROZEN_KG_DATASETS,
        "query_triples": 64,
        "contexts_per_class": (20, 50, 100, 200, 400),
        "allowed_statuses": ("development",),
    },
    FROZEN_KG_SOURCE_PANEL_ASSET_ID: {
        "asset_root": FROZEN_KG_SOURCE_PANEL_ASSET_ROOT,
        "manifest_sha256": FROZEN_KG_SOURCE_PANEL_MANIFEST_SHA256,
        "datasets": FROZEN_KG_SOURCE_PANEL_DATASETS,
        "query_triples": 64,
        "contexts_per_class": (20, 50, 100, 200, 400),
        "allowed_statuses": ("development",),
    },
}


def frozen_kg_asset(asset_id: str) -> dict:
    if asset_id in {
        FROZEN_KG_ULTRA57_ASSET_ID,
        FROZEN_KG_ULTRA57_HARD_CONTEXT_ASSET_ID,
        FROZEN_KG_GRAPHLET_CONTEXT_ASSET_ID,
        FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
        FROZEN_KG_NEGATIVE_ROBUSTNESS_ASSET_ID,
    }:
        override = os.environ.get("RTFM_FROZEN_KG_ASSET_ROOT")
        if not override:
            raise ValueError(
                "RTFM_FROZEN_KG_ASSET_ROOT is required for a runtime-registered "
                "ULTRA57 asset"
            )
        root = Path(override).expanduser().resolve()
        manifest_path = root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("asset_id") != asset_id:
            raise ValueError("the runtime ULTRA57 manifest has the wrong asset ID")
        datasets = {
            name: {
                "transfer_regime": item["transfer_regime"],
                "archive_sha256": item["archive_sha256"],
            }
            for name, item in manifest["datasets"].items()
        }
        return {
            "asset_root": str(root),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "datasets": datasets,
            "query_triples": 64,
            "contexts_per_class": (
                (200,)
                if asset_id
                in {
                    FROZEN_KG_ULTRA57_HARD_CONTEXT_ASSET_ID,
                    FROZEN_KG_GRAPHLET_CONTEXT_ASSET_ID,
                    FROZEN_KG_RELATION_CONTEXT_ASSET_ID,
                    FROZEN_KG_NEGATIVE_ROBUSTNESS_ASSET_ID,
                }
                else (20, 50, 100, 200)
            ),
            "context_modes": tuple(
                manifest.get("context_modes", ("original_uniform",))
            ),
            "allowed_statuses": ("development",),
        }
    try:
        return FROZEN_KG_ASSETS[asset_id]
    except KeyError as error:
        raise ValueError(f"unsupported frozen KG asset: {asset_id}") from error


# Minimum base-node count needed to draw 400 context and 400 evaluation rows per class.
MIN_GRAPH_SIZE: Final = {
    "reflexivity": 800,
    "symmetry": 60,
    "antisymmetry": 64,
    "transitivity": 44,
    "typed_composition_l2": 120,
    "typed_composition_l3": 160,
    "local_structural_roles": 64,
    "block_affinity_homophilic": 64,
    "block_affinity_heterophilic": 64,
    "discovery_reciprocity": 256,
    "discovery_typed_composition": 288,
    "discovery_structural_roles": 256,
    "discovery_block_affinity": 256,
}


def aligned_encoder(property_name: str) -> str:
    try:
        return PROPERTY_ALIGNED_ENCODER[property_name]
    except KeyError as error:
        raise ValueError(f"unsupported controlled property: {property_name}") from error


def validate_property_encoder(property_name: str, encoder: str) -> None:
    try:
        allowed = PROPERTY_ENCODERS[property_name]
    except KeyError as error:
        raise ValueError(f"unsupported controlled property: {property_name}") from error
    if encoder not in allowed:
        raise ValueError(
            f"encoder {encoder!r} is not registered for {property_name}; "
            f"allowed={list(allowed)}"
        )


def macro_graph_family(property_name: str) -> str:
    try:
        return PROPERTY_MACRO_GRAPH_FAMILY[property_name]
    except KeyError as error:
        raise ValueError(f"unsupported controlled property: {property_name}") from error
