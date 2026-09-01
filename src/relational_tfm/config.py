from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from relational_tfm.datasets.real import REAL_DATASET_ADAPTERS
from relational_tfm.datasets.rule_induction import SUPPORTED_RULES
from relational_tfm.decoders.classification import (
    frozen_tabpfn_identity,
    tabicl_frozen_identity,
    tabm_training_identity,
)
from relational_tfm.decoders.limix import frozen_model_identity as limix_frozen_model_identity
from relational_tfm.encoders.clutrr import CLUTRR_ENCODERS, CLUTRR_MAX_PATH_LENGTH
from relational_tfm.encoders.compositional_shift import (
    COMPOSITION_MAX_PATH_LENGTH,
    COMPOSITION_SHIFT_ENCODERS,
)
from relational_tfm.encoders.graphlog import (
    GRAPHLOG_ENCODERS,
    GRAPHLOG_MAX_PATH_LENGTH,
)
from relational_tfm.encoders.rule_induction import RULE_INDUCTION_ENCODERS
from relational_tfm.evaluation.query_relative import QUERY_TRANSFORMS
from relational_tfm.protocol import (
    FROZEN_KG_INTERFACES,
    MIN_GRAPH_SIZE,
    REAL_ALIGNED_ENCODERS,
    SUPPORTED_DECODERS,
    SUPPORTED_FROZEN_KG_DECODERS,
    SUPPORTED_NOISE_REGIMES,
    SUPPORTED_PROPERTIES,
    SUPPORTED_REAL_DECODERS,
    SUPPORTED_REAL_ENCODERS,
    SUPPORTED_TASK_FAMILIES,
    TASK_FAMILY_SCHEMAS,
    frozen_kg_asset,
    macro_graph_family,
    validate_property_encoder,
)


class ConfigError(ValueError):
    """Raised when a study configuration violates the frozen schema."""


def _nonempty_unique_strings(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"{key} must contain non-empty strings")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise ConfigError(f"{key} must not contain duplicates")
    return normalized


def _nonempty_unique_positive_ints(mapping: dict[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ConfigError(f"{key} must contain positive integers")
    normalized = tuple(value)
    if len(normalized) != len(set(normalized)):
        raise ConfigError(f"{key} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class PropertySpec:
    property_name: str
    encoders: tuple[str, ...]
    graph_sizes: tuple[int, ...]
    noise_regimes: tuple[str, ...]
    macro_graph_family: str | None = None


@dataclass(frozen=True)
class RealDatasetSpec:
    dataset_name: str
    asset_id: str
    macro_graph_family: str
    aligned_encoder: str
    encoders: tuple[str, ...]


@dataclass(frozen=True)
class FrozenKGDatasetSpec:
    dataset_name: str
    transfer_regime: str
    archive_sha256: str
    interfaces: tuple[str, ...]


@dataclass(frozen=True)
class RuleInductionSpec:
    rule_name: str
    encoders: tuple[str, ...]


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    study_id: str
    status: str
    task_family: str
    properties: tuple[str, ...]
    encoders: tuple[str, ...]
    decoders: tuple[str, ...]
    contexts_per_class: tuple[int, ...]
    context_negative_ratios: tuple[int, ...]
    graph_sizes: tuple[int, ...]
    noise_regimes: tuple[str, ...]
    seeds: tuple[int, ...]
    max_context_per_class: int | None
    evaluation_per_class: int | None
    evaluation_negative_ratio: int
    ceiling_seeds: tuple[int, ...]
    property_specs: tuple[PropertySpec, ...]
    real_dataset_specs: tuple[RealDatasetSpec, ...]
    frozen_kg_dataset_specs: tuple[FrozenKGDatasetSpec, ...]
    rule_specs: tuple[RuleInductionSpec, ...]
    resource_class: str | None
    model_identity: dict[str, Any] | None
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> StudyConfig:
        if not isinstance(mapping, dict):
            raise ConfigError("configuration root must be a mapping")
        schema_version = mapping.get("schema_version")
        if schema_version not in {1, 2, 3, 4, 5, 6, 7, 8}:
            raise ConfigError(
                "schema_version must be 1, 2, 3, 4, 5, 6, 7, or 8"
            )

        study_id = mapping.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ConfigError("study_id must be a non-empty string")

        status = mapping.get("status")
        if status not in {"smoke", "development", "confirmation"}:
            raise ConfigError("status must be smoke, development, or confirmation")

        task_family = mapping.get("task_family")
        if task_family not in SUPPORTED_TASK_FAMILIES:
            raise ConfigError(f"unsupported task_family: {task_family!r}")
        if schema_version not in TASK_FAMILY_SCHEMAS[task_family]:
            raise ConfigError(
                f"task_family {task_family!r} does not support schema_version {schema_version}"
            )

        if schema_version == 1:
            graph = mapping.get("graph")
            if not isinstance(graph, dict):
                raise ConfigError("graph must be a mapping")
            return cls(
                schema_version=1,
                study_id=study_id.strip(),
                status=status,
                task_family=task_family,
                properties=_nonempty_unique_strings(mapping, "properties"),
                encoders=_nonempty_unique_strings(mapping, "encoders"),
                decoders=_nonempty_unique_strings(mapping, "decoders"),
                contexts_per_class=_nonempty_unique_positive_ints(
                    mapping, "contexts_per_class"
                ),
                context_negative_ratios=(1,),
                graph_sizes=_nonempty_unique_positive_ints(graph, "sizes"),
                noise_regimes=_nonempty_unique_strings(graph, "noise_regimes"),
                seeds=_nonempty_unique_positive_ints(mapping, "seeds"),
                max_context_per_class=None,
                evaluation_per_class=None,
                evaluation_negative_ratio=1,
                ceiling_seeds=(),
                property_specs=(),
                rule_specs=(),
                real_dataset_specs=(),
                frozen_kg_dataset_specs=(),
                resource_class=None,
                model_identity=None,
                raw=mapping,
            )

        if schema_version == 3:
            return cls._from_real_mapping(mapping, study_id.strip(), status, task_family)
        if schema_version == 4:
            return cls._from_frozen_kg_mapping(
                mapping, study_id.strip(), status, task_family
            )
        if schema_version == 5:
            return cls._from_rule_induction_mapping(
                mapping, study_id.strip(), status, task_family
            )
        if schema_version == 6:
            return cls._from_clutrr_mapping(
                mapping, study_id.strip(), status, task_family
            )
        if schema_version == 7:
            return cls._from_compositional_shift_mapping(
                mapping, study_id.strip(), status, task_family
            )
        if schema_version == 8:
            return cls._from_graphlog_mapping(
                mapping, study_id.strip(), status, task_family
            )
        if task_family != "synthetic_relational":
            raise ConfigError("schema_version 2 currently supports synthetic_relational only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= SUPPORTED_DECODERS:
            raise ConfigError("decoders contain an unsupported decoder")
        if resource_class == "cpu" and set(decoders) & {"tabpfn", "limix2m"}:
            raise ConfigError("CPU plans must not contain a frozen GPU TFM")
        if resource_class == "gpu" and (
            len(decoders) != 1 or decoders[0] not in {"tabpfn", "limix2m"}
        ):
            raise ConfigError("GPU plans must contain exactly one frozen TFM decoder")
        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            if not isinstance(model_identity, dict):
                raise ConfigError("GPU plans require model_identity")
            expected_identity = (
                frozen_tabpfn_identity()
                if decoders == ("tabpfn",)
                else limix_frozen_model_identity()
            )
            if any(model_identity.get(key) != value for key, value in expected_identity.items()):
                raise ConfigError("GPU model_identity does not match the registered frozen TFM")
            checkpoint_hash = model_identity.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ConfigError("GPU model_identity checkpoint_sha256 is invalid")
        elif model_identity is not None:
            raise ConfigError("CPU plans must not declare a GPU model_identity")

        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_class")
        if tuple(sorted(contexts)) != contexts:
            raise ConfigError("contexts_per_class must be strictly increasing")
        ratios = (
            _nonempty_unique_positive_ints(mapping, "context_negative_ratios")
            if "context_negative_ratios" in mapping
            else (1,)
        )
        if tuple(sorted(ratios)) != ratios:
            raise ConfigError("context_negative_ratios must be strictly increasing")
        max_context = mapping.get("max_context_per_class")
        evaluation_size = mapping.get("evaluation_per_class")
        evaluation_negative_ratio = mapping.get("evaluation_negative_ratio", 1)
        if not isinstance(max_context, int) or isinstance(max_context, bool) or max_context <= 0:
            raise ConfigError("max_context_per_class must be a positive integer")
        if (
            not isinstance(evaluation_size, int)
            or isinstance(evaluation_size, bool)
            or evaluation_size <= 0
        ):
            raise ConfigError("evaluation_per_class must be a positive integer")
        if (
            not isinstance(evaluation_negative_ratio, int)
            or isinstance(evaluation_negative_ratio, bool)
            or evaluation_negative_ratio <= 0
        ):
            raise ConfigError("evaluation_negative_ratio must be a positive integer")
        if max(contexts) > max_context:
            raise ConfigError("a context budget exceeds max_context_per_class")
        if max(contexts) * max(ratios) > max_context:
            raise ConfigError("a negative context budget exceeds max_context_per_class")
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")
        ceiling_seeds = _nonempty_unique_positive_ints(mapping, "ceiling_seeds")
        if set(seeds) & set(ceiling_seeds):
            raise ConfigError("episode and ceiling seeds must be disjoint")

        raw_specs = mapping.get("property_specs")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ConfigError("property_specs must be a non-empty list")
        property_specs: list[PropertySpec] = []
        seen_properties: set[str] = set()
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict):
                raise ConfigError("each property_specs item must be a mapping")
            property_name = raw_spec.get("property")
            if property_name not in SUPPORTED_PROPERTIES:
                raise ConfigError(f"unsupported controlled property: {property_name!r}")
            if property_name in seen_properties:
                raise ConfigError(f"duplicate property spec: {property_name}")
            seen_properties.add(property_name)
            encoders = _nonempty_unique_strings(raw_spec, "encoders")
            for encoder in encoders:
                try:
                    validate_property_encoder(property_name, encoder)
                except ValueError as error:
                    raise ConfigError(str(error)) from error
            sizes = _nonempty_unique_positive_ints(raw_spec, "graph_sizes")
            if max_context >= 400 and evaluation_size >= 400:
                minimum = MIN_GRAPH_SIZE[property_name]
                if any(size < minimum for size in sizes):
                    raise ConfigError(
                        f"{property_name} graph size is below its registered minimum {minimum}"
                    )
            noise_regimes = _nonempty_unique_strings(raw_spec, "noise_regimes")
            if not set(noise_regimes) <= SUPPORTED_NOISE_REGIMES:
                raise ConfigError(f"{property_name} contains an unsupported noise regime")
            registered_macro = raw_spec.get("macro_graph_family")
            if registered_macro is not None:
                expected_macro = macro_graph_family(property_name)
                if registered_macro != expected_macro:
                    raise ConfigError(
                        f"{property_name} macro_graph_family must be {expected_macro!r}"
                    )
            property_specs.append(
                PropertySpec(property_name, encoders, sizes, noise_regimes, registered_macro)
            )

        return cls(
            schema_version=2,
            study_id=study_id.strip(),
            status=status,
            task_family=task_family,
            properties=tuple(spec.property_name for spec in property_specs),
            encoders=tuple(dict.fromkeys(e for spec in property_specs for e in spec.encoders)),
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=ratios,
            graph_sizes=tuple(
                dict.fromkeys(s for spec in property_specs for s in spec.graph_sizes)
            ),
            noise_regimes=tuple(
                dict.fromkeys(n for spec in property_specs for n in spec.noise_regimes)
            ),
            seeds=seeds,
            max_context_per_class=max_context,
            evaluation_per_class=evaluation_size,
            evaluation_negative_ratio=evaluation_negative_ratio,
            ceiling_seeds=ceiling_seeds,
            property_specs=tuple(property_specs),
            rule_specs=(),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_real_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "real_graph":
            raise ConfigError("schema_version 3 supports real_graph only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= SUPPORTED_REAL_DECODERS:
            raise ConfigError("decoders contain an unsupported decoder")
        if resource_class == "cpu" and set(decoders) & {"tabpfn", "limix2m", "tabm"}:
            raise ConfigError("CPU plans must not contain a GPU decoder")
        if resource_class == "gpu" and (
            len(decoders) != 1 or decoders[0] not in {"tabpfn", "limix2m", "tabm"}
        ):
            raise ConfigError("GPU plans must contain exactly one GPU decoder")
        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            if decoders == ("tabpfn",):
                expected_identity = frozen_tabpfn_identity()
            elif decoders == ("limix2m",):
                expected_identity = limix_frozen_model_identity()
            else:
                expected_identity = tabm_training_identity()
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected_identity.items()
            ):
                raise ConfigError("GPU model_identity does not match the registered decoder")
        elif model_identity is not None:
            raise ConfigError("CPU plans must not declare a GPU model_identity")

        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_class")
        ratios = _nonempty_unique_positive_ints(mapping, "context_negative_ratios")
        if tuple(sorted(contexts)) != contexts or tuple(sorted(ratios)) != ratios:
            raise ConfigError("real context axes must be strictly increasing")
        max_context = mapping.get("max_context_per_class")
        evaluation_size = mapping.get("evaluation_per_class")
        evaluation_ratio = mapping.get("evaluation_negative_ratio")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (max_context, evaluation_size, evaluation_ratio)
        ):
            raise ConfigError("real context/evaluation sizes must be positive integers")
        if max(contexts) * max(ratios) > max_context:
            raise ConfigError("a real negative context budget exceeds max_context_per_class")

        seeds = _nonempty_unique_positive_ints(mapping, "seeds")
        raw_specs = mapping.get("dataset_specs")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ConfigError("dataset_specs must be a non-empty list")
        specs = []
        for raw in raw_specs:
            if not isinstance(raw, dict) or raw.get("dataset") not in REAL_DATASET_ADAPTERS:
                raise ConfigError("unsupported real dataset spec")
            adapter = REAL_DATASET_ADAPTERS[str(raw["dataset"])]
            encoders = _nonempty_unique_strings(raw, "encoders")
            aligned = raw.get("aligned_encoder")
            if (
                not set(encoders) <= SUPPORTED_REAL_ENCODERS
                or aligned not in REAL_ALIGNED_ENCODERS
            ):
                raise ConfigError("invalid real encoder specification")
            if raw.get("asset_id") != adapter.asset_id:
                raise ConfigError(
                    f"{adapter.dataset_name} asset_id must be {adapter.asset_id!r}"
                )
            if raw.get("macro_graph_family") != adapter.macro_graph_family:
                raise ConfigError(
                    f"{adapter.dataset_name} macro_graph_family must be "
                    f"{adapter.macro_graph_family!r}"
                )
            specs.append(
                RealDatasetSpec(
                    dataset_name=str(raw["dataset"]),
                    asset_id=str(raw["asset_id"]),
                    macro_graph_family=str(raw["macro_graph_family"]),
                    aligned_encoder=str(aligned),
                    encoders=encoders,
                )
            )
        if len({spec.dataset_name for spec in specs}) != len(specs):
            raise ConfigError("real dataset specs must be unique")

        return cls(
            schema_version=3,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=tuple(spec.dataset_name for spec in specs),
            encoders=tuple(dict.fromkeys(e for spec in specs for e in spec.encoders)),
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=ratios,
            graph_sizes=(),
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=int(max_context),
            evaluation_per_class=int(evaluation_size),
            evaluation_negative_ratio=int(evaluation_ratio),
            ceiling_seeds=(),
            property_specs=(),
            rule_specs=(),
            real_dataset_specs=tuple(specs),
            frozen_kg_dataset_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_rule_induction_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "rule_induction":
            raise ConfigError("schema_version 5 supports rule_induction only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= SUPPORTED_DECODERS:
            raise ConfigError("decoders contain an unsupported decoder")
        if resource_class == "cpu" and set(decoders) & {"tabpfn", "limix2m"}:
            raise ConfigError("CPU plans must not contain a frozen GPU TFM")
        if resource_class == "gpu" and (
            len(decoders) != 1 or decoders[0] not in {"tabpfn", "limix2m"}
        ):
            raise ConfigError("GPU plans must contain exactly one frozen TFM decoder")
        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            expected_identity = (
                frozen_tabpfn_identity()
                if decoders == ("tabpfn",)
                else limix_frozen_model_identity()
            )
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected_identity.items()
            ):
                raise ConfigError("GPU model_identity does not match the registered frozen TFM")
        elif model_identity is not None:
            raise ConfigError("CPU plans must not declare a GPU model_identity")

        graph_sizes = _nonempty_unique_positive_ints(mapping, "graph_sizes")
        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_class")
        max_context = mapping.get("max_context_per_class")
        evaluation_size = mapping.get("evaluation_per_class")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (max_context, evaluation_size)
        ):
            raise ConfigError("rule_induction context/evaluation sizes must be positive integers")
        if max(contexts) > max_context:
            raise ConfigError("a rule_induction context budget exceeds max_context_per_class")
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")

        raw_specs = mapping.get("rule_specs")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ConfigError("rule_specs must be a non-empty list")
        specs = []
        for raw in raw_specs:
            if not isinstance(raw, dict) or raw.get("rule") not in SUPPORTED_RULES:
                raise ConfigError("unsupported rule_induction rule")
            encoders = _nonempty_unique_strings(raw, "encoders")
            if not set(encoders) <= RULE_INDUCTION_ENCODERS:
                raise ConfigError("invalid rule_induction encoder specification")
            specs.append(RuleInductionSpec(rule_name=str(raw["rule"]), encoders=encoders))
        if len({spec.rule_name for spec in specs}) != len(specs):
            raise ConfigError("rule_induction rule specs must be unique")

        return cls(
            schema_version=5,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=tuple(spec.rule_name for spec in specs),
            encoders=tuple(dict.fromkeys(e for spec in specs for e in spec.encoders)),
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=(1,),
            graph_sizes=graph_sizes,
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=int(max_context),
            evaluation_per_class=int(evaluation_size),
            evaluation_negative_ratio=1,
            ceiling_seeds=(),
            property_specs=(),
            rule_specs=tuple(specs),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_clutrr_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "clutrr_symbolic":
            raise ConfigError("schema_version 6 supports clutrr_symbolic only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= {
            "logistic",
            "logistic_tuned",
            "catboost",
            "catboost_tuned",
            "mlp",
            "gru_sequence",
            "tabpfn",
        }:
            raise ConfigError("CLUTRR contains an unsupported decoder")
        if resource_class == "cpu" and "tabpfn" in decoders:
            raise ConfigError("CPU CLUTRR plans must not contain TabPFN")
        if resource_class == "gpu" and decoders != ("tabpfn",):
            raise ConfigError("GPU CLUTRR plans must contain exactly TabPFN")
        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            expected = frozen_tabpfn_identity()
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected.items()
            ):
                raise ConfigError("GPU model_identity does not match frozen TabPFN")
            checkpoint_hash = model_identity.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ConfigError("GPU model_identity checkpoint_sha256 is invalid")
        elif model_identity is not None:
            raise ConfigError("CPU CLUTRR plans must not declare model_identity")

        dataset_config = mapping.get("dataset_config")
        registered_lengths = {
            "gen_train23_test2to10": ((2, 3), tuple(range(2, 11))),
            "gen_train234_context23_test2to10": (
                (2, 3),
                tuple(range(2, 11)),
            ),
            "gen_train234_test2to10": ((2, 3, 4), tuple(range(2, 11))),
        }
        if dataset_config not in registered_lengths:
            raise ConfigError("unsupported CLUTRR dataset_config")
        train_lengths = _nonempty_unique_positive_ints(mapping, "train_chain_lengths")
        evaluation_lengths = _nonempty_unique_positive_ints(
            mapping, "evaluation_chain_lengths"
        )
        if (train_lengths, evaluation_lengths) != registered_lengths[dataset_config]:
            raise ConfigError("CLUTRR chain-length protocol changed")

        encoders = _nonempty_unique_strings(mapping, "encoders")
        if not set(encoders) <= CLUTRR_ENCODERS:
            raise ConfigError("CLUTRR encoders contain an unsupported interface")
        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_relation")
        if tuple(sorted(contexts)) != contexts:
            raise ConfigError("contexts_per_relation must be strictly increasing")
        max_context = mapping.get("max_context_per_relation")
        if (
            not isinstance(max_context, int)
            or isinstance(max_context, bool)
            or max_context <= 0
            or max(contexts) > max_context
        ):
            raise ConfigError("invalid CLUTRR max_context_per_relation")
        context_sampling = mapping.get("context_sampling")
        if context_sampling is not None:
            if (
                not isinstance(context_sampling, dict)
                or set(context_sampling)
                != {
                    "protocol",
                    "base_chain_lengths",
                    "added_chain_lengths",
                    "added_per_relation",
                }
            ):
                raise ConfigError("invalid CLUTRR context_sampling specification")
            if context_sampling["protocol"] != "paired_additive_length_control_v1":
                raise ConfigError("unsupported CLUTRR context sampling protocol")
            base_lengths = tuple(context_sampling["base_chain_lengths"])
            added_lengths = tuple(context_sampling["added_chain_lengths"])
            if base_lengths != (2, 3) or added_lengths not in {(2, 3), (4,)}:
                raise ConfigError("invalid paired-additive CLUTRR chain lengths")
            if not set(base_lengths + added_lengths) <= set(train_lengths):
                raise ConfigError("context sampling requests unavailable chain lengths")
            added_per_relation = context_sampling["added_per_relation"]
            if (
                not isinstance(added_per_relation, int)
                or isinstance(added_per_relation, bool)
                or added_per_relation <= 0
            ):
                raise ConfigError("added_per_relation must be a positive integer")
            if max(contexts) + added_per_relation > max_context:
                raise ConfigError("paired-additive context exceeds registered maximum")
        max_path_length = mapping.get("max_path_length")
        if max_path_length != CLUTRR_MAX_PATH_LENGTH:
            raise ConfigError("CLUTRR max_path_length changed")
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")

        asset_files = mapping.get("asset_files")
        if not isinstance(asset_files, dict) or set(asset_files) != {
            "train",
            "validation",
            "test",
        }:
            raise ConfigError("CLUTRR asset_files must define train, validation, and test")
        for split, spec in asset_files.items():
            if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
                raise ConfigError(f"invalid CLUTRR {split} asset specification")
            path = spec["path"]
            checksum = spec["sha256"]
            if (
                not isinstance(path, str)
                or not path.startswith("data/raw/clutrr/")
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ConfigError(f"invalid CLUTRR {split} asset path")
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ConfigError(f"invalid CLUTRR {split} asset checksum")

        return cls(
            schema_version=6,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=(str(dataset_config),),
            encoders=encoders,
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=(1,),
            graph_sizes=(),
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=int(max_context),
            evaluation_per_class=None,
            evaluation_negative_ratio=1,
            ceiling_seeds=(),
            property_specs=(),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=(),
            rule_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_compositional_shift_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "compositional_length_shift":
            raise ConfigError(
                "schema_version 7 supports compositional_length_shift only"
            )
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= {"logistic", "catboost", "tabpfn"}:
            raise ConfigError("compositional shift contains an unsupported decoder")
        if resource_class == "cpu" and "tabpfn" in decoders:
            raise ConfigError("CPU compositional plans must not contain TabPFN")
        if resource_class == "gpu" and decoders != ("tabpfn",):
            raise ConfigError("GPU compositional plans must contain exactly TabPFN")

        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            expected = frozen_tabpfn_identity()
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected.items()
            ):
                raise ConfigError("GPU model_identity does not match frozen TabPFN")
            checkpoint_hash = model_identity.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ConfigError("GPU model_identity checkpoint_sha256 is invalid")
        elif model_identity is not None:
            raise ConfigError("CPU compositional plans must not declare model_identity")

        if mapping.get("algebra") != "dihedral_d8":
            raise ConfigError("the registered compositional algebra must be dihedral_d8")
        train_lengths = _nonempty_unique_positive_ints(
            mapping, "train_chain_lengths"
        )
        evaluation_lengths = _nonempty_unique_positive_ints(
            mapping, "evaluation_chain_lengths"
        )
        if train_lengths != (2, 3) or evaluation_lengths != tuple(range(4, 11)):
            raise ConfigError("registered D8 train/evaluation length split changed")
        if mapping.get("max_path_length") != COMPOSITION_MAX_PATH_LENGTH:
            raise ConfigError("registered D8 maximum path length changed")

        encoders = _nonempty_unique_strings(mapping, "encoders")
        if not set(encoders) <= COMPOSITION_SHIFT_ENCODERS:
            raise ConfigError("compositional shift contains an unsupported encoder")
        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_class")
        if tuple(sorted(contexts)) != contexts:
            raise ConfigError("contexts_per_class must be strictly increasing")
        max_context = mapping.get("max_context_per_class")
        if (
            not isinstance(max_context, int)
            or isinstance(max_context, bool)
            or max_context <= 0
            or max(contexts) > max_context
        ):
            raise ConfigError("invalid compositional max_context_per_class")
        evaluation_per_length = mapping.get("evaluation_per_class_per_length")
        if (
            not isinstance(evaluation_per_length, int)
            or isinstance(evaluation_per_length, bool)
            or evaluation_per_length <= 0
        ):
            raise ConfigError(
                "evaluation_per_class_per_length must be a positive integer"
            )
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")

        return cls(
            schema_version=7,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=("dihedral_d8",),
            encoders=encoders,
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=(1,),
            graph_sizes=(),
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=max_context,
            evaluation_per_class=evaluation_per_length,
            evaluation_negative_ratio=1,
            ceiling_seeds=(),
            property_specs=(),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=(),
            rule_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_graphlog_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "graphlog_symbolic":
            raise ConfigError("schema_version 8 supports graphlog_symbolic only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= {
            "logistic",
            "catboost",
            "catboost_depth1",
            "catboost_depth2",
            "catboost_depth3",
            "catboost_depth4",
            "catboost_depth5",
            "catboost_depth6",
            "catboost_depth7",
            "catboost_tuned",
            "tabpfn",
        }:
            raise ConfigError("GraphLog contains an unsupported decoder")
        if resource_class == "cpu" and "tabpfn" in decoders:
            raise ConfigError("CPU GraphLog plans must not contain TabPFN")
        if resource_class == "gpu" and decoders != ("tabpfn",):
            raise ConfigError("GPU GraphLog plans must contain exactly TabPFN")

        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            expected = frozen_tabpfn_identity()
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected.items()
            ):
                raise ConfigError("GPU model_identity does not match frozen TabPFN")
            checkpoint_hash = model_identity.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ConfigError("GPU model_identity checkpoint_sha256 is invalid")
        elif model_identity is not None:
            raise ConfigError("CPU GraphLog plans must not declare model_identity")

        if mapping.get("dataset") != "GraphLog/v1.1":
            raise ConfigError("registered GraphLog dataset identity changed")
        worlds = _nonempty_unique_strings(mapping, "worlds")
        expected_worlds = tuple(f"rule_{index}" for index in range(57))
        if worlds != expected_worlds:
            raise ConfigError("GraphLog confirmation must contain rule_0 through rule_56")
        if (
            mapping.get("context_inner_split") != "train"
            or mapping.get("evaluation_inner_split") != "test"
        ):
            raise ConfigError("registered GraphLog train/test split changed")
        train_lengths = _nonempty_unique_positive_ints(
            mapping, "train_chain_lengths"
        )
        evaluation_lengths = _nonempty_unique_positive_ints(
            mapping, "evaluation_chain_lengths"
        )
        if train_lengths != (2, 3) or evaluation_lengths != (4, 5):
            raise ConfigError("registered GraphLog length split changed")
        if mapping.get("max_path_length") != GRAPHLOG_MAX_PATH_LENGTH:
            raise ConfigError("registered GraphLog maximum path length changed")

        asset = mapping.get("asset")
        if not isinstance(asset, dict):
            raise ConfigError("GraphLog asset must be a mapping")
        path = asset.get("path")
        checksum = asset.get("sha256")
        if (
            not isinstance(path, str)
            or not path.startswith("data/raw/graphlog/")
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ConfigError("invalid GraphLog asset path")
        if not isinstance(checksum, str) or not re.fullmatch(
            r"[0-9a-f]{64}", checksum
        ):
            raise ConfigError("invalid GraphLog asset checksum")

        encoders = _nonempty_unique_strings(mapping, "encoders")
        if not set(encoders) <= GRAPHLOG_ENCODERS:
            raise ConfigError("GraphLog contains an unsupported encoder")
        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_relation")
        registered_contexts = (5, 10, 20, 40)
        if (
            tuple(sorted(contexts)) != contexts
            or not set(contexts) <= set(registered_contexts)
        ):
            raise ConfigError("GraphLog context budgets left the registered grid")
        max_context = mapping.get("max_context_per_relation")
        if max_context != 40:
            raise ConfigError("registered GraphLog maximum context changed")
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")

        return cls(
            schema_version=8,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=worlds,
            encoders=encoders,
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=(1,),
            graph_sizes=(),
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=max_context,
            evaluation_per_class=None,
            evaluation_negative_ratio=1,
            ceiling_seeds=(),
            property_specs=(),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=(),
            rule_specs=(),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )

    @classmethod
    def _from_frozen_kg_mapping(
        cls, mapping: dict[str, Any], study_id: str, status: str, task_family: str
    ) -> StudyConfig:
        if task_family != "frozen_kg_interface":
            raise ConfigError("schema_version 4 supports frozen_kg_interface only")
        resource_class = mapping.get("resource_class")
        if resource_class not in {"cpu", "gpu"}:
            raise ConfigError("resource_class must be cpu or gpu")
        decoders = _nonempty_unique_strings(mapping, "decoders")
        if not set(decoders) <= SUPPORTED_FROZEN_KG_DECODERS:
            raise ConfigError("decoders contain an unsupported decoder")
        if resource_class == "cpu" and set(decoders) & {"tabpfn", "limix2m", "tabicl"}:
            raise ConfigError("CPU plans must not contain a frozen GPU TFM")
        if resource_class == "gpu" and (
            len(decoders) != 1 or decoders[0] not in {"tabpfn", "limix2m", "tabicl"}
        ):
            raise ConfigError("GPU plans must contain exactly one frozen TFM decoder")
        model_identity = mapping.get("model_identity")
        if resource_class == "gpu":
            if decoders == ("tabpfn",):
                expected_identity = frozen_tabpfn_identity()
            elif decoders == ("limix2m",):
                expected_identity = limix_frozen_model_identity()
            else:
                expected_identity = tabicl_frozen_identity()
            if not isinstance(model_identity, dict) or any(
                model_identity.get(key) != value for key, value in expected_identity.items()
            ):
                raise ConfigError("GPU model_identity does not match the registered frozen TFM")
            checkpoint_hash = model_identity.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ConfigError("GPU model_identity checkpoint_sha256 is invalid")
        elif model_identity is not None:
            raise ConfigError("CPU plans must not declare a GPU model_identity")

        asset_id = mapping.get("asset_id")
        if not isinstance(asset_id, str):
            raise ConfigError("asset_id must be a string")
        try:
            asset = frozen_kg_asset(asset_id)
        except ValueError as error:
            raise ConfigError(str(error)) from error
        if status not in asset["allowed_statuses"]:
            raise ConfigError(f"{asset_id} does not support status={status}")
        if mapping.get("asset_manifest_sha256") != asset["manifest_sha256"]:
            raise ConfigError("RFMLink asset manifest hash mismatch")
        if mapping.get("query_triples") != asset["query_triples"]:
            raise ConfigError("the frozen RFMLink query endpoint changed")
        contexts = _nonempty_unique_positive_ints(mapping, "contexts_per_class")
        if contexts != asset["contexts_per_class"]:
            raise ConfigError("the frozen RFMLink context ladder changed")
        context_modes = mapping.get("context_modes", ["original_uniform"])
        if (
            not isinstance(context_modes, list)
            or not context_modes
            or not all(isinstance(mode, str) and mode.strip() for mode in context_modes)
            or len(context_modes) != len(set(context_modes))
        ):
            raise ConfigError("context_modes must be a non-empty unique string list")
        registered_context_modes = tuple(asset.get("context_modes", ("original_uniform",)))
        if not set(context_modes) <= set(registered_context_modes):
            raise ConfigError("the frozen RFMLink context modes changed")
        query_transforms = mapping.get("query_transforms", ["raw"])
        if (
            not isinstance(query_transforms, list)
            or not query_transforms
            or len(query_transforms) != len(set(query_transforms))
            or not set(query_transforms) <= set(QUERY_TRANSFORMS)
        ):
            raise ConfigError("query_transforms contains an unsupported transform")
        seeds = _nonempty_unique_positive_ints(mapping, "seeds")

        raw_specs = mapping.get("dataset_specs")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ConfigError("dataset_specs must be a non-empty list")
        specs: list[FrozenKGDatasetSpec] = []
        for raw in raw_specs:
            if not isinstance(raw, dict) or raw.get("dataset") not in asset["datasets"]:
                raise ConfigError("unsupported frozen RFMLink dataset spec")
            dataset_name = str(raw["dataset"])
            registered = asset["datasets"][dataset_name]
            interfaces = _nonempty_unique_strings(raw, "interfaces")
            if not set(interfaces) <= set(FROZEN_KG_INTERFACES):
                raise ConfigError("unsupported frozen RFMLink interface")
            if raw.get("transfer_regime") != registered["transfer_regime"]:
                raise ConfigError(f"{dataset_name} transfer regime mismatch")
            if raw.get("archive_sha256") != registered["archive_sha256"]:
                raise ConfigError(f"{dataset_name} archive hash mismatch")
            specs.append(
                FrozenKGDatasetSpec(
                    dataset_name=dataset_name,
                    transfer_regime=str(registered["transfer_regime"]),
                    archive_sha256=str(registered["archive_sha256"]),
                    interfaces=interfaces,
                )
            )
        if len({spec.dataset_name for spec in specs}) != len(specs):
            raise ConfigError("frozen RFMLink dataset specs must be unique")

        return cls(
            schema_version=4,
            study_id=study_id,
            status=status,
            task_family=task_family,
            properties=tuple(spec.dataset_name for spec in specs),
            encoders=tuple(dict.fromkeys(i for spec in specs for i in spec.interfaces)),
            decoders=decoders,
            contexts_per_class=contexts,
            context_negative_ratios=(1,),
            graph_sizes=(),
            noise_regimes=("none",),
            seeds=seeds,
            max_context_per_class=400,
            evaluation_per_class=None,
            evaluation_negative_ratio=1,
            ceiling_seeds=(),
            property_specs=(),
            rule_specs=(),
            real_dataset_specs=(),
            frozen_kg_dataset_specs=tuple(specs),
            resource_class=resource_class,
            model_identity=model_identity,
            raw=mapping,
        )


def load_study_config(path: Path) -> StudyConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"configuration does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    return StudyConfig.from_mapping(payload)
