from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from relational_tfm.decoders.classification import (
    TABPFN_PREDICTION_BATCH_SIZE,
    build_classifier,
    decoder_provenance,
    frozen_tabpfn_identity,
    positive_decision_orientation,
    positive_probability,
    positive_ranking_score,
    tabicl_frozen_identity,
    tabm_training_identity,
)
from relational_tfm.decoders.limix import frozen_model_identity


class _RecordingClassifier:
    classes_ = np.asarray([0, 1])
    _rtfm_prediction_batch_size = TABPFN_PREDICTION_BATCH_SIZE

    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.batch_lengths.append(len(values))
        positive = values[:, 0].astype(np.float64)
        return np.column_stack([1.0 - positive, positive])


def test_tabpfn_prediction_batches_preserve_row_order() -> None:
    row_count = 2 * TABPFN_PREDICTION_BATCH_SIZE + 1
    values = np.linspace(0.0, 1.0, row_count, dtype=np.float64).reshape(-1, 1)
    model = _RecordingClassifier()
    result = positive_probability(model, values)
    assert model.batch_lengths == [
        TABPFN_PREDICTION_BATCH_SIZE,
        TABPFN_PREDICTION_BATCH_SIZE,
        1,
    ]
    assert np.array_equal(result, values[:, 0])


def test_binary_ranking_prefers_unsaturated_decision_scores() -> None:
    class _SaturatedClassifier:
        classes_ = np.asarray([0, 1])

        def decision_function(self, values: np.ndarray) -> np.ndarray:
            return values[:, 0]

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            positive = np.ones(len(values), dtype=np.float64)
            return np.column_stack([1.0 - positive, positive])

    values = np.asarray([[40.0], [41.0], [42.0]])
    model = _SaturatedClassifier()
    assert np.array_equal(
        positive_ranking_score(model, values),
        values[:, 0],
    )
    assert np.unique(positive_probability(model, values)).size == 1
    assert positive_decision_orientation(model, values) == 1


def test_tabpfn_builder_uses_frozen_cache_configuration(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            captured.update({"version": version, **kwargs})
            return SimpleNamespace()

    tabpfn = ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = _Classifier
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = SimpleNamespace(V3="v3")
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)

    model = build_classifier("tabpfn", 123)
    assert captured["version"] == "v3"
    assert captured["fit_mode"] == "fit_with_cache"
    assert captured["keep_cache_on_device"] is True
    assert captured["n_estimators"] == 4
    assert captured["random_state"] == 123
    assert model._rtfm_prediction_batch_size == 50_000


def test_catboost_does_not_write_auxiliary_files(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    catboost = ModuleType("catboost")
    catboost.CatBoostClassifier = _Classifier
    monkeypatch.setitem(sys.modules, "catboost", catboost)

    build_classifier("catboost", 123)
    assert captured["allow_writing_files"] is False


def test_nominal_logistic_is_invariant_to_category_recoding() -> None:
    categories = np.tile(np.asarray([0.0, 1.0, 2.0]), 20)
    numerical = np.linspace(-1.0, 1.0, len(categories))
    values = np.column_stack([categories, numerical])
    labels = (categories == 1.0).astype(np.int64)
    recoded = values.copy()
    recoding = {0.0: 17.0, 1.0: 4.0, 2.0: 91.0}
    recoded[:, 0] = np.asarray([recoding[value] for value in categories])

    original = build_classifier("logistic_nominal", 123, (0,)).fit(values, labels)
    renamed = build_classifier("logistic_nominal", 123, (0,)).fit(recoded, labels)

    assert np.allclose(
        positive_probability(original, values),
        positive_probability(renamed, recoded),
        atol=1e-10,
    )


def test_nominal_catboost_registers_categorical_columns(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    catboost = ModuleType("catboost")
    catboost.CatBoostClassifier = _Classifier
    monkeypatch.setitem(sys.modules, "catboost", catboost)

    build_classifier("catboost_nominal", 123, (0, 16))
    assert captured["cat_features"] == [0, 16]
    assert captured["allow_writing_files"] is False


def test_catboost_multiclass_uses_multiclass_loss(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    catboost = ModuleType("catboost")
    catboost.CatBoostClassifier = _Classifier
    monkeypatch.setitem(sys.modules, "catboost", catboost)

    build_classifier("catboost", 123, multiclass=True)
    assert captured["loss_function"] == "MultiClass"


def test_catboost_depth1_is_capacity_restricted(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    catboost = ModuleType("catboost")
    catboost.CatBoostClassifier = _Classifier
    monkeypatch.setitem(sys.modules, "catboost", catboost)

    build_classifier("catboost_depth1", 123, multiclass=True)
    assert captured["depth"] == 1
    assert captured["iterations"] == 500
    assert captured["loss_function"] == "MultiClass"


def test_catboost_provenance_records_version_and_parameters(monkeypatch) -> None:
    class _Classifier:
        __module__ = "catboost.core"

        def get_params(self):
            return {"depth": 1, "iterations": 500, "ignored": "value"}

    monkeypatch.setattr(
        "relational_tfm.decoders.classification.importlib.metadata.version",
        lambda package: "1.2.test" if package == "catboost" else "unexpected",
    )
    provenance = decoder_provenance(_Classifier())
    assert provenance["package"] == "catboost"
    assert provenance["package_version"] == "1.2.test"
    assert provenance["parameters"] == {"depth": 1, "iterations": 500}


def test_tabm_builder_uses_frozen_context_only_training_semantics(monkeypatch) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    pytabkit = ModuleType("pytabkit")
    pytabkit.TabM_D_Classifier = _Classifier
    monkeypatch.setitem(sys.modules, "pytabkit", pytabkit)

    build_classifier("tabm", 123)
    identity = tabm_training_identity()
    assert captured == {
        "device": "cuda",
        "random_state": 123,
        "n_cv": identity["n_cv"],
        "n_refit": identity["n_refit"],
        "val_fraction": identity["val_fraction"],
        "n_threads": identity["n_threads"],
        "verbosity": 0,
    }
    assert identity["selection_population"] == "context_only_internal_validation"


def test_tabicl_builder_uses_pinned_frozen_configuration(
    monkeypatch, tmp_path
) -> None:
    captured: dict = {}

    class _Classifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    checkpoint = tmp_path / "tabicl-classifier-v2-20260212.ckpt"
    checkpoint.write_bytes(b"pinned-checkpoint")
    tabicl = ModuleType("tabicl")
    tabicl.TabICLClassifier = _Classifier
    monkeypatch.setitem(sys.modules, "tabicl", tabicl)
    monkeypatch.setenv("TABICL_MODEL_PATH", str(checkpoint))
    monkeypatch.setattr(
        "relational_tfm.decoders.classification.importlib.metadata.version",
        lambda package: "2.0.2" if package == "tabicl" else "unexpected",
    )
    monkeypatch.setattr(
        "relational_tfm.decoders.classification.TABICL_CHECKPOINT_SHA256",
        "2fc6be70a83fbb56d8e53a6b36995a3d59bd018b21de394c60ab1ea64d009942",
    )

    model = build_classifier("tabicl", 123)
    identity = tabicl_frozen_identity()
    assert captured == {
        "n_estimators": 8,
        "batch_size": 8,
        "average_logits": True,
        "support_many_classes": True,
        "model_path": str(checkpoint.resolve()),
        "allow_auto_download": False,
        "device": "cuda",
        "use_amp": True,
        "use_fa3": False,
        "offload_mode": False,
        "random_state": 123,
        "verbose": False,
    }
    assert identity["checkpoint_version"] == checkpoint.name
    assert model._rtfm_prediction_batch_size == 50_000
    assert model.rtfm_provenance()["checkpoints"][0]["path"] == str(
        checkpoint.resolve()
    )


def test_logistic_context_tuning_uses_only_context_folds() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(size=(90, 4))
    labels = np.repeat(np.arange(3), 30)
    values[:, 0] += labels
    model = build_classifier("logistic_tuned", 456, multiclass=True)
    model.fit(values, labels)
    probabilities = model.predict_proba(values[:7])
    provenance = decoder_provenance(model)
    assert probabilities.shape == (7, 3)
    assert provenance["selection_population"] == "context_only_stratified_cross_validation"
    assert provenance["selection_metric"] == "accuracy"
    assert provenance["folds"] == 3
    assert len(provenance["candidate_grid"]) == 4
    assert provenance["best_parameters"] in provenance["candidate_grid"]


def test_frozen_tfm_identities_include_execution_semantics() -> None:
    tabpfn = frozen_tabpfn_identity()
    assert tabpfn["fit_mode"] == "fit_with_cache"
    assert tabpfn["n_estimators"] == 4
    assert "prediction_batch_size" not in tabpfn
    assert "keep_cache_on_device" not in tabpfn

    limix = frozen_model_identity()
    assert limix["model_version"] == "LimiX-2M"
    assert limix["checkpoint_bytes"] == 9_558_253
    assert limix["retrieval_len"] == "dynamic"
    assert limix["prediction_batching"] == "complete-evaluation-set"
