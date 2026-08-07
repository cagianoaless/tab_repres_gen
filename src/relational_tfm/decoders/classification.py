from __future__ import annotations

import importlib.metadata
import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from relational_tfm.hashing import file_sha256

TABPFN_MODEL_VERSION = "V3"
TABPFN_FIT_MODE = "fit_with_cache"
TABPFN_PREDICTION_BATCH_SIZE = 50_000
TABPFN_KEEP_CACHE_ON_DEVICE = True
TABM_PACKAGE_VERSION = "1.7.3"
TABM_N_CV = 1
TABM_N_REFIT = 0
TABM_VAL_FRACTION = 0.1
TABM_N_THREADS = 1
TABICL_PACKAGE_VERSION = "2.0.2"
TABICL_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
TABICL_CHECKPOINT_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)
TABICL_N_ESTIMATORS = 8
TABICL_BATCH_SIZE = 8
TABICL_PREDICTION_BATCH_SIZE = 50_000
CONTEXT_TUNING_FOLDS = 3
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
CATBOOST_GRID = tuple(
    {"depth": depth, "l2_leaf_reg": l2_leaf_reg}
    for depth in (4, 7, 10)
    for l2_leaf_reg in (3.0, 10.0)
)
CLUTRR_MAX_PATH_LENGTH = 10
GRU_EMBEDDING_DIM = 32
GRU_HIDDEN_DIM = 64
GRU_MAX_EPOCHS = 250
GRU_PATIENCE = 30
GRU_LEARNING_RATE = 3e-3
GRU_WEIGHT_DECAY = 1e-4


def _ordered_path_tokens(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] <= 1:
        raise ValueError("ordered path input must contain length and relation coordinates")
    relation_columns = values.shape[1] - 1
    if relation_columns % CLUTRR_MAX_PATH_LENGTH:
        raise ValueError("ordered path width is incompatible with the registered maximum length")
    relation_count = relation_columns // CLUTRR_MAX_PATH_LENGTH
    lengths = np.rint(values[:, 0] * CLUTRR_MAX_PATH_LENGTH).astype(np.int64)
    if np.any(lengths < 1) or np.any(lengths > CLUTRR_MAX_PATH_LENGTH):
        raise ValueError("ordered path input contains an invalid normalized length")
    blocks = values[:, 1:].reshape(
        len(values), CLUTRR_MAX_PATH_LENGTH, relation_count
    )
    active = np.arange(CLUTRR_MAX_PATH_LENGTH)[None, :] < lengths[:, None]
    if not np.allclose(blocks.sum(axis=2)[active], 1.0):
        raise ValueError("active ordered path positions must be one-hot")
    if not np.allclose(blocks.sum(axis=2)[~active], 0.0):
        raise ValueError("inactive ordered path positions must be zero")
    tokens = np.argmax(blocks, axis=2).astype(np.int64) + 1
    tokens[~active] = 0
    return tokens, lengths, relation_count


class SharedSequenceGRUClassifier:
    """Context-only recurrent control with weights shared over path positions."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.classes_: np.ndarray | None = None
        self.model_: Any | None = None
        self.best_epoch_: int | None = None
        self.validation_history_: list[dict[str, float | int]] = []
        self.relation_count_: int | None = None

    @staticmethod
    def _model(relation_count: int, class_count: int) -> Any:
        import torch

        class SequenceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(
                    relation_count + 1,
                    GRU_EMBEDDING_DIM,
                    padding_idx=0,
                )
                self.gru = torch.nn.GRU(
                    GRU_EMBEDDING_DIM,
                    GRU_HIDDEN_DIM,
                    batch_first=True,
                )
                self.output = torch.nn.Linear(GRU_HIDDEN_DIM, class_count)

            def forward(self, tokens: Any, lengths: Any) -> Any:
                embedded = self.embedding(tokens)
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    embedded,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, hidden = self.gru(packed)
                return self.output(hidden[-1])

        return SequenceModel()

    def _stratified_validation_indices(
        self, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 71_003)
        training = []
        validation = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                raise ValueError("GRU context requires at least two rows per class")
            shuffled = indices[rng.permutation(len(indices))]
            validation_count = max(1, int(round(0.2 * len(indices))))
            validation.extend(map(int, shuffled[:validation_count]))
            training.extend(map(int, shuffled[validation_count:]))
        return (
            np.asarray(sorted(training), dtype=np.int64),
            np.asarray(sorted(validation), dtype=np.int64),
        )

    def _fit_epochs(
        self,
        tokens: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        *,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[Any, int]:
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(int(os.environ.get("RTFM_CPU_THREADS", "1")))
        model = self._model(int(self.relation_count_), len(self.classes_))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=GRU_LEARNING_RATE,
            weight_decay=GRU_WEIGHT_DECAY,
        )
        token_tensor = torch.as_tensor(tokens, dtype=torch.long)
        length_tensor = torch.as_tensor(lengths, dtype=torch.long)
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        best_state = None
        best_accuracy = -1.0
        best_epoch = 1
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(token_tensor, length_tensor),
                label_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if validation is None:
                continue
            validation_tokens, validation_lengths, validation_labels = validation
            model.eval()
            with torch.no_grad():
                predictions = model(
                    torch.as_tensor(validation_tokens, dtype=torch.long),
                    torch.as_tensor(validation_lengths, dtype=torch.long),
                ).argmax(dim=1)
            accuracy = float(
                np.mean(predictions.numpy() == validation_labels)
            )
            self.validation_history_.append(
                {
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "training_loss": float(loss.detach()),
                }
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= GRU_PATIENCE:
                break
        if validation is not None and best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    def fit(self, values: np.ndarray, labels: np.ndarray) -> SharedSequenceGRUClassifier:
        tokens, lengths, self.relation_count_ = _ordered_path_tokens(values)
        labels = np.asarray(labels, dtype=np.int64)
        self.classes_ = np.unique(labels)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("GRU requires contiguous zero-based class labels")
        training, validation = self._stratified_validation_indices(labels)
        _, self.best_epoch_ = self._fit_epochs(
            tokens[training],
            lengths[training],
            labels[training],
            GRU_MAX_EPOCHS,
            validation=(
                tokens[validation],
                lengths[validation],
                labels[validation],
            ),
        )
        self.model_, _ = self._fit_epochs(
            tokens,
            lengths,
            labels,
            self.best_epoch_,
        )
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("GRU classifier is not fitted")
        import torch

        tokens, lengths, relation_count = _ordered_path_tokens(values)
        if relation_count != self.relation_count_:
            raise ValueError("GRU evaluation relation schema changed")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(
                torch.as_tensor(tokens, dtype=torch.long),
                torch.as_tensor(lengths, dtype=torch.long),
            )
            probabilities = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities

    def rtfm_provenance(self) -> dict[str, Any]:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "architecture": "relation_embedding_then_single_layer_gru",
            "position_parameter_sharing": True,
            "embedding_dim": GRU_EMBEDDING_DIM,
            "hidden_dim": GRU_HIDDEN_DIM,
            "maximum_epochs": GRU_MAX_EPOCHS,
            "early_stopping_patience": GRU_PATIENCE,
            "learning_rate": GRU_LEARNING_RATE,
            "weight_decay": GRU_WEIGHT_DECAY,
            "selection_population": "context_only_stratified_holdout",
            "best_epoch": self.best_epoch_,
            "validation_history": self.validation_history_,
            "seed": self.seed,
        }


class ContextTunedClassifier:
    """Select a classical decoder using only folds of the supplied context."""

    def __init__(self, decoder: str, seed: int, *, multiclass: bool) -> None:
        if decoder not in {"logistic", "catboost"}:
            raise ValueError(f"unsupported context-tuned decoder: {decoder}")
        self.decoder = decoder
        self.seed = int(seed)
        self.multiclass = bool(multiclass)
        self.classes_: np.ndarray | None = None
        self.best_model_: Any | None = None
        self.best_parameters_: dict[str, Any] | None = None
        self.validation_scores_: list[dict[str, Any]] = []

    def _candidates(self) -> tuple[dict[str, Any], ...]:
        if self.decoder == "logistic":
            return tuple({"C": value} for value in LOGISTIC_C_GRID)
        return CATBOOST_GRID

    def _build(self, parameters: dict[str, Any], fold_seed: int) -> Any:
        if self.decoder == "logistic":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(parameters["C"]),
                    max_iter=2000,
                    random_state=fold_seed,
                ),
            )
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError as error:
            raise RuntimeError("install the 'tree' extra to use CatBoost") from error
        return CatBoostClassifier(
            iterations=500,
            depth=int(parameters["depth"]),
            l2_leaf_reg=float(parameters["l2_leaf_reg"]),
            learning_rate=0.05,
            loss_function="MultiClass" if self.multiclass else "Logloss",
            random_seed=fold_seed,
            thread_count=int(os.environ.get("RTFM_CPU_THREADS", "1")),
            allow_writing_files=False,
            verbose=False,
        )

    def _score(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
    ) -> float:
        if self.multiclass:
            predictions = classes[np.argmax(probabilities, axis=1)]
            return float(accuracy_score(labels, predictions))
        positive = np.flatnonzero(classes == 1)
        if positive.size != 1:
            raise ValueError("binary context tuning did not expose class 1")
        return float(roc_auc_score(labels, probabilities[:, int(positive[0])]))

    def fit(self, values: np.ndarray, labels: np.ndarray) -> ContextTunedClassifier:
        values = np.asarray(values)
        labels = np.asarray(labels, dtype=np.int64)
        class_counts = np.unique(labels, return_counts=True)[1]
        if int(class_counts.min()) < CONTEXT_TUNING_FOLDS:
            raise ValueError("context is too small for the frozen three-fold tuning protocol")
        splitter = StratifiedKFold(
            n_splits=CONTEXT_TUNING_FOLDS,
            shuffle=True,
            random_state=self.seed,
        )
        best_key: tuple[float, int] | None = None
        for candidate_index, parameters in enumerate(self._candidates()):
            fold_scores = []
            for fold_index, (train_indices, validation_indices) in enumerate(
                splitter.split(values, labels)
            ):
                model = self._build(parameters, self.seed + fold_index + 1)
                model.fit(values[train_indices], labels[train_indices])
                probabilities = np.asarray(
                    model.predict_proba(values[validation_indices]),
                    dtype=np.float64,
                )
                classes = np.asarray(model.classes_, dtype=np.int64)
                fold_scores.append(
                    self._score(labels[validation_indices], probabilities, classes)
                )
            mean_score = float(np.mean(fold_scores))
            self.validation_scores_.append(
                {
                    "parameters": dict(parameters),
                    "fold_scores": fold_scores,
                    "mean_score": mean_score,
                }
            )
            key = (mean_score, -candidate_index)
            if best_key is None or key > best_key:
                best_key = key
                self.best_parameters_ = dict(parameters)
        if self.best_parameters_ is None:
            raise RuntimeError("context tuning did not select a candidate")
        self.best_model_ = self._build(self.best_parameters_, self.seed)
        self.best_model_.fit(values, labels)
        self.classes_ = np.asarray(self.best_model_.classes_, dtype=np.int64)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("context-tuned decoder is not fitted")
        return np.asarray(self.best_model_.predict_proba(values), dtype=np.float64)

    def rtfm_provenance(self) -> dict[str, Any]:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "base_decoder": self.decoder,
            "selection_population": "context_only_stratified_cross_validation",
            "selection_metric": "accuracy" if self.multiclass else "roc_auc",
            "folds": CONTEXT_TUNING_FOLDS,
            "candidate_grid": [dict(parameters) for parameters in self._candidates()],
            "best_parameters": self.best_parameters_,
            "validation_scores": self.validation_scores_,
            "seed": self.seed,
        }


def _nominal_logistic_classifier(
    seed: int,
    categorical_features_indices: tuple[int, ...],
) -> Any:
    categorical = tuple(sorted(set(map(int, categorical_features_indices))))
    if not categorical:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        )
    encoder_options: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
        encoder_options["sparse_output"] = True
    else:
        encoder_options["sparse"] = True
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(**encoder_options), list(categorical)),
        ],
        remainder=StandardScaler(),
    )
    return make_pipeline(
        preprocessor,
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )


class NativeCategoricalCatBoostClassifier:
    """CatBoost wrapper preserving nominal columns through fit and prediction."""

    def __init__(
        self,
        seed: int,
        categorical_features_indices: tuple[int, ...],
        *,
        multiclass: bool,
    ) -> None:
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError as error:
            raise RuntimeError("install the 'tree' extra to use CatBoost") from error
        self.seed = int(seed)
        self.categorical_features_indices = tuple(
            sorted(set(map(int, categorical_features_indices)))
        )
        self.model = CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            loss_function="MultiClass" if multiclass else "Logloss",
            random_seed=self.seed,
            thread_count=int(os.environ.get("RTFM_CPU_THREADS", "4")),
            allow_writing_files=False,
            verbose=False,
            cat_features=list(self.categorical_features_indices),
        )
        self.classes_: np.ndarray | None = None

    def _frame(self, values: np.ndarray) -> Any:
        import pandas as pd

        values = np.asarray(values)
        frame = pd.DataFrame(values)
        for column in self.categorical_features_indices:
            observed = np.asarray(values[:, column], dtype=np.float64)
            if not np.isfinite(observed).all() or not np.allclose(
                observed,
                np.rint(observed),
            ):
                raise ValueError("nominal CatBoost columns must contain finite integers")
            frame[column] = np.rint(observed).astype(np.int64).astype(str)
        return frame

    def fit(
        self,
        values: np.ndarray,
        labels: np.ndarray,
    ) -> NativeCategoricalCatBoostClassifier:
        self.model.fit(self._frame(values), np.asarray(labels, dtype=np.int64))
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._frame(values)), dtype=np.float64)

    def rtfm_provenance(self) -> dict[str, Any]:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "base_decoder": "catboost",
            "nominal_handling": "native_cat_features",
            "categorical_features_indices": list(self.categorical_features_indices),
            "parameters": self.model.get_params(),
            "seed": self.seed,
        }


def tabm_training_identity() -> dict[str, Any]:
    """Frozen training semantics for the from-scratch TabM-D comparator."""

    return {
        "package": "pytabkit",
        "package_version": TABM_PACKAGE_VERSION,
        "model": "TabM_D_Classifier",
        "method": "TabM-D",
        "device_request": "cuda",
        "n_cv": TABM_N_CV,
        "n_refit": TABM_N_REFIT,
        "val_fraction": TABM_VAL_FRACTION,
        "n_threads": TABM_N_THREADS,
        "hyperparameter_policy": "pytabkit_1.7.3_tabm_d_defaults",
        "selection_population": "context_only_internal_validation",
    }


class TabMClassifier:
    """PyTabKit TabM-D classifier trained only on the supplied labeled context."""

    def __init__(self, seed: int, categorical_features_indices: tuple[int, ...]) -> None:
        try:
            from pytabkit import TabM_D_Classifier
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "make the pinned PyTabKit 1.7.3 package available to use TabM"
            ) from error
        self.seed = int(seed)
        self.categorical_features_indices = tuple(categorical_features_indices)
        identity = tabm_training_identity()
        self.model = TabM_D_Classifier(
            device=os.environ.get("RTFM_DEVICE", str(identity["device_request"])),
            random_state=self.seed,
            n_cv=int(identity["n_cv"]),
            n_refit=int(identity["n_refit"]),
            val_fraction=float(identity["val_fraction"]),
            n_threads=int(identity["n_threads"]),
            verbosity=0,
        )
        self.classes_: np.ndarray | None = None
        self.training_rows_: int | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> TabMClassifier:
        values = np.asarray(values)
        labels = np.asarray(labels, dtype=np.int64)
        categorical = np.zeros(values.shape[1], dtype=bool)
        if self.categorical_features_indices:
            categorical[list(self.categorical_features_indices)] = True
        self.model.fit(
            values,
            labels,
            cat_indicator=categorical if categorical.any() else None,
        )
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        self.training_rows_ = len(labels)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TabM classifier is not fitted")
        return np.asarray(self.model.predict_proba(np.asarray(values)), dtype=np.float64)

    def rtfm_provenance(self) -> dict[str, Any]:
        observed_version = importlib.metadata.version("pytabkit")
        if observed_version != TABM_PACKAGE_VERSION:
            raise RuntimeError(
                f"PyTabKit version changed: {observed_version} != {TABM_PACKAGE_VERSION}"
            )
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **tabm_training_identity(),
            "random_state": self.seed,
            "categorical_features_indices": list(self.categorical_features_indices),
            "training_rows": self.training_rows_,
        }


def tabicl_frozen_identity() -> dict[str, Any]:
    """Pinned inference semantics for the frozen TabICL classifier."""

    return {
        "package": "tabicl",
        "package_version": TABICL_PACKAGE_VERSION,
        "checkpoint_version": TABICL_CHECKPOINT_VERSION,
        "checkpoint_sha256": TABICL_CHECKPOINT_SHA256,
        "n_estimators": TABICL_N_ESTIMATORS,
        "batch_size": TABICL_BATCH_SIZE,
        "prediction_batch_size": TABICL_PREDICTION_BATCH_SIZE,
        "average_logits": True,
        "support_many_classes": True,
        "use_amp": True,
        "use_fa3": False,
        "offload_mode": False,
        "kv_cache": True,
    }


@lru_cache(maxsize=4)
def _verified_tabicl_checkpoint(path: str) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"TabICL checkpoint is missing: {checkpoint}")
    observed_hash = file_sha256(checkpoint)
    if observed_hash != TABICL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "TabICL checkpoint hash changed: "
            f"{observed_hash} != {TABICL_CHECKPOINT_SHA256}"
        )
    return {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": observed_hash,
    }


class FrozenTabICLClassifier:
    """TabICL classifier conditioned only on the registered labeled context."""

    def __init__(self, seed: int) -> None:
        try:
            from tabicl import TabICLClassifier
        except ModuleNotFoundError as error:
            raise RuntimeError("install the pinned TabICL package to use TabICL") from error
        observed_version = importlib.metadata.version("tabicl")
        if observed_version != TABICL_PACKAGE_VERSION:
            raise RuntimeError(
                f"TabICL version changed: {observed_version} != {TABICL_PACKAGE_VERSION}"
            )
        checkpoint_path = os.environ.get("TABICL_MODEL_PATH")
        if not checkpoint_path:
            raise RuntimeError("TABICL_MODEL_PATH must point to the pinned classifier checkpoint")
        self.checkpoint = _verified_tabicl_checkpoint(checkpoint_path)
        self.seed = int(seed)
        identity = tabicl_frozen_identity()
        self.model = TabICLClassifier(
            n_estimators=int(identity["n_estimators"]),
            batch_size=int(identity["batch_size"]),
            average_logits=bool(identity["average_logits"]),
            support_many_classes=bool(identity["support_many_classes"]),
            model_path=self.checkpoint["path"],
            allow_auto_download=False,
            device=os.environ.get("RTFM_DEVICE", "cuda"),
            use_amp=bool(identity["use_amp"]),
            use_fa3=bool(identity["use_fa3"]),
            offload_mode=bool(identity["offload_mode"]),
            random_state=self.seed,
            verbose=False,
        )
        self._rtfm_prediction_batch_size = int(identity["prediction_batch_size"])
        self.classes_: np.ndarray | None = None
        self.training_rows_: int | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> FrozenTabICLClassifier:
        values = np.asarray(values)
        labels = np.asarray(labels, dtype=np.int64)
        self.model.fit(values, labels, kv_cache=True)
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        self.training_rows_ = len(labels)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TabICL classifier is not fitted")
        return np.asarray(self.model.predict_proba(np.asarray(values)), dtype=np.float64)

    def rtfm_provenance(self) -> dict[str, Any]:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **tabicl_frozen_identity(),
            "device_request": os.environ.get("RTFM_DEVICE", "cuda"),
            "random_state": self.seed,
            "training_rows": self.training_rows_,
            "checkpoints": [dict(self.checkpoint)],
        }


def frozen_tabpfn_identity() -> dict[str, Any]:
    return {
        "package": "tabpfn",
        "model_version": TABPFN_MODEL_VERSION,
        "n_estimators": 4,
        "fit_mode": TABPFN_FIT_MODE,
        "inference_precision": "auto",
        "memory_saving_mode": "auto",
    }


def tabpfn_prediction_batch_size() -> int:
    value = int(
        os.environ.get(
            "RTFM_TABPFN_PREDICTION_BATCH_SIZE",
            str(TABPFN_PREDICTION_BATCH_SIZE),
        )
    )
    if value <= 0:
        raise ValueError("RTFM_TABPFN_PREDICTION_BATCH_SIZE must be positive")
    return value


def build_classifier(
    decoder: str,
    seed: int,
    categorical_features_indices: tuple[int, ...] = (),
    *,
    multiclass: bool = False,
) -> Any:
    if decoder == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        )
    if decoder == "logistic_nominal":
        return _nominal_logistic_classifier(seed, categorical_features_indices)
    if decoder == "logistic_tuned":
        return ContextTunedClassifier("logistic", seed, multiclass=multiclass)
    if decoder == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                early_stopping=True,
                max_iter=500,
                random_state=seed,
            ),
        )
    if decoder == "gru_sequence":
        if not multiclass:
            raise ValueError("gru_sequence is registered only for multiclass path tasks")
        return SharedSequenceGRUClassifier(seed)
    if decoder in {"catboost", "catboost_depth1"}:
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError as error:
            raise RuntimeError("install the 'tree' extra to use CatBoost") from error
        return CatBoostClassifier(
            iterations=500,
            depth=1 if decoder == "catboost_depth1" else 7,
            learning_rate=0.05,
            loss_function="MultiClass" if multiclass else "Logloss",
            random_seed=seed,
            thread_count=int(os.environ.get("RTFM_CPU_THREADS", "4")),
            allow_writing_files=False,
            verbose=False,
        )
    if decoder == "catboost_nominal":
        return NativeCategoricalCatBoostClassifier(
            seed,
            categorical_features_indices,
            multiclass=multiclass,
        )
    if decoder == "catboost_tuned":
        return ContextTunedClassifier("catboost", seed, multiclass=multiclass)
    if decoder == "tabm":
        if multiclass:
            raise ValueError("the registered TabM comparison is binary only")
        return TabMClassifier(seed, categorical_features_indices)
    if decoder == "tabicl":
        if multiclass:
            raise ValueError("the registered TabICL comparison is binary only")
        return FrozenTabICLClassifier(seed)
    if decoder == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ModuleNotFoundError as error:
            raise RuntimeError("install the 'tfm' extra to use TabPFN") from error
        options = {
            "device": os.environ.get("RTFM_DEVICE", "cuda"),
            "n_estimators": 4,
            "auto_scale_n_estimators": False,
            "random_state": seed,
            "ignore_pretraining_limits": True,
            "inference_precision": "auto",
            "fit_mode": TABPFN_FIT_MODE,
            "memory_saving_mode": "auto",
            "keep_cache_on_device": TABPFN_KEEP_CACHE_ON_DEVICE,
            "n_preprocessing_jobs": 1,
            "categorical_features_indices": list(categorical_features_indices),
        }
        signature = inspect.signature(TabPFNClassifier.__init__)
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            options = {
                key: value for key, value in options.items() if key in signature.parameters
            }
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3, **options)
        model._rtfm_requested_options = options
        model._rtfm_prediction_batch_size = tabpfn_prediction_batch_size()
        return model
    if decoder == "limix2m":
        from relational_tfm.decoders.limix import LimiX2MClassifier

        return LimiX2MClassifier(seed)
    raise ValueError(f"decoder is not implemented: {decoder}")


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    batch_size = getattr(model, "_rtfm_prediction_batch_size", None)
    if batch_size is None or len(values) <= int(batch_size):
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    else:
        probabilities = np.concatenate(
            [
                np.asarray(
                    model.predict_proba(values[start : start + batch_size]),
                    dtype=np.float64,
                )
                for start in range(0, len(values), int(batch_size))
            ],
            axis=0,
        )
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or matches.size != 1:
        raise ValueError("classifier does not expose a unique positive-class probability")
    result = probabilities[:, int(matches[0])]
    if not np.isfinite(result).all():
        raise ValueError("classifier returned non-finite probabilities")
    return result


def positive_ranking_score(model: Any, values: np.ndarray) -> np.ndarray:
    """Return the least-compressed score available for binary ranking.

    Binary ranking depends only on score order.  A logistic sigmoid can round
    distinct large decision values to identical probabilities, introducing
    artificial ties.  Prefer the model's decision function when its orientation
    relative to class 1 is explicit; otherwise retain the probability fallback.
    """
    values = np.asarray(values)
    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(values), dtype=np.float64)
        classes = np.asarray(model.classes_)
        if scores.ndim == 1 and classes.shape == (2,):
            matches = np.flatnonzero(classes == 1)
            if matches.size != 1:
                raise ValueError("classifier does not expose a unique positive class")
            if int(matches[0]) == 0:
                scores = -scores
            if not np.isfinite(scores).all():
                raise ValueError("classifier returned non-finite decision scores")
            return scores
    return positive_probability(model, values)


def positive_decision_orientation(model: Any, values: np.ndarray) -> int:
    """Return whether a one-column binary decision score is increasing or decreasing."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("decision orientation requires a one-column interface")
    finite = values[np.isfinite(values[:, 0]), 0]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return 0
    probes = np.asarray([[finite.min()], [finite.max()]], dtype=values.dtype)
    scores = positive_ranking_score(model, probes)
    difference = float(scores[1] - scores[0])
    return int(np.sign(difference))


def class_probabilities(
    model: Any, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    batch_size = getattr(model, "_rtfm_prediction_batch_size", None)
    if batch_size is None or len(values) <= int(batch_size):
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    else:
        probabilities = np.concatenate(
            [
                np.asarray(
                    model.predict_proba(values[start : start + batch_size]),
                    dtype=np.float64,
                )
                for start in range(0, len(values), int(batch_size))
            ],
            axis=0,
        )
    classes = np.asarray(model.classes_, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(values), len(classes))
        or len(np.unique(classes)) != len(classes)
    ):
        raise ValueError("classifier exposes invalid class probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("classifier returned non-finite probabilities")
    return probabilities, classes


def decoder_provenance(model: Any) -> dict[str, Any]:
    if hasattr(model, "rtfm_provenance"):
        return dict(model.rtfm_provenance())
    payload: dict[str, Any] = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
    }
    if model.__class__.__module__.startswith("catboost"):
        parameters = model.get_params()
        payload.update(
            {
                "package": "catboost",
                "package_version": importlib.metadata.version("catboost"),
                "parameters": {
                    key: parameters[key]
                    for key in (
                        "allow_writing_files",
                        "depth",
                        "iterations",
                        "learning_rate",
                        "loss_function",
                        "random_seed",
                        "thread_count",
                    )
                    if key in parameters
                },
            }
        )
    if model.__class__.__module__.startswith("tabpfn"):
        payload.update(
            {
                "model_version": TABPFN_MODEL_VERSION,
                "package": "tabpfn",
                "package_version": importlib.metadata.version("tabpfn"),
                "n_estimators": int(model.n_estimators),
                "actual_n_estimators": int(
                    getattr(model, "n_estimators_", model.n_estimators)
                ),
                "auto_scale_n_estimators": getattr(
                    model, "auto_scale_n_estimators", "unsupported_by_package"
                ),
                "prediction_batch_size": int(model._rtfm_prediction_batch_size),
                "device_request": str(model.device),
                # `devices_` is a fitted attribute. A cell whose context features are
                # all constant is unfittable for TabPFN, and the runner records it at the
                # collision-conditioned ceiling rather than aborting the shard; an empty
                # list there means no device was ever resolved because no fit ran. The
                # checkpoint identity below comes from `model_path` and is still verified.
                "resolved_devices": [
                    str(device) for device in getattr(model, "devices_", ())
                ],
                "fit_mode": str(model.fit_mode),
                "inference_precision": str(model.inference_precision),
                "memory_saving_mode": str(model.memory_saving_mode),
                "keep_cache_on_device": getattr(
                    model, "keep_cache_on_device", "unsupported_by_package"
                ),
                "random_state": int(model.random_state),
            }
        )
        model_paths = model.model_path if isinstance(model.model_path, list) else [model.model_path]
        checkpoints = []
        for model_path in model_paths:
            path = Path(model_path).expanduser().resolve()
            checkpoint: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                checkpoint["bytes"] = path.stat().st_size
                checkpoint["sha256"] = file_sha256(path)
            checkpoints.append(checkpoint)
        payload["checkpoints"] = checkpoints
        if len(checkpoints) == 1 and "sha256" in checkpoints[0]:
            payload["checkpoint_sha256"] = checkpoints[0]["sha256"]
    return payload
