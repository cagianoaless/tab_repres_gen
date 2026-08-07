from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from relational_tfm.hashing import file_sha256

LIMIX_REPOSITORY = "https://github.com/limix-ldm-ai/LimiX.git"
LIMIX_SOURCE_COMMIT = "61ede97e37d3eaae9e4500c0bba02965a2370eba"
LIMIX_CHECKPOINT_REPOSITORY = "stable-ai/LimiX-2M"
LIMIX_CHECKPOINT_REVISION = "46dcbce344f61237ddf6c1b41c9ecb3c358e8f14"
LIMIX_CHECKPOINT_FILENAME = "LimiX-2M.ckpt"
LIMIX_CHECKPOINT_SHA256 = "16f385d5201dc17cd6a8eb18becb0b4f8b3603838423b9f06062106acc965513"
LIMIX_CHECKPOINT_BYTES = 9_558_253
LIMIX_CONFIG_RELATIVE_PATH = Path("config/cls_default_2M_retrieval.json")
LIMIX_CONFIG_SHA256 = "51dab9e5bfcc58e8a6d76317a8a921dcc8411ad0b6107d1bd714656c19232917"
LIMIX_PIPELINES = 4


def frozen_model_identity() -> dict[str, Any]:
    return {
        "package": "external-source",
        "package_version": LIMIX_SOURCE_COMMIT,
        "model_version": "LimiX-2M",
        "n_estimators": LIMIX_PIPELINES,
        "source_repository": LIMIX_REPOSITORY,
        "source_commit": LIMIX_SOURCE_COMMIT,
        "checkpoint_repository": LIMIX_CHECKPOINT_REPOSITORY,
        "checkpoint_revision": LIMIX_CHECKPOINT_REVISION,
        "checkpoint_filename": LIMIX_CHECKPOINT_FILENAME,
        "checkpoint_sha256": LIMIX_CHECKPOINT_SHA256,
        "checkpoint_bytes": LIMIX_CHECKPOINT_BYTES,
        "inference_config": str(LIMIX_CONFIG_RELATIVE_PATH),
        "inference_config_sha256": LIMIX_CONFIG_SHA256,
        "retrieval": True,
        "retrieval_len": "dynamic",
        "retrieval_dynamic_ratio": 0.5,
        "retrieval_cluster_count": 47,
        "joint_context_evaluation_preprocessing": True,
        "prediction_batching": "complete-evaluation-set",
    }


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_predictor(source_dir: Path) -> type[Any]:
    sys.path.insert(0, str(source_dir))
    try:
        module = importlib.import_module("inference.predictor")
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).resolve().is_relative_to(source_dir):
        raise ValueError("loaded LimiX predictor from an unexpected source")
    return module.LimiXPredictor


class LimiX2MClassifier:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._predictor: Any | None = None
        self._provenance: dict[str, Any] | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> LimiX2MClassifier:
        import torch

        source_dir = Path(os.environ["RTFM_LIMIX_SOURCE_DIR"]).resolve()
        checkpoint = Path(os.environ["RTFM_LIMIX_CHECKPOINT"]).resolve()
        config = source_dir / LIMIX_CONFIG_RELATIVE_PATH
        if _git(source_dir, "rev-parse", "HEAD") != LIMIX_SOURCE_COMMIT:
            raise ValueError("LimiX source commit does not match the frozen identity")
        if _git(source_dir, "status", "--porcelain"):
            raise ValueError("LimiX source checkout is dirty")
        if checkpoint.stat().st_size != LIMIX_CHECKPOINT_BYTES:
            raise ValueError("LimiX checkpoint size does not match the frozen identity")
        if file_sha256(checkpoint) != LIMIX_CHECKPOINT_SHA256:
            raise ValueError("LimiX checkpoint hash does not match the frozen identity")
        if file_sha256(config) != LIMIX_CONFIG_SHA256:
            raise ValueError("LimiX inference config does not match the frozen identity")
        pipelines = json.loads(config.read_text(encoding="utf-8"))
        if len(pipelines) != LIMIX_PIPELINES or not all(
            item["retrieval_config"]["use_retrieval"] for item in pipelines
        ):
            raise ValueError("LimiX requires the frozen four-pipeline retrieval configuration")
        retrieval_configs = [item["retrieval_config"] for item in pipelines]
        if any(
            item.get("retrieval_len") != "dynamic"
            or item.get("dynamic_ratio") != 0.5
            or item.get("use_cluster") is not True
            or item.get("cluster_num") != 47
            or item.get("use_threshold") is not True
            for item in retrieval_configs
        ):
            raise ValueError("LimiX retrieval settings do not match the frozen identity")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("LimiX-2M requires exactly one visible CUDA device")

        predictor_class = _load_predictor(source_dir)
        self._predictor = predictor_class(
            device=torch.device("cuda"),
            model_path=str(checkpoint),
            inference_config=str(config),
            mix_precision=True,
            seed=self.seed,
        )
        self._x_train = np.asarray(values, dtype=np.float64)
        self._y_train = np.asarray(labels, dtype=np.int64)
        if set(np.unique(self._y_train)) != {0, 1}:
            raise ValueError("LimiX-2M requires both binary classes in context")
        self._provenance = {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **frozen_model_identity(),
            "mix_precision": True,
            "task_type": "Classification",
            "seed": self.seed,
            "predictor_reuse_scope": "none; fresh predictor per cell",
            "checkpoints": [
                {
                    "path": str(checkpoint),
                    "exists": True,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": file_sha256(checkpoint),
                }
            ],
        }
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self._predictor is None or self._x_train is None or self._y_train is None:
            raise ValueError("LimiX-2M must be fitted before prediction")
        output = np.asarray(
            self._predictor.predict(
                self._x_train,
                self._y_train,
                np.asarray(values, dtype=np.float64),
                task_type="Classification",
            ),
            dtype=np.float64,
        )
        if output.shape != (len(values), 2):
            raise ValueError(f"unexpected LimiX prediction shape: {output.shape}")
        if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("LimiX returned invalid class probabilities")
        return output

    def rtfm_provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            raise ValueError("LimiX-2M provenance is unavailable before fitting")
        return self._provenance
