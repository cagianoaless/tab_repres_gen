from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from relational_tfm.protocol import FROZEN_KG_REVIEWER_MATCHED_FILTER_ASSET_ID

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "generate_ultra4g_control_registration.py"
)
SPEC = importlib.util.spec_from_file_location("reviewer_kg_registration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(path: Path) -> Path:
    payload = {
        "asset_id": FROZEN_KG_REVIEWER_MATCHED_FILTER_ASSET_ID,
        "context_modes": ["unfiltered_uniform", "filtered_uniform"],
        "datasets": {
            "tiny": {
                "num_entities": 5,
                "transfer_regime": "transductive",
                "archive_sha256": "a" * 64,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_profiles_encode_distinct_interventions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    for profile in MODULE.PROFILES:
        output = tmp_path / f"{profile}.yaml"
        MODULE.generate_config(manifest, output, profile=profile)
        config = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert config["dataset_specs"][0]["interfaces"] == list(MODULE.INTERFACES)
        if profile == "matched-filter-tabpfn":
            assert config["context_modes"] == [
                "unfiltered_uniform",
                "filtered_uniform",
            ]
            assert "query_transforms" not in config
        else:
            assert config["context_modes"] == ["filtered_uniform"]
            assert config["query_transforms"] == [
                "raw",
                "within_query_zscore",
                "within_query_percentile",
            ]
