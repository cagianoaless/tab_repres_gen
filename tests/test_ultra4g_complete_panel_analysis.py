import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "analysis" / "analyze_ultra4g_q64_complete_panel.py"
SPEC = importlib.util.spec_from_file_location("rfmlink_complete57", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_provenance_map_has_57_unique_datasets_and_16_families():
    lookup = MODULE.provenance_lookup()
    assert len(lookup) == 57
    assert len(set(lookup.values())) == 16
    assert lookup["dbpedia100k"] == "dbpedia100k"
    assert lookup["yago310"] == "yago310"


def test_effect_summary_weights_families_equally():
    effects = pd.DataFrame(
        {
            "provenance_family": ["large", "large", "large", "small"],
            "effect": [0.9, 0.9, 0.9, -0.1],
        }
    )
    row, family = MODULE.summarize_effect(
        effects, contrast="test", direction="combined", context=20
    )
    assert np.isclose(row["equal_family_effect"], 0.4)
    assert np.isclose(row["equal_dataset_effect"], 0.65)
    assert set(family["provenance_family"]) == {"large", "small"}


def test_complete_panel_contract_is_frozen():
    assert MODULE.CONTEXTS == (20, 50, 100, 200)
    assert MODULE.SEEDS == (101, 102, 103)
    assert len(MODULE.INTERFACES) == 9
