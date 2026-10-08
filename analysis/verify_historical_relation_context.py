"""Reproduce the historical 60-test correction from immutable derived tables only.

Scientific formulas and sign-flip logic are adapted from historical analysis
commit 02c15111be5b3d622f8358a673bdfcc0225fd05f. No raw-run reconstruction,
model execution, prospective fixed-negative requirement, or output writing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("experiments/ultra4g/results/relation-context-historical")
MANIFEST_SHA256 = "4dbff7d9ca488ca95aae1645f5cf2c7d1aaa85164abed8321e6a459791d5303d"
DECODERS = ("catboost", "logistic", "tabpfn")
SCHEMAS = (
    "b0_local_only",
    "b0_local_ultra",
    "b0_local_no_relation_id",
    "b0_local_ultra_no_relation_id",
)
MODES = ("frequency_random_filtered_uniform", "relation_matched_filtered_uniform")
SEEDS = (101, 102, 103)
CATEGORY_COUNTS = {
    "context_sampling": 12,
    "relation_id": 12,
    "ultra_score": 12,
    "sampling_x_relation_id": 6,
    "sampling_x_ultra_score": 6,
    "decoder_x_sampling": 12,
}
KEY = "sampling|catboost|b0_local_no_relation_id"
KEY_EFFECT = 0.013937919134311694
KEY_T = 3.7901691175274235
KEY_P = 0.036895751953125
IDENTITY_RANGE = (-0.003127454409378321, 0.002987582098983631)
COVERAGE_COLUMNS = (
    "evaluation_relation_type_coverage",
    "evaluation_relation_mass_coverage",
    "relation_distribution_tv",
)
COVERAGE = {
    MODES[0]: (0.66736765661346564, 0.6934286714804293, 0.446971985479798),
    MODES[1]: (0.87069340856803934, 0.89126346472537876, 0.18960336766098487),
}


@dataclass(frozen=True)
class Contrast:
    category: str
    # Coefficient, decoder, schema, context mode: all effects are treatment-reference.
    terms: tuple[tuple[int, str, str, str], ...]


def contrast_definitions() -> dict[str, Contrast]:
    """The historical six groups, not head/tail duplicates or negative-policy tests."""
    definitions: dict[str, Contrast] = {}

    def sampling(decoder: str, schema: str) -> tuple[tuple[int, str, str, str], ...]:
        return ((1, decoder, schema, MODES[1]), (-1, decoder, schema, MODES[0]))

    def difference(
        left: tuple[tuple[int, str, str, str], ...],
        right: tuple[tuple[int, str, str, str], ...],
    ) -> tuple[tuple[int, str, str, str], ...]:
        return left + tuple((-coefficient, d, s, m) for coefficient, d, s, m in right)

    pairs = {
        "relation_id": {
            "without_ultra": ("b0_local_only", "b0_local_no_relation_id"),
            "with_ultra": ("b0_local_ultra", "b0_local_ultra_no_relation_id"),
        },
        "ultra_score": {
            "with_relation_id": ("b0_local_ultra", "b0_local_only"),
            "without_relation_id": ("b0_local_ultra_no_relation_id", "b0_local_no_relation_id"),
        },
    }
    for decoder in DECODERS:
        for schema in SCHEMAS:
            definitions[f"sampling|{decoder}|{schema}"] = Contrast(
                "context_sampling", sampling(decoder, schema)
            )
        for category, variants in pairs.items():
            for status, (treatment, reference) in variants.items():
                for mode in MODES:
                    definitions[f"{category}|{decoder}|{status}|{mode}"] = Contrast(
                        category, ((1, decoder, treatment, mode), (-1, decoder, reference, mode))
                    )
                interaction = f"sampling_x_{category}"
                definitions[f"{interaction}|{decoder}|{status}"] = Contrast(
                    interaction,
                    difference(sampling(decoder, treatment), sampling(decoder, reference)),
                )
    for schema in SCHEMAS:
        for left, right in (
            ("tabpfn", "catboost"),
            ("tabpfn", "logistic"),
            ("catboost", "logistic"),
        ):
            definitions[f"decoder_x_sampling|{left}-{right}|{schema}"] = Contrast(
                "decoder_x_sampling", difference(sampling(left, schema), sampling(right, schema))
            )
    return definitions


def require_close(actual: object, expected: object, label: str, tolerance: float = 1e-15) -> None:
    if not np.allclose(actual, expected, atol=tolerance, rtol=0):
        raise ValueError(f"historical derived values disagree: {label}")


def validate_cells(cells: pd.DataFrame) -> None:
    keys = ["dataset", "decoder", "interface", "context_mode", "seed"]
    datasets = sorted(cells.dataset.unique())
    expected = set(itertools.product(datasets, DECODERS, SCHEMAS, MODES, SEEDS))
    observed = set(cells[keys].itertuples(index=False, name=None))
    if len(datasets) != 57 or len(cells) != 4104 or observed != expected:
        raise ValueError("historical 4104-cell matrix is incomplete or duplicated")
    if cells.cell_id.nunique() != 4104 or cells.provenance_family.nunique() != 16:
        raise ValueError("historical cell IDs or provenance families are invalid")
    if not cells.groupby("dataset").provenance_family.nunique().eq(1).all():
        raise ValueError("dataset/family assignment changes across cells")
    widths = {SCHEMAS[0]: 18, SCHEMAS[1]: 19, SCHEMAS[2]: 17, SCHEMAS[3]: 18}
    if not cells.feature_count.eq(cells.interface.map(widths)).all():
        raise ValueError("historical feature widths changed")
    if not cells.context_per_class.eq(200).all():
        raise ValueError("historical context budget changed")
    for status in ("decoder_fit_status", "decoder_prediction_status"):
        if not cells[status].eq("succeeded").all():
            raise ValueError(f"unsuccessful historical cells: {status}")
    for direction in ("combined", "head", "tail"):
        values = cells[f"{direction}_mrr"]
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError("invalid historical MRR")
    require_close(cells.combined_mrr, (cells.head_mrr + cells.tail_mrr) / 2, "head/tail mean")


def family_matrix(family: pd.DataFrame) -> pd.DataFrame:
    selected = family.loc[family.direction == "combined"]
    definitions = contrast_definitions()
    if len(selected) != 960 or set(selected.contrast_id) != set(definitions):
        raise ValueError("historical combined contrasts are incomplete or duplicated")
    if selected.groupby("contrast_id").contrast_family.nunique().ne(1).any():
        raise ValueError("inconsistent contrast categories")
    categories = selected.drop_duplicates("contrast_id").set_index("contrast_id").contrast_family
    if categories.value_counts().to_dict() != CATEGORY_COUNTS:
        raise ValueError("historical 60-contrast category counts changed")
    if any(categories[key] != definition.category for key, definition in definitions.items()):
        raise ValueError("historical contrast category/ID mismatch")
    if selected.duplicated(["provenance_family", "contrast_id"]).any():
        raise ValueError("duplicate historical family effects")
    matrix = selected.pivot(index="provenance_family", columns="contrast_id", values="effect")
    matrix = matrix.sort_index(axis=1)
    if matrix.shape != (16, 60) or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("historical 16 x 60 matrix is incomplete or invalid; no contrasts dropped")
    return matrix


def t_statistic(values: np.ndarray) -> np.ndarray:
    """Historical one-sample t, including its zero-variance convention."""
    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    statistic = np.divide(means, errors, out=np.zeros_like(means), where=errors > 0.0)
    constant = (errors == 0.0) & (means != 0.0)
    statistic[constant] = np.sign(means[constant]) * np.inf
    return statistic


def exact_max_t(base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same sign per provenance family across all contrasts, with exact enumeration."""
    if base.ndim != 2 or not 2 <= base.shape[0] <= 16 or base.shape[1] < 1:
        raise ValueError("invalid family-effect matrix dimensions")
    if not np.isfinite(base).all():
        raise ValueError("nonfinite family effects")
    observed = t_statistic(base)
    raw = np.zeros(base.shape[1], dtype=np.int64)
    adjusted = np.zeros(base.shape[1], dtype=np.int64)
    for signs in itertools.product((-1.0, 1.0), repeat=base.shape[0]):
        statistic = np.abs(t_statistic(base * np.asarray(signs, dtype=float)[:, None]))
        raw += statistic >= np.abs(observed) - 1e-12
        adjusted += statistic.max() >= np.abs(observed) - 1e-12
    return observed, raw / (2 ** base.shape[0]), adjusted / (2 ** base.shape[0])


def verify(root: Path = ROOT) -> dict[str, object]:
    directory = root / RESULTS
    manifest_bytes = (directory / "analysis_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("historical package manifest hash changed")
    manifest = json.loads(manifest_bytes)
    analysis_commit = "02c15111be5b3d622f8358a673bdfcc0225fd05f"
    if manifest["analysis_commit"] != analysis_commit:
        raise ValueError("historical analysis identity changed")
    if len(manifest["files"]) != 12:
        raise ValueError("historical package inventory changed")
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("unsafe historical package path")
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
            raise ValueError(f"historical package hash changed: {name}")

    audit = json.loads((directory / "completion_audit.json").read_text())
    for key, expected in {
        "status": "PASS",
        "cells": 4104,
        "datasets": 57,
        "provenance_families": 16,
        "combined_contrasts": 60,
        "analysis_commit": analysis_commit,
        "asset_id": "rfmlink_ultra57_relation_context_v1",
        "asset_manifest_sha256": "8fef670b0cdf4574da81bf5d35b7eca8613cdac515988a03e1c6f0ca6764cd9e",
    }.items():
        if audit[key] != expected:
            raise ValueError(f"historically recorded audit identity changed: {key}")
    # Recorded identities only: missing upstream plans are not independently rehashed.
    for name, cells_expected, plan_hash in (
        ("cpu_run", 2736, "bd836b248fdb2812206087fae0c5ee88f5e9035cb9450c5d6d5da7893e9ccee2"),
        ("tabpfn_run", 1368, "09461de58b61803baad894e8d3fb4e1da3980c13410fe36decac195c43553b7a"),
    ):
        record = audit[name]
        if (
            record["cells"] != cells_expected
            or record["plan_sha256"] != plan_hash
            or record["source_commit"] != "a48dd52dfdedcf38428c489909b1f31be27ca3d9"
        ):
            raise ValueError(f"historically recorded execution identity changed: {name}")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(directory / name, float_precision="round_trip")

    cells = read("cell_metrics.csv")
    validate_cells(cells)
    family = read("contrast_by_family.csv")
    matrix = family_matrix(family)
    definitions = contrast_definitions()
    # Rebuild combined-direction effects from seed-averaged historical cell MRR.
    dataset = (
        cells.groupby(["dataset", "provenance_family", "decoder", "interface", "context_mode"])
        .combined_mrr.mean()
        .unstack(["decoder", "interface", "context_mode"])
    )
    effects = pd.DataFrame(
        {
            key: sum(coefficient * dataset[(d, s, m)] for coefficient, d, s, m in spec.terms)
            for key, spec in definitions.items()
        }
    )
    dataset_effects = read("contrast_by_dataset.csv")
    selected = dataset_effects.loc[dataset_effects.direction == "combined"]
    if len(selected) != 3420 or selected.duplicated(["dataset", "contrast_id"]).any():
        raise ValueError("historical dataset contrasts incomplete or duplicated")
    stored_dataset = selected.pivot(index="dataset", columns="contrast_id", values="effect")
    rebuilt_dataset = effects.reset_index("provenance_family", drop=True)
    stored_dataset = stored_dataset.reindex(index=rebuilt_dataset.index, columns=effects.columns)
    require_close(rebuilt_dataset.to_numpy(), stored_dataset.to_numpy(), "dataset contrasts")
    rebuilt = (
        effects.groupby("provenance_family")
        .mean()
        .reindex(index=matrix.index, columns=matrix.columns)
    )
    require_close(rebuilt.to_numpy(), matrix.to_numpy(), "equal-family effects")
    summary = read("contrast_summary.csv")
    summary = summary.loc[summary.direction == "combined"].set_index("contrast_id")
    if len(summary) != 60 or not summary.index.is_unique or set(summary.index) != set(definitions):
        raise ValueError("historical combined summary incomplete or duplicated")
    summary = summary.reindex(matrix.columns)
    require_close(matrix.mean().to_numpy(), summary.equal_family_effect, "contrast summary")

    observed, raw, adjusted = exact_max_t(matrix.to_numpy(dtype=float))
    stored = read("combined_exact_max_t.csv").set_index("contrast_id")
    if len(stored) != 60 or not stored.index.is_unique or set(stored.index) != set(definitions):
        raise ValueError("historical max-t table incomplete or duplicated")
    stored = stored.reindex(matrix.columns)
    for table in (stored, summary):
        if not table.sign_patterns.eq(65536).all() or not table.joint_contrasts.eq(60).all():
            raise ValueError("historical correction size changed")
        require_close(observed, table.observed_t, "observed t", tolerance=1e-12)
        if not np.array_equal(raw, table.exact_two_sided_p):
            raise ValueError("historical exact sign-flip p-values disagree")
        if not np.array_equal(adjusted, table.exact_max_t_adjusted_p):
            raise ValueError("historical exact 60-test adjusted p-values disagree")
    require_close(summary.loc[KEY, "equal_family_effect"], KEY_EFFECT, "key effect")
    require_close(stored.loc[KEY, "observed_t"], KEY_T, "key t", tolerance=1e-12)
    if stored.loc[KEY, "exact_max_t_adjusted_p"] != KEY_P:
        raise ValueError("historical key adjusted p changed")
    identities = summary.loc[summary.contrast_family == "relation_id"]
    identity_range = (identities.equal_family_effect.min(), identities.equal_family_effect.max())
    if len(identities) != 12 or identities.exact_max_t_adjusted_p.lt(0.05).any():
        raise ValueError("historical identity contrast count/significance changed")
    require_close(identity_range, IDENTITY_RANGE, "with-minus-without identity range")

    coverage = read("context_distribution_audit.csv")
    keys = ["dataset", "context_mode", "seed"]
    expected = set(itertools.product(sorted(cells.dataset.unique()), MODES, SEEDS))
    if len(coverage) != 342 or set(coverage[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("historical coverage matrix incomplete or duplicated")
    mapping = cells.groupby("dataset").provenance_family.first()
    if not coverage.provenance_family.eq(coverage.dataset.map(mapping)).all():
        raise ValueError("coverage provenance assignment differs from cell matrix")
    coverage_dataset = coverage.groupby(["dataset", "provenance_family", "context_mode"])[
        list(COVERAGE_COLUMNS)
    ].mean()
    coverage_family = coverage_dataset.groupby(["provenance_family", "context_mode"]).mean()
    aggregate = coverage_family.groupby("context_mode").mean()
    for mode, expected_values in COVERAGE.items():
        require_close(aggregate.loc[mode].to_numpy(), expected_values, "equal-family coverage")
    return {
        "status": "PASS",
        "boundary": "historical derived-result reproducibility, not raw-run reproducibility",
        "cells": len(cells),
        "datasets": cells.dataset.nunique(),
        "provenance_families": matrix.shape[0],
        "category_counts": CATEGORY_COUNTS,
        "joint_contrasts": matrix.shape[1],
        "sign_patterns": 65536,
        "coverage": aggregate.to_dict(orient="index"),
        "key_effect": float(summary.loc[KEY, "equal_family_effect"]),
        "key_observed_t": float(stored.loc[KEY, "observed_t"]),
        "key_adjusted_p": float(stored.loc[KEY, "exact_max_t_adjusted_p"]),
        "identity_direction": "with identity minus without identity",
        "identity_range": identity_range,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
