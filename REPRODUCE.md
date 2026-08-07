# Reproduction Guide

## Environment

Python 3.11 or 3.12 is supported. With `uv`:

```bash
uv sync --extra dev --extra tree --extra plots
```

Install the frozen tabular decoder only when rerunning its inference cells:

```bash
uv sync --extra dev --extra tree --extra plots --extra tfm
```

## Validate the artifact

```bash
python tools/verify_artifact.py
MPLCONFIGDIR=.matplotlib uv run pytest -q
```

The first command checks the final manifest, anonymity-sensitive strings, required result matrices, and panel cardinalities.

## Recreate manuscript figures

```bash
mkdir -p generated/figures
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_component_audit_anatomy.py
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_clutrr_ablation.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_composition_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_context_robustness_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_query_evidence_figure.py
```

Outputs are written to `generated/figures/`.

## Rerunning cells

The package entry point is `rtfm`. Study YAML files in `configs/studies/` specify the controlled, CLUTRR, GraphLog, and dihedral experiments. The 57-KG analyses use checksum-locked frozen arrays created by the asset builders in `analysis/`; these arrays and model checkpoints are not duplicated in this repository because of size and licensing constraints.

The committed per-dataset, per-family, per-world, per-seed, and query-stratum result tables allow all paper-level statistics to be recomputed without model inference. Raw per-candidate predictions belong to the archival artifact and are intentionally omitted from this compact review repository.
