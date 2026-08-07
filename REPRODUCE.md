# Reproduction Guide

## Environment

Python 3.11 or 3.12:

```bash
uv sync --frozen --extra dev --extra tree --extra plots
```

For decoder inference:

```bash
uv sync --extra dev --extra tree --extra plots --extra tfm
```

## Validate

```bash
uv run python tools/verify_artifact.py
MPLCONFIGDIR=.matplotlib uv run pytest -q
```

## Recreate figures

```bash
mkdir -p generated/figures
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_component_audit_anatomy.py
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_clutrr_ablation.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_composition_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_context_robustness_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_query_evidence_figure.py
```

## Rerunning cells

The package entry point is `rtfm`; study configurations are in `configs/studies/`. Scripts requiring raw arrays or checkpoints cannot run from this compact repository. Aggregate analyses and figures use the committed tables.

## Maintainer check

After changing tracked files:

```bash
python tools/build_manifest.py
python tools/verify_artifact.py
```
