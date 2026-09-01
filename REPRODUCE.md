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

The package entry point is `rtfm`; study configurations are in `configs/studies/`.

## Additional protocol controls

The checked-in outputs can be inspected without private datasets or model
checkpoints. The following commands regenerate the analyses once the corresponding
frozen inputs are available.

### Collision-conditioned CLUTRR ceiling

```bash
uv run python analysis/audit_clutrr_singleton_mass.py \
  --output generated/results/clutrr-collision-conditioned/summary.json
```

### GraphLog capacity sweep

First freeze the plan and execute its shards, then summarize the complete run:

```bash
uv run rtfm plan \
  --config configs/studies/graphlog_symbolic_cpu_catboost_capacity_sweep_reviewer_v1.yaml \
  --output generated/plans/graphlog-capacity-sweep.json
uv run python analysis/run_graphlog_capacity_sweep.py \
  --repo . \
  --plan generated/plans/graphlog-capacity-sweep.json \
  --output generated/runs/graphlog-capacity-sweep
uv run python analysis/summarize_graphlog_capacity_sweep.py \
  --run-root generated/runs/graphlog-capacity-sweep \
  --output-dir generated/results/graphlog-capacity-sweep
```

### Frozen-KG pointwise, query-relative, and filtering controls

`analysis/build_ultra4g_matched_filter_asset.py` constructs the matched context
asset. It preserves positive rows, context query identities, seeds, evaluation
queries, masks, targets, and feature schemas, while changing only whether sampled
negative entities are rejected by the all-known-positive mask. The resulting
manifest is passed to `analysis/generate_ultra4g_control_registration.py`, and the
registered cells are run with `analysis/run_ultra4g_controls.py`. See `--help` on
these scripts for the required frozen asset roots.

The final summaries are regenerated with:

```bash
uv run python analysis/analyze_ultra4g_controls.py \
  --query-cpu-root <query-cpu-run> \
  --query-tabpfn-root <query-tabpfn-run> \
  --matched-tabpfn-root <matched-filter-run> \
  --asset-manifest <matched-asset-manifest> \
  --provenance-map <provenance-map> \
  --output-dir generated/results/ultra4g-controls

uv run python analysis/analyze_ultra4g_aggregation_sensitivity.py
```
