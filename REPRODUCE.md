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

These checks need no model checkpoint or KG runtime archive. `MANIFEST.sha256`
covers all release files except itself. During development it also covers new
non-ignored files before they are staged. Refresh it with
`uv run python tools/build_manifest.py` after intentional release edits.
Historical result tables must not be overwritten to satisfy verification.
`tools/historical_results.json` separately pins all 188 original result and
publication files at the audit HEAD. Refreshing the release manifest cannot
silently change these immutable historical bytes.

## A. Rerunnable from public assets

Stage the CLUTRR assets (all six CSVs) and GraphLog ZIP using the URLs and SHA-256
values in `data/manifests/external_assets.yaml`:

```bash
uv run rtfm stage-data --asset clutrr_v1_gen23_train \
  --asset clutrr_v1_gen23_validation --asset clutrr_v1_gen23_test \
  --asset clutrr_v1_gen234_train --asset clutrr_v1_gen234_validation \
  --asset clutrr_v1_gen234_test --asset graphlog_v1_1_author_mirror
```

Synthetic calibration and dihedral populations are generated from frozen config
seeds. CPU logistic/CatBoost cells use public/generated data; the recurrent GRU
control additionally requires PyTorch. TabPFN needs the pinned TabPFN-3 checkpoint
and its access terms. Installing the `tfm` extra does not guarantee model access.
The default suite/CI uses small CPU model fixtures, never TabPFN/ULTRA checkpoint
inference or GPU experiments.

For example, freeze a CLUTRR plan and run a small CPU shard:

```bash
uv run rtfm plan --config configs/studies/clutrr_symbolic_cpu_confirmation_v2.yaml \
  --output generated/plans/clutrr-cpu.json
uv run rtfm run-shard --plan generated/plans/clutrr-cpu.json \
  --shard-index 0 --cells-per-shard 1 --output-dir generated/runs/clutrr/shard-0000
```

Use the same source code/configs and retain plan, shard, runtime and prediction
manifests. CPU reruns can still be substantial; the capacity sweep has 2394 cells.
Some aggregate scripts require historical raw per-cell archives which are absent;
they can consume new complete runs but cannot invert checked-in aggregates.

## B. Downstream reruns requiring frozen KG assets

The supported boundary is:

```text
raw KG -> support graph -> ULTRA inference -> local/global feature construction
       [external upstream pipeline]
    -> frozen RFMLink manifest + dataset NPZ archives
       [repository loader -> decoder cells -> analyses -> publication inputs]
```

Set `RTFM_FROZEN_KG_ASSET_ROOT` to a supplied archive directory containing
`manifest.json` and `<dataset>.npz`. Runtime registration in
`src/relational_tfm/protocol.py` checks the asset ID, reads its dataset archive
hashes and freezes the actual manifest-byte hash in each plan. The loader checks
both manifest and archive bytes. Runtime registration has no universal fixed
large-panel hash: individual historical analyses may pin a particular source
manifest/plan. Do not substitute hashes from the three-dataset pilot.

| Asset ID | Purpose |
|---|---|
| `rfmlink_frozen_interface_ultra57_v2` | 57-KG q64 encoder ladder |
| `rfmlink_ultra4g_negative_robustness_v1` | Filtered-uniform vs role/degree negative contexts |
| `rfmlink_ultra57_reviewer_matched_filtering_v1` | Matched all-known-positive filtering |
| `rfmlink_ultra57_relation_context_v1` | Positive-context matching and relation-ID controls (not supplied) |
| `rfmlink_frozen_interface_ultra57_hard_context_v1` | Legacy hard-context study |
| `rfmlink_graphlet_context_pilot_v1` | Legacy graphlet pilot |

Already recorded historical identities (not newly inferred provenance): the
context-robustness audit pins manifest
`0c886c53bcc6866dac8e21503b5bac816ab47c472ae932d9e2ab6145c0f255e7`.
`analysis/analyze_ultra4g_decoder_compatibility.py` requires manifest
`6c3977f6e14e89a636415d96a78ad638f0cc550df5b04b960a8b423a9bebb317`
and CPU plan
`1efdbf85f1bf4c3db9997b6b5d78e86d6ed14415b5fffda1272dea303357fac0`.
These are different source registrations; their input archives/runs are not
included. Use each analysis's own registration, not a universal replacement hash.

The separately registered pilots `rfmlink_frozen_interface_q64_v1` and
`rfmlink_source_panel_q64_v1` have literal manifest/archive SHA-256 values in
`protocol.py`; they are not the complete 57-KG panel. q64 means 64 sampled test
triples, expanded to head/tail queries, ranked against every entity with frozen
filtered masks. It is not a candidate shortlist. Positive/negative context
selection and equal-family/equal-dataset averaging retain their original rules.

Archive fields used downstream include `candidate_b0`, `candidate_b2v2`,
`num_entities`, `filtered`, `target_entities`, `side_codes`, `anchor_entities`,
and per-seed context matrices/labels/triples. Feature names are in the manifest.
Context prefixes are `context_<seed>` or `context_<mode>_<seed>`.
Query-relative controls additionally require context directional query IDs.
Negative-context and matched-filter builders require the original external q512
candidate/query inputs and extraction implementation; their `--help` options list
the inputs. The complete upstream feature/ULTRA materializer is not shipped.

The positive-context analysis is described in
`experiments/ultra4g/results/positive-context/README.md`. Its required experiment
matrix and frozen archives are absent, so the real analysis cannot be reproduced
from this clone alone. Acceptance targets are never encoded as output data.

## C. Checked-in evidence and table-only reproduction

`tools/verify_artifact.py` validates release hashes, required inventory and
headline endpoints without rerunning models. Figure scripts below read checked-in
aggregate/publication inputs. Aggregation sensitivity can also be regenerated
directly from the checked-in dataset table:

```bash
uv run python analysis/analyze_ultra4g_aggregation_sensitivity.py
```

Outputs belong in `generated/`; keep `experiments/*/results/` and
`publication_data/` historical. See `PROVENANCE.md` for unresolved source SHAs.

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
