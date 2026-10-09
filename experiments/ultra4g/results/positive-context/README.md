# Prospective positive-context and relation-identity sensitivity utility

## Purpose

`analysis/analyze_ultra4g_positive_context.py` implements a deterministic
prospective analysis using the existing loader selection and statistical helpers.
It compares positive-context constructions while holding exact negative rows
fixed, distinct from the paper's context sampler. No result CSVs are bundled
for this sensitivity.

## Input contract

The analysis requires registered prospective inputs:

- A `manifest.json` with asset ID `rfmlink_ultra57_relation_context_v1`, `datasets` archive hashes,
  19 `b0_feature_names` (relation ID at column 0, ULTRA score at column 18), and
  13 `b2v2_feature_names`, following the existing runtime asset schema. Each
  dataset record needs `transfer_regime`; the manifest also supplies registered
  `context_modes`, `query_selection`, `context_selection` and `scientific_status`.
- One hash-matching `<dataset>.npz` for each of 57 datasets. Evaluation fields:
  `candidate_b0`, `candidate_b2v2`, `filtered`, `num_entities`, `num_relations`,
  `target_entities`, `anchor_entities`, `side_codes`. The 64 test triples have
  128 directional queries with all-entity candidates and frozen filtering masks.
- For each registered mode and seed, `context_<mode>_<seed>_X`, `_b2v2`, `_y`,
  `_triples`; `original_uniform` uses `context_<seed>` instead. Triples have
  head/relation/tail columns. Positives change; selected negative triples and
  features are identical between arms. Budget and evaluation archive remain fixed.
- Registered prospective plans and complete checksum-verified `results/shard-*` outputs from
  the existing runner, including metrics, prediction NPZs, and shard manifests.
  Matrix: 57 datasets × 3 decoders × 4 schema variants × 2 modes × 3 seeds
  (4,104 cells). This prospective contract is distinct from the paper's
  reference matrix, which uses a fixed negative-sampling policy rather than
  exact-fixed-negative pairing.
- A dataset/family CSV with columns `dataset,provenance_family`, covering exactly
  the observed 57 datasets and 16 families. The original mode names, decoder
  settings/checkpoints, sampling seeds, and multiple-testing scope are specified
  by the supplied prospective registration. Acceptance targets serve only as
  validation criteria.

Decoder IDs are preserved, including `logistic_nominal` / `catboost_nominal`
where registered. Nominal and numeric treatments of relation IDs are not
interchangeable; each decoder retains its registered settings.

The four variants are Local, Local+score and their no-relation-ID counterparts:
`b0_local_only`, `b0_local_ultra`, `b0_local_no_relation_id`,
`b0_local_ultra_no_relation_id`. The last two remove only column 0; tests validate
all remaining values, ordering and categorical indices.

## Statistical scope

An explicit correction-scope choice is required:

- `separate` applies one exact max-|t| correction over the 12 positive-context
  contrasts and a separate exact max-|t| correction over the 12 relation-identity
  contrasts (stratified by context mode).
- `joint` applies one exact max-|t| correction over all 24 contrasts and is
  retained as a stricter prospective sensitivity analysis.

Neither scope reproduces the paper's 60-test correction. Both are prospective
sensitivities. The scope is specified independently of observed significance.

Coverage is recomputed from selected positive relation IDs against the fixed
evaluation relation distribution: covered evaluation types / evaluation types;
evaluation probability mass on covered types; and half the L1 distance between
normalized positive/evaluation distributions on their union. Equal head/tail
relation mass is checked. Seed means precede dataset means; both equal-dataset
and equal-family summaries are emitted. MRR uses stored combined directional
ranks; dataset effects are averaged within families before exact paired
2^16-sign-pattern max-|t| correction. Practical ties remain ±0.005.

## Usage

Example prospective invocation, using registered mode names:
Set `RTFM_FROZEN_KG_ASSET_ROOT` to the same directory passed to `--asset-root`;
the existing plan validator requires this runtime registration. The current
relation-context protocol permits 200 examples per class (400 total context
rows); fixture tests use smaller budgets only for unit testing.

```bash
uv run python -m analysis.analyze_ultra4g_positive_context \
  --run-root /path/to/verified/run \
  --asset-root "$RTFM_FROZEN_KG_ASSET_ROOT" \
  --provenance /path/to/dataset_family_map.csv \
  --reference-mode pool_random --matched-mode evaluation_relation_matched \
  --correction-scope separate --output generated/positive-context
```

`--run-root` is repeatable for separate decoder registrations.

The script checks asset, plan, shard, prediction and selection hashes; rejects
incomplete/unpaired matrices and varying evaluation/context budgets; and refuses
to overwrite results. After validation it emits coverage by dataset/seed,
coverage summary, dataset contrasts, contrast summary and an input-hashed audit.
Synthetic fixtures test the analysis, not the manuscript values.

## Relation to the paper experiment

The paper experiment resamples negative rows under a fixed policy, so its
outputs are not paired under this exact-row contract. Its 4,104-cell reference
package is in
[`../relation-context-historical/`](../relation-context-historical/REFERENCE.md).
The paper uses one joint 60-test correction family, rather than either scope
offered here. The end-to-end workflow is in
[REPRODUCE.md](../../../../REPRODUCE.md). The bundled reference tables and
analysis code verify the paper's procedure:

```bash
uv run python -m analysis.verify_historical_relation_context
```
