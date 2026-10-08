# Prospective positive-context and relation-identity sensitivity utility

Status: **prospective/sensitivity analysis**, not historical manuscript
reproduction. No result CSVs or `audit.json` are manufactured here. The genuine
historical 4,104-cell derived package is recovered separately in
[`../relation-context-historical/`](../relation-context-historical/RECOVERY.md).
Its correction is one 60-test family and its negative rows are resampled under
a fixed policy. Raw frozen archives, predictions and execution files remain
unavailable. The manuscript's acceptance values are never used as output data.

`analysis/analyze_ultra4g_positive_context.py` implements a deterministic
downstream contract using the existing loader selection and statistical helpers.
It does **not** reconstruct the historical context sampler. Its contract holds
exact negative rows fixed, stricter than the historical policy-only invariant.
Do not use it to interpret historical outputs as if those rows were identical.
It requires appropriately registered prospective inputs:

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
  features must remain identical. Budget and evaluation archive remain fixed.
- Registered prospective plans and complete checksum-verified `results/shard-*` outputs from
  the existing runner, including metrics, prediction NPZs, and shard manifests.
  Matrix: 57 datasets × 3 decoders × 4 schema variants × 2 modes × 3 seeds
  (4,104 cells). This prospective contract is distinct from the recovered
  historical matrix, which does not establish this exact-fixed-negative pairing.
- A dataset/family CSV with columns `dataset,provenance_family`, covering exactly
  the observed 57 datasets and 16 families. The original mode names, decoder
  settings/checkpoints, sampling seeds, and multiple-testing scope must come
  from the supplied prospective registration, not inference from acceptance targets.

Decoder IDs are preserved, including `logistic_nominal` / `catboost_nominal`
where registered. Nominal and numeric treatments of relation IDs are not
interchangeable: supply the original settings, never substitute a decoder based
only on its family name.

The four variants are Local, Local+score and their no-relation-ID counterparts:
`b0_local_only`, `b0_local_ultra`, `b0_local_no_relation_id`,
`b0_local_ultra_no_relation_id`. The last two remove only column 0; tests validate
all remaining values, ordering and categorical indices.

Example prospective invocation (replace the mode names with registered names):
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

`--run-root` is repeatable for separate decoder registrations. An explicit
correction-scope choice remains required:

- `separate` applies one exact max-|t| correction over the 12 positive-context
  contrasts and a separate exact max-|t| correction over the 12 relation-identity
  contrasts (stratified by context mode).
- `joint` applies one exact max-|t| correction over all 24 contrasts and is
  retained as a stricter prospective sensitivity analysis.

Neither scope reproduces the historical 60-test correction. Both are prospective
sensitivities, and an explicit choice remains required. The example is not a
manuscript-facing reproduction command and must not be chosen to optimize
significance. The recovered historical analysis code and correction tables
establish the actual 60-test procedure despite absent raw execution files:

```bash
uv run python -m analysis.verify_historical_relation_context
```

Coverage is recomputed from selected positive relation IDs against the fixed
evaluation relation distribution: covered evaluation types / evaluation types;
evaluation probability mass on covered types; and half the L1 distance between
normalized positive/evaluation distributions on their union. Equal head/tail
relation mass is checked. Seed means precede dataset means; both equal-dataset
and equal-family summaries are emitted. MRR uses stored combined directional
ranks; dataset effects are averaged within families before exact paired
2^16-sign-pattern max-|t| correction. Practical ties remain ±0.005.

The script checks asset, plan, shard, prediction and selection hashes; rejects
incomplete/unpaired matrices and varying evaluation/context budgets; and refuses
to overwrite results. After validation it emits coverage by dataset/seed,
coverage summary, dataset contrasts, contrast summary and an input-hashed audit.
Synthetic fixtures test the analysis, not the manuscript values.
