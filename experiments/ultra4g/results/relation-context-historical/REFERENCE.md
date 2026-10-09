# Relation-context reference package

The 13 reference output files are unchanged from
`results/ultra57/relation-context/`:
result commit `5311736a48f9a1780bff4cc6378f73a13727da2a`,
analysis commit `02c15111be5b3d622f8358a673bdfcc0225fd05f`.
Their committed bytes and the 12 internal manifest hashes match.
The directory's legacy name is retained for compatibility.

## Verify reported results

```bash
uv run python -m analysis.verify_historical_relation_context
```

The CPU verifier checks all 4,104 cells:
57 datasets × 3 decoders × 4 schemas × 2 context modes × 3 seeds,
with 16 provenance families. It reconstructs seed-averaged dataset/family MRR
effects, coverage summaries and all observed t / exact corrected p-values.

| Joint correction category | Count | Effect direction |
|---|---:|---|
| Positive context | 12 | relation-matched minus frequency-random |
| Relation identity | 12 | with minus without identity |
| Frozen score | 12 | with minus without score |
| Context × identity | 6 | context effect with identity minus without |
| Context × score | 6 | context effect with score minus without |
| Context × decoder | 12 | context effect for first decoder minus second |

One **60-test** family uses combined head/tail MRR, with all 65,536 shared
family sign patterns across the 16×60 matrix. No incomplete contrast is dropped.
These are not duplicated directions or negative-policy comparisons.
CatBoost `b0_local_no_relation_id` has effect 0.013937919134311694,
t=3.7901691175274235 and adjusted p=0.036895751953125.
Identity effects range from -0.003127454409378321 to +0.002987582098983631,
**with identity minus without identity**; none survives correction.

The context-robustness **30-test** family is separate (18 encoder + 12
negative-context effects), not a subset of this family. The one fully specified
comparison shared by the two packages is TabPFN Local+score minus Local under
`relation_matched_filtered_uniform`; stored vectors need not be identical.

## Context semantics

Both arms use 200 positive + 200 negative directional rows and the same
filtered-uniform negative policy. The relation-matched arm resamples negative rows
for its selected positive queries. Evaluation queries, candidate features,
targets and filtering arrays stay fixed. Exact negative rows are not an invariant.
Literal modes are `frequency_random_filtered_uniform` and
`relation_matched_filtered_uniform`.

The unchanged reference README's phrase "negative sampling are fixed" means
policy, not identical rows. The prospective `positive-context/` utility instead
fixes exact negative rows and offers 12+12 or 24-test sensitivities. Neither
replaces this 60-test procedure.

## Source and execution

The supplied end-to-end source constructs the public KG representations,
RFMLink assets and contexts, runs the decoders and reproduces this analysis.
See [REPRODUCE.md](../../../../REPRODUCE.md).

`reproduction/REFERENCE_EXECUTION.json` records execution source
`a48dd52dfdedcf38428c489909b1f31be27ca3d9`, checkpoint and plan identities.
The reference asset manifest SHA-256 is
`8fef670b0cdf4574da81bf5d35b7eca8613cdac515988a03e1c6f0ca6764cd9e`.
All 57 listed archive hashes and the complete 2,736 CPU + 1,368 TabPFN
metrics/prediction pairs were verified on the execution system.
Large assets, predictions and serialized execution plans are not bundled in Git.

The recorded CPU and TabPFN plan hashes identify canonical scientific plan
bodies, excluding `created_at_utc` and `plan_sha256`, not serialized JSON
file bytes. Both identity types are recorded separately in the compact metadata.
Fresh runs use new registrations and their actual source/asset/plan hashes.
