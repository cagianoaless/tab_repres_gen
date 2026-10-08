# Historical relation-context package: recovery and interpretation

The other 13 files in this directory are immutable, byte-for-byte historical
derived outputs from `results/ultra57/relation-context/` in the original study
repository. Result-package commit: `5311736a48f9a1780bff4cc6378f73a13727da2a`.
Analysis-code commit: `02c15111be5b3d622f8358a673bdfcc0225fd05f`.
Every file was compared with its historical Git object. The 12 files inventoried
by `analysis_manifest.json` additionally match its byte counts and SHA-256s.
All 13 files, including that manifest, are pinned in `tools/historical_results.json`.
No CSV formatting, identifiers or numerical values have been rewritten.

## Derived-result reproduction

```bash
uv run python -m analysis.verify_historical_relation_context
```

The CPU-only verifier checks the complete 4,104-cell Cartesian matrix:
57 datasets × 3 decoders (`catboost`, `logistic`, `tabpfn`) × 4 schemas ×
2 modes × 3 seeds (101--103), assigned to 16 provenance families.
It rebuilds combined-direction contrasts from seed-averaged cell MRR,
checks dataset and family effects, and reproduces all 60 exact corrected
p-values from the stored 16 × 60 family matrix. Coverage is reaggregated
from recovered dataset/seed diagnostics, not recomputed from absent raw triples.

| Historical correction group | Count | Direction |
|---|---:|---|
| Positive-context effects | 12 | relation-matched minus frequency-random |
| Relation-identity effects | 12 | with identity minus without identity |
| Frozen-score additions | 12 | with score minus without score |
| Context × identity interactions | 6 | context effect with identity minus without identity |
| Context × score interactions | 6 | context effect with score minus without score |
| Context × decoder interactions | 12 | context effect for first decoder minus second decoder |

Total: **12 + 12 + 12 + 6 + 6 + 12 = 60**, in one correction family.
The same sign for a provenance family is applied to all 60 combined head/tail
contrasts. All 2^16 = 65,536 signs are enumerated; the one-sample t is recomputed
for each contrast, and the maximum absolute t supplies the exact adjustment.
No incomplete contrast is dropped. These are not head/tail duplicates or
negative-policy contrasts. The key CatBoost `b0_local_no_relation_id` effect is
0.013937919134311694 (t = 3.7901691175274235; adjusted p = 0.036895751953125).
The 12 identity effects range from -0.003127454409378321 to +0.002987582098983631,
**with minus without** identity; none survives the historical 60-test correction.

The separate `context-robustness/` 30-test analysis remains unchanged:
18 encoder comparisons + 12 negative-context comparisons. It uses its own
16 × 30 matrix, is not a subset of this 60-test family, and must not be pooled
with it. The one fully specified comparison in common is TabPFN Local+score
minus Local under `relation_matched_filtered_uniform`; separate packages need
not have identical stored effect vectors.

## Negative-row semantics and prospective distinction

The historical random and relation-matched context constructions use the same
filtered-uniform negative-sampling policy and the same context budget (200
positive + 200 negative directional rows), while the relation-matched arm
resamples negative rows for its selected positive queries. Evaluation queries,
candidate features, targets and filtering arrays remain fixed. **Exact negative
rows are not a historical invariant.** Literal mode identifiers are
`frequency_random_filtered_uniform` and `relation_matched_filtered_uniform`.

The historical README is preserved without editing. Its phrase "negative
sampling are fixed" describes the policy, not identical negative rows; its
"only positive relation coverage changes" does not mean negatives are copied.
The prospective `positive-context/` tool requires identical negative rows and
uses 12+12 or 24-test sensitivity scopes. Neither reproduces this historical
60-test analysis. See that directory's README for its stricter contract.

## Recorded upstream identities, not currently revalidated files

| Identity | Historical record |
|---|---|
| Execution source | `a48dd52dfdedcf38428c489909b1f31be27ca3d9` |
| Asset ID | `rfmlink_ultra57_relation_context_v1` |
| Asset manifest SHA-256 | `8fef670b0cdf4574da81bf5d35b7eca8613cdac515988a03e1c6f0ca6764cd9e` |
| CPU plan SHA-256 | `bd836b248fdb2812206087fae0c5ee88f5e9035cb9450c5d6d5da7893e9ccee2` |
| TabPFN plan SHA-256 | `09461de58b61803baad894e8d3fb4e1da3980c13410fe36decac195c43553b7a` |
| TabPFN checkpoint SHA-256 | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |

Execution source, manifest/plan and shard identities are recorded in
`completion_audit.json`. The checkpoint identity is specified by the historical
registration generator at the execution-source revision; it is not a newly
rehashable checkpoint or independent evidence of the runtime installation.

Raw frozen relation-context archives, raw metrics/predictions and execution
plans/registrations as files are **not recovered**. The hashes above identify
historically recorded inputs; those absent files were not independently rehashed.
Full reruns require these external assets through `RTFM_FROZEN_KG_ASSET_ROOT`,
as well as execution files and model access. The current verification reproduces
the historical **derived analysis only**, not the raw execution or archive-level
pairing. No upstream reconstruction is claimed.
