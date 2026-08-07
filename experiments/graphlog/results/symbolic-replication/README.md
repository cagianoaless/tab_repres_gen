# GraphLog CPU compositional replication

## Integrity

- Status: complete and verified.
- Frozen plan:
  `dc6872565a8726a715a9437cf5029db6c733c469b08e15fc097b5fd9dc8e7006`.
- Cells: 8,208/8,208.
- Shards: 8/8.
- Worlds: all 57 GraphLog v1.1 worlds.
- Context seeds: 101, 102, and 103.
- Decoders: logistic regression and CatBoost.
- Execution source recorded by the shard manifests:
  `artifact-source-revision`.
- All metric payloads, prediction arrays, cell specifications, task audits,
  and shard manifests passed checksum and schema verification.

The runtime used Python 3.11.14, NumPy 2.4.2, pandas 2.3.3,
scikit-learn 1.8.0, and CatBoost 1.2.10. Exact runtime metadata remains in
the local raw shard manifests; the compact tables in this directory are the
publication artifacts.

## Registered endpoint

The endpoint is equal-length macro balanced accuracy over official inner-test
descriptor lengths four and five. Each contrast first averages the three
context seeds within a world and then weights the 57 worlds equally.
Intervals are fixed-panel hierarchical bootstrap sensitivities, not
population-level confidence intervals.

At 40 context rows per target relation:

| Decoder | Registered contrast | Mean delta | Fixed-panel interval | W/T/L |
|---|---|---:|---:|---:|
| Logistic | shared multiset minus absolute path | +0.0646 | [+0.0339, +0.0969] | 41/0/16 |
| Logistic | block-permuted minus aligned combined | +0.0199 | [-0.0044, +0.0449] | 32/0/25 |
| CatBoost | shared multiset minus absolute path | -0.1575 | [-0.1876, -0.1266] | 6/0/51 |
| CatBoost | block-permuted minus aligned combined | -0.0916 | [-0.1143, -0.0686] | 7/0/50 |

Logistic regression meets the two registered sign-and-majority criteria.
CatBoost does not and instead shows a strong reversal. Consequently, the CPU
stream does not establish a decoder-invariant replication.

## Interpretation

The external benchmark supports a sharper claim than a universal
shared-coordinate advantage. With logistic regression, the shared multiset
contrast moves from -0.0071 at five rows per relation to +0.0646 at forty,
while the absolute interface remains nearly flat. CatBoost prefers the
aligned absolute representation at every budget; its shared-minus-absolute
contrast ranges from -0.1174 to -0.1575.

Thus, the usefulness of position sharing is conditional on decoder bias and
context. The blockwise control also has decoder-dependent meaning: it removes
the complete ordered block, including supported prefix positions, so its
negative CatBoost effect is evidence that CatBoost uses aligned prefix order,
not evidence against the validity of the null.

The frozen TabPFN plan is required to locate the TFM within this
decoder-specific reversal. It cannot alter the completed CPU conclusion.

## Files

- `summary.json`: integrity, bootstrap specification, and registered
  decisions.
- `primary_contrasts.csv`: four registered CPU endpoint contrasts.
- `aggregate_context_curves.csv`: absolute context curves.
- `contrast_context_curves.csv`: paired contrast curves.
- `contrast_by_length.csv`: length-four and length-five diagnostics.
- `primary_contrasts_by_world.csv`: fixed-world paired deltas.
- `cell_metrics_by_length.csv`: compact per-cell, per-length metrics.
