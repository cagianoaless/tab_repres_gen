# GraphLog

All 57 GraphLog worlds use the official symbolic path descriptor. Context paths
have length 2--3 and evaluation paths length 4--5. Metrics are equal-world
balanced accuracy at 40 examples per relation.

| Decoder | Shared multiset | Absolute path | Shared - absolute | Worlds W/T/L |
|---|---:|---:|---:|---:|
| Logistic | 0.298 | 0.234 | +0.065 | 41/0/16 |
| CatBoost | 0.125 | 0.282 | -0.157 | 6/0/51 |
| CatBoost depth 1 | 0.201 | 0.271 | -0.070 | - |
| Tuned CatBoost | 0.127 | 0.276 | -0.149 | - |
| TabPFN | 0.159 | 0.238 | -0.079 | - |

Mean balanced-accuracy chance is `0.105`. The representation ordering reverses
with decoder class: the linear decoder favors shared counts, whereas nonlinear
decoders favor the absolute path on this panel. Rule-overlap descriptors do not
predict the reversal out of sample (all leave-cluster-out `R2 < 0`).

## Results

- [`symbolic-replication/`](symbolic-replication): complete logistic and
  CatBoost cells and primary contrasts.
- [`decoder-sensitivity/`](decoder-sensitivity): depth-1, tuned CatBoost, and
  TabPFN contrasts.
- [`rule-overlap/`](rule-overlap): absolute levels, chance normalization,
  similarity clusters, and predictive analyses.
- [`fixed-panel/`](fixed-panel): panel-level audit.
- [`protocol/`](protocol): asset and schedule audits.
