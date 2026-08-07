# CLUTRR ordered-coordinate interference audit

Status: **PASS**.

The prospectively registered control completed 90/90 cells in 20/20 verified
shards with zero failures. It preserves the candidate-aligned relation
multiset and jointly permutes only the 140 absolute-position columns within
chain length.

At 80 context examples per relation, equal-length OOD accuracy over lengths
4--10 is:

| Decoder | Multiset only | Ordered + multiset | Multiset + permuted ordered | New - ordered | New - multiset |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.4295 | 0.2663 | 0.3997 | +0.1334 | -0.0298 |
| CatBoost | 0.2725 | 0.1630 | 0.2555 | +0.0925 | -0.0170 |
| TabPFN-3 | 0.2894 | 0.1874 | 0.2917 | +0.1043 | +0.0023 |

Every seed-level new-minus-ordered contrast is positive at this context for
all three primary decoders. Against multiset-only, TabPFN is effectively
restored, while logistic and CatBoost retain small deficits.

At smaller contexts, especially 20 examples per relation, the additional
permuted block remains a burden relative to multiset-only. The result
therefore separates two phenomena:

1. within this parameterization, candidate-aligned absolute-position
   associations are a major contributor to high-context interference
   relative to the same block after permutation;
2. merely carrying the extra nuisance block can still increase estimation
   burden when context is limited.

This is a paired descriptive sensitivity on one fixed official CLUTRR
population and three context seeds. Permutation may also alter cross-block
consistency, so it does not support pure, sampled-task, or
coordinate-invariant causal inference.

Frozen plan hashes:

- CPU: `473f626ccd779dce6333db30ae318878b95b098036ead7e7b7038ae6b3b93a0d`
- TabPFN: `145101a0cb89d06fdaac16e55d0fd28af30dee4afb5973aa8209f3770b07f2bd`

The machine-readable summary, cell table, paired comparisons, registrations,
plans, terminal statuses, and metric-free registration audit are stored in
this directory.
