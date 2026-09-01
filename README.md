# Graph-to-Table Interface Audit

Code, configurations, aggregate results, and figure data for **Graph-to-Table Interfaces Control Relational Generalisation**.

The artifact also contains the additional protocol controls used to test the main
conclusions: collision-conditioned CLUTRR ceilings, a depth sweep for GraphLog,
query-relative KG features, matched negative filtering, and dataset-weighted
aggregation sensitivity.

| Directory | Role |
|---|---|
| `experiments/controlled/` | Row-equivalence and permutation calibration |
| `experiments/clutrr/` | Short-to-long composition controls |
| `experiments/graphlog/` | Representation ordering across decoders |
| `experiments/dihedral16/` | Dihedral composition control |
| `experiments/ultra4g/` | Frozen `ultra_4g` analyses on 57 KGs |
| `publication_data/` | Inputs for manuscript figures |

Start with [`REPRODUCE.md`](REPRODUCE.md) for validation and rerun commands, and
[`ARTIFACT_MAP.md`](ARTIFACT_MAP.md) for the mapping from claims to evidence.
