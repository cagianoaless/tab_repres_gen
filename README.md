# Graph-to-Table Interface Audit

Code, configurations, aggregate results, and figure data for **Graph-to-Table Interfaces Control Relational Generalisation**.

| Directory | Role |
|---|---|
| `experiments/controlled/` | Row-equivalence and permutation calibration |
| `experiments/clutrr/` | Short-to-long composition controls |
| `experiments/graphlog/` | Representation ordering across decoders |
| `experiments/dihedral16/` | Dihedral composition control |
| `experiments/ultra4g/` | Frozen `ultra_4g` analyses on 57 KGs |
| `publication_data/` | Inputs for manuscript figures |

## Start here

1. Locate evidence in [`ARTIFACT_MAP.md`](ARTIFACT_MAP.md).
2. Follow [`REPRODUCE.md`](REPRODUCE.md).
3. Run `uv run python tools/verify_artifact.py`.

Committed tables reproduce the manuscript statistics and figures without model inference. Raw predictions, benchmark assets, and checkpoints are omitted; configurations and checksums are retained.
