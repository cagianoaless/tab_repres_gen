# Graph-to-Table Interface Audit

This anonymous artifact contains the code, configurations, committed result tables, and figure-source data for the paper **Graph-to-Table Interfaces Control Relational Generalisation**.

The repository is intentionally publication-scoped. It contains only the experiments used in the manuscript:

| Directory | Role |
|---|---|
| `experiments/controlled/` | Calibration of row-equivalence ceilings and matched permutation checks |
| `experiments/clutrr/` | Short-to-long composition and absolute-position shortcut controls |
| `experiments/graphlog/` | Decoder-dependent representation ordering over 57 worlds |
| `experiments/dihedral16/` | Order-16 dihedral composition boundary control |
| `experiments/ultra4g/` | Frozen `ultra_4g` interface, context, and query-level analyses over 57 KGs |
| `publication_data/` | Compact CSV inputs used by the manuscript figures |

Operational logs, failed runs, superseded analyses, remote-machine scripts, unrelated model sweeps, rendered manuscript files, and repository history are deliberately excluded. No reported result is reconstructed from a partial run or an imputed cell.

## Start here

1. Read [`ARTIFACT_MAP.md`](ARTIFACT_MAP.md) to locate every reported figure and table.
2. Read [`REPRODUCE.md`](REPRODUCE.md) for environment, checks, and figure commands.
3. Run `python tools/verify_artifact.py` before analysis.

The committed tables are sufficient to reproduce the paper-level analyses without rerunning GPU inference. Raw per-candidate predictions are excluded from this reviewer-facing repository; the retained per-seed, per-world, per-dataset, per-family, and query-stratum tables are the smallest data products needed to inspect the reported comparisons. Large benchmark assets and model checkpoints are not duplicated; their identities and checksums are recorded in `data/manifests/`, and the experiment configurations needed to regenerate cells are in `configs/studies/`.
