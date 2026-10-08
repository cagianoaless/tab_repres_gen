# Graph-to-Table Interfaces Control Relational Generalisation

This repository provides the end-to-end reproduction pipeline for
*Graph-to-Table Interfaces Control Relational Generalisation*. It includes the
source and configurations for constructing the paper's graph-to-table
representations, running decoder experiments, evaluating filtered rankings,
and reproducing the reported analyses.

The experiments separate the graph-to-table encoder phi, labelled context
constructor S, and decoder D. The complete interface is `(phi, S, D)`.
On the 57-KG panel the complete representation is:

```text
Full32 = Local18 + frozen ULTRA score1 + Global13
```

Local18 contains relation, degree, pair, path and cluster statistics. Global13
contains PageRank, component statistics, and component-wise spectral heat and
diffusion features at scales 0.1, 1, 10 and 100. The source pipeline uses public
ULTRA datasets and the pinned ULTRA-4G checkpoint; weights and generated
multi-gigabyte intermediates are external to this lightweight repository.

## Getting started

Follow [REPRODUCE.md](REPRODUCE.md) for environment/checkpoint setup and the
pipeline from public KGs to features, RFMLink assets, decoder runs and analyses.
To verify the bundled reference outputs without running models:

```bash
uv sync --frozen --extra dev --extra tree --extra plots
uv run python tools/verify_artifact.py
MPLCONFIGDIR=.matplotlib uv run pytest -q
```

| Directory | Contents |
|---|---|
| `reproduction/` | Checksummed feature producers, portable entry points and execution identities |
| `src/relational_tfm/`, `configs/studies/` | Context construction, decoders, evaluation and study configurations |
| `experiments/controlled/` | Row-equivalence and permutation calibration |
| `experiments/clutrr/` | Short-to-long composition and recurrent controls |
| `experiments/graphlog/` | Representation ordering and decoder-capacity controls |
| `experiments/dihedral16/` | Dihedral composition control |
| `experiments/ultra4g/` | 57-KG encoder ladder and context/filtering controls |
| `analysis/`, `publication_data/` | Statistical/figure code and reference publication inputs |

Reference outputs are checksummed and must not be overwritten by reruns.
Write new outputs to `generated/` or a separate reproduction workspace.
The [artifact map](ARTIFACT_MAP.md) links scientific claims to execution and
analysis code. [PROVENANCE.md](PROVENANCE.md) records source, checkpoint and
plan identities; [TERMINOLOGY.md](TERMINOLOGY.md) explains compatible schema
identifiers. See [LICENSING.md](LICENSING.md) for code, dataset and model terms.

The prospective fixed-negative-row sensitivity utility is distinct from the
paper's relation-context experiment; its 12/24-test options do not replace
the paper's joint 60-test correction. The context-robustness analysis uses
a separate 30-test family.

Model dependencies: [ULTRA](https://github.com/DeepGraphLearning/ULTRA) and
[TabPFN](https://github.com/PriorLabs/TabPFN). Built with PriorLabs-TabPFN.
