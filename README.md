# Graph-to-Table Interfaces Control Relational Generalisation

`tab_repres_gen` is the scientific artifact for this paper: paired experiments
that vary graph-to-table encoding, labelled context construction, and tabular
decoding while keeping evaluation fixed.

| Directory | Role |
|---|---|
| `experiments/controlled/` | Row-equivalence and permutation calibration |
| `experiments/clutrr/` | Short-to-long composition controls |
| `experiments/graphlog/` | Representation ordering across decoders |
| `experiments/dihedral16/` | Dihedral composition control |
| `experiments/ultra4g/` | Frozen `ultra_4g` analyses on 57 KGs |
| `publication_data/` | Inputs for manuscript figures |

CLUTRR and GraphLog can be rerun after staging their checksummed public data;
synthetic and dihedral controls generate their data. TabPFN runs additionally
require the pinned model package/checkpoint and GPU resources. Full 57-KG reruns
require external frozen ULTRA/RFMLink archives, supplied through
`RTFM_FROZEN_KG_ASSET_ROOT`. This repository starts downstream of upstream KG
preparation, ULTRA inference, and feature materialization.

Checked-in `experiments/*/results/` and `publication_data/` are immutable
historical evidence, validated without running models. Write reruns to
`generated/` and compare them explicitly. Genuine historical derived tables are
now recovered for the [CLUTRR GRU](experiments/clutrr/results/recurrent-gru/README.md)
and the complete [relation-context / identity control](experiments/ultra4g/results/relation-context-historical/RECOVERY.md).
Their endpoints and historical 60-test correction are reproducible from these
tables; missing raw execution files prevent full raw-run reconstruction.
The [positive-context tool](experiments/ultra4g/results/positive-context/README.md)
remains a stricter prospective 12/24-test sensitivity, not that historical analysis.

In the paper, phi is the graph-to-table encoder, S constructs the labelled
context, and D is the decoder. The complete interface is `(phi, S, D)`.
Legacy `interface` fields identify feature schemas, thus encoder variants phi;
serialized identifiers remain compatible. See [TERMINOLOGY.md](TERMINOLOGY.md).

Start with [REPRODUCE.md](REPRODUCE.md) and the claim-based
[ARTIFACT_MAP.md](ARTIFACT_MAP.md). [PROVENANCE.md](PROVENANCE.md) records unresolved
historical provenance; [LICENSING.md](LICENSING.md) records distribution decisions
still required. [CAMERA_READY_AUDIT.md](CAMERA_READY_AUDIT.md) records the audit.
