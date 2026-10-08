# Historical provenance

Audit starting HEAD: `4968c7a632815816a075dd2080839a8a4bac76eb`.
The initial working tree was clean.

The literal `artifact-source-revision` is unresolved historical provenance, not
a commit SHA. Every existing occurrence is retained. It appears in CLUTRR
coordinate coverage, additive context, shared-coordinate, metric-robustness and
shortcut artifacts; dihedral CPU/GPU artifacts; GraphLog replication and depth-1
sensitivity; KG ladder, context robustness and ranking correction; publication
provenance; and their analysis producers.

The artifact repository's available Git history begins at
`9955c64903a67bd801588b48f8328135317e3eea`, a compact artifact import which already
contains these placeholders. Subsequent history, plans and result metadata do
not establish the original execution SHA. The import SHA proves when files
entered this repository, not which upstream revision executed the experiments.
It must not replace an execution/source-result commit.

Plan hashes, checksums, cell IDs, dataset counts and source paths already present
in historical records retain their original meaning. `MANIFEST.sha256` verifies
current release bytes; it does not repair missing historical source provenance.
Some legacy analyses compare the placeholder string to source registrations;
their historical compatibility checks remain intact. New analysis outputs hash
their actual input files and never manufacture historical commit identifiers.

To resolve those placeholder-bearing artifacts, supply original run
registrations/runtime manifests
and corresponding upstream Git objects. No historical provenance placeholder
was resolved during this audit.

The additional review-era `source_repository: anonymous-source-tree` in
`publication_data/query_evidence_provenance.json` also occurs in the initial
import. Its upstream repository identity cannot be proven; it is preserved.
The current artifact repository name is not evidence of that source's identity.

## Occurrence inventory at audit HEAD

Each entry was checked against the available history and source metadata. All
remain unresolved; none is replaced with the audit HEAD. Counts refer to matching
lines (a line can reference more than one revision).

| File | Matching lines | Resolution |
|---|---:|---|
| `analysis/analyze_rfmlink_negative_robustness.py` | 1 | unresolved; present in initial import |
| `analysis/analyze_ultra4g_q64_complete_panel.py` | 1 | unresolved; present in initial import |
| `analysis/audit_clutrr_metric_robustness.py` | 2 | unresolved; present in initial import |
| `analysis/summarize_clutrr_shortcut_diagnostic.py` | 1 | unresolved; present in initial import |
| `analysis/summarize_dihedral16.py` | 1 | unresolved; present in initial import |
| `analysis/summarize_dihedral16_tabpfn.py` | 1 | unresolved; present in initial import |
| `analysis/summarize_graphlog_decoder_sensitivity.py` | 1 | unresolved; present in initial import |
| `experiments/clutrr/results/additive-context/README.md` | 1 | unresolved; present in initial import |
| `experiments/clutrr/results/coordinate-coverage/README.md` | 1 | unresolved; present in initial import |
| `experiments/clutrr/results/metric-robustness/summary.json` | 2 | unresolved; present in initial import |
| `experiments/clutrr/results/shared-coordinates/README.md` | 1 | unresolved; present in initial import |
| `experiments/clutrr/results/shortcut-diagnostic/summary.json` | 1 | unresolved; present in initial import |
| `experiments/dihedral16/results/cpu/summary.json` | 1 | unresolved; present in initial import |
| `experiments/dihedral16/results/tabpfn/analysis_manifest.json` | 1 | unresolved; present in initial import |
| `experiments/dihedral16/results/tabpfn/summary.json` | 1 | unresolved; present in initial import |
| `experiments/graphlog/results/decoder-sensitivity/summary.json` | 1 | unresolved; present in initial import |
| `experiments/graphlog/results/symbolic-replication/README.md` | 1 | unresolved; present in initial import |
| `experiments/ultra4g/results/context-robustness/audit.json` | 1 | unresolved; present in initial import |
| `experiments/ultra4g/results/logistic-ranking-correction/audit.json` | 2 | unresolved; present in initial import |
| `experiments/ultra4g/results/q64-encoder-ladder/README.md` | 1 | unresolved; present in initial import |
| `experiments/ultra4g/results/q64-encoder-ladder/audit.json` | 1 | unresolved; present in initial import |
| `publication_data/context_robustness_provenance.json` | 1 | unresolved; present in initial import |
| `publication_data/query_evidence_provenance.json` | 1 | unresolved; present in initial import |

## Historical derived-result recovery after the initial audit

Reconciliation starting HEAD: `eef4b705a5cc1ba3c80785a07b18c8bbd9ed6beb`.
The original study repository supplied independently checked historical Git
objects and derived tables for two previously missing controls:

| Recovered evidence | Historical source | Current artifact location |
|---|---|---|
| Two GRU derived CSVs | Result commit `7b3acb87da64571d35060da83f290e1bed3266cb`; original paths `results/audits/acceptance_controls_v3/` | `experiments/clutrr/results/recurrent-gru/` |
| Thirteen relation-context derived package files | Result commit `5311736a48f9a1780bff4cc6378f73a13727da2a`; original path `results/ultra57/relation-context/` | `experiments/ultra4g/results/relation-context-historical/` |
| KG statistical analysis logic | Analysis-code commit `02c15111be5b3d622f8358a673bdfcc0225fd05f` | Adapted derived-only verifier `analysis/verify_historical_relation_context.py` |

Every recovered file matches its historical committed bytes. The GRU README
lists its two SHA-256s; the KG `analysis_manifest.json` supplies 12 internal
hashes, cross-checked before copying, and `tools/historical_results.json` pins
all 15 recovered files including the KG manifest itself. All original 188
entries remain intact. A result-package commit is not an execution-source commit.
The pin file's original `audit_head`/`source` describe its original inventory;
the added `recovered_sources` records separately identify the 15 additions.

The GRU config matches the historical config byte-for-byte, but its original
cell metrics, predictions, plan and runtime provenance remain unavailable.
No GRU execution-source revision can be asserted from these tables alone.

KG `completion_audit.json` records execution source
`a48dd52dfdedcf38428c489909b1f31be27ca3d9`, asset manifest, CPU/TabPFN plan and
shard-manifest identities. `relation-context-historical/RECOVERY.md` records
those exact hashes and the checkpoint specified by the historical registration
generator. Actual frozen archives, raw metrics/predictions and execution
plans/registrations were not recovered and were **not independently rehashed**.
Verification reproduces derived effects/coverage and the historical 60-test
correction, not raw-run execution or archive-level pairing.

This recovery does not resolve or rewrite any placeholder in the inventory
above. It replaces the earlier absence assessment only for these two controls;
historical/prospective statistical and negative-row contracts are distinguished
explicitly in the new recovery note and prospective README.
