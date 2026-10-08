# Historical provenance

Audit starting HEAD: `4968c7a632815816a075dd2080839a8a4bac76eb`.
The initial working tree was clean.

The literal `artifact-source-revision` is unresolved historical provenance, not
a commit SHA. Every existing occurrence is retained. It appears in CLUTRR
coordinate coverage, additive context, shared-coordinate, metric-robustness and
shortcut artifacts; dihedral CPU/GPU artifacts; GraphLog replication and depth-1
sensitivity; KG ladder, context robustness and ranking correction; publication
provenance; and their analysis producers.

The available Git history begins at
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

To resolve this limitation, supply original run registrations/runtime manifests
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
+| `analysis/analyze_rfmlink_negative_robustness.py` | 1 | unresolved; present in initial import |
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
