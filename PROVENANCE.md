# Source provenance

## Pipeline sources

`reproduction/SOURCE_MANIFEST.json` inventories the byte-identical producer
source, required import dependencies, launcher metadata and numerical reference
gates. Each entry records its path, role, size, SHA-256 and a source commit where
one is established. Manifest paths are relative to `reproduction/`.
Producer files without an upstream committed Git identity
are identified by their verified SHA-256; no commit is inferred for them.

| Component | Source identity |
|---|---|
| Local18 | `d63ad3d0b8d3d1cb3d9cc1e160f66de9d276016e950f3dbf6b9b347d1e9dfb4d` |
| Global13 hybrid module | `d00aeb2e467d50649d326bc5dc384c7abaefc1aae5b7ab84abb2c0f89e9f0dda` |
| Global13 inherited module | `b21ff322409aefe256845272a1e5131254348c2253f99142c53da307c3e14194` |
| Global implementation digest | `da165b3f24a8e8c9f9b5ed840400301bc30f30134274ab10c947ff32e76ed470` |
| RFMLink exporter, context builder and registration/worker code | `a48dd52dfdedcf38428c489909b1f31be27ca3d9` |
| Relation-context analysis | `02c15111be5b3d622f8358a673bdfcc0225fd05f` |
| Relation-context reference tables | `5311736a48f9a1780bff4cc6378f73a13727da2a` |
| GRU reference tables | `7b3acb87da64571d35060da83f290e1bed3266cb` |
| GRU execution registration | `35ce37623d2f4bd1baee66b0ded837af63d8e17a` |

The portable wrappers adapt paths and, for a fresh analysis, the actual execution
revision checked against registrations. They do not alter feature formulas,
random streams, decoders or statistical estimands. The module/directory names
in the source closure remain compatible with the original entry points.
`reproduction/reference/frozen_numeric_protocol.json` checks the Global implementation
digest and its ten original source files.

## Dependencies and checkpoints

ULTRA source: [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA),
commit `427966ad8ed60420eef034063d44f3153addff90`.
TabPFN package: `8.1.0`, V3 classifier, four estimators for the KG decoder matrix.
Checkpoint identities are independently verified reference inputs:

| Checkpoint | SHA-256 |
|---|---|
| `ultra_4g.pth` | `48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0` |
| `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |

Weights remain external; acquisition and verification are described in
[REPRODUCE.md](REPRODUCE.md). Producer direct dependency versions are recorded
separately from the artifact's `uv.lock`; neither claims byte-identical results
across hardware and numerical libraries.

## Plan and reference execution identities

`reproduction/REFERENCE_EXECUTION.json` records reference execution source,
manifest and plan identities. A **scientific plan hash** is SHA-256 of canonical
JSON after excluding `created_at_utc` and `plan_sha256`:
`ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':')`.
It is not the SHA-256 of the serialized JSON file; both are listed separately
in that metadata.

| Plan | Scientific-plan-body SHA-256 |
|---|---|
| Relation-context CPU | `bd836b248fdb2812206087fae0c5ee88f5e9035cb9450c5d6d5da7893e9ccee2` |
| Relation-context TabPFN | `09461de58b61803baad894e8d3fb4e1da3980c13410fe36decac195c43553b7a` |
| GRU | `bded7fc607c0fff7036f617599f721d3954d18f0d24df506dd94a2683bf686f5` |

The reference relation-context manifest has SHA-256
`8fef670b0cdf4574da81bf5d35b7eca8613cdac515988a03e1c6f0ca6764cd9e`.
Its 57 listed archives and the complete reference executions (2,736 CPU +
1,368 TabPFN cells; 15 GRU cells) were verified on the execution system.
Large archives, weights and prediction trees are external generated/reference
outputs, not prerequisites bundled into Git. The lightweight tables and their
verifiers support direct paper-result verification.

The numerical stability shard files are unchanged reference gates, not new
runs. The compact graph-consistency record retains the original gate's SHA-256
and decision without machine-specific runtime paths. Fresh executions record
their own paths, package versions, plans and source revision.

## Reference outputs and compatibility

The release manifest checks current release bytes. The separate pin inventory
protects 203 reference result/publication files; all prior entries remain intact.
Result-package commits identify table versions, not necessarily execution source.
Serialized legacy filenames containing `historical` are retained for compatibility.

Some reference files contain the literal `artifact-source-revision`, rather
than a verified execution SHA. The artifact import
`9955c64903a67bd801588b48f8328135317e3eea` already contains those fields; it
does not establish the revision that executed each experiment. They remain
unchanged rather than being replaced with the current repository HEAD.
The additional `source_repository: anonymous-source-tree` field in the pinned
query-evidence metadata likewise remains an unresolved source identifier.
These metadata fields do not prevent numerical verification or fresh execution
with the supplied pipeline.

### Unresolved source-revision fields

Counts refer to matching lines, not distinct revisions.

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
