# Camera-ready repository audit

Starting HEAD: `4968c7a632815816a075dd2080839a8a4bac76eb` (clean working tree).
No manuscript files were read or modified for this audit.

## Before modifications

The audit covered documentation, packaging, manifest/verifier, protocol, runner,
RFMLink loader, CLUTRR/GraphLog encoders, decoders, analysis inventory, study
configs, checked-in results, publication data, tests, and all eight available
Git commits. Initial manifest verification passed;
`uv sync --frozen --extra dev --extra tree --extra plots` succeeded and all 110
initial CPU tests passed. The manifest covered 360 tracked files excluding itself.

Packaging and verification contained anonymity assumptions. The map relied on
figure numbers and omitted current controls. Historical review identifiers are
execution identifiers and are preserved.

The raw-data pipeline is present for synthetic calibration, CLUTRR, GraphLog,
and dihedral composition. TabPFN runs require a checkpoint; not all historical
raw per-cell predictions are shipped. Analyses requiring omitted raw runs cannot
regenerate their results from aggregates. Figures and selected downstream
analyses use checked-in tables directly.

The 57-KG loader begins at frozen candidate/context archives. Raw KG support
construction, upstream ULTRA inference and full local/global materialization
are absent. Negative-context and matched-filter builders also need external
q512 candidate/query assets. Runtime manifests supply large-panel hashes.

Positive-context coverage: `rfmlink_ultra57_relation_context_v1` exists only as
a supported runtime asset ID. Neither current files nor Git history contain its
archives, run matrix, coverage results or dedicated analysis. This is case B:
a documented deterministic downstream analysis is added, with no real CSVs
fabricated. Relation-ID variants already work, but their result matrix is absent.
Historical source placeholders cannot be resolved; see `PROVENANCE.md`.

## Scientific interpretation boundaries

CLUTRR context fit is in-sample resubstitution. For TabPFN the queried row's label
is present in the conditioning context. It does not establish a transferable
prefix rule. The shortcut diagnostic's TabPFN multiset OOD mean (0.2847388)
differs from the primary metric-robustness mean (0.2893819); these historical
sources are preserved independently. The suffix intervention summary records
an accuracy change, not a maximum over individual prediction probabilities.

Query-relative transforms are label-free input controls. Context groups contain
sampled positive/negative rows; evaluation groups contain all entities. They do
not directly test a ranking-trained model. Cross-fit/full-support changes do
not establish the strict covariate-shift condition of an unchanged conditional
label distribution. Prefix mechanisms require bounded interpretation of the
interventions, not context fit alone.

## Release changes and limits

The claim map, reproduction boundaries, terminology, unresolved provenance and
licensing are documented. Encoder-schema names gain compatibility aliases.
Verification gains publication inventory, complete depth-matrix and primary
endpoint checks. Tests cover semantics and separate permutation streams. New
positive-context analysis rejects incomplete/unpaired input.

All historical result and publication-data bytes are preserved. No GPU experiment
is rerun. Upstream assets and author-approved license/citation metadata remain
maintainer actions, not inferred provenance.

## Final CPU validation

Frozen dependency sync succeeded. The expanded suite passes 151 tests (110
before changes); changed Python files pass Ruff. The verifier retains all
original scientific checks, checks the complete 2,394-cell depth matrix and
primary endpoints, and pins 188 original result/publication files through
`tools/historical_results.json`. The release manifest includes new non-ignored
files without staging or committing them.

Table-only aggregation regeneration exactly matches the historical JSON,
including equal-dataset effect -0.03461606091149509 and equal-family effect
0.006384521668912228. All five documented figure pipelines execute on CPU.
Four generated PDFs were rendered and inspected with macOS Quick Look because
Poppler was unavailable; the context-robustness PNG was also inspected. Two
plots issue a benign semibold-font fallback. Pre-existing query-evidence outputs
were preserved by using audit-specific filenames. Generated validation outputs
are ignored and are not part of the release manifest.

The added CI workflow installs only CPU validation extras and runs the verifier
and tests; it has not been executed by GitHub Actions during this local audit.
The real positive-context and identity MRR analyses remain unavailable. Tests
cover synthetic data and strict downstream contracts, not the missing paper
results or historical sampler/correction scope. The suffix result verifies
aggregate accuracy invariance; individual historical prediction archives are
not supplied. No historical scientific output/config/runner or lockfile changed.

All 26 placeholder-bearing lines in 23 historical files remain unresolved and
are listed in `PROVENANCE.md`; the additional anonymous upstream repository
identity is also unresolved. No manuscript files were modified. No commit or
push was performed.

## Changed-file inventory

Modified (14):

```text
.gitignore
ARTIFACT_MAP.md
MANIFEST.sha256
README.md
REPRODUCE.md
pyproject.toml
src/relational_tfm/datasets/rfmlink.py
src/relational_tfm/protocol.py
tests/test_clutrr.py
tests/test_graphlog.py
tests/test_query_relative.py
tests/test_rfmlink_ladder_interfaces.py
tools/build_manifest.py
tools/verify_artifact.py
```

Added (10):

```text
.github/workflows/artifact.yml
CAMERA_READY_AUDIT.md
LICENSING.md
PROVENANCE.md
TERMINOLOGY.md
analysis/analyze_ultra4g_positive_context.py
experiments/ultra4g/results/positive-context/README.md
tests/test_artifact_verification.py
tests/test_positive_context_analysis.py
tools/historical_results.json
```

Generated validation figures and aggregation tables are under ignored
`generated/`, not checked-in result directories. Existing generated query
evidence files were not overwritten. No original file was deleted or moved.
