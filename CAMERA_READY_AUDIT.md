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

At the initial artifact-repository-only audit, positive-context coverage:
`rfmlink_ultra57_relation_context_v1` existed only as a supported runtime asset
ID in this checkout. Neither its current files nor its Git history contained its
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

At the time of the local audit, frozen dependency sync succeeded. The expanded
suite passed 151 tests (110 before changes); changed Python files passed Ruff.
The verifier retains all
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
and tests. It had not yet been executed by GitHub Actions at the time of the
local audit. It subsequently ran successfully for commit
`63d28ca987d6c041507e9806bada29b983da37a8`: dependency sync,
`tools/verify_artifact.py`, and `pytest -q` all passed.
At that time, the real positive-context and identity MRR analyses were
unavailable. Those tests covered synthetic data and strict downstream contracts,
not the missing paper results or historical sampler/correction scope.
The suffix result verifies
aggregate accuracy invariance; individual historical prediction archives are
not supplied. No historical scientific output/config/runner or lockfile changed.

All 26 placeholder-bearing lines in 23 historical files remain unresolved and
are listed in `PROVENANCE.md`; the additional anonymous upstream repository
identity is also unresolved. No manuscript files were modified. At the time of
the local audit, no commit or push had yet been performed. The camera-ready
refactor was subsequently committed as
`63d28ca987d6c041507e9806bada29b983da37a8` and pushed to
`origin/camera-ready-refactor`.

## Initial refactor changed-file inventory

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

## Final historical-derived reconciliation

Starting branch: `camera-ready-refactor`; HEAD:
`eef4b705a5cc1ba3c80785a07b18c8bbd9ed6beb`, clean before integration.
Read-only inspection of the original study repository recovered two genuine
historical derived-result controls:

- GRU: two committed CSVs (result commit
  `7b3acb87da64571d35060da83f290e1bed3266cb`), both exact-hash and Git verified.
  All 135 length records reconstruct five OOD means and the length-4/10 endpoints.
  Historical and artifact configs are byte-identical. Raw cells/predictions,
  execution plan and runtime provenance remain unrecovered.
- Relation-context: 13 committed files (result commit
  `5311736a48f9a1780bff4cc6378f73a13727da2a`), all Git verified, with the 12
  manifest-listed byte counts/hashes also verified. Complete 4,104-cell derived
  matrix, coverage diagnostics and 60-test correction outputs are recovered.
  Raw frozen archives, predictions and execution plans/registrations remain
  unavailable; recorded upstream identities are not newly rehashed assets.

The historical correction is **one 60-test family**: 12 positive-context,
12 with-minus-without identity, 12 score, 6 context×identity, 6 context×score
and 12 context×decoder contrasts. The 16×60 matrix uses all 65,536 family
sign patterns. The derived-only verifier reconstructs effects from historical
cell MRR and reproduces all observed t and exact adjusted p-values. The separate
context-robustness 30-test family remains unchanged.

The earlier prospective reconstruction's 12+12 correction assumption and
identical-negative-row contract do not describe the historical experiment.
Historical matched positives trigger newly sampled negatives under the same
filtered-uniform policy and 200+200-row budget; evaluation remains fixed.
The original historical README is preserved, with its policy-versus-row wording
clarified in `RECOVERY.md`. The prospective utility retains its stricter contract
and both 12+12/24 sensitivity scopes; neither is historical reproduction.

Pinning adds 15 recovered files to the unchanged 188 original entries (203
total). The release manifest and CPU artifact verifier cover the new evidence.
CPU validation passed: both dedicated verifiers, `tools/verify_artifact.py`,
and `MPLCONFIGDIR=.matplotlib uv run pytest -q` (165 passed). Ruff passes on all
five changed Python files and `git diff --check` is clean. The manifest covers
390 release files excluding itself. Independent comparison with starting HEAD
confirms unchanged original 188 pins and 228 protected result/publication/config/
runner/lockfile files. Every new historical copy matches its source hash; no
private absolute filesystem paths occur in changed or added files.
No manuscript, existing scientific result, config, runner or lockfile is changed.
No scientific experiment is rerun; only derived-table calculations are performed.
