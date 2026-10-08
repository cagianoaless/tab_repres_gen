# Reproduction guide

## 1. Environment and quick verification

Python 3.11 or 3.12 is supported for the artifact package:

```bash
uv sync --frozen --extra dev --extra tree --extra plots
uv run python tools/verify_artifact.py
MPLCONFIGDIR=.matplotlib uv run pytest -q
```

These CPU checks verify all 203 pinned reference files, representation/source
identities, matrix completeness and paper endpoints. They do not load models.
The release manifest covers all non-ignored release files except itself;
refresh it after intentional source/documentation changes with
`uv run python tools/build_manifest.py`. Never overwrite reference results.

The complete reproduction pipeline and reference outputs have been validated
for source identity, configuration, and reported-result consistency. Artifact
preparation did not include an additional full-scale rerun from a fresh environment.

## 2. ULTRA and model setup

Feature materialization runs on a Linux CUDA machine. The original producer
uses CUDA explicitly; changing devices is not part of this reproduction path.
Large graphs require substantial GPU memory, CPU RAM and disk space. The three
intermediate trees occupy approximately 17.5 + 139 + 95 GB; allow additional
space for model caches, exports and runs.

Keep the producer environment separate from the artifact's locked CPU/decoder
environment. Its direct dependency versions are in
`reproduction/requirements-upstream.txt`:

```bash
uv venv generated/upstream-env --python 3.11
UP="$PWD/generated/upstream-env/bin/python"
uv pip install --python "$UP" -r reproduction/requirements-upstream.txt
git clone https://github.com/DeepGraphLearning/ULTRA.git generated/ULTRA
git -C generated/ULTRA checkout --detach 427966ad8ed60420eef034063d44f3153addff90
```

Use a CUDA-capable PyTorch 2.10.0 installation appropriate to the host.
The project-owned scatter/rspmm adapter preserves the checkpoint's aggregation
orientation; it does not substitute ordinary forward PyG message passing.
ULTRA remains an external dependency, not vendored source.

The pinned ULTRA checkout provides `ckpts/ultra_4g.pth`; its
[upstream README](https://github.com/DeepGraphLearning/ULTRA#checkpoints)
also describes supported checkpoint acquisition. Obtain TabPFN V3 through
TabPFN 8.1.0's own downloader, after accepting its model access terms:

```bash
export TABPFN_MODEL_CACHE_DIR="$PWD/generated/checkpoints"
export TABPFN_CHECKPOINT="$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt"
"$UP" -c 'import os; from pathlib import Path; from tabpfn.constants import ModelVersion; from tabpfn.model_loading import download_model; result = download_model(Path(os.environ["TABPFN_CHECKPOINT"]), version=ModelVersion.V3, which="classifier", model_name="tabpfn-v3-classifier-v3_default.ckpt"); assert result == "ok", result'
```

Do not substitute newer default checkpoints. Initialization verifies these hashes:

| Checkpoint | SHA-256 |
|---|---|
| `ultra_4g.pth` | `48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0` |
| `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |

## 3. Public KG → Local18 and frozen score → q512

Choose a workspace outside the source directory, for example:

```bash
WORK="$PWD/generated/paper-pipeline"
pipe() { "$UP" -m reproduction.run_pipeline --work-root "$WORK" "$@"; }
pipe init --ultra-root "$PWD/generated/ULTRA" --tabpfn-checkpoint "$TABPFN_CHECKPOINT"
pipe batch1 -- --register
pipe batch1 -- --download
pipe batch1 -- --prepare
pipe batch1 -- --summarize
pipe q512 -- --register
pipe q512 -- --prepare
```

These are the original CLI actions, with paths adapted by
`reproduction/run_pipeline.py`. Initialization stages the unchanged producer
source in the workspace, links the external ULTRA checkout and records model
paths. It also supplies explicitly labelled reference numerical gates used by
the original entry points; these are not presented as freshly executed checks.
No earlier experimental launcher is invoked.
The batch1 summary records the current completion state and is required by
the original q512 registration; it does not run decoder experiments.

The 57 names/classes are fixed in `rfmlink/ultra57_registry.py`.
ULTRA's public loaders acquire the raw data. The batch1 producer constructs
canonical target splits with stable duplicate removal and train/validation/test
priority, preserving raw files. It constructs the train-only graph plus inverse
relations and ULTRA's relation graph, then scores all entities for head/tail
queries. Positive-context scoring masks exactly one triple per batch.

The Local producer computes 18 fields in the declared order using train-only
statistics and clusters (at most 24, L2-normalized signatures, KMeans n_init=50,
full-support seed 7). Context features use 10 shuffled folds with the original
seed streams. ULTRA scores occupy B0 column 18. The q512 producer changes only
the evaluation cap and reuses verified batch1 context files.

Outputs:

```text
$WORK/results/ultra57_rfmlink_batch1_v1/immutable_assets/
$WORK/results/ultra57_rfmlink_q512_v1/immutable_assets/
```

For per-dataset execution the producer accepts `--dataset <registered-slug>`.
Each phase must finish for all 57 datasets before export.

## 4. q512 → Global13 → Full32

Run the original numerical gates and materializer:

```bash
pipe global --script cutoff_audit
for shard in 0 1 2 3 4 5; do
  pipe global --script development_large_stability -- \
    --output-root "$WORK/results/analysis/g2t_q512_hybrid_solver_metric_free_v1/development_large_stability" \
    --shard-id "$shard" --num-shards 6
done
pipe global --script spectral_audit -- --run
pipe global --script materialize -- --shard-id 0 --num-shards 1
pipe global --script spectral_audit -- --finalize-materialized
pipe global --script finalize_numeric_protocol
```

Global13 comprises two directed binary-graph PageRanks, three undirected
component fields and eight spectral pair features. The component fraction is
`log1p(component_size) / log1p(num_entities)`. The undirected binary graph has
no self-loops; its symmetric normalized Laplacian is solved per component.
At most eight positive modes are retained, with a component-wise zero mode
removed and protection against splitting repeated eigenspaces. Dense eigh is
used at component size ≤512; larger components use deterministic eigsh SM,
k=10, tolerance 1e-10 and maxiter=max(10000,20*n).

At scales 0.1/1/10/100, the exact producer emits component-size-scaled heat
similarities and squared diffusion distances, zero across distinct components.
It retains the original serialized `diffusion_distance_*` names.
Do not replace these with a square root or a full untruncated heat kernel.

Outputs are under
`$WORK/results/g2t_q512_hybrid_solver_metric_free_v1/feature_assets/`.
The exporter concatenates Local18, frozen score1 and Global13 to form Full32.

## 5. Export q64 RFMLink assets and construct contexts

```bash
SOURCE="$WORK/rfmlink_frozen_interface_ultra57_v2"
ASSET="$WORK/rfmlink_ultra57_relation_context_v1"
Q512="$WORK/results/ultra57_rfmlink_q512_v1/immutable_assets"
GLOBAL="$WORK/results/g2t_q512_hybrid_solver_metric_free_v1/feature_assets"
pipe study --script export_rfmlink_source_panel_q64 -- \
  --rfmlink-workspace "$WORK" --output-root "$SOURCE" --all-datasets --workers 1
pipe study --script build_rfmlink_relation_context_parallel -- \
  --reference-root "$SOURCE" --q512-root "$Q512" --b2v2-root "$GLOBAL" \
  --output-root "$ASSET" --workers 4
pipe study --script build_rfmlink_relation_context_asset -- \
  --reference-root "$SOURCE" --q512-root "$Q512" --b2v2-root "$GLOBAL" \
  --output-root "$ASSET" --finalize
```

q64 means 64 sampled test triples expanded into head/tail queries, with full
filtered all-entity ranking, not a candidate shortlist. The two literal context
modes are `frequency_random_filtered_uniform` and
`relation_matched_filtered_uniform`. Both use 200 positive + 200 negative
directional rows and the same filtered-uniform policy. The matched arm
resamples negative rows for its selected positive queries. Evaluation queries,
candidate features, targets and filtering arrays remain fixed.

## 6. Register and execute the decoder matrix

Use the artifact's decoder environment, including the `tfm` extra:

```bash
uv sync --frozen --extra dev --extra tree --extra plots --extra tfm
export RTFM_FROZEN_KG_ASSET_ROOT="$ASSET"
PY="$(uv run python -c 'import sys; print(sys.executable)')"
for profile in cpu tabpfn; do
  uv run python -m reproduction.run_relation_context decoders -- \
    --repo-root "$PWD" --asset-root "$ASSET" --output-root "$WORK/runs/$profile" \
    --python "$PY" --profile "$profile" --tabpfn-cache "$TABPFN_MODEL_CACHE_DIR" --prepare-only
  uv run python -m reproduction.run_relation_context decoders -- \
    --repo-root "$PWD" --asset-root "$ASSET" --output-root "$WORK/runs/$profile" \
    --python "$PY" --profile "$profile" --tabpfn-cache "$TABPFN_MODEL_CACHE_DIR"
done
```

The unchanged launcher records actual source/config/asset/plan identities,
creates 2,736 CPU and 1,368 TabPFN cells and verifies completed shards.
`--worker-index` / `--worker-count` distribute the same registered shards.
The four schemas, two modes and seeds 101–103 are fixed.
The existing runner performs filtered ranking, saves predictions and checks
checkpoint identity. No decoder behavior is changed by the portability wrapper.

## 7. Aggregate and reproduce the statistical analysis

```bash
uv run python -m reproduction.run_relation_context analyze \
  --execution-source "$(git rev-parse HEAD)" -- \
  --cpu-run-root "$WORK/runs/cpu" --tabpfn-run-root "$WORK/runs/tabpfn" \
  --asset-manifest "$ASSET/manifest.json" --output "$WORK/analysis/relation-context" \
  --analysis-commit "$(git rev-parse HEAD)"
```

Fresh runs are checked against their actual execution revision; they are never
attributed to the reference execution SHA. The original analysis validates
all 4,104 cells, audits positive-relation coverage, computes dataset/family
effects and applies one exact 60-test max-|t| correction across 16 provenance
families (65,536 shared sign patterns). Identity effects are with minus without
identity. The separate context-robustness 30-test family remains separate.
The prospective `analysis/analyze_ultra4g_positive_context.py` utility fixes
exact negative rows and offers 12/24-test sensitivities; it is not the paper path.

For all other KG controls, the generated source manifest can feed the existing
builders/registrations in `analysis/`; see [ARTIFACT_MAP.md](ARTIFACT_MAP.md).
Use each script's `--help` and its own registered asset/plan identities.

## 8. CLUTRR, GraphLog and generated controls

Stage public data using the checksummed declarations:

```bash
uv run rtfm stage-data --asset clutrr_v1_gen23_train \
  --asset clutrr_v1_gen23_validation --asset clutrr_v1_gen23_test \
  --asset clutrr_v1_gen234_train --asset clutrr_v1_gen234_validation \
  --asset clutrr_v1_gen234_test --asset graphlog_v1_1_author_mirror
uv run rtfm plan --config configs/studies/clutrr_symbolic_cpu_confirmation_v2.yaml \
  --output generated/plans/clutrr-cpu.json
uv run rtfm run-shard --plan generated/plans/clutrr-cpu.json \
  --shard-index 0 --cells-per-shard 1 --output-dir generated/runs/clutrr/shard-0000
```

Execute all registered shards to complete a study. Substitute the GRU,
GraphLog, dihedral or controlled-study config from `configs/studies/` as
appropriate; PyTorch is needed for GRU. Synthetic and dihedral populations are
generated from the configured seeds. The claim map identifies each aggregate
script and its required run inputs. The GraphLog capacity sweep has 2,394 cells.
TabPFN studies use the pinned V3 checkpoint through the model cache above.

## 9. Verify paper outputs and render figures

```bash
uv run python -m reproduction.verify_source_pipeline
uv run python -m analysis.verify_historical_gru
uv run python -m analysis.verify_historical_relation_context
uv run python tools/verify_artifact.py
mkdir -p generated/figures
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_component_audit_anatomy.py
MPLCONFIGDIR=.matplotlib uv run python analysis/plot_clutrr_ablation.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_composition_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_context_robustness_figure.py
MPLCONFIGDIR=.matplotlib uv run python analysis/generate_query_evidence_figure.py
```

These result verifiers reconstruct reported endpoints and corrected contrasts
from bundled reference tables. The source verifier also checks source hashes,
feature order, solver policy and checkpoint identities. Large generated assets
and checkpoints are not bundled. [PROVENANCE.md](PROVENANCE.md) distinguishes
scientific-plan-body hashes from serialized-file hashes and documents reference
execution identities. Keep reruns separate from checksummed reference outputs.
