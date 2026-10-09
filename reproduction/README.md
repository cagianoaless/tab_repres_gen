# Paper reproduction pipeline

The source closure constructs Local18, the frozen ULTRA score and Global13
from public ULTRA KGs, exports the Full32 RFMLink representation, and runs
the decoder/analysis pipeline. Follow [REPRODUCE.md](../REPRODUCE.md).

| Path | Role |
|---|---|
| `pipeline/RFMLink/rfmlink/vectorized_invariants.py` | Local18 producer |
| `pipeline/RFMLink/rfmlink/ultra_wrapper.py`, `torch_scatter_shim.py` | Checkpoint-compatible ULTRA inference |
| `pipeline/RFMLink/scripts/run_ultra57_rfmlink*.py` | Dataset canonicalization and batch1/q512 materialization |
| `pipeline/RFMLink/side_experiments/g2t_q512_hybrid_solver_metric_free/` | Global13 solver, numerical gates and materialization |
| `pipeline/RFMLink/side_experiments/g2t_q512_eigengap_safe/features.py` | Inherited pair-feature formulas |
| `study/scripts/` | RFMLink export, contexts, decoder registration/execution and analysis |
| `run_pipeline.py`, `run_relation_context.py` | Thin path/provenance adapters |
| `SOURCE_MANIFEST.json` | Exact transfer hashes and source roles |
| `REFERENCE_EXECUTION.json`, `reference/` | Execution identities and numerical reference gates |
| `verify_source_pipeline.py` | CPU source/schema/solver/checkpoint-identity checks |

The unchanged import closure includes supporting modules from earlier producer
versions. They are import dependencies, not additional experiments to execute.
Shell launchers and the prediction-batch benchmark are retained where the
original registration hashes their bytes. Use the portable Python entry points,
not machine-specific shell defaults.

Producer sources remain byte-identical to `SOURCE_MANIFEST.json`; portable
wrappers adapt paths and runtime configuration. Numerical gate files are
reference checks from the paper execution. Checkouts, weights and large
generated intermediates remain external; raw plans/predictions are not bundled.

Relation-context uses 60 jointly corrected combined-direction contrasts.
Context robustness uses a separate 30-test family. The prospective
fixed-negative-row analyzer is not this pipeline.
