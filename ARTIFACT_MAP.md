# Scientific claim to artifact map

Paths are relative to the repository root. Entries identify scientific experiments,
not figure numbers. Historical aggregates are evidence, not substitutes for raw
prediction archives. `public` means rerunnable after staging checksummed public
data (TabPFN additionally needs a checkpoint); `frozen` means external KG runtime
archives/runs are required; `tables` means the specified downstream step runs
directly on checked-in tables; `missing` means the required source is not shipped.

| Claim / experiment | Checked-in evidence | Analysis / figure | Configuration / execution source | Reproduction boundary |
|---|---|---|---|---|
| Paired graph-to-table diagnostic framework | `TERMINOLOGY.md`, protocol and runner | `analysis/plot_component_audit_anatomy.py` | `src/relational_tfm/protocol.py`, `runner.py` | tables / conceptual |
| Controlled row-equivalence calibration | `experiments/controlled/results/` | `audit_finite_population_ceilings.py`, `summarize_property_discovery_confirmation.py`, `plot_property_discovery_confirmation.py` | `configs/studies/property_discovery*.yaml` | generated data; aggregate regeneration needs cell runs |
| CLUTRR short-to-long support shift | `experiments/clutrr/results/metric-robustness/`, `coordinate-coverage/`, `length-only/` | `audit_clutrr_metric_robustness.py`, `audit_clutrr_coordinate_coverage.py`, `audit_clutrr_length_only_tabpfn.py` | `clutrr_symbolic_*confirmation*.yaml`, `*length_only_replay*.yaml` | public; historical per-cell archives not shipped |
| CLUTRR ordered-block interference | `experiments/clutrr/results/ordered-interference/` | `summarize_clutrr_ordered_interference.py`, `plot_clutrr_ablation.py` | `clutrr_symbolic_*ordered_interference_v1.yaml` | public rerun; plot uses tables |
| CLUTRR supported-prefix intervention | `experiments/clutrr/results/supported-prefix/`, `shortcut-diagnostic/` | `summarize_clutrr_supported_prefix.py`, `summarize_clutrr_shortcut_diagnostic.py` | `clutrr_symbolic_*supported_prefix_v1.yaml`, `*shortcut_diagnostic_v1.yaml` | public rerun; context fit is resubstitution |
| CLUTRR length-4 exposure | `experiments/clutrr/results/additive-context/` | `summarize_clutrr_additive_context_control.py` | `clutrr_symbolic_*additive_length4_v1.yaml`, `*additive_short_control_v1.yaml` | public rerun; separate paired addition arms |
| CLUTRR recurrent sequence control | No dedicated checked-in recurrent result table | GRU in `src/relational_tfm/decoders/classification.py` | `clutrr_symbolic_cpu_recursive_gru_v1.yaml` | public execution source present; historical output missing |
| CLUTRR collision-conditioned ceilings | `experiments/clutrr/results/collision-conditioned-ceilings/` | `audit_clutrr_singleton_mass.py` | CLUTRR public asset/config and encoder execution source | public; no model inference required |
| GraphLog decoder sensitivity | `experiments/graphlog/results/decoder-sensitivity/`, `symbolic-replication/`, `rule-overlap/` | `summarize_graphlog_decoder_sensitivity.py`, `analyze_graphlog_rule_overlap.py`, `generate_composition_figure.py` | `graphlog_symbolic_*confirmation_v1.yaml`, `*depth1_endpoint_v1.yaml`, `*tuned_endpoint_v1.yaml` | public rerun; figures use tables |
| GraphLog CatBoost depth sweep | `experiments/graphlog/results/capacity-sweep/` | `run_graphlog_capacity_sweep.py`, `summarize_graphlog_capacity_sweep.py` | `graphlog_symbolic_cpu_catboost_capacity_sweep_reviewer_v1.yaml` | public; 2394 CPU cells |
| GraphLog fixed-panel representation diagnostics | `experiments/graphlog/results/fixed-panel/` | `audit_graphlog_schedule.py`, `audit_graphlog_representation_by_length.py`, `summarize_graphlog_fixed_panel_audit.py` | GraphLog public asset and registered configs; original schedule-audit input not shipped | public regeneration; historical sign summary requires its original contrast source |
| Dihedral-16 boundary control | `experiments/dihedral16/results/cpu/`, `tabpfn/` | `summarize_dihedral16.py`, `summarize_dihedral16_tabpfn.py` | `dihedral16_*confirmation_v1.yaml` | generated populations; aggregate regeneration needs runs |
| 57-KG complete encoder ladder | `experiments/ultra4g/results/q64-encoder-ladder/`, `multiplicity/`, `native-ranking/` | `analyze_ultra4g_q64_complete_panel.py`, `analyze_rfmlink_q64_multiplicity_complete57.py` | Runtime plans from frozen manifest; loader/runner | frozen; full original run tables not shipped |
| Frozen ULTRA score additions | `experiments/ultra4g/results/context-robustness/contrast_summary.csv`, ladder contrast tables | `analyze_rfmlink_negative_robustness.py` | `generate_rfmlink_negative_robustness_registration.py` | frozen; checked-in effects validate directly |
| Context robustness | `experiments/ultra4g/results/context-robustness/`, `publication_data/context_*.csv` | `build_rfmlink_negative_robustness_asset.py`, `analyze_rfmlink_negative_robustness.py`, `generate_context_robustness_figure.py` | `generate_rfmlink_negative_robustness_registration.py` | frozen q512 support and archive assets; figure uses tables |
| Query-relative pointwise control | `experiments/ultra4g/results/pointwise-and-filtering-controls/query_*.csv`, `score_rank_checks.csv` | `analyze_ultra4g_controls.py`; `evaluation/query_relative.py` | `generate_ultra4g_control_registration.py`, `run_ultra4g_controls.py` | frozen; tests representation changes, not ranking loss |
| Positive-context relation coverage | Dependency note only: `experiments/ultra4g/results/positive-context/README.md` | `analyze_ultra4g_positive_context.py` | `rfmlink_ultra57_relation_context_v1` registration support; historical plan absent | missing archives and 4104-cell matrix; no real output claimed |
| Relation-identity ablation | Same dependency note; encoder semantics in loader/tests | `analyze_ultra4g_positive_context.py` | `b0_local_no_relation_id`, `b0_local_ultra_no_relation_id` | missing matrix; implementation and fixture tests present |
| Matched all-known-positive filtering | `experiments/ultra4g/results/pointwise-and-filtering-controls/matched_filter_*.csv` | `build_ultra4g_matched_filter_asset.py`, `analyze_ultra4g_controls.py` | `generate_ultra4g_control_registration.py`, `run_ultra4g_controls.py` | frozen q512 and paired context assets |
| Full-vs-native ULTRA aggregation sensitivity | `experiments/ultra4g/results/aggregation-sensitivity/` | `analyze_ultra4g_aggregation_sensitivity.py` | Checked-in ladder `dataset_metrics.csv` | tables; no models/assets needed |
| Query endpoint-connectivity strata | `experiments/ultra4g/results/logistic-ranking-correction/query_evidence_strata.csv`, `publication_data/query_evidence_absolute.csv` | `analyze_ultra4g_logistic_ranking_correction.py`, `generate_query_evidence_figure.py` | Historical ladder prediction archives + provenance map | frozen for extraction; figure uses tables |
| ULTRA pretraining-lineage restriction | `experiments/ultra4g/results/pretraining-lineage/` | `analyze_ultra4g_pretraining_lineage_sensitivity.py` | Original wide provenance-family contrast table (input hash in summary; not shipped in that format) | historical retained-family outputs validate directly; original full-input regeneration requires the registered source table |

Analysis filenames without a prefix are under `analysis/`; config filenames are
under `configs/studies/`. Wildcards indicate existing related configurations,
not missing files. `ARTIFACT_MAP.md` intentionally includes absent sources and
does not substitute acceptance targets for evidence.

## Endpoints

- CLUTRR: three paired context seeds; nested 5, 10, 20, 40, and 80 examples per target relation; evaluation lengths 4--10.
- GraphLog: all 57 official worlds; nested 5, 10, 20, and 40 examples per present class; evaluation lengths 4--5.
- Dihedral-16: ten generated populations; context lengths 2--3 and evaluation lengths 4--10.
- Frozen `ultra_4g`: 57 datasets in 16 provenance families; q64 test triples with full filtered entity ranking; three context seeds.
- Context robustness: 4,104 complete cells = 57 datasets x 4 interfaces x 2 context constructions x 3 decoders x 3 seeds.
- GraphLog capacity sweep: 2,394 complete cells = 57 worlds x 2 encoders x 7 depths x 3 seeds.
- Query-relative KG control: 4,617 complete cells across raw, within-query z-score, and within-query percentile representations.
- Matched filtering control: 1,026 complete TabPFN cells with identical positive context rows and evaluation.
