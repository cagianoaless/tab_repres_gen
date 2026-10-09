# Scientific claim to artifact map

Paths are relative to the repository root. The reproduction pipeline is:

```text
public ULTRA datasets + checkpoints → canonical source panel → Local18/score/Global13
→ Full32 RFMLink assets → contexts/decoders → filtered rankings → analyses → paper outputs
```

Source is bundled under `reproduction/`; model weights and large generated
intermediates remain external. Reference tables validate paper endpoints
without running models. `public` means public/generated data plus the required
model access; `generated KG` means assets constructed by the supplied upstream
pipeline; `frozen` means those archives are held fixed within a registered run;
`tables` means a direct check/plot from bundled reference outputs.
See [REPRODUCE.md](REPRODUCE.md) for the complete command sequence.

| Claim / experiment | Checked-in evidence | Analysis / figure | Configuration / execution source | Reproduction boundary |
|---|---|---|---|---|
| Paired graph-to-table diagnostic framework | `TERMINOLOGY.md`, protocol and runner | `analysis/plot_component_audit_anatomy.py` | `src/relational_tfm/protocol.py`, `runner.py` | tables / conceptual |
| Controlled row-equivalence calibration | `experiments/controlled/results/` | `audit_finite_population_ceilings.py`, `summarize_property_discovery_confirmation.py`, `plot_property_discovery_confirmation.py` | `configs/studies/property_discovery*.yaml` | generated data; aggregate regeneration needs cell runs |
| CLUTRR short-to-long support shift | `experiments/clutrr/results/metric-robustness/`, `coordinate-coverage/`, `length-only/` | `audit_clutrr_metric_robustness.py`, `audit_clutrr_coordinate_coverage.py`, `audit_clutrr_length_only_tabpfn.py` | `clutrr_symbolic_*confirmation*.yaml`, `*length_only_replay*.yaml` | public; fresh complete cell runs required for aggregate regeneration |
| CLUTRR ordered-block interference | `experiments/clutrr/results/ordered-interference/` | `summarize_clutrr_ordered_interference.py`, `plot_clutrr_ablation.py` | `clutrr_symbolic_*ordered_interference_v1.yaml` | public rerun; plot uses tables |
| CLUTRR supported-prefix intervention | `experiments/clutrr/results/supported-prefix/`, `shortcut-diagnostic/` | `summarize_clutrr_supported_prefix.py`, `summarize_clutrr_shortcut_diagnostic.py` | `clutrr_symbolic_*supported_prefix_v1.yaml`, `*shortcut_diagnostic_v1.yaml` | public rerun; context fit is resubstitution |
| CLUTRR length-4 exposure | `experiments/clutrr/results/additive-context/` | `summarize_clutrr_additive_context_control.py` | `clutrr_symbolic_*additive_length4_v1.yaml`, `*additive_short_control_v1.yaml` | public rerun; separate paired addition arms |
| CLUTRR recurrent sequence control | `experiments/clutrr/results/recurrent-gru/` | `verify_historical_gru.py`; GRU decoder | `clutrr_symbolic_cpu_recursive_gru_v1.yaml` | public rerun; reference tables verify five OOD means and length endpoints; execution identities in `PROVENANCE.md` |
| CLUTRR collision-conditioned ceilings | `experiments/clutrr/results/collision-conditioned-ceilings/` | `audit_clutrr_singleton_mass.py` | CLUTRR public asset/config and encoder execution source | public; no model inference required |
| GraphLog decoder sensitivity | `experiments/graphlog/results/decoder-sensitivity/`, `symbolic-replication/`, `rule-overlap/` | `summarize_graphlog_decoder_sensitivity.py`, `analyze_graphlog_rule_overlap.py`, `generate_composition_figure.py` | `graphlog_symbolic_*confirmation_v1.yaml`, `*depth1_endpoint_v1.yaml`, `*tuned_endpoint_v1.yaml` | public rerun; figures use tables |
| GraphLog CatBoost depth sweep | `experiments/graphlog/results/capacity-sweep/` | `run_graphlog_capacity_sweep.py`, `summarize_graphlog_capacity_sweep.py` | `graphlog_symbolic_cpu_catboost_capacity_sweep_reviewer_v1.yaml` | public; 2394 CPU cells |
| GraphLog fixed-panel representation diagnostics | `experiments/graphlog/results/fixed-panel/` | `audit_graphlog_schedule.py`, `audit_graphlog_representation_by_length.py`, `summarize_graphlog_fixed_panel_audit.py` | GraphLog public asset and registered configs; original schedule-audit input not shipped | public regeneration; reference sign summary uses its registered contrast source |
| Dihedral-16 boundary control | `experiments/dihedral16/results/cpu/`, `tabpfn/` | `summarize_dihedral16.py`, `summarize_dihedral16_tabpfn.py` | `dihedral16_*confirmation_v1.yaml` | generated populations; aggregate regeneration needs runs |
| 57-KG complete encoder ladder | `experiments/ultra4g/results/q64-encoder-ladder/`, `multiplicity/`, `native-ranking/` | `analyze_ultra4g_q64_complete_panel.py`, `analyze_rfmlink_q64_multiplicity_complete57.py` | Bundled upstream producers/exporter; manifest-based plans and loader/runner | generated KG; reference summaries validate directly |
| Frozen ULTRA score additions | `experiments/ultra4g/results/context-robustness/contrast_summary.csv`, ladder contrast tables | `analyze_rfmlink_negative_robustness.py` | `generate_rfmlink_negative_robustness_registration.py` | frozen; checked-in effects validate directly |
| Context robustness | `experiments/ultra4g/results/context-robustness/`, `publication_data/context_*.csv` | `build_rfmlink_negative_robustness_asset.py`, `analyze_rfmlink_negative_robustness.py`, `generate_context_robustness_figure.py` | `generate_rfmlink_negative_robustness_registration.py` | frozen q512 support and archive assets; figure uses tables; separate 30-test correction (18 encoder + 12 negative-context effects), unchanged |
| Query-relative pointwise control | `experiments/ultra4g/results/pointwise-and-filtering-controls/query_*.csv`, `score_rank_checks.csv` | `analyze_ultra4g_controls.py`; `evaluation/query_relative.py` | `generate_ultra4g_control_registration.py`, `run_ultra4g_controls.py` | frozen; tests representation changes, not ranking loss |
| Positive-context relation coverage | `experiments/ultra4g/results/relation-context-historical/` (4104-cell reference matrix, coverage and contrasts) | `verify_historical_relation_context.py`; `reproduction/study/scripts/analyze_rfmlink_relation_context.py` | `reproduction/run_pipeline.py`, `run_relation_context.py` | end-to-end generated KG; 16-family/60-test correction; reference tables verified directly |
| Relation-identity ablation | Same reference package; 12 with-minus-without identity effects | Same 60-test analysis; loader/tests | `b0_local_no_relation_id`, `b0_local_ultra_no_relation_id` | same joint 60-test family; generated KG |
| Prospective fixed-negative positive-context sensitivity | `experiments/ultra4g/results/positive-context/README.md`; fixture tests, not real results | `analyze_ultra4g_positive_context.py` | Supplied prospective plans and fixed-negative frozen assets | frozen: stricter exact-negative-row contract; separate 12+12 or joint 24 tests, neither is the paper's 60-test procedure |
| Matched all-known-positive filtering | `experiments/ultra4g/results/pointwise-and-filtering-controls/matched_filter_*.csv` | `build_ultra4g_matched_filter_asset.py`, `analyze_ultra4g_controls.py` | `generate_ultra4g_control_registration.py`, `run_ultra4g_controls.py` | frozen q512 and paired context assets |
| Full-vs-native ULTRA aggregation sensitivity | `experiments/ultra4g/results/aggregation-sensitivity/` | `analyze_ultra4g_aggregation_sensitivity.py` | Checked-in ladder `dataset_metrics.csv` | tables; no models/assets needed |
| Query endpoint-connectivity strata | `experiments/ultra4g/results/logistic-ranking-correction/query_evidence_strata.csv`, `publication_data/query_evidence_absolute.csv` | `analyze_ultra4g_logistic_ranking_correction.py`, `generate_query_evidence_figure.py` | Ladder prediction archives + provenance map | frozen for extraction; figure uses tables |
| ULTRA pretraining-lineage restriction | `experiments/ultra4g/results/pretraining-lineage/` | `analyze_ultra4g_pretraining_lineage_sensitivity.py` | Original wide provenance-family contrast table (input hash in summary; not shipped in that format) | reference retained-family outputs validate directly; original full-input regeneration requires the registered source table |

Analysis filenames without a prefix are under `analysis/`; config filenames are
under `configs/studies/`. Wildcards indicate existing related configurations,
not missing files. New runs go to a generated workspace; reference outputs
remain checksummed. Acceptance targets are validation criteria, not
experimental data.

## Representation construction

| Stage | Bundled source | Generated output / dependency |
|---|---|---|
| Public 57-KG loading and canonicalization | `reproduction/pipeline/RFMLink/scripts/run_ultra57_rfmlink.py`, registry | Pinned external ULTRA checkout / public datasets |
| Local18 and score1 | `rfmlink/vectorized_invariants.py`, `ultra_wrapper.py`, confirmation engine under `reproduction/pipeline/RFMLink/` | `ultra57_rfmlink_batch1_v1/immutable_assets/`; ULTRA-4G checkpoint |
| q512 candidate materialization | `run_ultra57_rfmlink_q512.py` | `ultra57_rfmlink_q512_v1/immutable_assets/` |
| Global13 numerical features | Hybrid/eigengap-safe modules under `reproduction/pipeline/RFMLink/side_experiments/` | `g2t_q512_hybrid_solver_metric_free_v1/feature_assets/` |
| Full32 / q64 export | `reproduction/study/scripts/export_rfmlink_source_panel_q64.py` | `rfmlink_frozen_interface_ultra57_v2` manifest + NPZs |
| Context sampling | Original relation-context builders under `reproduction/study/scripts/`; other control builders under `analysis/` | Asset-specific context NPZs/manifests |
| Decoder plans, execution and rankings | `reproduction/run_relation_context.py`; `src/relational_tfm/` | Plans, runtime records, metrics and prediction shards |
| Aggregation and exact correction | Original analysis under `reproduction/study/scripts/`, dedicated reference verifiers | Dataset/family tables, 60-test relation-context results |
| Paper inputs | `analysis/`, `publication_data/`, reference result packages | Lightweight checksummed verification/figure inputs |

The CLUTRR shortcut-diagnostic and primary metric-robustness TabPFN multiset
tables represent separate experiments (approximately 0.284739 and 0.289382
OOD accuracy respectively); the primary verifier uses the primary table.

## Endpoints

- CLUTRR: three paired context seeds; nested 5, 10, 20, 40, and 80 examples per target relation; evaluation lengths 4--10.
- GraphLog: all 57 official worlds; nested 5, 10, 20, and 40 examples per present class; evaluation lengths 4--5.
- Dihedral-16: ten generated populations; context lengths 2--3 and evaluation lengths 4--10.
- Frozen `ultra_4g`: 57 datasets in 16 provenance families; q64 test triples with full filtered entity ranking; three context seeds.
- Context robustness: 4,104 complete cells = 57 datasets x 4 interfaces x 2 context constructions x 3 decoders x 3 seeds.
- Relation-context: a separate complete 4,104-cell matrix; one 60-test correction across 16 families; generated from the supplied pipeline.
- Recurrent GRU: 135 budget/seed/length reference records and five OOD means; public-data execution config and source supplied.
- GraphLog capacity sweep: 2,394 complete cells = 57 worlds x 2 encoders x 7 depths x 3 seeds.
- Query-relative KG control: 4,617 complete cells across raw, within-query z-score, and within-query percentile representations.
- Matched filtering control: 1,026 complete TabPFN cells with identical positive context rows and evaluation.
