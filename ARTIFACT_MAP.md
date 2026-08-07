# Manuscript to Artifact Map

Paths are relative to the repository root.

| Manuscript item | Evidence | Analysis or figure code |
|---|---|---|
| Figure 1, paired audit | Conceptual construction | `analysis/plot_component_audit_anatomy.py` |
| Figure 2, CLUTRR | `experiments/clutrr/results/ordered-interference/`, `additive-context/`, `supported-prefix/` | `analysis/plot_clutrr_ablation.py`; CLUTRR audit and summary scripts in `analysis/` |
| Figure 3, representation contrast | CLUTRR endpoint tables; `experiments/graphlog/results/decoder-sensitivity/`; `experiments/dihedral16/results/` | `analysis/generate_composition_figure.py` |
| Figure 4, context robustness | `experiments/ultra4g/results/context-robustness/` | `analysis/generate_context_robustness_figure.py` |
| Figure 5, query evidence | `experiments/ultra4g/results/logistic-ranking-correction/query_evidence_strata.csv` and `publication_data/query_evidence_absolute.csv` | `analysis/generate_query_evidence_figure.py` |
| Appendix, controlled calibration | `experiments/controlled/results/` | `analysis/audit_finite_population_ceilings.py`, `summarize_property_discovery_confirmation.py`, `plot_property_discovery_confirmation.py` |
| Appendix, CLUTRR levels and robustness | `experiments/clutrr/results/` | `analysis/audit_clutrr_*.py`, `analysis/summarize_clutrr_*.py` |
| Appendix, GraphLog levels and ceilings | `experiments/graphlog/results/fixed-panel/`, `decoder-sensitivity/`, `rule-overlap/` | `analysis/audit_graphlog_*.py`, `analysis/summarize_graphlog_*.py`, `analysis/analyze_graphlog_rule_overlap.py` |
| Appendix, dihedral levels | `experiments/dihedral16/results/cpu/` and `tabpfn/` | `analysis/summarize_dihedral16.py`, `analysis/summarize_dihedral16_tabpfn.py` |
| Appendix, complete nine-interface `ultra_4g` ladder | `experiments/ultra4g/results/q64-encoder-ladder/`, `multiplicity/`, `native-ranking/` | `analysis/analyze_ultra4g_q64_complete_panel.py`, `analysis/analyze_rfmlink_q64_multiplicity_complete57.py` |
| Appendix, complete context matrix | `experiments/ultra4g/results/context-robustness/` | `analysis/analyze_rfmlink_negative_robustness.py` |

## Endpoints

- CLUTRR: three paired context seeds; nested 5, 10, 20, 40, and 80 examples per target relation; evaluation lengths 4--10.
- GraphLog: all 57 official worlds; nested 5, 10, 20, and 40 examples per present class; evaluation lengths 4--5.
- Dihedral-16: ten generated populations; context lengths 2--3 and evaluation lengths 4--10.
- Frozen `ultra_4g`: 57 datasets in 16 provenance families; q64 test triples with full filtered entity ranking; three context seeds.
- Context robustness: 4,104 complete cells = 57 datasets x 4 interfaces x 2 context constructions x 3 decoders x 3 seeds.
