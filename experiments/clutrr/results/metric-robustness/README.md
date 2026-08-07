# CLUTRR metric-robustness audit

Status: **PASS**.

This audit reuses the complete corrected CLUTRR CPU and TabPFN-3 prediction
populations. It does not fit a model or select a configuration. The estimand is
the equal-length mean over fixed OOD lengths 4--10 at 80 context examples per
target relation, followed by an equal mean over seeds 101--103.

## Result

The relation multiset outperforms the ordered absolute-position path for every
decoder on all six registered metrics:

| Decoder | Accuracy | Balanced accuracy | Macro-F1 | Top-3 accuracy | Log loss reduction | Brier reduction |
|---|---:|---:|---:|---:|---:|---:|
| Logistic | +0.273 | +0.236 | +0.224 | +0.403 | 4.208 | 0.482 |
| CatBoost | +0.163 | +0.132 | +0.155 | +0.293 | 2.002 | 0.459 |
| TabPFN | +0.186 | +0.201 | +0.153 | +0.304 | 2.510 | 0.540 |

Positive values always favor the multiset; for log loss and Brier score they
are `ordered - multiset` reductions.

The accuracy and macro-F1 ordering holds in all 21 decoder-by-OOD-length
comparisons. Balanced accuracy agrees in 20/21; CatBoost at length 9 is the
single reversal (`-0.0194`). At the target-relation level the multiset wins for
16/18 relations under logistic regression, 16/18 under CatBoost, and 15/18
under TabPFN. The result is therefore broad across metrics and lengths, but not
uniform across target relations.

## Provenance

- CPU study: `clutrr_symbolic_cpu_confirmation_v2`, 150/150 cells, plan
  `cca4114229dbfac4938d707252c4257817149817becabb4b8e8b43a7bf86df98`.
- TabPFN study: `clutrr_symbolic_gpu_tabpfn_confirmation_v2`, 75/75 cells,
  plan `d118695450af93907a347aab5760dfb6bb3067ed30ffe3255771722bfd348a4b`.
- Source result commits and exact prediction checksums are recorded in
  `summary.json`.

`primary_metric_robustness.csv` contains the six metric comparisons,
`primary_per_length.csv` contains accuracy, balanced accuracy, and macro-F1
for each OOD length, and `primary_per_target_relation.csv` contains all
18 target-relation effects.

The scope is one fixed official CLUTRR test population and three nested-context
seeds. These results do not support inference to a population of datasets or
relation ontologies.
