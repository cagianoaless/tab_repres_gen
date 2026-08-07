# CLUTRR shared-coordinate TabPFN audit

Status: complete and audited.

- Study: `clutrr_symbolic_gpu_tabpfn_shared_coordinates_v1`
- Execution source: `artifact-source-revision`
- Plan SHA-256:
  `f911307adcd680acaeb25db68429fd8357c1b190544f6f318fa06b59f92229d5`
- Cells: 60/60
- Shards: 15/15
- Failures: 0
- Decoder: TabPFN-3 8.1.0, four estimators, `fit_with_cache`, `cuda:0`

At 80 context examples per relation, equal-length OOD accuracy over chain
lengths 4--10 is:

| Interface | Accuracy | Matched null | Difference |
|---|---:|---:|---:|
| Directed transition counts | 0.1562 | 0.0755 | +0.0806 |
| Ordered path plus relation multiset | 0.1874 | 0.0865 | +0.1009 |

All 30 aligned-minus-null contrasts are positive across five contexts, three
seeds, and the two interfaces. Relative to the previously registered absolute
ordered-path TabPFN endpoint (`0.1032`), the largest-context improvements are
`+0.0530` for transition counts and `+0.0842` for ordered-plus-multiset.

This confirms with the primary TFM that shared coordinates improve extraction
relative to the absolute-position table. It does not establish generic
systematic composition: both tabular interfaces remain well below the
shared-parameter GRU (`0.7055`) and the logistic transition-count readout
(`0.4085`) on the same fixed CLUTRR population.

Regenerate the compact tables from the immutable raw result root with:

```bash
python analysis/summarize_clutrr_shared_tabpfn.py \
  --results-root /path/to/clutrr_symbolic_gpu_tabpfn_shared_coordinates_v1 \
  --output-dir results/clutrr/shared-coordinates
```

`cells.csv`, `aggregate.csv`, and `contrasts.csv` contain every registered
cell and pairing. `summary.json` records the largest-context endpoint and
scope.
