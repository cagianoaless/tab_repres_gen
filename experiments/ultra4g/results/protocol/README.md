# Frozen-ULTRA q64 protocol audit

`audit.json` is generated directly from the immutable 57-dataset asset archive
by:

```bash
python analysis/audit_rfmlink_ultra57_protocol.py \
  --asset-root /path/to/rfmlink_frozen_interface_ultra57_v2 \
  --output results/ultra57/protocol/audit.json
```

The audit checks the nine interface schemas, both q64 query directions,
400+400 source contexts, deterministic nested budgets, and exact positive-query
disjointness between context and evaluation. It records rather than hides
duplicate context triples or random negatives that coincide with evaluation
positives. It does not characterize negative hardness or validate model scores.
