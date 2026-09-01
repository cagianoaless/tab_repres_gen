# Pointwise, query-relative, and filtering controls

The query-relative control compares raw, within-query z-score, and within-query
percentile rows. The filtering control changes only whether context negatives are
rejected by the all-known-positive mask.

The `*_absolute_mrr.csv`, `*_contrasts.csv`, and `*_dataset_effects.csv` files
contain the corresponding summaries. `summary.json` records completeness, plan
hashes, asset hashes, and contamination counts.
