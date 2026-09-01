# Pointwise, query-relative, and filtering controls

The query-relative control compares raw rows with within-query z-score and
within-query percentile transforms over the same 57 datasets. It contains 4,617
complete cells across logistic regression, CatBoost, and TabPFN.

The matched filtering control compares uniform negatives that exclude only the
target entity with uniform negatives rejected by the all-known-positive mask.
Positive rows, source queries, seeds, evaluation, features, and decoder settings
are fixed. The unfiltered arm contains 456 known-positive negatives among 34,200
negative context rows.

`query_transform_contrasts.csv` and `matched_filter_contrasts.csv` contain the
primary paired effects. `summary.json` records completeness, frozen plan hashes,
asset hashes, and contamination counts.
