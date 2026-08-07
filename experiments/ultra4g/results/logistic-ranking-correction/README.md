# Logistic Ranking Audit

Binary ranking uses decision values rather than saturated probabilities. For the one-column `ultra_score_only` representation, the fitted coefficient supplies only the orientation; candidates are ranked by the original frozen `ultra_4g` score so upstream ties are preserved exactly.

The audit covers all `57 x 9 x 4 x 3 = 6,156` logistic cells and verifies exact equality between score-only logistic and native `ultra_4g` ranks in all 684 score-only cells.

## Files

- `corrected_logistic_dataset_metrics.csv`: seed-averaged metrics for all nine representations and four context budgets.
- `query_evidence_strata.csv`: the 24 family-blocked query-stratum summaries used by the manuscript figure.
- `audit.json`: completeness and rank-equivalence checks.
