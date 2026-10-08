# Relation-context compatibility

Post-hoc fixed-panel factorial on 57 knowledge graphs. Evaluation rows, q64 candidates, filtered masks, feature matrices, context size, and negative sampling are fixed; only positive relation coverage changes.

The simultaneous sensitivity uses exact family-level sign flips and max-|t| adjustment over all 60 combined-direction contrasts.

## Relation-matched minus frequency-random context

| Decoder | Interface | Equal-family MRR | W/T/L | max-|t| p |
|---|---|---:|---:|---:|
| catboost | b0_local_no_relation_id | +0.0139 | 31/14/12 | 0.03690 |
| catboost | b0_local_only | +0.0084 | 32/12/13 | 0.50769 |
| catboost | b0_local_ultra | +0.0062 | 28/14/15 | 0.99823 |
| catboost | b0_local_ultra_no_relation_id | +0.0080 | 31/8/18 | 0.99695 |
| logistic | b0_local_no_relation_id | +0.0025 | 26/14/17 | 1.00000 |
| logistic | b0_local_only | +0.0023 | 27/13/17 | 1.00000 |
| logistic | b0_local_ultra | +0.0025 | 15/14/28 | 1.00000 |
| logistic | b0_local_ultra_no_relation_id | +0.0004 | 17/10/30 | 1.00000 |
| tabpfn | b0_local_no_relation_id | +0.0036 | 32/12/13 | 0.99951 |
| tabpfn | b0_local_only | +0.0018 | 34/9/14 | 1.00000 |
| tabpfn | b0_local_ultra | +0.0053 | 20/22/15 | 0.99884 |
| tabpfn | b0_local_ultra_no_relation_id | +0.0022 | 22/19/16 | 1.00000 |

Full cell, dataset, family, distribution, interaction, and multiplicity tables are stored as CSV files in this directory.
