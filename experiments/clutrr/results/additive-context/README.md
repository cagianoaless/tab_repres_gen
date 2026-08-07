# Paired additive CLUTRR context control

Status: complete and audited.

- Execution source: `artifact-source-revision`
- CPU cells: 120/120, zero failures
- TabPFN cells: 60/60, zero failures
- Paired base-context hash audit: PASS for both decoder streams
- Test population: fixed official `gen_train234_test2to10` test split
- Added rows: 20 per relation in both arms

At the largest base context (80 examples per relation), replacing the 20
additional length-2/3 examples with 20 length-4 examples changes ordered-path
accuracy as follows:

| Decoder | Length 4 | Equal-length mean, lengths 5--10 |
|---|---:|---:|
| TabPFN-3 | +0.7576 | +0.0326 |
| Logistic regression | +0.4892 | -0.0007 |
| CatBoost | +0.3723 | +0.0099 |

The direct length-4 repair is large for all three decoders. Generalization to
still-unseen lengths 5--10 is small for the classical decoders; TabPFN is
positive in all three seeds at this endpoint, but its magnitude remains much
smaller than the direct repair.

The result rules out two explanations of the earlier same-test intervention:
neither a larger total context nor removal of length-2/3 rows creates the
length-4 effect. It remains fixed-population mechanism evidence and does not
identify coordinate support as the only causal mechanism.

`cpu/` and `tabpfn/` contain the complete paired summaries. `execution/`
contains the exact plans, registration records, and terminal status records.
Atomic metrics and predictions remain preserved in the archived source output
root and can be regenerated from the registered source commit.
