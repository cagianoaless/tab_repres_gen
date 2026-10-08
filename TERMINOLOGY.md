# Terminology and compatibility

The complete graph-to-table interface is `(phi, S, D)`: an encoder phi, a labelled
context constructor S, and a decoder D. A feature schema specifies an encoder
variant; it does not specify the complete interface.

Legacy KG configurations use `interface` for this schema. The field appears
in cell IDs, immutable plan hashes, result columns, and cache keys. It remains
unchanged for compatibility. New Python code can use `encode_feature_schema` and
`FROZEN_KG_ENCODER_VARIANTS`; `_interface` and `FROZEN_KG_INTERFACES` remain aliases.

Local = 18 B0 columns; score = the frozen ULTRA score (B0 column 18);
Global = 13 B2v2 columns; Full = Local + score + Global = 32 columns.
`b0_local_no_relation_id` removes only B0 column 0 (17 columns).
`b0_local_ultra_no_relation_id` retains the score (18 columns).
The remaining nominal feature at original column 16 moves to column 15.
Precisely, the no-ID slices are original B0 `1:18` and `1:19` respectively;
categorical indices change from `(0,16)` to `(15,)`. Paper relation-identity
effects report **with identity minus without identity**, not the reverse.

The relation-context correction uses one 60-contrast family, distinct
from the context-robustness 30-test family. Fixed negative sampling
means a fixed policy; exact negative rows are resampled for matched positives.
The prospective positive-context utility instead fixes exact rows and offers
12+12/24-test sensitivity scopes, not the paper's reproduction procedure.

CLUTRR and GraphLog context encoding uses `encoder_seed`; evaluation encoding
uses `encoder_seed + 1`: deterministic, separate paired permutation streams.
Permutations are within path length; a block permutation moves entire positional
rows and retains the multiset block where specified. This behavior is intentional.

`context_fit_metrics` reports resubstitution predictions after fitting on the
same labelled context, including for TabPFN. The query row's label is still in
the conditioning context. It is a fit diagnostic, not leave-one-out accuracy
or evidence of generalisation. Within-query normalization changes the pointwise
input representation; it does not introduce a learning-to-rank loss.

Cross-fit versus full-support representations need not satisfy strict
covariate shift with an unchanged conditional label distribution. Prefix
interventions support bounded diagnostic interpretations, not a causal rule
inferred from context fit alone. The supported-prefix reference summary checks
aggregate accuracy invariance, not a maximum individual probability change.
Separate diagnostic and primary multiset outputs use their own reference
tables and are not interchangeable endpoints.
