# Terminology and compatibility

The complete graph-to-table interface is `(phi, S, D)`: an encoder phi, a labelled
context constructor S, and a decoder D. A feature schema specifies an encoder
variant; it does not specify the complete interface.

Historical KG configurations use `interface` for this schema. The field appears
in cell IDs, immutable plan hashes, result columns, and cache keys. It remains
unchanged for compatibility. New Python code can use `encode_feature_schema` and
`FROZEN_KG_ENCODER_VARIANTS`; `_interface` and `FROZEN_KG_INTERFACES` remain aliases.

Local = 18 B0 columns; score = the frozen ULTRA score (B0 column 18);
Global = 13 B2v2 columns; Full = Local + score + Global = 32 columns.
`b0_local_no_relation_id` removes only B0 column 0 (17 columns).
`b0_local_ultra_no_relation_id` retains the score (18 columns).
The remaining nominal feature at original column 16 moves to column 15.

CLUTRR and GraphLog context encoding uses `encoder_seed`; evaluation encoding
uses `encoder_seed + 1`: deterministic, separate paired permutation streams.
Permutations are within path length; a block permutation moves entire positional
rows and retains the multiset block where specified. This behavior is intentional.

`context_fit_metrics` reports resubstitution predictions after fitting on the
same labelled context, including for TabPFN. The query row's label is still in
the conditioning context. It is a fit diagnostic, not leave-one-out accuracy
or evidence of generalisation. Within-query normalization changes the pointwise
input representation; it does not introduce a learning-to-rank loss.
