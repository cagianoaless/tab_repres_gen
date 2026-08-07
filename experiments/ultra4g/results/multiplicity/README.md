# RFMLink q64 multiplicity sensitivity

Status: **POST-HOC FIXED-PANEL SENSITIVITY**.

This audit evaluates the eight selected frozen-ULTRA panel contrasts together.
The unit is the metric-free provenance family. Every exact sign pattern applies
the same family sign to all contrasts, preserving their observed cross-contrast
dependence. Statistics are studentized family means.

The output reports:

- an exact two-sided family sign-flip p-value for each contrast;
- Holm adjustment across the eight reported contrasts;
- a joint max-|t| adjustment from the shared exact sign-flip distribution.
- every leave-one-provenance-family-out estimate and its range.

The sign-flip analysis assumes symmetry of the family-level effects under the
zero null. It is a simultaneous sensitivity for this fixed panel, not uncertainty
over a population of graph families and not a replacement for the unadjusted
effect estimates and fixed-panel bootstrap intervals.
The eight contrasts were selected post hoc as the reported scientific family.
The max-|t| adjustment controls this family only, not every contrast that could
have been computed after inspecting the panel.

## Contrasts surviving joint maxT at 0.05

- `local_relational_vs_entity_tokens_c200_combined`: effect +0.2336, joint maxT p=0.00226
- `ultra_score_given_global_structure_c200_combined`: effect +0.1078, joint maxT p=0.00003
- `global_structure_given_ultra_c200_tail`: effect +0.0106, joint maxT p=0.04434

The direction-specific global-feature result and the global-by-score interaction
do not survive the eight-contrast simultaneous sensitivity and must remain
exploratory.

## Leave-one-family-out

- `local_relational_vs_entity_tokens_c200_combined`: full +0.2336; leave-one-family-out range [+0.1855, +0.2447]; same sign throughout: true.
- `ultra_score_given_global_structure_c200_combined`: full +0.1078; leave-one-family-out range [+0.1027, +0.1150]; same sign throughout: true.
- `global_structure_given_ultra_c20_combined`: full -0.0101; leave-one-family-out range [-0.0127, -0.0073]; same sign throughout: true.
- `global_structure_given_ultra_c200_combined`: full +0.0048; leave-one-family-out range [+0.0036, +0.0066]; same sign throughout: true.
- `global_structure_given_ultra_c200_head`: full -0.0010; leave-one-family-out range [-0.0030, +0.0014]; same sign throughout: false.
- `global_structure_given_ultra_c200_tail`: full +0.0106; leave-one-family-out range [+0.0085, +0.0119]; same sign throughout: true.
- `global_structure_x_ultra_score_c200_combined`: full +0.0087; leave-one-family-out range [+0.0072, +0.0102]; same sign throughout: true.
- `full_interface_vs_native_ultra_c200_combined`: full +0.0064; leave-one-family-out range [-0.0414, +0.0146]; same sign throughout: false.

## Provenance

- source table: `results/ultra57/q64-encoder-ladder/contrast_by_family.csv`
- source SHA256: `3ce32ed9b54bc9debb3077a6388923f94cfdc5c6a1e2fcdcf8cb220469787796`
- source study: `rfmlink_ultra57_q64_complete57_v1`
- contrast count: 8
- family count: 16
- exact joint sign patterns: 65,536
