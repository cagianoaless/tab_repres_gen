# Frozen `ultra_4g` Panel

The fixed panel contains 57 knowledge graphs grouped into 16 provenance families. Each directional query is evaluated by full filtered entity ranking; q64 refers to the relation-stratified sample of test triples, not to a 64-candidate ranking approximation.

Outputs include the nine-interface ladder, multiplicity and leave-one-family-out analyses, native-ranking comparisons, nominal logistic ranking correction, query-level evidence, and the 4,104-cell context-robustness matrix.

The two context constructions share positive rows and evaluation exactly. `relation_matched_filtered_uniform` draws relation-matched corruptions uniformly after filtering known positives. `relation_matched_role_degree` favours entities observed in the required relation role and graph partition, then selects candidates with endpoint degrees similar to the replaced true entity. The frozen `ultra_4g` score is not used to select these negatives.

`results/pointwise-and-filtering-controls/` contains three further checks. The
query-relative arms apply within-query z-score or percentile transforms using the
same label-free transformation for context and evaluation. The score-only
logistic baseline remains rank preserving, while more flexible pointwise
decoders generally do not recover native ranking. The matched filtering arm
shows that filtering changes local-only TabPFN MRR by +0.008 under equal-family
weighting, with a fixed-panel sensitivity interval of [0.002, 0.015], while the
effects after adding the frozen score are much smaller.

`results/aggregation-sensitivity/` reports both estimands for the full readout
against native ranking. The contrast is +0.006 with equal provenance-family
weighting and -0.035 with equal dataset weighting. At a 0.005 practical tie
band, the dataset panel contains 5 wins, 4 ties, and 48 losses. These are
fixed-panel sensitivities, not population intervals.
