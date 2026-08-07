# Frozen `ultra_4g` Panel

The fixed panel contains 57 knowledge graphs grouped into 16 provenance families. Each directional query is evaluated by full filtered entity ranking; q64 refers to the relation-stratified sample of test triples, not to a 64-candidate ranking approximation.

Outputs include the nine-interface ladder, multiplicity and leave-one-family-out analyses, native-ranking comparisons, nominal logistic ranking correction, query-level evidence, and the 4,104-cell context-robustness matrix.

The two context constructions share positive rows and evaluation exactly. `relation_matched_filtered_uniform` draws relation-matched corruptions uniformly after filtering known positives. `relation_matched_role_degree` favours entities observed in the required relation role and graph partition, then selects candidates with endpoint degrees similar to the replaced true entity. The frozen `ultra_4g` score is not used to select these negatives.
