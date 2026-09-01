# CLUTRR

Contexts are sampled without replacement from the official training split, stratified by the 18 target relations and nested across budgets. Context paths have lengths 2--3; evaluation uses the official test paths of lengths 4--10. The path-edge vocabulary has 14 directed relation symbols.

Results cover relation multisets, absolute-position paths, complete-block within-length permutations, supported-prefix controls, directed transition counts, metric robustness, and additive length-4 exposure.

`results/collision-conditioned-ceilings/` separates singleton mass from the
Bayes ceiling on non-singleton rows. On the equally weighted length-4--10
evaluation population, the relation multiset has singleton mass 0.757 and
non-singleton mass 0.243. Its ceiling restricted to the non-singleton mass is
0.712. Absolute position is injective on this population, so its conditional
non-singleton ceiling is undefined rather than reported as an informative value.
