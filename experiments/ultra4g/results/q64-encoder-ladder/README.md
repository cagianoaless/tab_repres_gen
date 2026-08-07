# RFMLink q64 complete-panel findings

Status: **POST-HOC COMPLETE-PANEL ANALYSIS OF A FROZEN SWEEP**.

This analysis treats all 57 datasets in the immutable execution plan as the
fixed completed panel. The primary sensitivity gives equal weight to each of
16 metric-free provenance families;
equal-dataset summaries are secondary. Intervals are fixed-panel
source-family bootstrap sensitivity intervals, not population confidence
intervals.

## Audited design

- 57 datasets, 16 provenance families, and 6,156 complete TabPFN cells;
- nine frozen interfaces, four nested contexts per class, and three seeds;
- relation-stratified q64 filtered ranking in both head and tail directions;
- native zero-shot ULTRA computed on the same candidate rows;
- source result commit: `artifact-source-revision`;
- source `cell_metrics.csv` SHA256: `c7f314221f1b4482ebdd6d91815ae5da8757f9912d11e6a0f4eed9a21b91ce59`;
- explicit practical tie band for descriptive wins/ties/losses: +/-0.005 MRR.

## Findings

1. **Stable relational coordinates matter more than opaque entity tokens.**
   At 200 examples per class, B0 local features exceed token-only rows by
   +0.2336 [+0.1550, +0.3465] MRR; all 16 source families have a positive raw effect.

2. **Context is an extraction resource, not merely extra rows.** From 20 to
   200 examples per class, equal-family MRR changes by
   +0.0011 for opaque entity tokens,
   +0.0704 for the 18 local relational
   coordinates, and +0.0542 for the full
   32-column interface. The richer reusable interfaces benefit much more from
   context.

3. **The frozen KGFM score is the dominant broadly reusable signal.** Adding
   ULTRA to local+global structure yields +0.1078 [+0.0814, +0.1332] MRR at the largest
   context and is positive in all 16 families. This is an interface analysis,
   not evidence that the handcrafted table replaces the KGFM.

4. **The 13 global descriptors have a context- and direction-dependent
   incremental effect.** Conditional on ULTRA they are harmful at 20 examples
   per class (-0.0101 [-0.0210, +0.0001] combined MRR). At 200 examples per class the
   tail effect is +0.0106 [+0.0040, +0.0177], whereas the head effect is
   -0.0010 [-0.0083, +0.0056]. The combined endpoint remains small and its interval
   includes zero. Its equal-dataset estimate is
   +0.0109; provenance weighting therefore
   materially attenuates the apparent gain.

5. **Global descriptors complement the score more than they work alone.**
   The difference-in-differences between their effect with and without ULTRA is
   +0.0087 [+0.0026, +0.0149] at 200 examples per class. This post-hoc mechanism
   result is consistent with score conditioning/calibration, not standalone
   graph reasoning.

6. **The global-feature effect is regime-heterogeneous.** At 200 examples per
   class it is +0.0157 [+0.0059, +0.0265] in
   entity+relation-inductive graphs, +0.0075 [+0.0027, +0.0128] in
   entity-inductive graphs, and -0.0011 [-0.0079, +0.0055] in transductive
   graphs. These are descriptive fixed-panel strata, not a causal regime
   interaction.

7. **The direct table readout does not generally beat native ULTRA.** At 200
   examples per class, the equal-family full-interface contrast is
   +0.0064 [-0.0531, +0.1084], but it is positive on only
   2/16 families and
   5/57 datasets outside the practical tie band.
   The positive equal-family mean is driven by MetaFam; excluding that family
   gives -0.0414 MRR. Native ULTRA remains the appropriate
   system baseline.

8. **Sparse handcrafted interfaces expose a concrete coverage failure.**
   194 cells use the registered constant-input fallback. These
   occur only in dyadic/path interfaces and concentrate in entity+relation
   inductive graphs; their frequency falls sharply as context grows.

## Reading rule

The robust paper-level evidence is the representation--context pattern:
reusable relational coordinates and the frozen score support extraction, while
opaque tokens, constant interfaces, and insufficient context do not. The tail
specificity and B2v2-by-ULTRA interaction are mechanism evidence. The study does
not establish universal TabPFN superiority, native-ULTRA improvement, or a
population-level law over knowledge graphs.

## Files

- `audit.json`: frozen panel and lineage checks;
- `dataset_metrics.csv`: seed-averaged dataset cells;
- `absolute_mrr_equal_family.csv` and `absolute_mrr_equal_dataset.csv`;
- `contrast_summary.csv`: all registered-style post-hoc contrasts;
- `contrast_by_dataset.csv`, `contrast_by_family.csv`, and
  `contrast_by_regime.csv`;
- `context_scaling_summary.csv` and `context_scaling_by_family.csv`;
- `constant_fallback_summary.csv`;
- `native_ultra_sensitivity.csv`;
- `finding_summary.json`: compact machine-readable headline values.
