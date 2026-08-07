# CLUTRR shortcut diagnostic

This post-hoc diagnostic tests a specific account of the CLUTRR reversal at the
largest registered context (80 examples per target relation).

- `cells.csv` records in-context fit and evaluation accuracy for the multiset,
  absolute ordered path, their concatenation, and the concatenation with only
  context-supported absolute positions retained.
- `aggregate.csv` averages the three paired context seeds.
- `representation_audit.csv` reports the exact finite-population ceiling and
  singleton mass for the multiset and supported-prefix concatenation.
- `matched_length_4_5.csv` restricts the original CLUTRR comparison to the same
  evaluation lengths used by GraphLog.
- `dihedral16_matched_context.csv` reports the independent dihedral-composition
  contrasts at 20 and 50 examples per class.

The absolute-prefix interfaces fit the length-2/3 context nearly perfectly but
transfer worse than the coarser multiset. Because the supported-prefix
concatenation has the higher ceiling, this is an extraction reversal rather
than information loss. It is consistent with a transfer-unstable prefix
shortcut, but remains a post-hoc mechanism diagnosis on one CLUTRR population.
