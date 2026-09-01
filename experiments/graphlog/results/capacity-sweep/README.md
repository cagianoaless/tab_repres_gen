# GraphLog CatBoost capacity sweep

This complete fixed-panel control contains all 57 worlds, both symbolic
encoders, depths 1--7, and seeds 101--103. No world is selected using its
evaluation outcome.

`capacity_curve.csv` is the compact endpoint. `paired_by_seed.csv` and
`paired_by_world.csv` retain the matched contrasts, while `cells.csv` retains all
2,394 cell-level results. The multiset-minus-absolute evaluation contrast widens
from -0.0719 at depth 1 to -0.1598 at depth 7 as both context accuracies approach
one.
