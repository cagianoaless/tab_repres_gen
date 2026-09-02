# GraphLog CatBoost capacity sweep

The fixed panel contains 57 worlds, two symbolic encoders, CatBoost depths 1--7,
and seeds 101--103. `cells.csv` contains all registered cells,
`paired_by_seed.csv` and `paired_by_world.csv` contain matched contrasts, and
`capacity_curve.csv` contains the depth-level summary. Evaluation results use
equal-length macro balanced accuracy over the official inner-test lengths 4--5,
which is the primary GraphLog endpoint used in the paper.
