# D8 Composition

The generated order-16 dihedral-group task tests chain lengths 4--10 over ten
independent populations. The primary context contains 400 examples per class.

| Decoder | Shared lossy - absolute | 95% interval | Permuted - aligned block | 95% interval |
|---|---:|---:|---:|---:|
| Logistic | -0.0004 | [-0.0028, 0.0020] | +0.0011 | [-0.0011, 0.0035] |
| CatBoost | +0.0126 | [0.0106, 0.0147] | -0.0019 | [-0.0043, 0.0003] |
| TabPFN | -0.0011 | [-0.0042, 0.0023] | +0.0085 | [0.0024, 0.0147] |

The broad CLUTRR shared-coordinate advantage does not reproduce. The D8
multiset is genuinely lossy (`ceiling = 0.796`), while ordered and transition
interfaces are nearly or fully distinguishing.

## Results

- [`cpu/aggregate.csv`](cpu/aggregate.csv): logistic and CatBoost curves.
- [`cpu/primary_seed_contrasts.csv`](cpu/primary_seed_contrasts.csv): paired
  population-level contrasts.
- [`tabpfn/aggregate.csv`](tabpfn/aggregate.csv): TabPFN curves.
- [`tabpfn/primary_seed_contrasts.csv`](tabpfn/primary_seed_contrasts.csv):
  paired TabPFN contrasts.
- [`cpu/ceiling_summary.csv`](cpu/ceiling_summary.csv): representation
  ceilings and collision diagnostics.
