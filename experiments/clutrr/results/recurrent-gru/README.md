# Historical recurrent GRU: derived-result recovery

`gru_by_context.csv` and `gru_by_length.csv` are genuine committed historical
derived outputs, copied byte-for-byte from the historical study repository.
Result commit: `7b3acb87da64571d35060da83f290e1bed3266cb`; original commit paths
are `results/audits/acceptance_controls_v3/`, subsequently moved to
`results/cross-benchmark/mechanism-and-matrix/`. Both committed copies were
checked against the recovered files before integration.

| File | SHA-256 |
|---|---|
| `gru_by_context.csv` | `65a39c0af72be4cb38187b3964c348c32df1ee230f04d12681d1d3b3dcbff3ef` |
| `gru_by_length.csv` | `7e1ea39a98eb11c34b5dd56173b53e783056b32f2cb9d1da627a00482c763430` |

The current `configs/studies/clutrr_symbolic_cpu_recursive_gru_v1.yaml` is
byte-identical to the recovered historical configuration: budgets 5, 10, 20,
40, 80; seeds 101, 102, 103; embedding 32, hidden size 64, one recurrent layer,
shared parameters across positions, context-only stratified early stopping.
The length table has 135 unique budget/seed/length records (lengths 2--10).
OOD accuracy averages lengths 4--10 within each seed, then averages the three
seeds. The context table also stores seed minima/maxima, which are checked.

```bash
uv run python -m analysis.verify_historical_gru
```

This reconstructs all five OOD means and the largest-budget length-4/length-10
endpoints from the length table, with 1e-15 tolerance for floating-point
aggregation. It runs no models and writes no files. Both CSV hashes are immutable
in `tools/historical_results.json` as well as the release manifest.

The original 15 cell-level metric files, prediction NPZs, execution plan and
runtime provenance were **not recovered**. No execution-source SHA is inferred
from the result commit. These tables support historical derived-endpoint
verification, not full raw-run reconstruction or prediction-level validation.
A new run using the public config must be labelled a fresh reproduction.
