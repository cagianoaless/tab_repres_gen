# CLUTRR recurrent GRU reference outputs

The two byte-identical reference CSVs come from result commit
`7b3acb87da64571d35060da83f290e1bed3266cb`, originally under
`results/audits/acceptance_controls_v3/`, then
`results/cross-benchmark/mechanism-and-matrix/`.

| File | SHA-256 |
|---|---|
| `gru_by_context.csv` | `65a39c0af72be4cb38187b3964c348c32df1ee230f04d12681d1d3b3dcbff3ef` |
| `gru_by_length.csv` | `7e1ea39a98eb11c34b5dd56173b53e783056b32f2cb9d1da627a00482c763430` |

The supplied `configs/studies/clutrr_symbolic_cpu_recursive_gru_v1.yaml`
matches the paper configuration: budgets 5,10,20,40,80; seeds 101–103;
embedding 32, hidden size 64, one shared recurrent layer and context-only
stratified early stopping. There are 135 unique budget/seed/length records.
OOD accuracy averages lengths 4–10 within each seed, then the three seeds.

```bash
uv run python -m analysis.verify_historical_gru
```

The verifier reconstructs all five OOD means and largest-budget length-4/10
endpoints, and checks the reference hashes. It runs no model and writes no files.
The execution registration identifies source
`35ce37623d2f4bd1baee66b0ded837af63d8e17a`; its 15 complete metrics/prediction
pairs and plan were verified on the execution system. They are not bundled.
Plan-body and serialized-file identities are in
`reproduction/REFERENCE_EXECUTION.json`. Use the public-data procedure in
[REPRODUCE.md](../../../../REPRODUCE.md) to run the supplied config afresh.
