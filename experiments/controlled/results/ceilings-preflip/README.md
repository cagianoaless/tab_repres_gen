# Pre-flip finite-population ceiling audit

This metric-free audit regenerates the frozen controlled confirmation episodes
and computes every equivalence-class statistic twice:

1. with the observed labels after the registered deterministic 10% symmetric
   flip; and
2. with the latent labels before that flip, reconstructed from the frozen
   positive/negative query-ID partition.

It does not read decoder predictions. The command was:

```bash
PYTHONPATH=src:. python analysis/audit_finite_population_ceilings.py \
  --config configs/studies/property_discovery_cpu_confirmation.yaml \
  --config configs/studies/property_discovery_encoder_ladder_cpu_confirmation.yaml \
  --output-dir results/controlled-properties/ceilings-preflip \
  --workers 1
```

The audit covers 390 graph-level encoder populations in 39
property-by-encoder groups.

## Paper-facing result

| Interface / property | Observed-label ceiling | Latent-label ceiling | Singleton mass | Latent collision mass |
|---|---:|---:|---:|---:|
| Relation only / reciprocity | 0.5189 | 0.5199 | 0.0000 | 1.0000 |
| Reverse edge / reciprocity | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Typed paths / composition | 0.9073 | 1.0000 | 0.0665 | 0.0000 |
| Local roles / structural role | 0.9004 | 1.0000 | 0.0000 | 0.0000 |
| Block affinity / community | 0.9164 | 0.9239 | 0.6553 | 0.1908 |

For typed composition and structural roles, the observed ceiling is determined
almost entirely by the registered label flip; their latent representations are
collision-free on the finite evaluation populations. The perfect reciprocity
ceiling is entirely singleton-driven and therefore only permits lookup. The
relation-only control and block-affinity interface exhibit genuine latent-label
collisions.

The expression `singleton_mass + (1 - singleton_mass) * 0.9` reproduces the
observed ceiling within 0.0006 for typed composition and structural roles. It is
not exact in general because the deterministic flips need not be distributed
uniformly inside every representation class.

## Integrity

- `summary.json` SHA-256:
  `8c712171f1273e962475d38c5ff25dccb33c4a4b91910584b1b1ae3f83b8c013`
- `per_graph.csv` SHA-256:
  `7caa5fc365782cab2130391a47f8986ff41e12d89ac2d4b72ac28295aa973574`
- `equivalence_class_histograms.jsonl` SHA-256:
  `844ad1b2e95eae5b542583789c9ea8764e4e9bc6c19cde669be4709fd349ef75`
- `cross_graph_ceilings.json` SHA-256:
  `3ff5db84f95d1ea9a60ea53fe3423d33d62c73c5b33df7fa9adf412bcf118ba4`
