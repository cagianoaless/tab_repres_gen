# Singleton-aware ceiling audit

This metric-free audit regenerates the frozen controlled confirmation episodes
from:

- `property_discovery_cpu_confirmation.yaml`; and
- `property_discovery_encoder_ladder_cpu_confirmation.yaml`.

It does not read decoder predictions. The command was:

```bash
PYTHONPATH=src:. python analysis/audit_finite_population_ceilings.py \
  --config configs/studies/property_discovery_cpu_confirmation.yaml \
  --config configs/studies/property_discovery_encoder_ladder_cpu_confirmation.yaml \
  --output-dir results/controlled-properties/ceilings-singletons \
  --workers 1
```

The output contains 390 per-graph encoder audits across 39
property-by-encoder groups. `cross_graph_ceilings.json` independently pools
three generated and relabelled graph populations for each group.

## Main diagnostic

| Property-aligned interface | Within-graph accuracy ceiling | Singleton row mass | Pooled cross-graph ceiling | Pooled singleton mass |
|---|---:|---:|---:|---:|
| reciprocity evidence | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| typed-composition evidence | 0.9073 | 0.0665 | 0.7935 | 0.0187 |
| structural-role evidence | 0.9004 | 0.0000 | 0.8865 | 0.0000 |
| topology evidence | 0.9164 | 0.6553 | 0.8013 | 0.3733 |

The perfect reciprocity ceiling is entirely singleton-driven, including after
pooling independent graphs. It is therefore an oracle memorization ceiling and
does not demonstrate transferable sufficiency. Identifier and random-embedding
interfaces show the same pathology. The typed-composition and structural-role
interfaces retain repeated equivalence classes and their pooled ceilings are
more informative. Topology evidence is intermediate and must be interpreted
with its substantial singleton mass.

This audit changes the reading of the ceiling, not any predictive result:
perfect or near-perfect values must always be accompanied by singleton mass and
conditional entropy.
