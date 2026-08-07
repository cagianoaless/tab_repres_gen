# Controlled Properties

The controlled panel measures whether row equivalence classes preserve the
latent label distinction. Ceilings are finite-population quantities; singleton
mass identifies ceilings driven by lookup.

| Representation / property | Latent ceiling | Singleton mass | Mixed-label collision mass |
|---|---:|---:|---:|
| Relation only / reciprocity | 0.520 | 0.000 | 1.000 |
| Reverse edge / reciprocity | 1.000 | 1.000 | 0.000 |
| Typed paths / composition | 1.000 | 0.067 | 0.000 |
| Local roles / structural role | 1.000 | 0.000 | 0.000 |
| Block affinity / community | 0.924 | 0.655 | 0.191 |

Matched representations expose the generating statistic. Off-diagonal failures
show that dimension or injectivity alone does not preserve the required
distinction.

## Results

- [`property-discovery/encoder-ladder.csv`](property-discovery/encoder-ladder.csv):
  full encoder-by-property metrics.
- [`property-discovery/cells.csv`](property-discovery/cells.csv): cell metrics.
- [`ceilings-preflip/`](ceilings-preflip): latent and flipped-label ceilings.
- [`ceilings-singletons/`](ceilings-singletons): equivalence-class diagnostics.
- [`alignment-null/`](alignment-null): aligned and permutation-null scores.
