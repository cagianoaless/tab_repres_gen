# CLUTRR

CLUTRR supplies symbolic relation paths of length 2--3 in context and evaluates
lengths 4--10. The experiment compares absolute-position coordinates with
coordinates shared across path positions.

## Encoder Ordering

Largest-context accuracy on the fixed OOD population:

| Decoder | Ordered path | Relation multiset | Multiset advantage |
|---|---:|---:|---:|
| Logistic | 0.157 | 0.430 | +0.273 |
| CatBoost | 0.109 | 0.273 | +0.163 |
| TabPFN | 0.103 | 0.289 | +0.186 |

The multiset advantage has the same direction for balanced accuracy, macro-F1,
top-3 accuracy, log loss, and Brier score.

## Ordered-Block Interference

Adding the aligned absolute-position block to the multiset reduces accuracy.
Permuting that block preserves its dimension and marginal structure while
recovering most of the loss.

| Decoder | Multiset | Aligned ordered block | Permuted ordered block | Recovery |
|---|---:|---:|---:|---:|
| Logistic | 0.430 | 0.266 | 0.400 | +0.133 |
| CatBoost | 0.273 | 0.163 | 0.255 | +0.093 |
| TabPFN | 0.289 | 0.187 | 0.292 | +0.104 |

Adding length-4 context repairs length 4 much more than still-unseen lengths
5--10: `+0.489/-0.001` for logistic, `+0.372/+0.010` for CatBoost,
and `+0.758/+0.033` for TabPFN.

## Results

- [`metric-robustness/`](metric-robustness): metrics by decoder, length, and
  target relation.
- [`ordered-interference/`](ordered-interference): aligned/permuted block cells.
- [`additive-context/`](additive-context): length-4 exposure intervention.
- [`supported-prefix/`](supported-prefix): supported-prefix control.
- [`shortcut-diagnostic/`](shortcut-diagnostic): context-fit diagnostics.
- [`shared-coordinates/`](shared-coordinates): position-shared interfaces.
- [`coordinate-coverage/`](coordinate-coverage): active-coordinate audit.
- [`length-only/`](length-only): length-only conditional-shift diagnostic.
