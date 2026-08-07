# CLUTRR coordinate-support audit

This label-free audit was executed at source commit
`artifact-source-revision` against the immutable
`gen_train23_test2to10` assets. Its SHA-256 is
`7e69e258c5074b4e7ef3d94a2c748fbe8970f592aabf062015ce9d48d4e414c0`.

At 80 context examples per relation, active-cell coverage is:

| Encoder | Coverage | Fully supported evaluation rows |
|---|---:|---:|
| Length only | 1.000 | 1.000 |
| Relation multiset | 1.000 | 1.000 |
| Absolute ordered path | 0.549 | 0.125 |
| Relation-transition counts | 0.977 | 0.878 |
| Ordered path plus multiset | 0.719 | 0.125 |

For the absolute ordered path, active-cell coverage falls deterministically
from 1.000 at lengths 2--3 to 0.800 at length 4 and 0.364 at length 10.
Transition-count coverage remains between 0.977 and 0.987 at OOD lengths
4--10. These numbers use no labels or decoder outputs. They measure
coordinate support only, not relational sufficiency or learnability.
