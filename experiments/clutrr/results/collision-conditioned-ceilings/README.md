# Collision-conditioned CLUTRR ceilings

`summary.json` decomposes each encoder's empirical ceiling into singleton and
non-singleton evaluation mass. The non-singleton ceiling is computed only from
feature rows shared by at least two queries. The `ood_equal_length` endpoint
equally weights official test lengths 4--10.

The relation multiset endpoint has singleton mass 0.7567, non-singleton mass
0.2433, and a conditional non-singleton ceiling of 0.7116. Absolute position has
singleton mass 1.0 on the same endpoint, so no non-singleton conditional ceiling
exists.
