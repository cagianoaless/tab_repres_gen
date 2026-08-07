# GraphLog

All 57 official worlds use one global vocabulary of 20 directed relation symbols. The shared row has 21 columns: path length plus 20 relation counts normalized by the maximum path length 5. The absolute row has 101 columns: path length plus five 20-column positional blocks. Neither representation contains entity identifiers or the target label.

Contexts are nested, stratified by present target class, and sampled without replacement from each world's internal training split. Evaluation uses the complete length-4--5 test population.

