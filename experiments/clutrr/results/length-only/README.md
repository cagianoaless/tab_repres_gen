# CLUTRR length-only TabPFN audit

The audit consumes the 75-cell immutable result branch
`results/clutrr_symbolic_gpu_tabpfn_confirmation_v2_cluster` and selects all
15 `path_length_only` cells.

It passes checksum, probability normalization, canonical class-order,
all-class context coverage, and exact metric-recomputation checks. At 80
context examples per relation, the three-seed mean accuracy is
`0.010762070971495057`, below the fixed-test majority accuracy
`0.13699825479930192` and uniform-random expectation `1/18`.

Reconstruction from the checksummed official assets adds a representation-level
diagnosis. Every length-only row at a fixed chain length is identical, while
the conditional label distribution changes sharply between the training
context and test population even at the nominally observed lengths 2 and 3.
At the largest context, both the transferred context-mode classifier and
TabPFN obtain zero mean accuracy on these two lengths. TabPFN assigns an
identical probability vector to every row of the same length, as the interface
requires. The mean total-variation distance between the context and test
conditional label distributions is `0.8102`.

The endpoint is therefore compatible with an insufficient scalar
representation under conditional label shift, rather than evidence by itself
of a class-index or metric fault. A fresh three-cell GPU replay remains the
independent software-level check and is frozen in
`configs/studies/clutrr_symbolic_gpu_tabpfn_length_only_replay_v1.yaml`.
