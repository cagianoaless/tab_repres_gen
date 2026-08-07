# Native ULTRA-4G scores

This folder contains only native zero-shot ULTRA-4G filtered-ranking metrics from the completed
RFMLink 57-dataset evaluations. It excludes TabPFN, classical decoders, score blending, and
per-candidate score tensors.

## Files

- `native_ultra_q64.csv`: 64 relation-stratified queries per evaluated direction.
- `native_ultra_q512.csv`: 512 relation-stratified queries per evaluated direction.
- `manifest.json`: source protocols, source-summary hashes, exported-file hashes, completion
  counts, and the ULTRA checkpoint hash.

Both source protocols completed all 684 registered jobs and contain all 57 datasets.

## Metrics

Each CSV has one row per dataset. `primary_side` follows the registered official evaluation:

- `combined` for datasets evaluated with head-and-tail prediction;
- `tail` for datasets registered as tail-only.

`primary_mrr` is the corresponding official-direction MRR. The remaining columns report all
available head, tail, and combined values for MRR, Hits@1, Hits@3, Hits@10, mean rank, and query
count.

The native ULTRA values do not depend on the context seed or decoder. RFMLink stored the same
native baseline under four decoder arms; the export verifies exact agreement and retains one copy
per dataset.

## Interpretation

The q64 and q512 files evaluate different deterministic relation-stratified query subsets. They
are separate estimands and must not be averaged or treated as repeated measurements. Both use
complete all-entity filtered ranking with the frozen `ultra_4g.pth` checkpoint identified in the
manifest.
