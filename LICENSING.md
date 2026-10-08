# Code, data and model licensing

This repository does not currently establish a code license or an author-approved
copyright holder list. This audit therefore does not grant a new license. The
maintainers must choose and approve a code license before public distribution;
that license must not purport to relicense external data or model weights.

Existing checksummed source declarations are in
`data/manifests/external_assets.yaml`:

| Resource | Recorded terms / boundary |
|---|---|
| CLUTRR Hugging Face CSV packaging | CC-BY-NC-4.0 |
| GraphLog author mirror | Upstream repository recorded as MIT; verify data distribution terms separately |
| MovieLens | Follow the dataset's own README terms |
| SNAP email dataset | Follow SNAP terms and citation requirements |
| auto-kbc / facebookresearch kbc | Pinned source repositories; verify each dataset's terms |
| TabPFN and ULTRA checkpoints | Separate model/package terms; weights are not bundled |
| Frozen RFMLink archives | Not bundled; upstream dataset/model redistribution permissions must be checked |

Python dependency versions are pinned by `uv.lock`; their licenses remain their
own. Installed metadata inspected during this audit records BSD-style terms for
NumPy, pandas, SciPy and scikit-learn; MIT for PyYAML, plotnine, pytest and Ruff;
and Apache-2.0 for CatBoost. NetworkX's metadata does not populate a license
field; its bundled license must be consulted. Optional TabPFN/cluster packages
and model checkpoints were not installed or redistributed by this audit.
These metadata observations are not a redistribution clearance. No external
data or model weights are relicensed here.

Citation title: **Graph-to-Table Interfaces Control Relational Generalisation**.
The repository does not provide a verified publication author list, DOI or
proceedings record. A `CITATION.cff` with publication attribution is deferred
until maintainers provide these details; no authors are inferred from Git names.
