# Code, data and model terms

The repository does not yet specify a code license or citation metadata.
These require an author decision.

The bundled producer/adaptor source is project-owned source used by the
reproduction pipeline. ULTRA is an external dependency with its own MIT license;
its repository is not vendored. No model weights or raw datasets are bundled.
TabPFN package/model access is governed by Prior Labs' terms, including applicable
attribution requirements. Built with PriorLabs-TabPFN.

External dataset declarations are in `data/manifests/external_assets.yaml`:

| Resource | Terms / boundary |
|---|---|
| CLUTRR Hugging Face CSV packaging | Recorded CC-BY-NC-4.0 |
| GraphLog author mirror | Upstream repository recorded as MIT; data terms apply separately |
| MovieLens | Dataset README terms |
| SNAP email | SNAP terms and citation requirements |
| auto-kbc / facebookresearch kbc | Pinned sources and each dataset's terms |
| Public ULTRA KGs | Original dataset terms; ULTRA's code license does not relicense data |
| ULTRA and TabPFN checkpoints | Upstream model terms; weights remain external |

Python dependencies retain their own licenses. A future repository code license
must not relicense datasets, models or other dependencies.
