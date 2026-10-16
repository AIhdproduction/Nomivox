# Third-party notices

Nomivox - E57 SAM 3 Point Cloud Classifier is licensed under the
AIhdproduction Education and Research License 1.1 in [LICENSE](LICENSE). That
license covers all code, scripts, presets and documentation in this repository.
The components below are not part of the repository; they are downloaded during
setup or obtained by the user and keep their own terms.

This repository does not include SAM 3 code, model weights or an installed
Python environment. The package versions and the SAM 3 source revision that
`install.bat` and `start_classifier.bat` install are pinned in
[requirements.lock](requirements.lock).

## SAM 3 by Meta

Nomivox uses SAM 3 for text-prompted image segmentation. We acknowledge the
SAM Materials provided by Meta.

- Source: https://github.com/facebookresearch/sam3
- Installed source revision: `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`.
- Upstream license at that revision:
  https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE
- Unmodified license copy from the installed pinned distribution:
  [docs/LICENSE_SAM3.txt](docs/LICENSE_SAM3.txt).
- Official model access: https://huggingface.co/facebook/sam3

The installer obtains the SAM 3 source directly from the upstream repository.
Users download the checkpoint themselves into `models/` and comply with the
applicable model access and license terms. The Nomivox license does not
replace or relax Meta's SAM License, and the SAM License does not lift the
purpose restrictions on Nomivox itself.

The SAM License has its own conditions. Among others, research publications
that use the SAM Materials must acknowledge them, and the license contains
trade-control and prohibited-use restrictions. Read the full text before using
the model.

## Installed Python packages

The installer downloads the following principal packages from their upstream
sources; none of them is redistributed in this repository. The license
identifiers are those declared by the pinned distributions.

| Component | Used for | Declared license |
|---|---|---|
| PyTorch, torchvision | GPU inference | BSD-3-Clause |
| PySide6, shiboken6 (Qt for Python) | Desktop interface | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| pye57 (bundles libE57Format and Xerces-C++) | Reading E57 files | MIT; bundled libraries under their own terms |
| laspy | Writing LAS/LAZ | BSD-2-Clause |
| lazrs | LAZ compression | MIT |
| NumPy, SciPy | Numerical processing | BSD-3-Clause |
| Pillow | Image decoding | MIT-CMU |
| psutil | Resource monitoring | BSD-3-Clause |

SAM 3 pulls in further packages such as timm and huggingface_hub (Apache-2.0).
Their own license texts and bundled notices remain applicable. Installed
distribution metadata and accompanying license files are available in
`.cache/venv/Lib/site-packages/` after setup.

This notice identifies the principal dependencies; it does not substitute for
the license texts of every transitive dependency. Packaging an installed
environment or executable requires retaining the applicable component notices
for that distribution as well.

## Provenance of the SAM adapter

`e57cls/seg/backend.py` was developed from the SAM 3 backend of an earlier
classifier by Domenic Hochstrasser, doing business as AIhdproduction. It is original AIhdproduction
material, not third-party code, and is covered by [LICENSE](LICENSE) like the
rest of this repository. The module-by-module reuse assessment is in
[docs/reuse_map.md](docs/reuse_map.md). Agisoft Metashape is not a dependency of
Nomivox.
