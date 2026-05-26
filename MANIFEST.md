# Archive Manifest

This manifest describes the public code archive for:

**Hyperbolic neural networks for automated classification of white blood cells and leukemia subtypes in peripheral blood smears**

## Source Code

- `src/wbc_classification/models/classic_cnn.py` - Euclidean ResNet-18 classifier.
- `src/wbc_classification/models/hyperbolic_cnn.py` - Hyperbolic prototype classifier.
- `src/wbc_classification/model_zoo.py` - Packaged model discovery and loading helpers.
- `data/dataloaders.py` - ImageFolder split loading, transforms, and deterministic DataLoaders.
- `utils/` - Reproducibility, metrics, and report generation helpers.

## CLI Entrypoints

- `scripts/make_grouped_splits.py` - Generate acquisition-aware grouped split indices.
- `scripts/cnn_fine_tuned.py` - Train/evaluate the Euclidean baseline.
- `scripts/hyperbolic_cnn_fine_tuned.py` - Train/evaluate the hyperbolic classifier.
- `scripts/verify_archive.py` - Validate public-release hygiene and packaged model checksums.

## Configurations

- `configs/cnn_grouped_single.json`
- `configs/cnn_grouped_grid.json`
- `configs/hyperbolic_grouped_single.json`
- `configs/hyperbolic_grouped_grid.json`

## Packaged Models

The archive includes two PyTorch checkpoints, both below GitHub's 100 MB per-file limit.

| Model | Path | Size | SHA256 |
| --- | --- | ---: | --- |
| Hyperbolic prototype classifier | `models/hyperbolic/best.pt` | 44 MB | `a20491d2b58525fd6f10af44692bc3d6a84283f6a48b853c69a3819b00b8b92b` |
| Euclidean CNN baseline | `models/euclid/best.pt` | 44 MB | `cb46451ceca531b574c348812d64a76946f51f630ba2d98fff7bc658a409012b` |

Each checkpoint folder also contains `metadata.json` and `test_classification_report.csv`.

## Excluded By Design

- Dataset images downloaded from Zenodo and any reconstructed source image folders.
- Generated training outputs and sweep directories.
- Local virtual environments, caches, logs, and notebook checkpoints.
- Exploratory notebooks and duplicate development scripts not needed for the reproducible CLI workflow.
