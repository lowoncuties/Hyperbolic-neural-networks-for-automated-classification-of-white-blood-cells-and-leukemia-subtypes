# Reproducibility Protocol

This document gives the canonical public workflow for reproducing the experiments.

## 1. Environment

Use Python 3.13.3 or newer. Install the package and dependencies from the repository root:

```bash
python3 -m pip install -e .
```

Runtime dependencies are declared in `pyproject.toml`. For GPU runs, install the PyTorch build that matches the local CUDA driver before installing this package if needed.

## 2. Dataset

Download the curated dataset from Zenodo:

https://doi.org/10.5281/zenodo.19126698

The dataset is a derived work based on public sources. Users must comply with the original licenses of the AML-Cytomorphology dataset, Raabin-WBC dataset, and multi-focus WBC dataset. Image data are not stored in this Git repository.

Arrange the downloaded data as an ImageFolder dataset:

```text
datasets/WBC_Our_dataset_extended/<class_name>/*.jpg
```

The expected 14 class folders are listed in the main `README.md`.

## 3. Grouped Splits

Generate acquisition-aware split indices:

```bash
python3 scripts/make_grouped_splits.py \
  --root datasets/WBC_Our_dataset_extended \
  --split-output-dir outputs \
  --persist-splits-dir splits_grouped_acq_v1 \
  --seed 42 \
  --train-frac 0.7 \
  --val-frac 0.15 \
  --test-frac 0.15 \
  --img-size 224 \
  --threshold 110
```

The generated `grouped_split_meta.json` records split sizes, class counts, group-kind counts, and a leakage check.

## 4. Single-Run Reproduction

Euclidean baseline:

```bash
python3 scripts/cnn_fine_tuned.py \
  --mode single \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/cnn_grouped_single.json \
  --run-name cnn_grouped_single
```

Hyperbolic classifier:

```bash
python3 scripts/hyperbolic_cnn_fine_tuned.py \
  --mode single \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/hyperbolic_grouped_single.json \
  --run-name hyperbolic_grouped_single
```

## 5. Sweep Reproduction

Euclidean sweep:

```bash
python3 scripts/cnn_fine_tuned.py \
  --mode sweep \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/cnn_grouped_single.json \
  --grid configs/cnn_grouped_grid.json \
  --runs-root outputs/runs/cnn_grouped \
  --results-csv outputs/results/cnn_grouped_summary.csv
```

Hyperbolic sweep:

```bash
python3 scripts/hyperbolic_cnn_fine_tuned.py \
  --mode sweep \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/hyperbolic_grouped_single.json \
  --grid configs/hyperbolic_grouped_grid.json \
  --runs-root outputs/runs/hyperbolic_grouped \
  --results-csv outputs/results/hyperbolic_grouped_summary.csv
```

## 6. Packaged Model Loading

The included checkpoints can be loaded without regenerating the dataset:

```python
from wbc_classification.model_zoo import load_model

model, checkpoint, metadata = load_model("hyperbolic")
```

Use `load_model("euclid")` for the Euclidean baseline.

## 7. Public Archive Validation

Before publishing the repository, run:

```bash
python3 scripts/verify_archive.py
```

This validates required files, JSON metadata, checkpoint checksums, file-size limits, and accidental hard-coded local paths.
