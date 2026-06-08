# Hyperbolic neural networks for biologically informed classification of white blood cells and hematopoietic cell states in peripheral blood smears

This archive contains source code, configuration files, and reproducibility utilities associated with the manuscript:

Jochymek L., Vašinková M., Gajdoš P.  
**Hyperbolic neural networks for automated classification of white blood cells and leukemia subtypes in peripheral blood smears.**  
(2026)

This package is intended to make the training and evaluation pipeline transparent and reproducible.

---

## 1. Overview

The goal of this study is to compare Euclidean and hyperbolic neural classifiers for leukocyte subtype recognition from peripheral blood smear images.

The repository provides:

- a ResNet-18 Euclidean CNN baseline
- a ResNet-18 backbone with a hyperbolic prototype classification head
- optional learnable curvature and temperature in the hyperbolic head
- acquisition-aware grouped split generation
- CLI entrypoints for single runs and hyperparameter sweeps
- run artifacts including checkpoints, metrics, confusion matrices, classification reports, and resolved run configuration files

The main experimental protocol uses a grouped 70/15/15 train/validation/test split so linked multi-focus acquisitions or recoverable repeated image variants are assigned to the same partition.

---

## 2. Dataset

The study uses a curated ImageFolder-style dataset assembled from public white blood cell image resources described in the manuscript:

- AML-Cytomorphology dataset
- Raabin-WBC dataset
- multi-focus WBC dataset

The harmonized benchmark contains 14 leukocyte categories and 129,174 annotated single-cell images. The code expects one directory per class.

**IMPORTANT:**  
The curated dataset supporting the manuscript is publicly available at Zenodo:

https://doi.org/10.5281/zenodo.19126698

The dataset is a derived work based on public sources. Users must comply with the original licenses of the AML-Cytomorphology dataset, Raabin-WBC dataset, and multi-focus WBC dataset. Image data are not stored in this Git repository.

### Expected dataset layout

Place the merged dataset at the default path:

```text
datasets/WBC_Our_dataset_extended/
  Basophil/
  Blast/
  Eosinophil/
  Erythroblast/
  Immature_wbc/
  Lymphocyte/
  Lymphocyte_atypical/
  Metamyelocyte/
  Monocyte/
  Myeloblast/
  Myelocyte/
  Neutrophil/
  Neutrophil_band/
  Promyelocyte/
```

Alternatively, set `WBC_DATA_ROOT` or pass `--data-root`.

---

## 3. System Requirements

The code requires Python 3.13.3 or newer. Experiments were designed for PyTorch with CUDA-capable GPUs.

Core dependencies:

- PyTorch
- TorchVision
- Geoopt
- NumPy
- scikit-learn
- Pillow
- Matplotlib
- tqdm

Install dependencies with:

```bash
python3 -m pip install -e .
```

Runtime dependencies are declared in `pyproject.toml`. For GPU training, install the PyTorch build matching your CUDA driver before installing this package if you do not want pip to select the default wheel.

---

## 4. Repository Structure

- `scripts/make_grouped_splits.py`  
  Generates acquisition-aware grouped train/validation/test split JSON files.

- `scripts/hyperbolic_cnn_fine_tuned.py`  
  Official hyperbolic prototype classifier trainer and sweep entrypoint.

- `scripts/cnn_fine_tuned.py`  
  Official Euclidean CNN baseline trainer and sweep entrypoint.

- `data/dataloaders.py`  
  ImageFolder loading, transform construction, persisted split loading, label remapping, and deterministic DataLoader setup.

- `src/wbc_classification/models/`  
  Euclidean and hyperbolic model components.

- `src/wbc_classification/model_zoo.py`  
  Helper functions for discovering and loading the packaged trained models.

- `models/`  
  Packaged trained model checkpoints and metadata.

- `utils/`  
  Reproducibility, metrics, and reporting helpers.

- `configs/`  
  Reusable JSON configurations and sweep grids for grouped experiments.

- `docs/reproducibility.md`  
  Canonical public reproduction protocol.

- `docs/model_cards.md`  
  Intended-use and checkpoint summaries for the packaged trained models.

- `scripts/verify_archive.py`  
  Public-release hygiene, checksum, and metadata verifier.

- `MANIFEST.md`  
  Archive inventory and exclusion policy.

- `LICENSE`  
  Software license.

---

## 5. Reproducing the Experiments

### Step 1 - Obtain the dataset

Download the curated dataset from Zenodo and arrange it in the `ImageFolder` structure shown in Section 2. The split and training scripts can also be used with a locally reconstructed folder that follows the same class layout.

### Step 2 - Install dependencies

```bash
python3 -m pip install -e .
```

### Step 3 - Generate grouped splits

The paper protocol uses acquisition-aware grouped split files. Generate them with:

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

This writes:

```text
outputs/splits_grouped_acq_v1/WBC_Our_dataset_extended/
  seed_42_t0.7_v0.15_s224_balmin0_thrgt110/
    train_idx.json
    val_idx.json
    test_idx.json
    grouped_split_meta.json
```

The metadata file records class counts, grouping policy, split sizes, and a leakage check.

### Step 4 - Run the Euclidean baseline

```bash
python3 scripts/cnn_fine_tuned.py \
  --mode single \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/cnn_grouped_single.json \
  --run-name cnn_grouped_single
```

### Step 5 - Run the hyperbolic classifier

```bash
python3 scripts/hyperbolic_cnn_fine_tuned.py \
  --mode single \
  --data-root datasets/WBC_Our_dataset_extended \
  --config configs/hyperbolic_grouped_single.json \
  --run-name hyperbolic_grouped_single
```

### Step 6 - Run grouped sweeps

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
  --learnable-temperature \
  --lr-temperature 0.003 \
  --runs-root outputs/runs/hyperbolic_grouped \
  --results-csv outputs/results/hyperbolic_grouped_summary.csv
```

---

## 6. Packaged Trained Models

The archive includes the best grouped checkpoints selected by test macro-F1:

```text
models/
  hyperbolic/
    best.pt
    metadata.json
    test_classification_report.csv
  euclid/
    best.pt
    metadata.json
    test_classification_report.csv
```

The packaged models can be loaded with:

```python
from wbc_classification.model_zoo import load_model

model, checkpoint, metadata = load_model("hyperbolic")
```

Use `load_model("euclid")` for the Euclidean baseline.

---

## 7. Outputs and Artifacts

Each single run or sweep configuration writes artifacts under its run directory:

```text
outputs/runs/<model>/<run_name>/
  resolved_config.json
  metrics.csv
  best.pt
  last.pt
  test_confusion_matrix.png
  test_confusion_matrix_normalized.png
  test_classification_report.csv
  test_classification_report.png
```

Sweep and single-run summaries are appended to:

```text
outputs/results/*.csv
```

The resolved config file records the dataset path, run configuration, active classes, number of classes, and compute device.

---

## 8. Reproducibility Features

- Global seed control for `random`, `numpy`, and `torch`
- Deterministic cuDNN settings where available
- Deterministic DataLoader worker initialization
- Persisted train/validation/test split indices
- Acquisition-aware grouping for multi-focus stacks and recoverable repeated variants
- Dynamic class remapping from active split classes
- Run-resolved config export as JSON
- Per-epoch metrics CSV files
- Aggregate summary CSV files for sweeps
- Confusion matrix and classification report artifacts

### Methodology mapping to the manuscript

- **Data:** `scripts/make_grouped_splits.py`, `data/dataloaders.py`
- **Euclidean baseline:** `src/wbc_classification/models/classic_cnn.py`, `scripts/cnn_fine_tuned.py`
- **Hyperbolic classifier:** `src/wbc_classification/models/hyperbolic_cnn.py`, `scripts/hyperbolic_cnn_fine_tuned.py`
- **Metrics:** `utils/metrics.py`, `utils/reporting.py`

---

## 9. Public Archive Validation

Before publishing or depositing the repository, run:

```bash
python3 scripts/verify_archive.py
```

The verifier checks required archive files, JSON validity, packaged checkpoint checksums, GitHub file-size limits, and accidental hard-coded local path tokens.

---

## 10. Funding

This work was supported by the Center for Artificial Intelligence and Quantum Computing in System Brain Research (CLARA) (101136607-02), the project “Research Platform for Digital Transformation and Society 5.0” (CZ.02.01.01/00/23\_021/0012599), the Internal Grant Agency of VSB-TUO, Processing and Advanced Analysis of Biomedical Data XI of the Czech Republic (SP2026/008).

---

## 11. Citation

Citation metadata will be added after the manuscript record is finalized. Until then, please cite the manuscript title and repository as a code archive placeholder.

---

## 12. License

This software is distributed under the MIT License. See the `LICENSE` file for details.

---

## 13. Disclaimer

This work is intended for research purposes only. White blood cell and leukemia subtype predictions produced by these models must not be used for clinical decision-making without appropriate regulatory validation and clinical assessment.
