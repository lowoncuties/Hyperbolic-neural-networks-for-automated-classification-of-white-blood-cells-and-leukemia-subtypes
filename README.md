# Hyperbolic neural networks for automated classification of white blood cells and leukemia subtypes in peripheral blood smears

This repository contains two complementary training pipelines for automated white blood cell and leukemia subtype classification:

- `hyperbolic_cnn_fine_tuned.py` – a ResNet-50 backbone with a hyperbolic prototype head.
- `cnn_fine_tuned.py` – a Euclidean baseline that mirrors the data pipeline and sweep utilities.

Both scripts now expose a modern CLI that supports single-run experiments and full sweeps with reproducible CSV logging. Helper utilities live in `cli_utils.py`.

## Dataset and splits

The trainers assume an ImageFolder-compatible directory structure where each cell class has its own folder under `DATA_ROOT`. Pre-computed splits are resolved automatically by matching the `split_seed`, thresholding, and balancing arguments to the on-disk metadata. Use `--split-output-dir` and `--persist-splits-dir` to point at custom split repositories if required.

## Running sweeps

Sweeps iterate over a hyper-parameter grid encoded as a JSON dictionary (`parameter_name -> [values...]`). The JSON can be supplied inline or via file path:

```bash
python hyperbolic_cnn_fine_tuned.py \
  --mode sweep \
  --data-root /path/to/WBC_Our_dataset \
  --grid configs/hyperbolic_grid.json \
  --runs-root /experiments/hyperbolic_runs \
  --results-csv /experiments/hyperbolic_runs/results.csv
```

```bash
python cnn_fine_tuned.py \
  --mode sweep \
  --data-root /path/to/WBC_Our_dataset \
  --grid '{"feature_dim": [128, 256], "learning_rate": [1e-4, 1e-3]}' \
  --runs-root /experiments/cnn_runs \
  --results-csv /experiments/cnn_runs/results.csv
```

Each run receives its own artifact directory (`runs-root/run_name`). A unified CSV (`results-csv`) is created automatically with run metadata, timestamps, and the full metric suite for validation and test splits.

## Running a single configuration

Single runs read a JSON configuration (inline or file) that maps directly onto the corresponding `FinetuneConfig` dataclass. Unknown fields are ignored, and unspecified fields fall back to the defaults defined in each script.

```bash
python hyperbolic_cnn_fine_tuned.py \
  --mode single \
  --data-root /path/to/WBC_Our_dataset \
  --config '{"feature_dim": 256, "lr_head": 0.01, "batch_size": 512}' \
  --run-name hyperbolic_baseline \
  --runs-root /experiments/hyperbolic_runs
```

```bash
python cnn_fine_tuned.py \
  --mode single \
  --data-root /path/to/WBC_Our_dataset \
  --config configs/cnn_single.json \
  --epochs 50 \
  --run-name euclidean_long_run
```

If `--single-out-dir` is omitted, artifacts default to `runs-root/run-name`. Results are still appended to the CSV so the sweep and single-run histories share the same reporting surface.

## Common CLI arguments

Both scripts expose the same convenience flags:

| Argument | Description |
| --- | --- |
| `--data-root` | Absolute path to the ImageFolder dataset. |
| `--runs-root` | Root directory where per-run checkpoints, confusion matrices, and reports are stored. |
| `--results-csv` | CSV file that aggregates metrics across runs (created if missing). |
| `--grid` | JSON dictionary describing a sweep grid. Accepts file paths or inline JSON strings. |
| `--config` | JSON object that overrides `FinetuneConfig` fields for sweeps and single runs. |
| `--epochs` | Optional global override for the epoch budget. |
| `--threshold` | Intensity threshold used when generating split names. Accepts `null`/`None` to disable. |
| `--balance-to-min` / `--no-balance-to-min` | Toggle the class-balancing logic used to create splits. |
| `--balance-cap` | Optional cap applied when balancing; accepts `null`/`None` to remove the cap. |
| `--split-seed`, `--train-frac`, `--val-frac` | Parameters that must match the metadata embedded in your persisted splits. |
| `--split-output-dir`, `--persist-splits-dir` | Advanced knobs for pointing at custom split repositories. |
| `--run-name`, `--single-out-dir` | Single-run naming and artifact placement controls. |

The CLI automatically casts inline `null`/`None` strings to Python `None` for optional integer arguments.

## Custom configurations

- `--grid` JSON example:
  ```json
  {
    "feature_dim": [128, 256, 512],
    "learning_rate": [0.0001, 0.001],
    "seed": [13, 42]
  }
  ```
- `--config` JSON example:
  ```json
  {
    "batch_size": 256,
    "dropout_rate": 0.25,
    "threshold": null,
    "balance_to_min": true
  }
  ```

These configs can live in version-controlled files (recommended for papers) or be provided inline during quick experiments.

## Outputs

Each run directory contains:

- `best.pt` and `last.pt` checkpoints when `out_dir` is set.
- Confusion matrices (raw and normalized) and classification report CSV/PNGs.
- Logged CSV rows with timestamps and metrics for downstream analysis.

The refactor keeps your experiments reproducible and portable, enabling automated sweeps directly from the CLI, which is ideal for paper-quality experimentation.
