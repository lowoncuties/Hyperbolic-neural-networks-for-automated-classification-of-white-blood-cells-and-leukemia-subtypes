# Hyperbolic neural networks for automated classification of white blood cells and leukemia subtypes in peripheral blood smears

This repository keeps the clean modular training library as the canonical implementation.

Official entrypoints:

- `scripts/hyperbolic_cnn_fine_tuned.py`: ResNet-18 backbone with a hyperbolic prototype head
- `scripts/cnn_fine_tuned.py`: ResNet-18 Euclidean baseline with the same split/CLI utilities

Reference-only migration scripts:

- `scripts/hyperbolic_cnn_learnable_temp.py`
- `scripts/cnn_fine_tuned_new.py`

The migration behavior from those scripts has been folded into the clean library without copying their duplicated data, metrics, and reporting code back into the main implementation.

## Defaults

The clean scripts now resolve paths through `project_paths.py`. Each value can be
overridden with an environment variable or the existing CLI flag.

| Setting | Hyperbolic / CNN default |
| --- | --- |
| `DATA_ROOT` | `$WBC_DATA_ROOT` or `./datasets/WBC_Our_dataset_extended` |
| `SPLIT_OUTPUT_DIR` | `$WBC_SPLIT_OUTPUT_DIR` or `./outputs` |
| `PERSIST_SPLITS_DIR` | `$WBC_PERSIST_SPLITS_DIR` or `splits` |
| Hyperbolic `RUNS_DIR` | `$WBC_HYPERBOLIC_RUNS_DIR` or `./outputs/runs/hyperbolic` |
| Hyperbolic `RESULTS_CSV` | `$WBC_HYPERBOLIC_RESULTS_CSV` or `./outputs/results/hyperbolic_summary.csv` |
| CNN `RUNS_DIR` | `$WBC_CNN_RUNS_DIR` or `./outputs/runs/cnn` |
| CNN `RESULTS_CSV` | `$WBC_CNN_RESULTS_CSV` or `./outputs/results/cnn_summary.csv` |

All of these can still be overridden from the CLI.

## Model Notes

- Both trainers use `torchvision.models.resnet18(weights=None)`.
- The hyperbolic model learns curvature through `raw_c -> softplus -> clamp`.
- The hyperbolic trainer supports two temperature modes:
  - fixed tau baseline
  - optional learnable tau, parameterized as `tau = softplus(raw_tau) + eps`
- The Euclidean trainer remains the clean modular implementation already present in the library; only its defaults were updated to repo-local/configurable paths.

## Dataset And Splits

The trainers assume an `ImageFolder` directory structure with one class directory per label. Persisted splits are resolved by matching:

- `split_seed`
- `train_frac`
- `val_frac`
- `img_size`
- `threshold`
- `balance_to_min`
- `balance_cap`

Use `--split-output-dir` and `--persist-splits-dir` if your split metadata lives elsewhere.

## Running Sweeps

Sweeps accept either an inline JSON grid or a path to a JSON file.

Hyperbolic baseline sweep:

```bash
python scripts/hyperbolic_cnn_fine_tuned.py \
  --mode sweep \
  --data-root /path/to/WBC_Our_dataset_extended \
  --grid configs/hyperbolic_grid.json \
  --runs-root /experiments/hyperbolic_runs \
  --results-csv /experiments/hyperbolic_runs/results.csv
```

Hyperbolic sweep with learnable temperature:

```bash
python scripts/hyperbolic_cnn_fine_tuned.py \
  --mode sweep \
  --data-root /path/to/WBC_Our_dataset_extended \
  --learnable-temperature \
  --lr-temperature 1e-3 \
  --grid '{"feature_dim":[256],"init_curvature":[2.0],"temperature":[1.0],"batch_size":[256],"lr_backbone":[1e-4],"lr_head":[5e-3],"lr_curvature":[3e-3],"img_size":[224],"seed":[42]}' \
  --runs-root /experiments/hyperbolic_runs \
  --results-csv /experiments/hyperbolic_runs/results.csv
```

CNN sweep:

```bash
python scripts/cnn_fine_tuned.py \
  --mode sweep \
  --data-root /path/to/WBC_Our_dataset_extended \
  --grid '{"feature_dim":[128,256],"learning_rate":[1e-4,1e-3]}' \
  --runs-root /experiments/cnn_runs \
  --results-csv /experiments/cnn_runs/results.csv
```

Each run receives its own artifact directory under `runs-root/run_name`.

## Running A Single Configuration

Single runs read a JSON configuration that maps directly to the corresponding `FinetuneConfig`.

Hyperbolic fixed-tau single run:

```bash
python scripts/hyperbolic_cnn_fine_tuned.py \
  --mode single \
  --data-root /path/to/WBC_Our_dataset_extended \
  --config '{"feature_dim":256,"lr_head":0.01,"batch_size":256}' \
  --run-name hyperbolic_baseline \
  --runs-root /experiments/hyperbolic_runs
```

Hyperbolic learnable-tau single run:

```bash
python scripts/hyperbolic_cnn_fine_tuned.py \
  --mode single \
  --data-root /path/to/WBC_Our_dataset_extended \
  --learnable-temperature \
  --lr-temperature 1e-3 \
  --config '{"feature_dim":256,"temperature":1.0,"lr_head":0.01,"batch_size":256}' \
  --run-name hyperbolic_learnable_tau \
  --runs-root /experiments/hyperbolic_runs
```

CNN single run:

```bash
python scripts/cnn_fine_tuned.py \
  --mode single \
  --data-root /path/to/WBC_Our_dataset_extended \
  --config configs/cnn_single.json \
  --epochs 50 \
  --run-name euclidean_long_run
```

If `--single-out-dir` is omitted, artifacts default to `runs-root/run-name`.

## Common CLI Arguments

Both official scripts expose the same configuration pattern:

- `--data-root`: dataset root
- `--runs-root`: directory for run artifacts
- `--results-csv`: aggregate CSV for sweep or single-run summaries
- `--mode`: `sweep` or `single`
- `--grid`: JSON grid for sweeps
- `--config`: JSON config overrides
- `--epochs`: optional epoch override
- `--threshold`: split threshold; accepts `null`/`None`
- `--balance-to-min` / `--no-balance-to-min`: split balancing toggle
- `--balance-cap`: optional balancing cap; accepts `null`/`None`
- `--split-seed`, `--train-frac`, `--val-frac`: split metadata controls
- `--split-output-dir`, `--persist-splits-dir`: alternate split locations
- `--run-name`, `--single-out-dir`: single-run naming and artifact placement

Hyperbolic-only temperature arguments:

- `--learnable-temperature`: enable learnable tau
- `--fixed-temperature`: force fixed tau
- `--lr-temperature`: learning rate for `raw_tau` when learnable temperature is enabled

## Outputs

Each run directory contains:

- `best.pt` and `last.pt` checkpoints when `out_dir` is set
- raw and normalized confusion matrices
- classification report CSV/PNG artifacts
- `metrics.csv` with per-epoch metrics

For the hyperbolic trainer, the run artifacts and summary CSV also record:

- curvature value
- whether learnable temperature was enabled
- final tau value
