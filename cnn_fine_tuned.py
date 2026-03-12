#!/usr/bin/env python3
"""
Euclidean Distance-Based Classifier for White Blood Cell and Leukemia Subtype Classification

This script converts the hyperbolic classifier to use classical Euclidean
distance for inference while maintaining all data loaders, random freezes,
and reproducibility.
"""

import argparse
import csv
import itertools
import json
import datetime as dt
from dataclasses import dataclass
import os
from typing import Tuple, List, Optional, Dict, Any
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from cli_utils import (
    append_row_to_csv,
    build_config_dict,
    ensure_csv_with_header,
    instantiate_config,
    optional_int,
    parse_grid_argument,
)
from data.dataloaders import build_dataloaders_from_splits
from models.classic_cnn import CNNClassifier
from utils.metrics import (
    compute_sensitivity_specificity_multiclass,
    f1_macro_from_preds_labels,
)
from utils.reporting import (
    save_classification_report_csv_and_image,
    save_confusion_matrix_figure,
)
from utils.reproducibility import set_global_seed


# ----------------------------
# Paths & configuration (mirrors hyperbolic script)
# ----------------------------
DATA_ROOT = "/data3/datasets/WBC_Our_dataset"
SPLIT_OUTPUT_DIR = "/data2/joc0027/venv/JYOT"
PERSIST_SPLITS_DIR = "splits"
RUNS_DIR = "/data2/joc0027/venv/JYOT/cnn_sweep_runs_donwsampled"
RESULTS_CSV = "/data2/joc0027/venv/JYOT/cnn_sweep_results_wbc_downsampled.csv"

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 42
IMG_SIZE_DEFAULT = 224

THRESHOLD: Optional[int] = 110
BALANCE_TO_MIN: bool = False
BALANCE_CAP: Optional[int] = None

DEFAULT_PARAM_GRID = {
    "feature_dim": [64, 128, 256],
    "batch_size": [512],
    "learning_rate": [1e-4, 1e-3, 5e-3],
    "img_size": [224],
    "seed": [42],
    "dropout_rate": [0.2],
    "weight_decay": [1e-4],
}

CSV_SUMMARY_FIELDS = [
    "run_name",
    "timestamp_iso",
    "best_epoch",
    "stopped_due_to_overfit",
    "gen_gap_val_train_loss",
    "test_loss",
    "test_acc",
    "test_f1_macro",
    "val_TP_total",
    "val_FP_total",
    "val_TN_weighted",
    "val_FN_total",
    "val_weighted_sensitivity",
    "val_weighted_specificity",
    "test_TP_total",
    "test_FP_total",
    "test_TN_weighted",
    "test_FN_total",
    "test_weighted_sensitivity",
    "test_weighted_specificity",
]


# ----------------------------
# Data (uses pre-generated JSON splits, identical to hyperbolic script)
# ----------------------------
# ----------------------------
# Evaluation helpers
# ----------------------------
@torch.no_grad()
def evaluate_with_cm(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
):
    """Evaluate model and return accuracy, confusion matrix, and raw preds."""
    model.eval()
    all_true = []
    all_pred = []
    num_correct = 0
    num_total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        num_correct += (preds == labels).sum().item()
        num_total += labels.numel()

        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    acc = num_correct / max(1, num_total)
    y_true = np.concatenate(all_true) if all_true else np.empty((0,), dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Ensure (C,C) shape even if some classes missing in subset
    if cm.shape != (num_classes, num_classes):
        full_cm = np.zeros((num_classes, num_classes), dtype=int)
        r, c = cm.shape
        full_cm[:r, :c] = cm
        cm = full_cm
    return acc, cm, y_true, y_pred


@torch.no_grad()
def evaluate_with_loss(
    model: nn.Module, loader: DataLoader, device: torch.device
):
    """
    Returns: avg_loss, accuracy, preds_tensor (N,), labels_tensor (N,)
    """
    model.eval()
    total_loss = 0.0
    n_total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        n_total += labels.numel()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    if n_total == 0:
        return float("nan"), 0.0, torch.empty(0), torch.empty(0)
    avg_loss = total_loss / n_total
    preds = (
        torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    )
    labs = (
        torch.cat(all_labels)
        if all_labels
        else torch.empty(0, dtype=torch.long)
    )
    acc = (preds == labs).float().mean().item() if preds.numel() else 0.0
    return avg_loss, acc, preds, labs


# ----------------------------
# Training configuration
# ----------------------------
@dataclass
class FinetuneConfig:
    feature_dim: int = 128
    batch_size: int = 32
    epochs: int = 25  # Max epochs
    learning_rate: float = 1e-3  # Single learning rate for all parameters
    img_size: int = 224
    workers: int = 4
    seed: int = 42
    patience: int = 7  # Increased patience
    overfit_patience: int = 3  # Early stopping after 3 overfitting epochs
    dropout_rate: float = 0.1  # Added dropout
    weight_decay: float = 1e-4  # Added weight decay
    out_dir: Optional[str] = None  # per-run directory; if None, no checkpoints
    threshold: Optional[int] = THRESHOLD
    balance_to_min: bool = BALANCE_TO_MIN
    balance_cap: Optional[int] = BALANCE_CAP
    split_seed: int = SPLIT_SEED
    train_frac: float = TRAIN_FRAC
    val_frac: float = VAL_FRAC
    split_output_dir: str = SPLIT_OUTPUT_DIR
    persist_splits_dir: str = PERSIST_SPLITS_DIR


# ----------------------------
# Training function
# ----------------------------
def train_one_config(
    data_root: str,
    cfg: FinetuneConfig,
):
    """Train a single configuration with early stopping."""
    set_global_seed(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        classes,
        _label_map,
    ) = build_dataloaders_from_splits(
        root=data_root,
        split_output_dir=cfg.split_output_dir,
        persist_splits_dir=cfg.persist_splits_dir,
        seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        threshold=cfg.threshold,
        balance_to_min=cfg.balance_to_min,
        balance_cap=cfg.balance_cap,
    )
    num_classes = len(classes)

    model = CNNClassifier(
        feature_dim=cfg.feature_dim,
        num_classes=num_classes,
        dropout_rate=cfg.dropout_rate,
    ).to(device)

    # Single optimizer for all parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Bookkeeping
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    overfit_streak = 0
    overfit_stopped = False

    hist = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for images, labels in tqdm(
            train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        # Validate after every epoch for proper early stopping
        val_loss, val_acc, val_preds, val_labels = evaluate_with_loss(
            model, val_loader, device
        )
        val_f1 = f1_macro_from_preds_labels(val_preds, val_labels, num_classes)

        # Get validation confusion matrix and detailed metrics
        val_acc_cm, val_cm, _, _ = evaluate_with_cm(
            model, val_loader, device, num_classes
        )
        val_TP, val_FP, val_TN, val_FN, val_wsens, val_wspec = (
            compute_sensitivity_specificity_multiclass(val_cm)
        )

        # Update learning rate
        scheduler.step(val_loss)

        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        # Overfitting detector (loss-based)
        if len(hist["train_loss"]) >= 2:
            train_down = train_loss < hist["train_loss"][-2] - 1e-6
            val_up = val_loss > hist["val_loss"][-2] + 1e-6
            if train_down and val_up:
                overfit_streak += 1
            else:
                overfit_streak = 0

        # Early stopping on best val loss
        improved = val_loss < best_val_loss - 1e-9
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Keep best state
            best_state = {
                "state_dict": model.state_dict(),
                "classes": classes,
                "feature_dim": cfg.feature_dim,
                "val_loss": best_val_loss,
                "epoch": epoch,
            }
            # Optional: save a checkpoint per run
            if cfg.out_dir:
                os.makedirs(cfg.out_dir, exist_ok=True)
                torch.save(best_state, os.path.join(cfg.out_dir, "best.pt"))
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(
                f"  -> No improvement for {epochs_no_improve} validation checks"
            )

        # Print epoch results
        print(
            f"Epoch {epoch}/{cfg.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}"
        )

        # Stop if patience reached OR overfit streak reached
        if (
            epochs_no_improve >= cfg.patience
            or overfit_streak >= cfg.overfit_patience
        ):
            overfit_stopped = overfit_streak >= cfg.overfit_patience
            if overfit_stopped:
                print(
                    f"  -> Early stopping due to overfitting (streak: {overfit_streak})"
                )
            else:
                print(
                    f"  -> Early stopping due to no improvement (patience: {cfg.patience})"
                )
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    # Test metrics
    test_loss, test_acc, _, _ = evaluate_with_loss(
        model, test_loader, device
    )

    # Get confusion matrix and detailed metrics
    test_acc_cm, test_cm, y_true_test, y_pred_test = evaluate_with_cm(
        model, test_loader, device, num_classes
    )
    test_TP, test_FP, test_TN, test_FN, test_wsens, test_wspec = (
        compute_sensitivity_specificity_multiclass(test_cm)
    )

    report_test = classification_report(
        y_true_test,
        y_pred_test,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    test_f1 = float(report_test.get("macro avg", {}).get("f1-score", 0.0))

    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(
        f"Test TP: {test_TP}, FP: {test_FP}, TN: {test_TN:.4f}, FN: {test_FN}"
    )
    print(f"Test Weighted Sensitivity: {test_wsens:.4f}")
    print(f"Test Weighted Specificity: {test_wspec:.4f}")

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        cm_png = os.path.join(cfg.out_dir, "test_confusion_matrix.png")
        cmn_png = os.path.join(cfg.out_dir, "test_confusion_matrix_normalized.png")
        rep_csv = os.path.join(cfg.out_dir, "test_classification_report.csv")
        rep_png = os.path.join(cfg.out_dir, "test_classification_report.png")

        save_confusion_matrix_figure(test_cm, classes, cm_png, normalize=False)
        save_confusion_matrix_figure(test_cm, classes, cmn_png, normalize=True)
        save_classification_report_csv_and_image(report_test, classes, rep_csv, rep_png)

    gen_gap = (
        float(best_val_loss - hist["train_loss"][best_epoch - 1])
        if best_epoch > 0
        else float("nan")
    )

    result = {
        "best_epoch": best_epoch,
        "stopped_due_to_overfit": int(overfit_stopped),
        "gen_gap_val_train_loss": gen_gap,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1_macro": test_f1,
        # Detailed validation metrics (from best epoch)
        "val_TP_total": val_TP,
        "val_FP_total": val_FP,
        "val_TN_weighted": val_TN,
        "val_FN_total": val_FN,
        "val_weighted_sensitivity": val_wsens,
        "val_weighted_specificity": val_wspec,
        # Detailed test metrics
        "test_TP_total": test_TP,
        "test_FP_total": test_FP,
        "test_TN_weighted": test_TN,
        "test_FN_total": test_FN,
        "test_weighted_sensitivity": test_wsens,
        "test_weighted_specificity": test_wspec,
        # Echo config
        "feature_dim": cfg.feature_dim,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "img_size": cfg.img_size,
        "seed": cfg.seed,
        "dropout_rate": cfg.dropout_rate,
        "weight_decay": cfg.weight_decay,
    }

    # Save last checkpoint too (optional)
    if cfg.out_dir:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "feature_dim": cfg.feature_dim,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_metrics": {
                    "loss": test_loss,
                    "acc": test_acc,
                    "f1_macro": test_f1,
                },
                "history": hist,
            },
            os.path.join(cfg.out_dir, "last.pt"),
        )

    return result


# ----------------------------
# Sweep utilities
# ----------------------------
def run_sweep_and_save_csv(
    data_root: str,
    out_root: str,
    csv_path: str,
    grid: Dict[str, List[Any]],
    base_config: Optional[Dict[str, Any]] = None,
    epochs: Optional[int] = None,
    hp_keys: Optional[List[str]] = None,
):
    """Run a hyper-parameter sweep and persist results."""
    os.makedirs(out_root, exist_ok=True)
    hp_order = hp_keys or sorted(grid.keys())
    fieldnames = hp_order + CSV_SUMMARY_FIELDS
    ensure_csv_with_header(csv_path, fieldnames)

    base_cfg = build_config_dict(FinetuneConfig)
    if base_config:
        base_cfg.update({k: v for k, v in base_config.items() if v is not None})

    combos = list(itertools.product(*[grid[k] for k in hp_order]))
    for i, combo in enumerate(combos, 1):
        params = {k: v for k, v in zip(hp_order, combo)}
        cfg_dict = dict(base_cfg)
        cfg_dict.update(params)
        if epochs is not None:
            cfg_dict["epochs"] = epochs
        run_name = "_".join(f"{k}={str(v)}" for k, v in params.items())
        cfg_dict["out_dir"] = os.path.join(out_root, run_name)
        cfg = FinetuneConfig(**cfg_dict)

        print(f"\n=== [{i}/{len(combos)}] Running: {run_name} ===")
        result = train_one_config(data_root, cfg)

        row = {k: cfg_dict.get(k) for k in hp_order}
        row.update(
            {
                "run_name": run_name,
                "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
        row.update(result)
        append_row_to_csv(csv_path, fieldnames, row)
        print(f"Logged to {csv_path}: {run_name}")


# ----------------------------
# CLI utilities
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNN baseline trainer with sweep and single-run modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Path to the dataset root")
    parser.add_argument(
        "--runs-root",
        default=RUNS_DIR,
        help="Directory where per-run artifacts are written",
    )
    parser.add_argument(
        "--results-csv",
        default=RESULTS_CSV,
        help="CSV file that collects sweep/single summaries",
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "single"],
        default="sweep",
        help="Run an entire grid sweep or a single configuration",
    )
    parser.add_argument(
        "--grid",
        help="JSON file or inline string describing the hyper-parameter grid",
    )
    parser.add_argument(
        "--config",
        help="JSON file or inline string with FinetuneConfig overrides",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch budget for every run",
    )
    parser.add_argument(
        "--threshold",
        type=optional_int,
        default=THRESHOLD,
        help="Intensity threshold used when building the persisted splits",
    )
    parser.add_argument(
        "--balance-to-min",
        dest="balance_to_min",
        action="store_true",
        default=BALANCE_TO_MIN,
        help="Enable balancing to the smallest class",
    )
    parser.add_argument(
        "--no-balance-to-min",
        dest="balance_to_min",
        action="store_false",
        help="Disable class-count balancing",
    )
    parser.add_argument(
        "--balance-cap",
        type=optional_int,
        default=BALANCE_CAP,
        help="Optional cap applied when balancing to the minimum",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=SPLIT_SEED,
        help="Seed component encoded inside the persisted split folders",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=TRAIN_FRAC,
        help="Training fraction used when the splits were created",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRAC,
        help="Validation fraction used when the splits were created",
    )
    parser.add_argument(
        "--split-output-dir",
        default=SPLIT_OUTPUT_DIR,
        help="Root directory containing the persisted split metadata",
    )
    parser.add_argument(
        "--persist-splits-dir",
        default=PERSIST_SPLITS_DIR,
        help="Relative directory inside the split root that stores JSON files",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for the single run (used only when mode=single)",
    )
    parser.add_argument(
        "--single-out-dir",
        default=None,
        help="Explicit directory for single-run artifacts (defaults to runs-root/run-name)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    grid = parse_grid_argument(args.grid, DEFAULT_PARAM_GRID)
    hp_keys = sorted(grid.keys())

    common_overrides = {
        "threshold": args.threshold,
        "balance_to_min": args.balance_to_min,
        "balance_cap": args.balance_cap,
        "split_seed": args.split_seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split_output_dir": args.split_output_dir,
        "persist_splits_dir": args.persist_splits_dir,
    }

    if args.mode == "sweep":
        base_cfg = build_config_dict(
            FinetuneConfig,
            config_arg=args.config,
            overrides=common_overrides,
        )
        print("Starting CNN classifier sweep...")
        run_sweep_and_save_csv(
            data_root=args.data_root,
            out_root=args.runs_root,
            csv_path=args.results_csv,
            grid=grid,
            base_config=base_cfg,
            epochs=args.epochs,
            hp_keys=hp_keys,
        )
        print("Sweep finished. CSV at:", args.results_csv)
        return

    run_name = args.run_name or "single_run"
    if args.single_out_dir:
        out_dir = args.single_out_dir
    elif args.runs_root:
        out_dir = os.path.join(args.runs_root, run_name)
    else:
        out_dir = None

    cfg = instantiate_config(
        FinetuneConfig,
        config_arg=args.config,
        overrides={**common_overrides, "epochs": args.epochs, "out_dir": out_dir},
    )
    metrics = train_one_config(args.data_root, cfg)
    print(
        f"Single run '{run_name}' finished with test_acc={metrics['test_acc']:.4f}"
    )
    fieldnames = hp_keys + CSV_SUMMARY_FIELDS
    row = {k: getattr(cfg, k, None) for k in hp_keys}
    row.update(
        {
            "run_name": run_name,
            "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    row.update(metrics)
    append_row_to_csv(args.results_csv, fieldnames, row)


if __name__ == "__main__":
    main()
