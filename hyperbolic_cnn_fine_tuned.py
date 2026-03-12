#!/usr/bin/env python3
"""
Hyperbolic Prototype Classifier for White Blood Cell and Leukemia Subtype Classification (dynamic #classes)

- Images: class-subfolder layout at /data3/datasets/WBC_Our_dataset
- Splits: pre-generated JSON index files (match the flags used when creating them)
- Head size: inferred from the splits after thresholding/balancing (no hard-coding)

Supported image formats: .jpg/.jpeg/.png/.tif/.tiff
"""

import argparse
import os
import json
import itertools
import datetime as dt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from cli_utils import (
    append_row_to_csv,
    build_config_dict,
    ensure_csv_with_header,
    instantiate_config,
    optional_int,
    parse_grid_argument,
)
from data.dataloaders import build_dataloaders_from_splits
from models.hyperbolic_cnn import HyperbolicClassifier
from utils.metrics import (
    compute_sensitivity_specificity_multiclass,
    f1_macro_from_report,
)
from utils.reporting import (
    save_classification_report_csv_and_image,
    save_confusion_matrix_figure,
)
from utils.reproducibility import set_global_seed


# ----------------------------
# Paths & configuration
# ----------------------------
DATA_ROOT = "/data3/datasets/WBC_Our_dataset"
SPLIT_OUTPUT_DIR = "/data2/joc0027/venv/JYOT"
PERSIST_SPLITS_DIR = "splits"

# Must match how splits were generated
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SPLIT_SEED = 42
IMG_SIZE_DEFAULT = 224

# These control which split folder we read (must match your dataloader flags)
THRESHOLD: Optional[int] = 110          # None => thrnone, else thrgt{THRESHOLD}
BALANCE_TO_MIN: bool = False           # False => balmin0, True => balmin_auto / balmin_cap{N}
BALANCE_CAP: Optional[int] = None      # only used if BALANCE_TO_MIN=True

# Model/output locations
RUNS_DIR   = "/data2/joc0027/venv/JYOT/hyperbolic_sweep_runs_wbc_downsampled_6"
RESULTS_CSV = "/data2/joc0027/venv/JYOT/sweep_new_wbc_downsampled_6.csv"

DEFAULT_PARAM_GRID = {
    "feature_dim": [256],
    "init_curvature": [2.0],
    "temperature": [1.0],
    "batch_size": [512],
    "lr_backbone": [1e-5, 3e-5, 1e-4],
    "lr_head": [5e-3, 1e-2, 2e-2],
    "lr_curvature": [1e-3, 3e-3, 5e-3],
    "img_size": [224],
    "seed": [42],
}

CSV_SUMMARY_FIELDS = [
    "run_name",
    "timestamp_iso",
    "best_epoch",
    "best_val_acc",
    "test_acc",
    "test_f1_macro",
    "learned_c",
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
# Eval helpers
# ----------------------------
@torch.no_grad()
def collect_preds(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.empty((0,), dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.empty((0,), dtype=int)
    return y_true, y_pred


@torch.no_grad()
def evaluate_with_cm(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    y_true, y_pred = collect_preds(model, loader, device)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if cm.shape != (num_classes, num_classes):
        full_cm = np.zeros((num_classes, num_classes), dtype=int)
        r, c = cm.shape
        full_cm[:r, :c] = cm
        cm = full_cm
    return acc, cm, y_true, y_pred


# ----------------------------
# Training configuration
# ----------------------------
@dataclass
class FinetuneConfig:
    feature_dim: int = 128
    init_curvature: float = 1.0
    temperature: float = 1.0
    batch_size: int = 64
    epochs: int = 25
    lr_backbone: float = 1e-4
    lr_head: float = 1e-2
    lr_curvature: float = 5e-3
    img_size: int = 224
    workers: int = 4
    seed: int = 42
    out_dir: Optional[str] = None
    split_seed: int = SPLIT_SEED
    train_frac: float = TRAIN_FRAC
    val_frac: float = VAL_FRAC
    threshold: Optional[int] = THRESHOLD
    balance_to_min: bool = BALANCE_TO_MIN
    balance_cap: Optional[int] = BALANCE_CAP
    split_output_dir: str = SPLIT_OUTPUT_DIR
    persist_splits_dir: str = PERSIST_SPLITS_DIR


# ----------------------------
# Train (returns metrics for sweep)
# ----------------------------
def train(
    data_root: str,
    cfg: FinetuneConfig,
) -> Dict[str, Any]:
    set_global_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train_loader, val_loader, test_loader, classes, _old2new = build_dataloaders_from_splits(
        root=data_root,
        split_output_dir=cfg.split_output_dir,
        persist_splits_dir=cfg.persist_splits_dir,
        seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        balance_to_min=cfg.balance_to_min,
        balance_cap=cfg.balance_cap,
        threshold=cfg.threshold,
    )

    num_classes = len(classes)  # <-- dynamic

    model = HyperbolicClassifier(
        feature_dim=cfg.feature_dim,
        num_classes=num_classes,
        init_curvature=cfg.init_curvature,
        temperature=cfg.temperature,
    ).to(device)

    # Param groups
    euclid_backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head_no_curv    = [p for n, p in model.named_parameters() if n.startswith("head.") and ("raw_c" not in n)]
    curv_param      = [model.head.raw_c]

    optimizer = torch.optim.Adam(
        [
            {"params": euclid_backbone, "lr": cfg.lr_backbone},
            {"params": head_no_curv,    "lr": cfg.lr_head},
            {"params": curv_param,      "lr": cfg.lr_curvature},
        ]
    )

    best_val = 0.0
    best_epoch = 0
    best_path = None

    # Per-run metrics CSV
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        metrics_csv = os.path.join(cfg.out_dir, "metrics.csv")
        if not os.path.exists(metrics_csv):
            import csv
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch","split","train_loss","accuracy","f1_macro","curvature_c",
                    "TP_total","FP_total","TN_weighted","FN_total","weighted_sensitivity","weighted_specificity"
                ])

    def _append_metrics_row(epoch, split, train_loss, acc, f1_macro, c_val,
                            TP_total, FP_total, TNw, FN_total, wsens, wspec):
        if not cfg.out_dir:
            return
        metrics_csv = os.path.join(cfg.out_dir, "metrics.csv")
        import csv
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, split,
                (None if train_loss is None else f"{train_loss:.6f}"),
                (None if acc is None else f"{acc:.6f}"),
                (None if f1_macro is None else f"{f1_macro:.6f}"),
                (None if c_val is None else f"{c_val:.8f}"),
                int(TP_total) if TP_total is not None else None,
                int(FP_total) if FP_total is not None else None,
                f"{TNw:.6f}" if TNw is not None else None,
                int(FN_total) if FN_total is not None else None,
                f"{wsens:.6f}" if wsens is not None else None,
                f"{wspec:.6f}" if wspec is not None else None,
            ])

    # ---- training
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0
        n_seen = 0

        for images, labels in pbar:
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

            with torch.no_grad():
                c_val = float(model.head.current_c().detach().cpu())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "c": f"{c_val:.4g}"})

        train_loss = running_loss / max(1, n_seen)

        # ---- validation metrics
        val_acc, val_cm, y_true_val, y_pred_val = evaluate_with_cm(model, val_loader, device, num_classes)
        report_val = classification_report(
            y_true_val, y_pred_val, labels=list(range(num_classes)),
            target_names=classes, output_dict=True, zero_division=0
        )
        val_f1_macro = f1_macro_from_report(report_val)

        TP_total, FP_total, TNw, FN_total, wsens, wspec = compute_sensitivity_specificity_multiclass(val_cm)
        c_now = float(model.head.current_c().detach().cpu())

        # Save best checkpoint by val accuracy
        if cfg.out_dir:
            os.makedirs(cfg.out_dir, exist_ok=True)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            if cfg.out_dir:
                best_path = os.path.join(cfg.out_dir, f"best_epoch{epoch}_val{best_val:.4f}.pt")
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "feature_dim": cfg.feature_dim,
                        "init_curvature": cfg.init_curvature,
                        "learned_c": c_now,
                        "temperature": cfg.temperature,
                        "epoch": epoch,
                        "val_acc": best_val,
                    },
                    best_path,
                )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1_macro={val_f1_macro:.4f} "
              f"(best_acc={best_val:.4f} @ {best_epoch}) c={c_now:.6f}")

        _append_metrics_row(epoch, "val", train_loss, val_acc, val_f1_macro, c_now,
                            TP_total, FP_total, TNw, FN_total, wsens, wspec)

    # ---- final TEST metrics + artifacts
    test_acc, test_cm, y_true_test, y_pred_test = evaluate_with_cm(model, test_loader, device, num_classes)
    report_test = classification_report(
        y_true_test, y_pred_test, labels=list(range(num_classes)),
        target_names=classes, output_dict=True, zero_division=0
    )
    test_f1_macro = f1_macro_from_report(report_test)
    print(f"Test accuracy: {test_acc:.4f} | Test F1_macro: {test_f1_macro:.4f}")

    T_TP, T_FP, T_TNw, T_FN, T_wsens, T_wspec = compute_sensitivity_specificity_multiclass(test_cm)
    c_final = float(model.head.current_c().detach().cpu())

    _append_metrics_row(best_epoch if best_epoch else cfg.epochs, "test",
                        None, test_acc, test_f1_macro, c_final,
                        T_TP, T_FP, T_TNw, T_FN, T_wsens, T_wspec)

    # ---- save artifacts
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        cm_png  = os.path.join(cfg.out_dir, "test_confusion_matrix.png")
        cmn_png = os.path.join(cfg.out_dir, "test_confusion_matrix_normalized.png")
        rep_csv = os.path.join(cfg.out_dir, "test_classification_report.csv")
        rep_png = os.path.join(cfg.out_dir, "test_classification_report.png")

        save_confusion_matrix_figure(test_cm, classes, cm_png, normalize=False)
        save_confusion_matrix_figure(test_cm, classes, cmn_png, normalize=True)
        save_classification_report_csv_and_image(report_test, classes, rep_csv, rep_png)

    if cfg.out_dir:
        last_ckpt = os.path.join(cfg.out_dir, "last.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "feature_dim": cfg.feature_dim,
                "init_curvature": cfg.init_curvature,
                "learned_c": c_final,
                "temperature": cfg.temperature,
                "epoch": cfg.epochs,
                "val_acc": best_val,
                "test_acc": test_acc,
                "test_f1_macro": test_f1_macro,
            },
            last_ckpt,
        )
        print(f"Saved last model to {last_ckpt}")
        if best_path:
            import shutil
            stable_best = os.path.join(cfg.out_dir, "best.pt")
            shutil.copyfile(best_path, stable_best)
            print(f"Saved best model to {stable_best}")

    return {
        "best_epoch": best_epoch if best_epoch else cfg.epochs,
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "learned_c": float(c_final),

        # Last-epoch VAL (for reference)
        "val_TP_total": int(TP_total),
        "val_FP_total": int(FP_total),
        "val_TN_weighted": float(TNw),
        "val_FN_total": int(FN_total),
        "val_weighted_sensitivity": float(wsens),
        "val_weighted_specificity": float(wspec),

        "test_TP_total": int(T_TP),
        "test_FP_total": int(T_FP),
        "test_TN_weighted": float(T_TNw),
        "test_FN_total": int(T_FN),
        "test_weighted_sensitivity": float(T_wsens),
        "test_weighted_specificity": float(T_wspec),
    }


# ----------------------------
# Sweep utilities
# ----------------------------
def iter_grid(grid: Dict[str, List[Any]]):
    keys = sorted(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def make_run_name(params: Dict[str, Any]) -> str:
    parts = [f"{k}={str(v)}" for k, v in sorted(params.items())]
    return "_".join(p.replace("/", "-") for p in parts)


def run_sweep_and_save_csv(
    data_root: str,
    out_root: str,
    csv_path: str,
    grid: Dict[str, List[Any]],
    epochs: Optional[int] = None,
    base_config: Optional[Dict[str, Any]] = None,
    hp_keys: Optional[List[str]] = None,
):
    os.makedirs(out_root, exist_ok=True)

    hp_order = hp_keys or sorted(grid.keys())
    fieldnames = hp_order + CSV_SUMMARY_FIELDS
    ensure_csv_with_header(csv_path, fieldnames)

    base_cfg = build_config_dict(FinetuneConfig)
    if base_config:
        base_cfg.update({k: v for k, v in base_config.items() if v is not None})

    for params in iter_grid(grid):
        cfg_dict = dict(base_cfg)
        cfg_dict.update(params)
        if epochs is not None:
            cfg_dict["epochs"] = epochs

        run_name = make_run_name(params)
        cfg_dict["out_dir"] = os.path.join(out_root, run_name)
        cfg = FinetuneConfig(**cfg_dict)

        print(f"\n=== Running: {run_name} ===")
        print(
            "Using splits with:",
            f"seed={cfg.split_seed}, t={cfg.train_frac}, v={cfg.val_frac}, s={cfg.img_size}, ",
            f"balance_to_min={cfg.balance_to_min}, balance_cap={cfg.balance_cap}, threshold={cfg.threshold}",
        )
        metrics = train(data_root=data_root, cfg=cfg)

        row = {k: cfg_dict.get(k) for k in hp_order}
        row.update(
            {
                "run_name": run_name,
                "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
        row.update(metrics)

        append_row_to_csv(csv_path, fieldnames, row)
        print(f"Logged to {csv_path}: {run_name}")


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperbolic classifier trainer with sweep and single-run modes.",
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
        print("Starting hyperbolic classifier sweep...")
        run_sweep_and_save_csv(
            data_root=args.data_root,
            out_root=args.runs_root,
            csv_path=args.results_csv,
            grid=grid,
            epochs=args.epochs,
            base_config=base_cfg,
            hp_keys=hp_keys,
        )
        print("Sweep finished.")
        return

    # Single run
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
    metrics = train(data_root=args.data_root, cfg=cfg)

    print(f"Single run '{run_name}' finished with test_acc={metrics['test_acc']:.4f}")
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
