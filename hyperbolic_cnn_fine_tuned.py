#!/usr/bin/env python3
"""
Hyperbolic Prototype Classifier for Leukocyte Classification (dynamic #classes)

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
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
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

# Restrict file types explicitly
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_global_seed(seed: int = 42, deterministic: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        import random
        s = base_seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------
# Metrics helpers
# ----------------------------
def compute_sensitivity_specificity_multiclass(cm: np.ndarray):
    num_classes = cm.shape[0]
    support = cm.sum(axis=1)
    total_samples = support.sum()

    tp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for i in range(num_classes):
        tp[i] = cm[i, i]
        fp[i] = cm[:, i].sum() - tp[i]
        fn[i] = cm[i, :].sum() - tp[i]
        tn[i] = cm.sum() - (tp[i] + fp[i] + fn[i])

        sensitivity[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0.0
        specificity[i] = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) != 0 else 0.0

    TP_total = int(np.trace(cm))
    FP_total = int((cm.sum(axis=0) - np.diag(cm)).sum())
    FN_total = int((cm.sum(axis=1) - np.diag(cm)).sum())
    weighted_TN = float(np.sum(tn * support) / total_samples) if total_samples != 0 else 0.0
    weighted_sensitivity = float(np.sum(sensitivity * support) / total_samples) if total_samples != 0 else 0.0
    weighted_specificity = float(np.sum(specificity * support) / total_samples) if total_samples != 0 else 0.0

    return TP_total, FP_total, weighted_TN, FN_total, weighted_sensitivity, weighted_specificity


def f1_macro_from_report(report_dict: Dict[str, Any]) -> float:
    return float(report_dict.get("macro avg", {}).get("f1-score", 0.0))


# ----------------------------
# Plotting helpers (matplotlib, no seaborn)
# ----------------------------
def save_confusion_matrix_figure(cm: np.ndarray, classes: List[str], out_path: str, normalize: bool = False):
    import matplotlib.pyplot as plt

    cm_plot = cm.astype(float)
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1e-12))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_plot, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm_plot.shape[1]),
           yticks=np.arange(cm_plot.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_classification_report_csv_and_image(report_dict: Dict[str, Any], classes: List[str],
                                             csv_path: str, img_path: str):
    import csv as _csv
    import matplotlib.pyplot as plt

    headers = ["class", "precision", "recall", "f1", "support"]
    rows = []

    for i, cls in enumerate(classes):
        if str(i) in report_dict:
            stats = report_dict[str(i)]
        elif cls in report_dict:
            stats = report_dict[cls]
        else:
            stats = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        rows.append([cls,
                     float(stats.get("precision", 0.0)),
                     float(stats.get("recall", 0.0)),
                     float(stats.get("f1-score", 0.0)),
                     int(stats.get("support", 0))])

    acc = float(report_dict.get("accuracy", 0.0))
    macro = report_dict.get("macro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
    wavg  = report_dict.get("weighted avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
    rows.append(["accuracy", acc, acc, acc, int(wavg.get("support", 0))])
    rows.append(["macro avg",
                 float(macro.get("precision", 0.0)),
                 float(macro.get("recall", 0.0)),
                 float(macro.get("f1-score", 0.0)),
                 int(macro.get("support", 0))])
    rows.append(["weighted avg",
                 float(wavg.get("precision", 0.0)),
                 float(wavg.get("recall", 0.0)),
                 float(wavg.get("f1-score", 0.0)),
                 int(wavg.get("support", 0))])

    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.35 * (len(rows) + 1)))
    ax.axis('off')
    table = ax.table(cellText=[[f"{c}" if i == 0 or isinstance(c, str) else f"{c:.4f}" if j in [1,2,3] else f"{c}"
                                for j, c in enumerate(r)] for i, r in enumerate(rows)],
                     colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title("Classification Report")
    fig.tight_layout()
    fig.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Model bits (ResNet18 + hyperbolic prototype head)
# ----------------------------
class HyperbolicPrototypeHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        init_curvature: float = 1.0,
        temperature: float = 1.0,
        min_c: float = 1e-4,
        max_c: float = 1e4,
        proto_init_std: float = 1e-3,
    ):
        super().__init__()
        from geoopt.manifolds.stereographic import math as hypmath
        self.hypmath = hypmath

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.min_c = float(min_c)
        self.max_c = float(max_c)
        self._eps_c = 1e-6

        self.raw_c = nn.Parameter(torch.log(torch.tensor(float(init_curvature))))
        self.proto_tan = nn.Parameter(torch.randn(num_classes, embedding_dim) * proto_init_std)

    def current_c(self) -> torch.Tensor:
        c = F.softplus(self.raw_c) + self._eps_c
        return torch.clamp(c, self.min_c, self.max_c)

    def forward(self, euclidean_feats: torch.Tensor) -> torch.Tensor:
        device = euclidean_feats.device
        c = self.current_c().to(device)
        k = -c

        z = self.hypmath.expmap0(euclidean_feats, k=k, dim=-1)
        p = self.hypmath.expmap0(self.proto_tan.to(device), k=k, dim=-1)
        d = self.hypmath.dist(z.unsqueeze(1), p.unsqueeze(0), k=k, dim=-1)
        logits = -d / self.temperature
        return logits


class Backbone(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        m = models.resnet18(weights=None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.encoder = m
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))


class HyperbolicClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        init_curvature: float = 1.0,
        temperature: float = 1.0,
        **head_kwargs,
    ):
        super().__init__()
        self.backbone = Backbone(out_dim=feature_dim)
        self.head = HyperbolicPrototypeHead(
            embedding_dim=feature_dim,
            num_classes=num_classes,
            init_curvature=init_curvature,
            temperature=temperature,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ----------------------------
# Data (uses pre-generated JSON splits)
# ----------------------------
def _is_valid_file(path: str) -> bool:
    return Path(path).suffix.lower() in ALLOWED_EXTS


def _resolve_split_dir(
    root: str,
    split_output_dir: str,
    persist_splits_dir: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    img_size: int,
    balance_to_min: bool,
    balance_cap: Optional[int],
    threshold: Optional[int],
) -> Path:
    dataset_name = Path(root).resolve().name
    base = Path(split_output_dir)
    split_base = base / persist_splits_dir / dataset_name

    bal_tag = "balmin0"
    if balance_to_min:
        bal_tag = "balmin_auto" if balance_cap is None else f"balmin_cap{int(balance_cap)}"
    thr_tag = "thrnone" if threshold is None else f"thrgt{int(threshold)}"

    # IMPORTANT: match the exact naming used by your dataloader
    return split_base / f"seed_{seed}_t{train_frac}_v{val_frac}_s{img_size}_{bal_tag}_{thr_tag}"


def _load_indices(split_dir: Path):
    def _load(name: str):
        p = split_dir / f"{name}_idx.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        return np.array(json.loads(p.read_text()), dtype=int)

    train_idx = _load("train")
    val_idx   = _load("val")
    test_idx  = _load("test")
    return train_idx, val_idx, test_idx


def _build_transforms(img_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    t_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return t_train, t_eval


def build_dataloaders_from_splits(
    root: str,
    split_output_dir: str = SPLIT_OUTPUT_DIR,
    persist_splits_dir: str = PERSIST_SPLITS_DIR,
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    img_size: int = IMG_SIZE_DEFAULT,
    batch_size: int = 64,
    workers: int = 4,
    balance_to_min: bool = BALANCE_TO_MIN,
    balance_cap: Optional[int] = BALANCE_CAP,
    threshold: Optional[int] = THRESHOLD,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[int, int]]:
    """
    Build loaders from saved splits AND remap labels to contiguous [0..K-1]
    based on classes actually present in the union of train/val/test.
    Returns:
      train_loader, val_loader, test_loader, active_class_names (new order), old_to_new_map
    """
    base = datasets.ImageFolder(root=root, is_valid_file=_is_valid_file)
    idx_to_name = {v: k for k, v in base.class_to_idx.items()}

    split_dir = _resolve_split_dir(
        root, split_output_dir, persist_splits_dir,
        seed, train_frac, val_frac, img_size,
        balance_to_min, balance_cap, threshold
    )
    train_idx, val_idx, test_idx = _load_indices(split_dir)

    # Determine active classes from union of used indices
    all_idx = np.unique(np.concatenate([train_idx, val_idx, test_idx]))
    targets_all = np.array(base.targets)
    active_old_ids = sorted(set(int(t) for t in targets_all[all_idx]))

    # Build old->new contiguous remap and corresponding names
    old_to_new = {old: new for new, old in enumerate(active_old_ids)}
    new_to_old = {v: k for k, v in old_to_new.items()}
    active_class_names = [idx_to_name[new_to_old[i]] for i in range(len(active_old_ids))]

    # Target transform applies remap at read time
    t_train, t_eval = _build_transforms(img_size)
    target_transform = (lambda y: old_to_new[int(y)])

    ds_train_full = datasets.ImageFolder(root=root, transform=t_train,
                                         is_valid_file=_is_valid_file, target_transform=target_transform)
    ds_val_full   = datasets.ImageFolder(root=root, transform=t_eval,
                                         is_valid_file=_is_valid_file, target_transform=target_transform)
    ds_test_full  = datasets.ImageFolder(root=root, transform=t_eval,
                                         is_valid_file=_is_valid_file, target_transform=target_transform)

    ds_train = Subset(ds_train_full, train_idx.tolist())
    ds_val   = Subset(ds_val_full,   val_idx.tolist())
    ds_test  = Subset(ds_test_full,  test_idx.tolist())

    worker_init = make_worker_init_fn(seed)
    gen = make_generator(seed)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin_mem,
        worker_init_fn=worker_init, generator=gen
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_mem,
        worker_init_fn=worker_init
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_mem,
        worker_init_fn=worker_init
    )

    # Stats: number of classes per split (post-remap)
    y_train_old = targets_all[train_idx]
    y_val_old   = targets_all[val_idx]
    y_test_old  = targets_all[test_idx]
    y_train_new = np.array([old_to_new[int(y)] for y in y_train_old])
    y_val_new   = np.array([old_to_new[int(y)] for y in y_val_old])
    y_test_new  = np.array([old_to_new[int(y)] for y in y_test_old])

    n_total = len(active_class_names)
    n_train = int(np.unique(y_train_new).size)
    n_val   = int(np.unique(y_val_new).size)
    n_test  = int(np.unique(y_test_new).size)

    print("\n[Splits]")
    print(f"  path   : {split_dir}")
    print(f"  classes: total_active={n_total}, train={n_train}, val={n_val}, test={n_test}")
    print(f"  names  : {active_class_names}")

    return train_loader, val_loader, test_loader, active_class_names, old_to_new


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
