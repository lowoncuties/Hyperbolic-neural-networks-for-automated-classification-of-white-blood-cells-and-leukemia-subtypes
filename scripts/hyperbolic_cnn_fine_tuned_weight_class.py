#!/usr/bin/env python3
"""
Hyperbolic Prototype Classifier for White Blood Cell and Leukemia Subtype Classification (dynamic #classes)
with Imbalance Handling:
- Class-balanced sampler (inverse-count^power)
- Class-Balanced Focal Loss (CB-Focal) with Deferred Re-Weighting (warm-up)
- Optional Logit Adjustment by priors

Images: class-subfolder layout at /data3/datasets/WBC_Our_dataset
Splits: pre-generated JSON index files (match the flags used when creating them)
"""


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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
THRESHOLD: Optional[int] = 41          # None => thrnone, else thrgt{THRESHOLD}
BALANCE_TO_MIN: bool = False           # False => balmin0, True => balmin_auto / balmin_cap{N}
BALANCE_CAP: Optional[int] = None      # only used if BALANCE_TO_MIN=True

# Model/output locations
RUNS_DIR   = "/data2/joc0027/venv/JYOT/hyperbolic_sweep_runs_wbc_downsampled_weighted"
RESULTS_CSV = "/data2/joc0027/venv/JYOT/sweep_new_wbc_downsampled_weighted.csv"

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
# Imbalance-aware losses
# ----------------------------
def effective_num_weights(counts: np.ndarray, beta: float) -> torch.Tensor:
    """
    Class-Balanced weight alpha_y = (1 - beta) / (1 - beta^n_y)
    For classes with count 0 (shouldn't happen for train), set weight 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = np.zeros_like(counts, dtype=np.float64)
    nz = counts > 0
    w[nz] = (1.0 - beta) / (1.0 - np.power(beta, counts[nz]))
    # Optional normalization to mean=1 to keep loss scale reasonable:
    if nz.any():
        w[nz] = w[nz] * (counts[nz].size / w[nz].sum())
    return torch.tensor(w, dtype=torch.float32)


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for multi-class (single-label).
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per sample
        ce = F.cross_entropy(logits, targets, reduction="none")
        # p_t for focal modulation
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1, 1)).squeeze(1).clamp_(1e-8, 1 - 1e-8)
        focal = (1.0 - pt).pow(self.gamma)
        # class weights
        alpha = self.class_weights.gather(0, targets)
        loss = alpha * focal * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


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
    # Slightly stronger, still label-preserving for cells
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
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
    weighted_sampler_power: Optional[float] = None,   # NEW: if not None, enable WeightedRandomSampler
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Build loaders from saved splits AND remap labels to contiguous [0..K-1]
    Returns:
      train_loader, val_loader, test_loader, active_class_names, old_to_new, train_counts, train_priors
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

    # ----- class stats on TRAIN (post-remap) -----
    y_train_old = targets_all[train_idx]
    y_val_old   = targets_all[val_idx]
    y_test_old  = targets_all[test_idx]
    y_train_new = np.array([old_to_new[int(y)] for y in y_train_old])
    y_val_new   = np.array([old_to_new[int(y)] for y in y_val_old])
    y_test_new  = np.array([old_to_new[int(y)] for y in y_test_old])

    num_classes = len(active_class_names)
    train_counts = np.bincount(y_train_new, minlength=num_classes)
    train_priors = train_counts / max(1, train_counts.sum())

    # ----- DataLoaders (optionally with WeightedRandomSampler) -----
    worker_init = make_worker_init_fn(seed)
    gen = make_generator(seed)
    pin_mem = torch.cuda.is_available()

    if weighted_sampler_power is not None:
        # per-sample weights: (1/n_y)^power
        counts = train_counts.copy().astype(np.float64)
        counts[counts == 0] = 1.0
        inv = np.power(counts, -float(weighted_sampler_power))
        sample_weights = inv[y_train_new]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=workers, pin_memory=pin_mem,
            worker_init_fn=worker_init, generator=gen
        )
    else:
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
    n_total = int(num_classes)
    n_train = int(np.unique(y_train_new).size)
    n_val   = int(np.unique(y_val_new).size)
    n_test  = int(np.unique(y_test_new).size)

    print("\n[Splits]")
    print(f"  path   : {split_dir}")
    print(f"  classes: total_active={n_total}, train={n_train}, val={n_val}, test={n_test}")
    print(f"  names  : {active_class_names}")
    print(f"[Train class counts] {train_counts.tolist()}  (priors sum={train_priors.sum():.4f})")

    return train_loader, val_loader, test_loader, active_class_names, old_to_new, train_counts, train_priors


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

    # ---- Imbalance knobs ----
    use_class_balanced_sampler: bool = True      # WeightedRandomSampler
    sampler_power: float = 0.5                   # 0.5 ~ inverse sqrt(count)
    use_cb_focal: bool = True                    # enable CB-Focal after warmup
    beta_cb: float = 0.9999                      # CB weight beta
    gamma_focal: float = 2.0                     # focal gamma
    warmup_epochs: int = 5                       # epochs with plain CE before CB-Focal
    logit_adjust_tau: float = 0.0                # 0 disables; else add tau*log(prior) to logits


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

    train_loader, val_loader, test_loader, classes, _old2new, train_counts, train_priors = build_dataloaders_from_splits(
        root=data_root,
        split_output_dir=SPLIT_OUTPUT_DIR,
        persist_splits_dir=PERSIST_SPLITS_DIR,
        seed=SPLIT_SEED,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        balance_to_min=BALANCE_TO_MIN,
        balance_cap=BALANCE_CAP,
        threshold=THRESHOLD,
        weighted_sampler_power=(cfg.sampler_power if cfg.use_class_balanced_sampler else None),
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

    # ---- imbalance-aware loss objects ----
    cb_weights = effective_num_weights(train_counts, beta=cfg.beta_cb).to(device)
    cb_focal = CBFocalLoss(cb_weights, gamma=cfg.gamma_focal, reduction="mean").to(device)
    log_prior = torch.tensor(np.log(np.clip(train_priors, 1e-12, 1.0)), dtype=torch.float32, device=device)

    def apply_logit_adjustment(logits: torch.Tensor) -> torch.Tensor:
        if cfg.logit_adjust_tau and cfg.logit_adjust_tau > 0.0:
            return logits + cfg.logit_adjust_tau * log_prior
        return logits

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

        use_cb_stage = (cfg.use_cb_focal and epoch > cfg.warmup_epochs)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            logits = apply_logit_adjustment(logits)

            if use_cb_stage:
                loss = cb_focal(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

            with torch.no_grad():
                c_val = float(model.head.current_c().detach().cpu())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "c": f"{c_val:.4g}",
                              "stage": "CB-Focal" if use_cb_stage else "CE"})

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
                        "train_counts": train_counts,
                        "train_priors": train_priors.tolist(),
                        "cb_beta": cfg.beta_cb,
                        "focal_gamma": cfg.gamma_focal,
                    },
                    best_path,
                )

        print(f"Epoch {epoch}: stage={'CB-Focal' if use_cb_stage else 'CE'} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1_macro={val_f1_macro:.4f} "
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
                "train_counts": train_counts,
                "train_priors": train_priors.tolist(),
                "cb_beta": cfg.beta_cb,
                "focal_gamma": cfg.gamma_focal,
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


def ensure_csv_with_header(path: str, fieldnames: List[str]):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def run_sweep_and_save_csv(
    data_root: str,
    out_root: str,
    csv_path: str,
    grid: Dict[str, List[Any]],
    epochs: Optional[int] = None,
):
    os.makedirs(out_root, exist_ok=True)

    hp_keys = sorted(grid.keys())
    summary_fields = [
        "run_name", "timestamp_iso",
        "best_epoch", "best_val_acc", "test_acc", "test_f1_macro", "learned_c",
        "val_TP_total","val_FP_total","val_TN_weighted","val_FN_total",
        "val_weighted_sensitivity","val_weighted_specificity",
        "test_TP_total","test_FP_total","test_TN_weighted","test_FN_total",
        "test_weighted_sensitivity","test_weighted_specificity",
    ]
    fieldnames = hp_keys + summary_fields
    ensure_csv_with_header(csv_path, fieldnames)

    for params in iter_grid(grid):
        run_epochs = epochs if epochs is not None else params.get("epochs", 25)
        run_name = make_run_name(params)
        run_out_dir = os.path.join(out_root, run_name)

        cfg = FinetuneConfig(
            feature_dim=params["feature_dim"],
            init_curvature=params["init_curvature"],
            temperature=params["temperature"],
            batch_size=params["batch_size"],
            epochs=run_epochs,
            lr_backbone=params["lr_backbone"],
            lr_head=params["lr_head"],
            lr_curvature=params["lr_curvature"],
            img_size=params["img_size"],
            workers=4,
            seed=params["seed"],
            out_dir=run_out_dir,

            # ---- imbalance defaults for sweep ----
            use_class_balanced_sampler=False,
            sampler_power=0.5,
            use_cb_focal=True,
            beta_cb=0.9999,
            gamma_focal=1.0,
            warmup_epochs=8,
            logit_adjust_tau=0.0,   # try 0.5–1.0 if you want LA as well
        )

        print(f"\n=== Running: {run_name} ===")
        print("Using splits with:",
              f"seed={SPLIT_SEED}, t={TRAIN_FRAC}, v={VAL_FRAC}, s={cfg.img_size}, "
              f"balance_to_min={BALANCE_TO_MIN}, balance_cap={BALANCE_CAP}, threshold={THRESHOLD}")
        metrics = train(data_root=data_root, cfg=cfg)

        row = {k: params[k] for k in hp_keys}
        row.update({
            "run_name": run_name,
            "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
        })
        row.update(metrics)

        import csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(f"Logged to {csv_path}: {run_name}")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    param_grid = {
        "feature_dim":   [256],
        "init_curvature":[2.0],
        "temperature":   [1.0],
        "batch_size":    [256],
        "lr_backbone":   [1e-5, 3e-5, 1e-4],
        "lr_head":       [5e-3, 1e-2, 2e-2],
        "lr_curvature":  [1e-3, 3e-3, 5e-3],
        "img_size":      [224],
        "seed":          [42],
    }

    print("Starting hyperbolic classifier sweep (dynamic classes) with imbalance handling...")
    run_sweep_and_save_csv(
        data_root=DATA_ROOT,
        out_root=RUNS_DIR,
        csv_path=RESULTS_CSV,
        grid=param_grid,
        epochs=25,
    )
    print("Sweep finished.")
