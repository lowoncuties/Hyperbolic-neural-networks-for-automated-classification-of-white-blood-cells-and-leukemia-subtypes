#!/usr/bin/env python3
"""
Euclidean Distance-Based Classifier for Leukocyte Detection and Classification

This script converts the hyperbolic classifier to use classical Euclidean
distance for inference while maintaining all data loaders, random freezes,
and reproducibility.
"""

import csv
import itertools
from dataclasses import dataclass
import os
from typing import Tuple, List, Optional
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix


# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_global_seed(seed: int = 42, deterministic: bool = True):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """Create worker initialization function for DataLoader."""

    def _init_fn(worker_id: int):
        s = base_seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded random generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------
# Metrics: sensitivity/specificity from CM
# ----------------------------
def compute_sensitivity_specificity_multiclass(cm: np.ndarray):
    """
    Computes sensitivity and specificity per class from a multi-class
    confusion matrix.

    Returns:
        TP_total, FP_total, weighted_TN, FN_total, weighted_sensitivity,
        weighted_specificity
    """
    num_classes = cm.shape[0]
    support = cm.sum(axis=1)  # count of samples of each class
    total_samples = support.sum()  # total_samples = cm.sum()

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

        sensitivity[i] = (
            tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0.0
        )
        specificity[i] = (
            tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) != 0 else 0.0
        )

    TP_total = int(np.trace(cm))
    FP_total = int((cm.sum(axis=0) - np.diag(cm)).sum())
    FN_total = int((cm.sum(axis=1) - np.diag(cm)).sum())
    weighted_TN = (
        float(np.sum(tn * support) / total_samples)
        if total_samples != 0
        else 0.0
    )
    weighted_sensitivity = (
        float(np.sum(sensitivity * support) / total_samples)
        if total_samples != 0
        else 0.0
    )
    weighted_specificity = (
        float(np.sum(specificity * support) / total_samples)
        if total_samples != 0
        else 0.0
    )

    return (
        TP_total,
        FP_total,
        weighted_TN,
        FN_total,
        weighted_sensitivity,
        weighted_specificity,
    )


def f1_macro_from_preds_labels(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> float:
    """
    Macro-F1 without sklearn. Handles empty-class safely.
    """
    if preds.numel() == 0:
        return 0.0
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


# ----------------------------
# Model components
# ----------------------------
class ClassificationHead(nn.Module):
    """
    Standard linear classification head.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute logits using standard linear layer.

        Args:
            features: (B, D) feature vectors

        Returns:
            logits: (B, K) logits for each class
        """
        return self.classifier(features)


class CNNBackbone(nn.Module):
    """
    Simple CNN backbone using ResNet18.
    """

    def __init__(self, out_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        # Use ResNet18 as base
        resnet = models.resnet18(weights=None)

        # Remove the final classification layer and get the feature dimension
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.encoder = resnet

        # Simple projection layer
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from ResNet (B, 512)
        features = self.encoder(x)

        # Project to final dimension
        return self.proj(features)  # (B, out_dim)


class CNNClassifier(nn.Module):
    """
    Complete CNN-based classifier with standard linear head.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = CNNBackbone(
            out_dim=feature_dim, dropout_rate=dropout_rate
        )
        self.head = ClassificationHead(
            embedding_dim=feature_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ----------------------------
# Data loading
# ----------------------------
def build_dataloaders(
    root: str,
    img_size: int = 224,
    batch_size: int = 64,
    workers: int = 4,
    seed: int = 42,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    List[str],
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    """Build train, validation, and test data loaders with augmentation.

    If `root` contains train/val/test subfolders, use them.
    Otherwise, assume a single folder-of-classes layout and create a
    reproducible 70/15/15 stratified split.
    """
    import numpy as np

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    has_split_dirs = all(os.path.isdir(p) for p in (train_dir, val_dir, test_dir))

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Evaluation transforms (no augmentation)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    worker_init = make_worker_init_fn(seed)
    train_gen = make_generator(seed)

    if has_split_dirs:
        # ---- original behavior
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        classes = train_ds.classes

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=worker_init,
            generator=train_gen,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=worker_init,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=worker_init,
        )
        return train_loader, val_loader, test_loader, classes, val_ds, test_ds

    # ---- fallback: single root with class subfolders (your WBC layout)
    base = datasets.ImageFolder(root, transform=None)  # no transform yet
    classes = base.classes
    targets = np.array(base.targets)
    n_classes = len(classes)

    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(targets))

    train_idx, val_idx, test_idx = [], [], []
    for c in range(n_classes):
        cls_idx = idx_all[targets == c]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        # 70/15/15 split per class (reproducible)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_train = min(n_train, n)  # safety
        n_val = min(n_val, n - n_train)
        # Remaining go to test
        t, v, te = np.split(cls_idx, [n_train, n_train + n_val])
        train_idx.extend(t.tolist())
        val_idx.extend(v.tolist())
        test_idx.extend(te.tolist())

    # Build three datasets that share the same samples but have different transforms
    train_ds_full = datasets.ImageFolder(root, transform=train_tf)
    val_ds_full = datasets.ImageFolder(root, transform=eval_tf)
    test_ds_full = datasets.ImageFolder(root, transform=eval_tf)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)
    test_ds = Subset(test_ds_full, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    return train_loader, val_loader, test_loader, classes, val_ds, test_ds


def make_fixed_random_eval_subset(
    ds, batch_size: int, max_batches: Optional[int], seed: int
) -> Optional[DataLoader]:
    """Create a fixed random subset for evaluation."""
    if max_batches is None:
        return None
    n = min(len(ds), max_batches * batch_size)
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=False)


# ----------------------------
# Evaluation helpers
# ----------------------------
@torch.no_grad()
def evaluate_with_cm(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
):
    """Evaluate model and return accuracy and confusion matrix."""
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
    return acc, cm


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

    train_loader, val_loader, test_loader, classes, _, _ = build_dataloaders(
        data_root,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        seed=cfg.seed,
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
        val_acc_cm, val_cm = evaluate_with_cm(
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
    test_loss, test_acc, preds, labs = evaluate_with_loss(
        model, test_loader, device
    )
    test_f1 = f1_macro_from_preds_labels(preds, labs, num_classes)

    # Get confusion matrix and detailed metrics
    test_acc_cm, test_cm = evaluate_with_cm(
        model, test_loader, device, num_classes
    )
    test_TP, test_FP, test_TN, test_FN, test_wsens, test_wspec = (
        compute_sensitivity_specificity_multiclass(test_cm)
    )

    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(
        f"Test TP: {test_TP}, FP: {test_FP}, TN: {test_TN:.4f}, FN: {test_FN}"
    )
    print(f"Test Weighted Sensitivity: {test_wsens:.4f}")
    print(f"Test Weighted Specificity: {test_wspec:.4f}")

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
    grid: dict,
):
    """
    Run hyperparameter sweep and save results to CSV.

    Args:
        data_root: Path to dataset
        out_root: Output directory for runs
        csv_path: Path to save CSV results
        grid: Dictionary of parameter_name -> list of values
    """
    os.makedirs(out_root, exist_ok=True)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    # CSV header
    fieldnames = [
        "feature_dim",
        "batch_size",
        "learning_rate",
        "img_size",
        "seed",
        "dropout_rate",
        "weight_decay",
        "best_epoch",
        "stopped_due_to_overfit",
        "gen_gap_val_train_loss",
        "test_loss",
        "test_acc",
        "test_f1_macro",
        # Detailed validation metrics
        "val_TP_total",
        "val_FP_total",
        "val_TN_weighted",
        "val_FN_total",
        "val_weighted_sensitivity",
        "val_weighted_specificity",
        # Detailed test metrics
        "test_TP_total",
        "test_FP_total",
        "test_TN_weighted",
        "test_FN_total",
        "test_weighted_sensitivity",
        "test_weighted_specificity",
    ]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for i, combo in enumerate(combos, 1):
            cfg_kwargs = {k: v for k, v in zip(keys, combo)}
            # Per-run output dir (optional, can comment out)
            # FIX: iterate (k, v) pairs correctly
            run_name = "_".join(f"{k}={str(v)}" for k, v in cfg_kwargs.items())
            cfg = FinetuneConfig(
                **cfg_kwargs, out_dir=os.path.join(out_root, run_name)
            )

            print(f"\n=== [{i}/{len(combos)}] Running: {run_name} ===")
            result = train_one_config(data_root, cfg)

            # Write to CSV
            row = {
                **{
                    k: getattr(cfg, k)
                    for k in [
                        "feature_dim",
                        "batch_size",
                        "learning_rate",
                        "img_size",
                        "seed",
                        "dropout_rate",
                        "weight_decay",
                    ]
                },
                **{
                    k: result[k]
                    for k in [
                        "best_epoch",
                        "stopped_due_to_overfit",
                        "gen_gap_val_train_loss",
                        "test_loss",
                        "test_acc",
                        "test_f1_macro",
                        # Detailed validation metrics
                        "val_TP_total",
                        "val_FP_total",
                        "val_TN_weighted",
                        "val_FN_total",
                        "val_weighted_sensitivity",
                        "val_weighted_specificity",
                        # Detailed test metrics
                        "test_TP_total",
                        "test_FP_total",
                        "test_TN_weighted",
                        "test_FN_total",
                        "test_weighted_sensitivity",
                        "test_weighted_specificity",
                    ]
                },
            }
            writer.writerow(row)
            f.flush()
            print("Wrote results to", csv_path)


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Paths
    PROJECT_ROOT = (
        Path(__file__).parent if "__file__" in globals() else Path.cwd()
    )

    # ✅ Your dataset with 13 class folders
    main_data_root = Path("/data3/datasets/WBC_Our_dataset").resolve()

    # ✅ Writeable locations for models & CSV
    sweep_out_root = "/data2/joc0027/venv/JYOT/cnn_sweep_runs_donwsampled"
    results_csv = "/data2/joc0027/venv/JYOT/cnn_sweep_results_wbc_downsampled.csv"

    # Optimized parameter grid for CNN classifier
    param_grid = {
        "feature_dim": [64, 128, 256],
        "batch_size": [512],
        "learning_rate": [1e-4, 1e-3, 5e-3],
        "img_size": [224],
        "seed": [42],
        "dropout_rate": [0.2],
        "weight_decay": [1e-4],
    }

    print("Starting CNN classifier sweep...")
    print(f"Data root: {main_data_root}")
    print(f"Output root: {sweep_out_root}")
    print(f"Results CSV: {results_csv}")

    run_sweep_and_save_csv(
        data_root=str(main_data_root),
        out_root=sweep_out_root,
        csv_path=results_csv,
        grid=param_grid,
    )
    print("Sweep finished. CSV at:", results_csv)
