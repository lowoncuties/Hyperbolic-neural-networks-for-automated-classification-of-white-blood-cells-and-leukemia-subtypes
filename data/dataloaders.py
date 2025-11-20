"""Dataset and DataLoader helpers shared by the training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.reproducibility import make_generator, make_worker_init_fn

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def is_valid_file(path: str, allowed_exts = ALLOWED_EXTS) -> bool:
    return Path(path).suffix.lower() in allowed_exts


def resolve_split_dir(
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

    return split_base / f"seed_{seed}_t{train_frac}_v{val_frac}_s{img_size}_{bal_tag}_{thr_tag}"


def _load_indices(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _load(name: str) -> np.ndarray:
        p = split_dir / f"{name}_idx.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        return np.array(json.loads(p.read_text()), dtype=int)

    train_idx = _load("train")
    val_idx = _load("val")
    test_idx = _load("test")
    return train_idx, val_idx, test_idx


def build_transforms(img_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    t_eval = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return t_train, t_eval


def build_dataloaders_from_splits(
    root: str,
    split_output_dir: str,
    persist_splits_dir: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    img_size: int,
    batch_size: int,
    workers: int,
    balance_to_min: bool,
    balance_cap: Optional[int],
    threshold: Optional[int],
    allowed_exts = ALLOWED_EXTS,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[int, int]]:
    base = datasets.ImageFolder(root=root, is_valid_file=lambda p: is_valid_file(p, allowed_exts))
    idx_to_name = {v: k for k, v in base.class_to_idx.items()}

    split_dir = resolve_split_dir(
        root,
        split_output_dir,
        persist_splits_dir,
        seed,
        train_frac,
        val_frac,
        img_size,
        balance_to_min,
        balance_cap,
        threshold,
    )
    train_idx, val_idx, test_idx = _load_indices(split_dir)

    all_idx = np.unique(np.concatenate([train_idx, val_idx, test_idx]))
    targets_all = np.array(base.targets)
    active_old_ids = sorted(set(int(t) for t in targets_all[all_idx]))

    old_to_new = {old: new for new, old in enumerate(active_old_ids)}
    new_to_old = {v: k for k, v in old_to_new.items()}
    active_class_names = [idx_to_name[new_to_old[i]] for i in range(len(active_old_ids))]

    t_train, t_eval = build_transforms(img_size)
    target_transform = lambda y: old_to_new[int(y)]

    def _dataset(transform):
        return datasets.ImageFolder(
            root=root,
            transform=transform,
            is_valid_file=lambda p: is_valid_file(p, allowed_exts),
            target_transform=target_transform,
        )

    ds_train_full = _dataset(t_train)
    ds_val_full = _dataset(t_eval)
    ds_test_full = _dataset(t_eval)

    ds_train = Subset(ds_train_full, train_idx.tolist())
    ds_val = Subset(ds_val_full, val_idx.tolist())
    ds_test = Subset(ds_test_full, test_idx.tolist())

    worker_init = make_worker_init_fn(seed)
    gen = make_generator(seed)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init,
        generator=gen,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init,
    )

    y_train_old = targets_all[train_idx]
    y_val_old = targets_all[val_idx]
    y_test_old = targets_all[test_idx]
    y_train_new = np.array([old_to_new[int(y)] for y in y_train_old])
    y_val_new = np.array([old_to_new[int(y)] for y in y_val_old])
    y_test_new = np.array([old_to_new[int(y)] for y in y_test_old])

    n_total = len(active_class_names)
    n_train = int(np.unique(y_train_new).size)
    n_val = int(np.unique(y_val_new).size)
    n_test = int(np.unique(y_test_new).size)

    print("\n[Splits]")
    print(f"  path   : {split_dir}")
    print(f"  classes: total_active={n_total}, train={n_train}, val={n_val}, test={n_test}")
    print(f"  names  : {active_class_names}")

    return train_loader, val_loader, test_loader, active_class_names, old_to_new


__all__ = [
    "ALLOWED_EXTS",
    "is_valid_file",
    "resolve_split_dir",
    "build_transforms",
    "build_dataloaders_from_splits",
]
