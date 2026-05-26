#!/usr/bin/env python3
"""Create acquisition-aware train/validation/test splits for a WBC ImageFolder.

The merged WBC dataset used for the paper contains repeated images from the
same recoverable acquisition in several filename patterns. This script assigns
those linked images to a shared group before splitting, so related files do not
appear in different train/validation/test partitions.

The resulting JSON files are compatible with the training scripts in this
repository:

    train_idx.json
    val_idx.json
    test_idx.json

The grouping is acquisition-aware, not fully patient-level, because patient
identifiers are not consistently recoverable from every merged source dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_OUTPUT_DIR


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
STACK_RE = re.compile(r"^(?P<img_num>\d+)_(?P<z>\d+)$")
VARIANT_RE = re.compile(r"^.+_\d+_\d+$")
DEFAULT_GROUPED_SPLITS_DIR = "splits_grouped_acq_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped split JSON files for the WBC ImageFolder dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_DATA_ROOT,
        help="Merged ImageFolder dataset root.",
    )
    parser.add_argument(
        "--split-output-dir",
        default=DEFAULT_SPLIT_OUTPUT_DIR,
        help="Base directory where persisted split indices will be written.",
    )
    parser.add_argument(
        "--persist-splits-dir",
        default=DEFAULT_GROUPED_SPLITS_DIR,
        help="Subdirectory used to store grouped split indices.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Recorded in the split directory name for trainer compatibility.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=110,
        help="Keep classes with count > threshold. Use -1 to disable filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the grouped split without writing files.",
    )
    return parser.parse_args()


def resolve_split_dir(
    root: str,
    split_output_dir: str,
    persist_splits_dir: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    img_size: int,
    threshold: int | None,
) -> Path:
    dataset_name = Path(root).expanduser().resolve().name
    split_base = Path(split_output_dir) / persist_splits_dir / dataset_name
    bal_tag = "balmin0"
    thr_tag = "thrnone" if threshold is None else f"thrgt{int(threshold)}"
    return split_base / f"seed_{seed}_t{train_frac}_v{val_frac}_s{img_size}_{bal_tag}_{thr_tag}"


def scan_imagefolder(root: str) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    classes = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class directories found under: {root_path}")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: List[Dict[str, object]] = []

    for class_name in classes:
        class_dir = root_path / class_name
        for dirpath, dirnames, filenames in os.walk(str(class_dir), followlinks=True):
            dirnames.sort()
            for fname in sorted(filenames):
                path = Path(dirpath) / fname
                if path.suffix.lower() not in ALLOWED_EXTS:
                    continue
                samples.append(
                    {
                        "index": len(samples),
                        "path": str(path),
                        "class_name": class_name,
                        "class_idx": class_to_idx[class_name],
                        "stem": path.stem,
                    }
                )

    return samples, class_to_idx


def infer_group(stem: str, class_name: str) -> Tuple[str, str]:
    """Return a class-local acquisition group id and its grouping kind."""

    stack_match = STACK_RE.fullmatch(stem)
    if stack_match is not None:
        z_index = int(stack_match.group("z"))
        if 0 <= z_index <= 9:
            return f"{class_name}::stack::{stack_match.group('img_num')}", "stack"

    if VARIANT_RE.fullmatch(stem):
        return f"{class_name}::variant::{stem.rsplit('_', 1)[0]}", "variant"

    return f"{class_name}::single::{stem}", "single"


def keep_by_threshold(
    samples: List[Dict[str, object]],
    threshold: int | None,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[str]]:
    class_counts = Counter(sample["class_name"] for sample in samples)
    if threshold is None:
        return samples, dict(class_counts), []

    kept_classes = {name for name, count in class_counts.items() if count > threshold}
    dropped_classes = sorted(
        name for name, count in class_counts.items() if count <= threshold
    )
    filtered = [sample for sample in samples if sample["class_name"] in kept_classes]
    kept_counts = Counter(sample["class_name"] for sample in filtered)
    return filtered, dict(kept_counts), dropped_classes


def build_groups(
    samples: Iterable[Dict[str, object]],
) -> Tuple[Dict[str, List[int]], Dict[str, str], Dict[str, Counter]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    group_kind: Dict[str, str] = {}
    kind_by_class: Dict[str, Counter] = defaultdict(Counter)

    for sample in samples:
        class_name = str(sample["class_name"])
        stem = str(sample["stem"])
        index = int(sample["index"])

        group_id, kind = infer_group(stem, class_name)
        groups[group_id].append(index)
        group_kind[group_id] = kind
        kind_by_class[class_name][kind] += 1

    return groups, group_kind, kind_by_class


def split_group_items_for_class(
    group_items: List[Tuple[str, List[int]]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> Dict[str, List[int]]:
    total = sum(len(indices) for _, indices in group_items)
    targets = {
        "train": train_frac * total,
        "val": val_frac * total,
        "test": test_frac * total,
    }

    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    assigned: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    priority = {"train": 0, "val": 1, "test": 2}

    for _, indices in group_items:
        size = len(indices)
        best_split = None
        best_score = None

        for split_name in ("train", "val", "test"):
            after = counts.copy()
            after[split_name] += size

            squared_error = sum((after[name] - targets[name]) ** 2 for name in after)
            overflow = sum(max(0.0, after[name] - targets[name]) for name in after)
            score = (squared_error, overflow, priority[split_name])

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None
        assigned[best_split].extend(indices)
        counts[best_split] += size

    return assigned


def grouped_stratified_split(
    samples: List[Dict[str, object]],
    groups: Dict[str, List[int]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int], Dict[str, Dict[str, int]]]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    sample_by_index = {int(sample["index"]): sample for sample in samples}
    class_to_group_items: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)

    for group_id, indices in groups.items():
        if not indices:
            continue
        class_name = str(sample_by_index[indices[0]]["class_name"])
        class_to_group_items[class_name].append((group_id, indices))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    class_split_counts: Dict[str, Dict[str, int]] = {}

    for class_name in sorted(class_to_group_items):
        rng = random.Random(f"{seed}:{class_name}")
        assigned = split_group_items_for_class(
            group_items=list(class_to_group_items[class_name]),
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            rng=rng,
        )

        train_idx.extend(assigned["train"])
        val_idx.extend(assigned["val"])
        test_idx.extend(assigned["test"])
        class_split_counts[class_name] = {
            "train": len(assigned["train"]),
            "val": len(assigned["val"]),
            "test": len(assigned["test"]),
        }

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx, class_split_counts


def summarize_group_leakage(
    groups: Dict[str, List[int]],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> Dict[str, int]:
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    leaking_groups = 0
    for indices in groups.values():
        present = 0
        if any(idx in train_set for idx in indices):
            present += 1
        if any(idx in val_set for idx in indices):
            present += 1
        if any(idx in test_set for idx in indices):
            present += 1
        if present > 1:
            leaking_groups += 1

    return {"total_groups": len(groups), "groups_with_leakage": leaking_groups}


def main() -> None:
    args = parse_args()
    threshold = None if args.threshold < 0 else args.threshold

    samples_all, class_to_idx = scan_imagefolder(args.root)
    samples, kept_counts, dropped_classes = keep_by_threshold(samples_all, threshold)
    if not samples:
        raise SystemExit("No samples remain after thresholding.")

    groups, group_kind, kind_by_class = build_groups(samples)
    train_idx, val_idx, test_idx, class_split_counts = grouped_stratified_split(
        samples=samples,
        groups=groups,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    leakage_summary = summarize_group_leakage(groups, train_idx, val_idx, test_idx)
    kind_counts = Counter(group_kind.values())
    split_dir = resolve_split_dir(
        root=args.root,
        split_output_dir=args.split_output_dir,
        persist_splits_dir=args.persist_splits_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        img_size=args.img_size,
        threshold=threshold,
    )

    meta = {
        "grouping_policy": "acquisition_aware_v1",
        "notes": [
            "stack groups: class-local numeric names like 123_0.jpg ... 123_9.jpg",
            "variant groups: class-local names sharing the same base before the final numeric suffix",
            "singleton groups: all remaining filenames",
            "patient-level separation is not guaranteed for all source datasets because merged filenames do not preserve patient ids everywhere",
        ],
        "root": str(Path(args.root).expanduser().resolve()),
        "split_dir": str(split_dir),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "img_size": args.img_size,
        "threshold": threshold,
        "class_to_idx": class_to_idx,
        "kept_class_counts": kept_counts,
        "dropped_classes": dropped_classes,
        "group_kind_counts": dict(kind_counts),
        "group_kind_counts_by_class": {
            name: dict(counter) for name, counter in kind_by_class.items()
        },
        "class_split_counts": class_split_counts,
        "leakage_check": leakage_summary,
    }

    print("[Grouped Split Generation]")
    print(f"  root              : {args.root}")
    print(f"  split dir         : {split_dir}")
    print(f"  threshold         : {threshold}")
    print(f"  kept classes      : {len(kept_counts)}")
    print(f"  dropped classes   : {dropped_classes if dropped_classes else '[]'}")
    print(f"  total samples     : {len(samples)}")
    print(f"  total groups      : {len(groups)}")
    print(f"  kind counts       : {dict(kind_counts)}")
    print(
        "  split sizes       : "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    print(f"  leakage check     : {leakage_summary}")

    if args.dry_run:
        print("  dry run           : no files written")
        return

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_idx.json").write_text(json.dumps(train_idx), encoding="utf-8")
    (split_dir / "val_idx.json").write_text(json.dumps(val_idx), encoding="utf-8")
    (split_dir / "test_idx.json").write_text(json.dumps(test_idx), encoding="utf-8")
    (split_dir / "grouped_split_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        "  wrote             : "
        "train_idx.json, val_idx.json, test_idx.json, grouped_split_meta.json"
    )


if __name__ == "__main__":
    main()
