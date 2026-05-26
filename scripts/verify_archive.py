#!/usr/bin/env python3
"""Validate public-release hygiene for the code archive."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024

REQUIRED_PATHS = [
    "README.md",
    "MANIFEST.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    "configs/cnn_grouped_grid.json",
    "configs/cnn_grouped_single.json",
    "configs/hyperbolic_grouped_grid.json",
    "configs/hyperbolic_grouped_single.json",
    "data/dataloaders.py",
    "scripts/cnn_fine_tuned.py",
    "scripts/hyperbolic_cnn_fine_tuned.py",
    "scripts/make_grouped_splits.py",
    "scripts/verify_archive.py",
    "src/wbc_classification/model_zoo.py",
    "src/wbc_classification/models/classic_cnn.py",
    "src/wbc_classification/models/hyperbolic_cnn.py",
    "models/README.md",
    "models/euclid/best.pt",
    "models/euclid/metadata.json",
    "models/euclid/test_classification_report.csv",
    "models/hyperbolic/best.pt",
    "models/hyperbolic/metadata.json",
    "models/hyperbolic/test_classification_report.csv",
    "docs/reproducibility.md",
    "docs/model_cards.md",
]

FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "datasets",
    "outputs",
    "runs",
    "wandb",
}

FORBIDDEN_PATHS = {
    "CITATION" + ".cff",
    "WBC_" + "dataloader.ipynb",
    "scripts/cnn_fine_tuned_" + "new.py",
    "scripts/hyperbolic_cnn_fine_tuned_" + "weight_class.py",
    "scripts/hyperbolic_cnn_" + "learnable_temp.py",
}

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}

LOCAL_USER_TOKEN = "jo" + "c0027"
LOCAL_WORKSPACE_TOKEN = "J" + "YOT"

FORBIDDEN_TEXT_PATTERNS = [
    re.compile("/" + "data" + r"\d?/"),
    re.compile("/" + "home" + "/"),
    re.compile(LOCAL_USER_TOKEN),
    re.compile(r"\b" + LOCAL_WORKSPACE_TOKEN + r"\b"),
]


def iter_repo_files() -> Iterable[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(part in FORBIDDEN_DIRS for part in rel_parts):
            continue
        yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_required_paths(errors: list[str]) -> None:
    for rel in REQUIRED_PATHS:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required path: {rel}")


def check_forbidden_files(errors: list[str]) -> None:
    forbidden_suffixes = {".ipynb", ".pyc", ".pyo", ".log"}
    for path in iter_repo_files():
        rel = path.relative_to(REPO_ROOT)
        rel_posix = rel.as_posix()
        if rel_posix in FORBIDDEN_PATHS:
            errors.append(f"forbidden non-current file: {rel_posix}")
        if path.suffix in forbidden_suffixes:
            errors.append(f"forbidden generated file: {rel}")
        if path.name in {".DS_Store"}:
            errors.append(f"forbidden system file: {rel}")
        if path.stat().st_size > MAX_GITHUB_FILE_BYTES:
            errors.append(f"file exceeds GitHub 100 MB limit: {rel}")


def check_json(errors: list[str]) -> None:
    for path in iter_repo_files():
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            rel = path.relative_to(REPO_ROOT)
            errors.append(f"invalid JSON in {rel}: {exc}")


def check_hard_paths(errors: list[str]) -> None:
    for path in iter_repo_files():
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            match = pattern.search(text)
            if match:
                rel = path.relative_to(REPO_ROOT)
                errors.append(f"hard/local path token '{match.group(0)}' in {rel}")
                break


def check_model_metadata(errors: list[str]) -> None:
    for model_name in ("euclid", "hyperbolic"):
        model_dir = REPO_ROOT / "models" / model_name
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = model_dir / metadata.get("checkpoint_file", "best.pt")
        if not checkpoint.exists():
            errors.append(f"missing checkpoint for {model_name}: {checkpoint}")
            continue
        expected = metadata.get("sha256")
        actual = sha256(checkpoint)
        if expected != actual:
            errors.append(
                f"checksum mismatch for {checkpoint.relative_to(REPO_ROOT)}: "
                f"expected {expected}, got {actual}"
            )
        if int(metadata.get("num_classes", 0)) != len(metadata.get("classes", [])):
            errors.append(f"num_classes/classes mismatch in {metadata_path}")


def main() -> int:
    errors: list[str] = []
    check_required_paths(errors)
    check_forbidden_files(errors)
    check_json(errors)
    check_hard_paths(errors)
    check_model_metadata(errors)

    if errors:
        print("Archive verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Archive verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
