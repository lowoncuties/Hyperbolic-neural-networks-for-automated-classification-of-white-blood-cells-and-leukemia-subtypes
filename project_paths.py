"""Shared default paths for datasets, splits, and training outputs."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_or_repo_path(env_name: str, *parts: str) -> str:
    """Return an environment override or a repo-local default path."""
    value = os.environ.get(env_name)
    if value:
        return value
    return str(REPO_ROOT.joinpath(*parts))


DEFAULT_DATA_ROOT = _env_or_repo_path(
    "WBC_DATA_ROOT", "datasets", "WBC_Our_dataset_extended"
)
DEFAULT_LEGACY_DATA_ROOT = _env_or_repo_path(
    "WBC_LEGACY_DATA_ROOT", "datasets", "WBC_Our_dataset"
)
DEFAULT_SPLIT_OUTPUT_DIR = _env_or_repo_path("WBC_SPLIT_OUTPUT_DIR", "outputs")
DEFAULT_PERSIST_SPLITS_DIR = os.environ.get("WBC_PERSIST_SPLITS_DIR", "splits")

DEFAULT_HYPERBOLIC_RUNS_DIR = _env_or_repo_path(
    "WBC_HYPERBOLIC_RUNS_DIR", "outputs", "runs", "hyperbolic"
)
DEFAULT_HYPERBOLIC_RESULTS_CSV = _env_or_repo_path(
    "WBC_HYPERBOLIC_RESULTS_CSV",
    "outputs",
    "results",
    "hyperbolic_summary.csv",
)

DEFAULT_CNN_RUNS_DIR = _env_or_repo_path(
    "WBC_CNN_RUNS_DIR", "outputs", "runs", "cnn"
)
DEFAULT_CNN_RESULTS_CSV = _env_or_repo_path(
    "WBC_CNN_RESULTS_CSV", "outputs", "results", "cnn_summary.csv"
)

DEFAULT_WEIGHTED_HYPERBOLIC_RUNS_DIR = _env_or_repo_path(
    "WBC_WEIGHTED_HYPERBOLIC_RUNS_DIR", "outputs", "runs", "hyperbolic_weighted"
)
DEFAULT_WEIGHTED_HYPERBOLIC_RESULTS_CSV = _env_or_repo_path(
    "WBC_WEIGHTED_HYPERBOLIC_RESULTS_CSV",
    "outputs",
    "results",
    "hyperbolic_weighted_summary.csv",
)
