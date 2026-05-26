#!/usr/bin/env python3
"""Shared helpers for command-line execution of the training scripts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import csv

T = TypeVar("T")


def _load_json_like(argument: str) -> Any:
    """Load JSON either from an inline string or a file path."""
    candidate = Path(argument)
    if candidate.exists():
        payload = candidate.read_text(encoding="utf-8")
    else:
        payload = argument
    return json.loads(payload)


def load_dict_argument(argument: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a dictionary parsed from ``argument`` if provided."""
    if not argument:
        return None
    data = _load_json_like(argument)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object for configuration arguments")
    return data


def parse_grid_argument(
    argument: Optional[str],
    default: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    """Return a hyper-parameter grid described by ``argument`` or ``default``."""
    if not argument:
        return default
    grid = load_dict_argument(argument)
    if grid is None:
        return default
    validated: Dict[str, List[Any]] = {}
    for key, values in grid.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Grid entry '{key}' must be a list, received {type(values).__name__}"
            )
        validated[key] = values
    return validated


def build_config_dict(
    config_cls: Type[T],
    *,
    config_arg: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a dictionary with defaults merged with CLI overrides."""
    base = asdict(config_cls())
    if config_arg:
        override_from_arg = load_dict_argument(config_arg)
        if override_from_arg:
            base.update(override_from_arg)
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return base


def instantiate_config(
    config_cls: Type[T],
    *,
    config_arg: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> T:
    """Instantiate ``config_cls`` after applying overrides."""
    cfg_dict = build_config_dict(config_cls, config_arg=config_arg, overrides=overrides)
    return config_cls(**cfg_dict)


def ensure_csv_with_header(path: str, fieldnames: Iterable[str]) -> None:
    """Create ``path`` if needed and write the header once."""
    if os.path.exists(path):
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()


def append_row_to_csv(path: str, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    """Append a row to ``path`` ensuring the header exists."""
    ensure_csv_with_header(path, fieldnames)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writerow(row)


def optional_int(value: str) -> Optional[int]:
    """``argparse`` helper that allows integers or textual ``None``."""
    normalized = value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return int(value)
