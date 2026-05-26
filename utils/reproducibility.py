"""Utility helpers to keep training runs reproducible."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across numpy, torch and python."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Return a worker init fn that offsets the base seed per worker."""

    def _init_fn(worker_id: int) -> None:
        s = base_seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator seeded with the provided value."""

    g = torch.Generator()
    g.manual_seed(seed)
    return g


__all__ = ["set_global_seed", "make_worker_init_fn", "make_generator"]
