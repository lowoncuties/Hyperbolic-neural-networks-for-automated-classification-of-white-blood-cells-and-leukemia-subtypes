"""Helpers for discovering and loading packaged trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .models import CNNClassifier, HyperbolicClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models"

MODEL_ALIASES = {
    "euclid": "euclid",
    "euclidean": "euclid",
    "cnn": "euclid",
    "hyperbolic": "hyperbolic",
    "hyp": "hyperbolic",
}


def canonical_model_name(name: str) -> str:
    """Normalize a user-facing model name to the artifact folder name."""

    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{name}'. Expected one of: {known}")
    return MODEL_ALIASES[key]


def model_dir(name: str, repo_root: Optional[str | Path] = None) -> Path:
    """Return the artifact directory for a packaged model."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / "models" / canonical_model_name(name)


def metadata_path(name: str, repo_root: Optional[str | Path] = None) -> Path:
    """Return the metadata JSON path for a packaged model."""

    return model_dir(name, repo_root) / "metadata.json"


def checkpoint_path(name: str, repo_root: Optional[str | Path] = None) -> Path:
    """Return the checkpoint path for a packaged model."""

    metadata = load_metadata(name, repo_root)
    return model_dir(name, repo_root) / metadata.get("checkpoint_file", "best.pt")


def load_metadata(name: str, repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the metadata stored next to a packaged checkpoint."""

    path = metadata_path(name, repo_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_models(repo_root: Optional[str | Path] = None) -> Dict[str, Dict[str, Any]]:
    """Return metadata for all packaged models with a metadata file."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    artifacts: Dict[str, Dict[str, Any]] = {}
    for path in sorted((root / "models").glob("*/metadata.json")):
        artifacts[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def build_model_from_metadata(metadata: Dict[str, Any]):
    """Instantiate the architecture described by a packaged model metadata file."""

    config = metadata.get("config", {})
    classes = metadata.get("classes", [])
    num_classes = int(metadata.get("num_classes") or len(classes))
    model_type = canonical_model_name(metadata["model_type"])

    if model_type == "euclid":
        return CNNClassifier(
            feature_dim=int(config.get("feature_dim", metadata.get("feature_dim", 256))),
            num_classes=num_classes,
            dropout_rate=float(config.get("dropout_rate", 0.2)),
        )

    return HyperbolicClassifier(
        feature_dim=int(config.get("feature_dim", metadata.get("feature_dim", 256))),
        num_classes=num_classes,
        init_curvature=float(config.get("init_curvature", 2.0)),
        temperature=float(config.get("temperature", 1.0)),
        learnable_temperature=bool(config.get("learnable_temperature", True)),
        feature_clip_norm=config.get("feature_clip_norm"),
    )


def load_model(
    name: str,
    repo_root: Optional[str | Path] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Load a packaged checkpoint and return ``(model, checkpoint, metadata)``."""

    import torch

    metadata = load_metadata(name, repo_root)
    checkpoint = torch.load(
        checkpoint_path(name, repo_root),
        map_location=map_location,
    )
    model = build_model_from_metadata(metadata)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    model.eval()
    return model, checkpoint, metadata


__all__ = [
    "available_models",
    "build_model_from_metadata",
    "canonical_model_name",
    "checkpoint_path",
    "load_metadata",
    "load_model",
    "metadata_path",
    "model_dir",
]
