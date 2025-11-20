"""CNN-based classifier pieces for Euclidean training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ClassificationHead(nn.Module):
    """Standard linear classification head."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.classifier(features)


class CNNBackbone(nn.Module):
    """ResNet18 backbone with a lightweight projection head."""

    def __init__(self, out_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)
        in_features = resnet.fc.in_features  # type: ignore[attr-defined]
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.encoder(x)
        return self.proj(features)


class CNNClassifier(nn.Module):
    """Complete classifier that combines the backbone and linear head."""

    def __init__(self, feature_dim: int, num_classes: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.backbone = CNNBackbone(out_dim=feature_dim, dropout_rate=dropout_rate)
        self.head = ClassificationHead(embedding_dim=feature_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(x)
        return self.head(features)


__all__ = ["ClassificationHead", "CNNBackbone", "CNNClassifier"]
