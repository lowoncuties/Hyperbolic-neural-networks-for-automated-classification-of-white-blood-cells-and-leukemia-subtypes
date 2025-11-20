"""Model components for the hyperbolic prototype classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HyperbolicPrototypeHead(nn.Module):
    """Hyperbolic prototype head operating on Euclidean features."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        init_curvature: float = 1.0,
        temperature: float = 1.0,
        min_c: float = 1e-4,
        max_c: float = 1e4,
        proto_init_std: float = 1e-3,
    ) -> None:
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
        self.proto_tan = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * proto_init_std
        )

    def current_c(self) -> torch.Tensor:
        """Return the current curvature value clipped to the configured range."""

        c = F.softplus(self.raw_c) + self._eps_c
        return torch.clamp(c, self.min_c, self.max_c)

    def forward(self, euclidean_feats: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        device = euclidean_feats.device
        c = self.current_c().to(device)
        k = -c

        z = self.hypmath.expmap0(euclidean_feats, k=k, dim=-1)
        p = self.hypmath.expmap0(self.proto_tan.to(device), k=k, dim=-1)
        d = self.hypmath.dist(z.unsqueeze(1), p.unsqueeze(0), k=k, dim=-1)
        logits = -d / self.temperature
        return logits


class Backbone(nn.Module):
    """Simple ResNet18 backbone with a projection head."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        m = models.resnet18(weights=None)
        in_dim = m.fc.in_features  # type: ignore[attr-defined]
        m.fc = nn.Identity()
        self.encoder = m
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.proj(self.encoder(x))


class HyperbolicClassifier(nn.Module):
    """Full classifier composed of the backbone + hyperbolic prototype head."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        init_curvature: float = 1.0,
        temperature: float = 1.0,
        **head_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = Backbone(out_dim=feature_dim)
        self.head = HyperbolicPrototypeHead(
            embedding_dim=feature_dim,
            num_classes=num_classes,
            init_curvature=init_curvature,
            temperature=temperature,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.backbone(x))


__all__ = ["HyperbolicPrototypeHead", "Backbone", "HyperbolicClassifier"]
