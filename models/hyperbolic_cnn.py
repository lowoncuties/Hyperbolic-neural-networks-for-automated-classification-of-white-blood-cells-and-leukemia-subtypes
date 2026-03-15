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
        learnable_temperature: bool = False,
        min_c: float = 1e-4,
        max_c: float = 1e4,
        proto_init_std: float = 1e-3,
        min_tau: float = 1e-4,
        max_tau: float = 1e4,
    ) -> None:
        super().__init__()
        from geoopt.manifolds.stereographic import math as hypmath

        self.hypmath = hypmath
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.learnable_temperature = bool(learnable_temperature)
        self.min_c = float(min_c)
        self.max_c = float(max_c)
        self._eps_c = 1e-6
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        self._eps_tau = 1e-6

        self.raw_c = nn.Parameter(torch.log(torch.tensor(float(init_curvature))))
        if self.learnable_temperature:
            self.register_buffer("fixed_tau", None, persistent=False)
            self.raw_tau = nn.Parameter(self._inverse_softplus(self.temperature))
        else:
            self.register_parameter("raw_tau", None)
            self.register_buffer(
                "fixed_tau",
                torch.tensor(self.temperature, dtype=torch.float32),
                persistent=False,
            )
        self.proto_tan = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * proto_init_std
        )

    @staticmethod
    def _inverse_softplus(value: float) -> torch.Tensor:
        value = max(float(value), 1e-6)
        tensor = torch.tensor(value, dtype=torch.float32)
        return torch.log(torch.expm1(tensor))

    def current_c(self) -> torch.Tensor:
        """Return the current curvature value clipped to the configured range."""

        c = F.softplus(self.raw_c) + self._eps_c
        return torch.clamp(c, self.min_c, self.max_c)

    def current_tau(self) -> torch.Tensor:
        """Return the active temperature value clipped to the configured range."""

        if self.raw_tau is None:
            tau = self.fixed_tau
        else:
            tau = F.softplus(self.raw_tau) + self._eps_tau
        return torch.clamp(tau, self.min_tau, self.max_tau)

    def forward(self, euclidean_feats: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        device = euclidean_feats.device
        c = self.current_c().to(device)
        k = -c

        z = self.hypmath.expmap0(euclidean_feats, k=k, dim=-1)
        p = self.hypmath.expmap0(self.proto_tan.to(device), k=k, dim=-1)
        d = self.hypmath.dist(z.unsqueeze(1), p.unsqueeze(0), k=k, dim=-1)
        tau = self.current_tau().to(device)
        logits = -d / tau
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
        learnable_temperature: bool = False,
        **head_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = Backbone(out_dim=feature_dim)
        self.head = HyperbolicPrototypeHead(
            embedding_dim=feature_dim,
            num_classes=num_classes,
            init_curvature=init_curvature,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.backbone(x))


__all__ = ["HyperbolicPrototypeHead", "Backbone", "HyperbolicClassifier"]
