from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
    ):
        super().__init__()
        dims: List[int] = [latent_dim, *list(hidden_dims), output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SparseGenerator(nn.Module):
    """Generate non-negative unit vectors with a learnable point mass at zero."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        gate_temperature: float = 0.5,
        logit_clamp: float = 10.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        dims: List[int] = [latent_dim, *hidden_dims]
        if latent_dim <= 0 or output_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("model dimensions must be greater than zero")
        if len(dims) < 2:
            raise ValueError("SparseGenerator requires at least one hidden dimension")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be greater than zero")
        if logit_clamp <= 0:
            raise ValueError("logit_clamp must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.magnitude_head = nn.Linear(dims[-1], output_dim)
        self.gate_head = nn.Linear(dims[-1], output_dim)
        self.gate_temperature = float(gate_temperature)
        self.logit_clamp = float(logit_clamp)
        self.eps = float(eps)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        # Draw in float32 under AMP: eps=1e-8 rounds to zero in float16,
        # which would leave log(0) capable of poisoning gate gradients.
        sample_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        u = torch.rand_like(sample_logits).clamp(self.eps, 1.0 - self.eps)
        logistic = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((sample_logits + logistic) / self.gate_temperature)
        hard = (soft > 0.5).to(soft.dtype)
        # Preserve the unit-norm contract even if every stochastic gate in a
        # row is off. This is vanishingly rare at 128 dimensions but otherwise
        # silently produces an invalid zero vector.
        empty = hard.sum(dim=1, keepdim=True) == 0
        fallback = F.one_hot(
            sample_logits.argmax(dim=1), sample_logits.shape[1]
        ).to(hard.dtype)
        hard = torch.where(empty, fallback, hard)
        return hard + soft - soft.detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        magnitude = F.softplus(self.magnitude_head(h))
        raw_logits = self.gate_head(h)
        logits = self.logit_clamp * torch.tanh(raw_logits / self.logit_clamp)
        x = self._sample_gate(logits) * magnitude
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=self.eps)


def build_generator(model_cfg: Dict, output_dim: int) -> nn.Module:
    """Build the configured generator, defaulting to the legacy MLP."""
    kind = model_cfg.get("generator_type", "mlp")
    common = {
        "latent_dim": int(model_cfg["latent_dim"]),
        "output_dim": output_dim,
        "hidden_dims": model_cfg["generator_hidden_dims"],
        "negative_slope": float(model_cfg["negative_slope"]),
    }
    if kind == "mlp":
        return Generator(**common)
    if kind == "sparse":
        return SparseGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
        )
    raise ValueError(f"Unknown generator_type: {kind}")
