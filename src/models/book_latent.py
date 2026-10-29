"""Book-latent generator and window critic for scout_kv v2.

A synthetic book has its own smooth random latent: per channel c, a sum of M
random Fourier features with length scale `l_c`,

    u_c(t) = sum_m (a_cm cos(w_cm t) + b_cm sin(w_cm t)) / sqrt(M),

with `w_cm ~ N(0,1) / l_c` drawn once from the seed and `a, b ~ N(0,1)` drawn per
book. The generator maps per-token noise, that latent and a body/tail flag to
one vector. A window critic scores W tokens spaced by one of several strides.
All three networks (generator, token critic, window critic) encode the flag
with the same two-column one-hot from `src.models.conditional`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.conditional import _append_one_hot
from src.models.critic import build_critic
from src.models.generator import Generator

N_FLAGS = 2


class DriftLatent(nn.Module):
    def __init__(
        self,
        seed: int,
        channels: int,
        n_features: int,
        min_length: float,
        max_length: float,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_features = int(n_features)
        g = torch.Generator().manual_seed(int(seed))
        ell = torch.logspace(
            math.log10(min_length), math.log10(max_length), self.channels
        )
        omega = torch.randn(self.channels, self.n_features, generator=g) / ell[:, None]
        self.register_buffer("omega", omega)

    def sample_coefficients(self, n: int, device, generator=None) -> Tensor:
        """Per-book Fourier coefficients, shape `(n, 2, channels, n_features)`."""
        return torch.randn(
            n, 2, self.channels, self.n_features, device=device, generator=generator
        )

    def forward(self, coef: Tensor, pos: Tensor) -> Tensor:
        """`coef (n, 2, C, M)`, `pos (n, L)` -> `(n, L, C)`."""
        phase = pos.float()[..., None, None] * self.omega
        a, b = coef[:, None, 0], coef[:, None, 1]
        return (a * phase.cos() + b * phase.sin()).sum(-1) / math.sqrt(self.n_features)


class BookLatentGenerator(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        output_dim: int,
        hidden_dims,
        negative_slope: float,
        n_flags: int,
        latent: DriftLatent,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_flags = int(n_flags)
        self.latent = latent
        self.net = Generator(
            latent_dim=self.latent_dim + latent.channels + self.n_flags,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )

    def forward(self, z: Tensor, u: Tensor, flags: Tensor) -> Tensor:
        """`z (..., latent_dim)`, `u (..., C)`, `flags (...)` -> `(..., output_dim)`."""
        lead = z.shape[:-1]
        x = torch.cat([z, u], dim=-1).reshape(-1, z.shape[-1] + u.shape[-1])
        x = _append_one_hot(x, flags.reshape(-1), self.n_flags)
        return self.net(x).reshape(*lead, -1)


def _mlp(dims: list[int], slope: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU(slope))
    return nn.Sequential(*layers)


class WindowCritic(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        n_strides: int,
        token_hidden_dims,
        window_hidden_dims,
        negative_slope: float,
        n_flags: int,
    ):
        super().__init__()
        self.window = int(window)
        self.n_strides = int(n_strides)
        self.n_flags = int(n_flags)
        self.negative_slope = float(negative_slope)
        tok = [int(d) for d in token_hidden_dims]
        win = [int(d) for d in window_hidden_dims]
        self.tok = _mlp([input_dim + self.n_flags, *tok], negative_slope)
        self.tok_head = nn.Linear(tok[-1], 1)
        self.win = _mlp(
            [self.window * tok[-1] + self.n_strides, *win, 1], negative_slope
        )

    def forward(self, x: Tensor, flags: Tensor, stride_idx: Tensor) -> Tensor:
        """`x (n, W, D)`, `flags (n, W)`, `stride_idx (n,)` -> `(n,)`."""
        n, w, d = x.shape
        flat = _append_one_hot(x.reshape(n * w, d), flags.reshape(n * w), self.n_flags)
        e = F.leaky_relu(self.tok(flat), self.negative_slope).reshape(n, w, -1)
        token_score = self.tok_head(e).mean(1)
        onehot = F.one_hot(stride_idx, self.n_strides).to(x.dtype)
        window_score = self.win(torch.cat([e.flatten(1), onehot], dim=-1))
        return (token_score + window_score).squeeze(-1)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def book_latent_settings(config: dict) -> dict:
    """The record a checkpoint stores, as plain lists and numbers."""
    model, data = config["model"], config["data"]
    return _plain(
        {
            "latent_dim": int(model["latent_dim"]),
            "generator_hidden_dims": list(model["generator_hidden_dims"]),
            "critic_hidden_dims": list(model["critic_hidden_dims"]),
            "negative_slope": float(model["negative_slope"]),
            "n_flags": int(model["n_flags"]),
            "book_latent": dict(model["book_latent"]),
            "window_critic": dict(model["window_critic"]),
            "descriptor_dim": int(data["descriptor_dim"]),
            "kv_n_sink": int(data["kv_n_sink"]),
            "kv_n_tail": int(data["kv_n_tail"]),
            "kv_books": [int(b) for b in data["kv_books"]],
            "kv_layer": int(data["kv_layer"]),
        }
    )


def build_book_latent(
    config: dict,
) -> tuple[BookLatentGenerator, nn.Module, WindowCritic]:
    model = config["model"]
    if model.get("generator_type") != "book_latent":
        raise ValueError(
            "build_book_latent needs model.generator_type 'book_latent', "
            f"got {model.get('generator_type')!r}"
        )
    if model.get("n_flags") != N_FLAGS:
        raise ValueError(
            f"build_book_latent needs model.n_flags 2, got {model.get('n_flags')!r}"
        )
    bl, wc = model["book_latent"], model["window_critic"]
    dim = int(config["data"]["descriptor_dim"])
    slope = float(model["negative_slope"])
    latent = DriftLatent(
        int(config["seed"]),
        int(bl["channels"]),
        int(bl["n_features"]),
        float(bl["min_length"]),
        float(bl["max_length"]),
    )
    gen = BookLatentGenerator(
        latent_dim=int(model["latent_dim"]),
        output_dim=dim,
        hidden_dims=model["generator_hidden_dims"],
        negative_slope=slope,
        n_flags=N_FLAGS,
        latent=latent,
    )
    token_critic = build_critic(model, dim)
    window_critic = WindowCritic(
        input_dim=dim,
        window=int(wc["window"]),
        n_strides=len(wc["strides"]),
        token_hidden_dims=wc["token_hidden_dims"],
        window_hidden_dims=wc["window_hidden_dims"],
        negative_slope=slope,
        n_flags=N_FLAGS,
    )
    return gen, token_critic, window_critic
