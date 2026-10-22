"""Position-flag conditioning for the generator and critic.

The wrappers take the network to condition as an argument and import only
torch, so `generator.py` and `critic.py` can import this module without a
cycle. The wrapped network is built with `n_flags` extra input columns; the
wrapper appends a one-hot of the flag to its input. `flags` has no default:
a call without it raises TypeError, so an unconditioned sampler cannot
silently read a flag-conditioned checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

BODY: int = 0
TAIL: int = 1


def _append_one_hot(x: Tensor, flags: Tensor, n_flags: int) -> Tensor:
    # F.one_hot raises RuntimeError for a flag outside [0, n_flags).
    return torch.cat([x, F.one_hot(flags, n_flags).to(x.dtype)], dim=1)


class FlagConditionedGenerator(nn.Module):
    """`base(cat([z, one_hot(flags)]))`; `base` takes `latent_dim + n_flags` inputs."""

    def __init__(self, base: nn.Module, n_flags: int):
        super().__init__()
        self.base = base
        self.n_flags = int(n_flags)

    def forward(self, z: Tensor, flags: Tensor) -> Tensor:
        return self.base(_append_one_hot(z, flags, self.n_flags))


class FlagConditionedCritic(nn.Module):
    """`base(cat([x, one_hot(flags)]))`; `base` takes `input_dim + n_flags` inputs."""

    def __init__(self, base: nn.Module, n_flags: int):
        super().__init__()
        self.base = base
        self.n_flags = int(n_flags)

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        return self.base(_append_one_hot(x, flags, self.n_flags))
