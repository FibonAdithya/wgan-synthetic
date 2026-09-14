from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
    ):
        super().__init__()
        dims: list[int] = [input_dim, *list(hidden_dims), 1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# Spec: docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md.
# k=20 separates the collapsed sheet from the corpus per row at 0.998 and
# keeps the real-vs-Gaussian ordering; the floor sits inside the measured
# gap between exact copies (~1e-8 after rounding) and the nearest genuine
# pair (first percentile 0.90), so it changes the feature of a copy and of
# nothing else.
DEFAULT_K = 20
DEFAULT_DISTANCE_FLOOR = 0.01


def _masked_squared_distances(
    x: Tensor, bank: Tensor | None, self_index: Tensor | None
) -> Tensor:
    """Squared L2 distances from each row of `x` to each row of `bank`
    (default: `x` itself), float32, with excluded entries set to +inf.

    Expanded squares with `clamp(min=0)` rather than `torch.cdist`: cdist's
    gradient is undefined at distance zero, and a single pair of exact
    copies -- present in NYTimes, and produced by a collapsing generator --
    would poison the whole backward pass. Self-exclusion is by index (an
    additive inf mask, so no in-place write on a tensor autograd saved), never
    by dropping the nearest column: an exact copy ties with the query at zero
    and would otherwise be dropped in its place.

    `self_index[i]` is the bank row that *is* `x[i]`, or -1 for "not in the
    bank". Only used when `bank` is given; within-batch the diagonal is the
    self index.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        q = x.float()
        b = q if bank is None else bank.float()
        sq_q = (q * q).sum(dim=1)
        sq_b = sq_q if bank is None else (b * b).sum(dim=1)
        d2 = sq_q[:, None] + sq_b[None, :] - 2.0 * (q @ b.T)
        d2 = d2.clamp(min=0.0)
        mask = torch.zeros_like(d2)
        if bank is None:
            mask.fill_diagonal_(float("inf"))
        elif self_index is not None:
            present = self_index >= 0
            rows = torch.arange(q.shape[0], device=q.device)[present]
            mask[rows, self_index[present]] = float("inf")
        return d2 + mask


def neighbourhood_distances(
    x: Tensor,
    k: int,
    floor: float,
    bank: Tensor | None = None,
    self_index: Tensor | None = None,
) -> Tensor:
    """Sorted distances from each row of `x` to its `k` nearest rows of
    `bank` (default: the other rows of `x`), each floored at `floor`.

    The floor is applied to the squared distances before the sqrt, so a
    zero distance never reaches sqrt (whose gradient there is infinite) and
    an exact copy reads as exactly `floor`, a bounded "tight pair" feature a
    continuous generator can match by producing near-copies.

    Raises when fewer than `k` candidates exist rather than truncating: the
    feature width is the critic's input width and must not change silently.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")
    if bank is None:
        n_candidates = x.shape[0] - 1
    else:
        n_candidates = bank.shape[0]
        if self_index is not None and (self_index >= 0).any():
            n_candidates -= 1
    if n_candidates < k:
        raise ValueError(
            f"need at least k={k} neighbour candidates per row, got {n_candidates} "
            f"({'within-batch, batch of ' + str(x.shape[0]) if bank is None else 'bank'})"
        )
    with torch.autocast(device_type=x.device.type, enabled=False):
        d2 = _masked_squared_distances(x, bank, self_index)
        r2, _ = torch.topk(d2, k, dim=1, largest=False, sorted=True)
        r2 = r2.clamp(min=float(floor) ** 2)
        return r2.sqrt()


def profile_features(r: Tensor) -> Tensor:
    """`[log(r_1/r_k), ..., log(r_{k-1}/r_k), log r_k]` per row: the Hill
    estimator's sufficient statistic (shape) plus the neighbourhood scale.
    `r` must be positive everywhere, which `neighbourhood_distances` ensures.
    """
    r_k = r[:, -1:]
    return torch.cat([torch.log(r[:, :-1] / r_k), torch.log(r_k)], dim=1)


class NeighbourhoodCritic(nn.Module):
    """The per-vector MLP on `[x_i, phi_i]`, where `phi_i` is row i's
    within-batch neighbourhood profile (`profile_features` of
    `neighbourhood_distances`).

    Batch-dependent by design: a per-vector critic cannot see local
    dimension, so a low-rank sheet with the right covariance envelope is,
    to it, the corpus (NYTimes v1 collapsed to LID 5 with the Wasserstein
    estimate under 0.06 throughout). Real rows are profiled among real
    batch-mates and fake among fake, so the within-batch bias in the
    distances is identical on both sides and cancels.

    Under `gradient_penalty` the per-row gradient is therefore of the
    batch's *summed* score, which includes how row i moves every other
    row's features. That is the intended Lipschitz constraint for a
    minibatch-dependent critic; do not "fix" it back to per-row.

    Requires `training.amp: false`. The neighbour maths runs in float32
    regardless, but under autocast the trainer hands the critic fp16-rounded
    fake rows and fp32 real rows, and a profile difference of that origin is
    not a difference in the data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        k: int = DEFAULT_K,
        distance_floor: float = DEFAULT_DISTANCE_FLOOR,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if k < 2:
            raise ValueError(f"critic_k must be at least 2 (one ratio entry), got {k}")
        if distance_floor <= 0.0:
            raise ValueError(
                f"critic_distance_floor must be positive, got {distance_floor}"
            )
        self.k = int(k)
        self.distance_floor = float(distance_floor)
        self.mlp = Critic(
            input_dim=input_dim + self.k,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )

    def features(self, x: Tensor) -> Tensor:
        r = neighbourhood_distances(x, self.k, self.distance_floor)
        return profile_features(r)

    def forward(self, x: Tensor) -> Tensor:
        phi = self.features(x).to(x.dtype)
        return self.mlp(torch.cat([x, phi], dim=1))


CRITIC_TYPES = ("per_vector", "neighbourhood")


def build_critic(model_cfg: Mapping[str, Any], input_dim: int) -> nn.Module:
    """Build the configured critic, defaulting to the per-vector `Critic`.

    Mirrors `build_generator`: `critic_type` selects the class, the common
    keys mean the same for every class, and an unknown value fails here
    rather than silently training the default.
    """
    kind = str(model_cfg.get("critic_type", "per_vector"))
    common = {
        "hidden_dims": model_cfg["critic_hidden_dims"],
        "negative_slope": float(model_cfg["negative_slope"]),
    }
    if kind == "per_vector":
        return Critic(input_dim=input_dim, **common)
    if kind == "neighbourhood":
        return NeighbourhoodCritic(
            input_dim=input_dim,
            k=int(model_cfg.get("critic_k", DEFAULT_K)),
            distance_floor=float(
                model_cfg.get("critic_distance_floor", DEFAULT_DISTANCE_FLOOR)
            ),
            **common,
        )
    raise ValueError(f"Unknown critic_type: {kind!r}; expected one of {CRITIC_TYPES}")
