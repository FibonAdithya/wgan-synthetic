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


# Spec: docs/ai/specs/2026-09-14-neighbourhood-critic-design.md.
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


def neighbourhood_indices(x: Tensor, k: int) -> Tensor:
    """Indices of each row's `k` nearest other rows of `x`, nearest first.

    Same masked distance matrix as `neighbourhood_distances`, so self is
    excluded by index and an exact copy stays a neighbour. Indices carry no
    gradient; the set critic gathers neighbour rows with them and the
    gradient flows through the gathered rows.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if x.shape[0] - 1 < k:
        raise ValueError(
            f"need at least k={k} neighbour candidates per row, got {x.shape[0] - 1} "
            f"(within-batch, batch of {x.shape[0]})"
        )
    with torch.no_grad():  # indices only; no reason to record the d2 graph
        d2 = _masked_squared_distances(x, None, None)
        _, idx = torch.topk(d2, k, dim=1, largest=False, sorted=True)
    return idx


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


def draw_real_bank_indices(num_rows: int, bank_size: int, seed: int) -> Tensor:
    """A fixed random subset of the training split, one draw per run seed.

    Seeded on its own generator, on the CPU, so the draw does not depend on
    how much of the global RNG the trainer consumed before building the
    critic. The result is stored in the checkpoint (`real_bank_indices`) so a
    resume rebuilds the same bank instead of redrawing it.
    """
    if bank_size > num_rows:
        raise ValueError(
            f"critic_bank_size={bank_size} exceeds the training split ({num_rows} rows)"
        )
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(num_rows, generator=g)[:bank_size]


class BankNeighbourhoodCritic(NeighbourhoodCritic):
    """Approach 2 of the neighbourhood-critic design: profile against a bank.

    Same features and MLP as `NeighbourhoodCritic`, but a row's neighbours
    come from a large detached bank rather than its batch-mates, so the
    profile is measured at a scale closer to the gate's (100-NN in 250k) and
    exact copies become visible at a rate a critic can learn from (about 3%
    of real rows at bank size 16k).

    Three populations, chosen by the caller:

    - `real`: the k nearest rows of the real bank, a fixed subset of the
      training split. A real row that is itself in the bank is excluded by
      index (`row_ids` -> `row_to_slot` -> `self_index`), never by distance,
      so a genuine copy elsewhere in the bank is still a neighbour.
    - `fake`: the k nearest rows of the fake bank, a ring of the most recent
      generator outputs, written by the trainer after each generator step
      from the batch it just scored, so a batch never sees itself.
    - `mixed` (the default, and what a bare `critic(x)` call gets): the
      union of both banks. `gradient_penalty` scores interpolated rows this
      way; an interpolate is neither population and the penalty only needs
      the critic to be smooth in the row, so the union is the neutral choice.

    Until the fake ring has been filled once, `fake` and `mixed` rows are
    profiled within-batch, exactly as `NeighbourhoodCritic` does. On a resume
    the ring starts empty again; the trainer records the step at which it
    first fills.

    Requires `training.amp: false`, as the parent does: under autocast the
    ring would hold fp16-rounded generator rows widened to float32 while the
    real bank holds exact training rows, turning the precision asymmetry
    into a persistent property of the two banks.

    Checkpoints carry `real_bank_indices` (persistent) and neither bank
    (non-persistent): the trainer rebuilds the real bank from the split.
    """

    POPULATIONS = ("real", "fake", "mixed")

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        k: int = DEFAULT_K,
        distance_floor: float = DEFAULT_DISTANCE_FLOOR,
        negative_slope: float = 0.2,
        bank_size: int = 16384,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            k=k,
            distance_floor=distance_floor,
            negative_slope=negative_slope,
        )
        if bank_size <= k:
            raise ValueError(
                f"critic_bank_size={bank_size} must exceed critic_k={k}: a real "
                "row excludes its own slot and still needs k neighbours"
            )
        self.bank_size = int(bank_size)
        self.register_buffer(
            "real_bank_indices", torch.full((self.bank_size,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "real_bank", torch.zeros(self.bank_size, input_dim), persistent=False
        )
        self.register_buffer(
            "fake_bank", torch.zeros(self.bank_size, input_dim), persistent=False
        )
        # A buffer, not a plain attribute, so `.to(device)` moves it with the
        # banks; non-persistent because it is rebuilt from the indices.
        self.register_buffer(
            "row_to_slot", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.fake_rows_written = 0

    @property
    def fake_bank_full(self) -> bool:
        return self.fake_rows_written >= self.bank_size

    @property
    def real_bank_set(self) -> bool:
        return self.row_to_slot.numel() > 0

    def set_real_bank(self, x_train: Tensor, indices: Tensor) -> None:
        """Fill the real bank with `x_train[indices]` and index it by row."""
        indices = indices.to(dtype=torch.long, device="cpu")
        if indices.shape != (self.bank_size,):
            raise ValueError(
                f"expected bank_size={self.bank_size} indices, got {tuple(indices.shape)}"
            )
        with torch.no_grad():
            self.real_bank_indices.copy_(indices.to(self.real_bank_indices.device))
            self.real_bank.copy_(x_train[indices].to(self.real_bank))
        slot = torch.full((x_train.shape[0],), -1, dtype=torch.long)
        slot[indices] = torch.arange(self.bank_size)
        self.row_to_slot = slot.to(self.real_bank.device)

    @torch.no_grad()
    def write_fake(self, rows: Tensor) -> None:
        """Append `rows` to the fake ring, overwriting the oldest entries."""
        n = rows.shape[0]
        if n > self.bank_size:
            raise ValueError(
                f"a batch of {n} rows does not fit a fake bank of {self.bank_size}"
            )
        rows = rows.detach().to(self.fake_bank)
        pos = self.fake_rows_written % self.bank_size
        end = pos + n
        if end <= self.bank_size:
            self.fake_bank[pos:end] = rows
        else:
            first = self.bank_size - pos
            self.fake_bank[pos:] = rows[:first]
            self.fake_bank[: n - first] = rows[first:]
        self.fake_rows_written += n

    def features(
        self,
        x: Tensor,
        population: str = "mixed",
        row_ids: Tensor | None = None,
    ) -> Tensor:
        if population not in self.POPULATIONS:
            raise ValueError(
                f"population must be one of {self.POPULATIONS}, got {population!r}"
            )
        needs_real_bank = population == "real" or (
            population == "mixed" and self.fake_bank_full
        )
        if needs_real_bank and not self.real_bank_set:
            raise RuntimeError(
                f"call set_real_bank() before scoring the {population!r} population"
            )
        if population == "real":
            self_index = (
                None
                if row_ids is None
                else self.row_to_slot[row_ids.to(self.row_to_slot.device)]
            )
            r = neighbourhood_distances(
                x,
                self.k,
                self.distance_floor,
                bank=self.real_bank,
                self_index=self_index,
            )
        elif not self.fake_bank_full:
            r = neighbourhood_distances(x, self.k, self.distance_floor)
        elif population == "fake":
            r = neighbourhood_distances(
                x, self.k, self.distance_floor, bank=self.fake_bank
            )
        else:
            r = neighbourhood_distances(
                x,
                self.k,
                self.distance_floor,
                bank=torch.cat([self.real_bank, self.fake_bank], dim=0),
            )
        return profile_features(r)

    def forward(
        self,
        x: Tensor,
        population: str = "mixed",
        row_ids: Tensor | None = None,
    ) -> Tensor:
        phi = self.features(x, population=population, row_ids=row_ids).to(x.dtype)
        return self.mlp(torch.cat([x, phi], dim=1))


def score_population(
    critic: nn.Module,
    x: Tensor,
    population: str,
    row_ids: Tensor | None = None,
) -> Tensor:
    """Score `x` as `population`; every critic but the bank one ignores the label.

    The trainer calls this at its three scoring sites so the loop reads the
    same for every `critic_type`. `gradient_penalty` keeps its bare
    `critic(interpolated)` call, which for the bank critic is the `mixed`
    population by default.
    """
    if isinstance(critic, BankNeighbourhoodCritic):
        return critic(x, population=population, row_ids=row_ids)
    return critic(x)


EDGE_POOLS = ("max", "mean")


class SetNeighbourhoodCritic(nn.Module):
    """One EdgeConv layer (Wang et al., DGCNN) over each row's k within-batch
    neighbours, pooled, then the per-vector MLP on `[x_i, pooled_i]`.

        e_ij = MLP_edge([x_i, x_j - x_i])     j in kNN(i), 2D -> H -> H
        a_i  = pool_j e_ij                     max (default) or mean
        s_i  = MLP_out([x_i, a_i])             D + H -> 1

    `x_j - x_i` is the local difference; its spread over j is the local
    tangent structure, whose singular spectrum is exactly what a collapsed
    sheet loses. Unlike `NeighbourhoodCritic` this class reads no
    distances, so it learns its own neighbourhood features. `distance_floor`
    is accepted for interface uniformity and unused: neighbour selection is
    by ordering, which a lower clamp does not change, and nothing here
    divides by a distance.

    Gradients flow through both the query row and the gathered neighbour
    rows, so under `gradient_penalty` the summed-score formulation is what
    bounds how fast a score can change when a neighbour moves.

    Requires `training.amp: false`, as `NeighbourhoodCritic` does: under
    autocast the trainer hands the critic fp16-rounded fake rows and fp32
    real rows, and a difference `x_j - x_i` of that origin is not a
    difference in the data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        k: int = DEFAULT_K,
        distance_floor: float = DEFAULT_DISTANCE_FLOOR,
        edge_dim: int = 128,
        edge_pool: str = "max",
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if k < 1:
            raise ValueError(f"critic_k must be positive, got {k}")
        if edge_dim < 1:
            raise ValueError(f"critic_edge_dim must be positive, got {edge_dim}")
        if edge_pool not in EDGE_POOLS:
            raise ValueError(
                f"critic_edge_pool must be one of {EDGE_POOLS}, got {edge_pool!r}"
            )
        self.k = int(k)
        self.distance_floor = float(distance_floor)
        self.edge_dim = int(edge_dim)
        self.edge_pool = str(edge_pool)
        self.edge = nn.Sequential(
            nn.Linear(2 * input_dim, self.edge_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(self.edge_dim, self.edge_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        self.mlp = Critic(
            input_dim=input_dim + self.edge_dim,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )

    def edge_features(self, x: Tensor, idx: Tensor) -> Tensor:
        """`(n, k, edge_dim)`: MLP_edge on `[x_i, x_j - x_i]` for each edge."""
        k = idx.shape[1]
        x_j = x[idx]  # (n, k, D), differentiable in x
        x_i = x[:, None, :].expand(-1, k, -1)
        return self.edge(torch.cat([x_i, x_j - x_i], dim=2))

    def pooled(self, x: Tensor, idx: Tensor) -> Tensor:
        e = self.edge_features(x, idx)
        if self.edge_pool == "max":
            return e.max(dim=1).values
        return e.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        idx = neighbourhood_indices(x, self.k)
        a = self.pooled(x, idx)
        # Defensive, a no-op under amp: false; unlike NeighbourhoodCritic.features
        # the edge MLP's output dtype is not float32-guaranteed.
        return self.mlp(torch.cat([x, a.to(x.dtype)], dim=1))


CRITIC_TYPES = (
    "per_vector",
    "neighbourhood",
    "neighbourhood_bank",
    "neighbourhood_set",
)


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
    if kind == "neighbourhood_bank":
        return BankNeighbourhoodCritic(
            input_dim=input_dim,
            k=int(model_cfg.get("critic_k", DEFAULT_K)),
            distance_floor=float(
                model_cfg.get("critic_distance_floor", DEFAULT_DISTANCE_FLOOR)
            ),
            bank_size=int(model_cfg.get("critic_bank_size", 16384)),
            **common,
        )
    if kind == "neighbourhood_set":
        return SetNeighbourhoodCritic(
            input_dim=input_dim,
            k=int(model_cfg.get("critic_k", DEFAULT_K)),
            distance_floor=float(
                model_cfg.get("critic_distance_floor", DEFAULT_DISTANCE_FLOOR)
            ),
            edge_dim=int(model_cfg.get("critic_edge_dim", 128)),
            edge_pool=str(model_cfg.get("critic_edge_pool", "max")),
            **common,
        )
    raise ValueError(f"Unknown critic_type: {kind!r}; expected one of {CRITIC_TYPES}")
