# Neighbourhood Critic, approach 3: learned set critic (`v2c`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train NYTimes `v2c`, whose critic learns its own neighbourhood features from the raw within-batch neighbour differences instead of a hand-chosen distance profile.

**Architecture:** `SetNeighbourhoodCritic` in `src/models/critic.py` finds each row's k within-batch neighbours (same masked distance matrix as the shared change, via a new `neighbourhood_indices`), runs one EdgeConv layer `MLP_edge([x_i, x_j - x_i])` over the k edges, pools over neighbours (max, or mean), and feeds `[x_i, pooled]` to the existing MLP. `forward(x)` keeps the plain calling convention, so the trainer does not change.

**Tech Stack:** Python 3, PyTorch, numpy, pytest, ruff. Local venv `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`; box `/opt/venvs/wgan-synthetic/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md`, sections "Shared" and "Approach 3".

**Prerequisite:** the shared change from `docs/superpowers/plans/2026-09-14-neighbourhood-critic-v2.md` Tasks 1 to 6 is merged on `nytimes-eda`. Start from that commit; do not edit `NeighbourhoodCritic` or `BankNeighbourhoodCritic`.

## Global Constraints

- All of the `v2` plan's constraints hold: `Critic` untouched, float32 neighbour maths with autocast disabled, no `torch.cdist`, self excluded by an `inf` mask, explicit `git add` paths, `make check` before each commit.
- Neighbour *selection* uses the masked squared distances; the floor plays no part in selection (ordering is unchanged by a lower clamp) and this class reads no distances, only neighbour rows, so `critic_distance_floor` is accepted for interface uniformity and unused. Say so in the docstring.
- `critic_edge_dim` default `128`; `critic_edge_pool` default `max`, alternative `mean`; anything else raises.
- The trainer is not modified by this plan. If the `v2b` plan has landed, its `score()` helper passes nothing to this class (it does not declare `population_aware`).

## Deviations from the spec

**Translation test zeroes the edge MLP's `x_i` columns too.** The spec's test says "with `MLP_out`'s `x_i` weights zeroed, scores are invariant to a translation `c`". The edge MLP also takes `x_i` as input, so that alone is not invariant. The test zeroes the `x_i` columns of both first layers; it then catches exactly what the spec wanted, an edge feature built from absolute `x_j` rather than `x_j - x_i`.

## File map

| file | change |
|---|---|
| `src/models/critic.py` | `neighbourhood_indices`, `SetNeighbourhoodCritic`, `"neighbourhood_set"` in `CRITIC_TYPES` and `build_critic` |
| `tests/test_set_critic.py` | new |
| `tests/test_critic_factory.py`, `tests/test_train_smoke.py` | extended |
| `configs/nytimes/v2c.yaml`, `configs/nytimes/v2c_seed42.yaml`, `scripts/nytimes_v2c_seed42_job.sh` | new |
| `tests/test_nytimes_configs.py` | extended |
| `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md` | documented |

---

### Task 1: `neighbourhood_indices`

**Files:**
- Modify: `src/models/critic.py`
- Test: `tests/test_set_critic.py` (new)

**Interfaces:**
- Consumes: `_masked_squared_distances(x, bank, self_index)` from the shared change.
- Produces: `neighbourhood_indices(x: Tensor, k: int) -> LongTensor (n, k)`, each row's k nearest other rows, nearest first, self excluded. No gradient (indices).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_set_critic.py
"""The learned set critic: one EdgeConv layer over within-batch neighbour
differences. Each test names the mutation it catches."""

import numpy as np
import pytest
import torch

from src.models.critic import neighbourhood_indices


def _unit(seed, n, dim):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def test_indices_match_brute_force_and_exclude_self():
    """Catches a lost diagonal mask (self would be index 0 of every row)."""
    x = _unit(0, n=9, dim=6)
    idx = neighbourhood_indices(x, k=3)
    full = torch.cdist(x, x)
    full.fill_diagonal_(float("inf"))
    _, expected = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    assert idx.shape == (9, 3) and idx.dtype == torch.long
    assert torch.equal(idx, expected)
    assert (idx != torch.arange(9)[:, None]).all()


def test_indices_keep_an_exact_copy_as_a_neighbour():
    """Catches self-exclusion by dropping the nearest column, which would
    drop a copy in the row's place."""
    x = _unit(1, n=8, dim=5)
    x[3] = x[0]
    idx = neighbourhood_indices(x, k=2)
    assert idx[0, 0] == 3 and idx[3, 0] == 0


def test_indices_require_more_than_k_rows():
    with pytest.raises(ValueError, match="k"):
        neighbourhood_indices(_unit(2, n=4, dim=3), k=4)
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_set_critic.py -v`. Expected: `ImportError: cannot import name 'neighbourhood_indices'`.

- [ ] **Step 3: Implement** (in `src/models/critic.py`, after `neighbourhood_distances`)

```python
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
    d2 = _masked_squared_distances(x, None, None)
    _, idx = torch.topk(d2, k, dim=1, largest=False, sorted=True)
    return idx
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_set_critic.py tests/test_neighbourhood_critic.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_set_critic.py
git commit -m "feat(critic): neighbourhood_indices, within-batch k-NN indices with self excluded"
```

---

### Task 2: `SetNeighbourhoodCritic`

**Files:**
- Modify: `src/models/critic.py`
- Test: `tests/test_set_critic.py`

**Interfaces:**
- Consumes: `neighbourhood_indices`, `Critic`, `DEFAULT_K`, `DEFAULT_DISTANCE_FLOOR`.
- Produces: `SetNeighbourhoodCritic(input_dim, hidden_dims, k=DEFAULT_K, distance_floor=DEFAULT_DISTANCE_FLOOR, edge_dim=128, edge_pool="max", negative_slope=0.2)` with `edge: nn.Sequential` (first layer `Linear(2*D, edge_dim)`), `mlp: Critic` (input width `D + edge_dim`), `edge_features(x, idx) -> (n, k, edge_dim)`, `pooled(x, idx) -> (n, edge_dim)`, `forward(x) -> (n,)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_set_critic.py`; add `SetNeighbourhoodCritic` to the import and `from src.train.train_wgan_gp import gradient_penalty`)

```python
def _set_critic(k=3, dim=4, edge_dim=8, pool="max"):
    torch.manual_seed(0)
    return SetNeighbourhoodCritic(
        input_dim=dim, hidden_dims=[6], k=k, edge_dim=edge_dim, edge_pool=pool
    )


def test_one_score_per_row():
    assert _set_critic()(_unit(3, 7, 4)).shape == (7,)


def test_layer_widths():
    """Catches the edge MLP fed x_j alone (width D) or the head not
    receiving the pooled features."""
    c = _set_critic(dim=4, edge_dim=8)
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    first_head = next(m for m in c.mlp.net if isinstance(m, torch.nn.Linear))
    assert first_edge.in_features == 2 * 4
    assert first_head.in_features == 4 + 8


def test_scores_depend_on_the_rest_of_the_batch():
    """Mirror of test_critic.py's independence test."""
    c = _set_critic(k=7)
    x = _unit(4, 8, 4)
    base = c(x)
    moved = x.clone()
    moved[7] = moved[7] + 1.0
    assert (c(moved)[:7] - base[:7]).abs().max() > 1e-6


def test_permuting_the_batch_permutes_the_scores():
    c = _set_critic(k=3)
    x = _unit(5, 10, 4)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(c(x[perm]), c(x)[perm], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("pool", ["max", "mean"])
def test_pooling_is_invariant_to_neighbour_order(pool):
    """Catches a flatten or a positional treatment of the k neighbours."""
    c = _set_critic(k=4, pool=pool)
    x = _unit(6, 9, 4)
    idx = neighbourhood_indices(x, k=4)
    shuffled = idx[:, torch.tensor([2, 0, 3, 1])]
    torch.testing.assert_close(c.pooled(x, idx), c.pooled(x, shuffled))


def test_edge_features_use_the_difference_not_the_absolute_neighbour():
    """With the x_i columns of BOTH first layers zeroed, the only input left
    is x_j - x_i, which a translation cannot change. Catches edge features
    built from absolute x_j."""
    c = _set_critic(k=3, dim=4)
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    first_head = next(m for m in c.mlp.net if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        first_edge.weight[:, :4].zero_()
        first_head.weight[:, :4].zero_()
    x = _unit(7, 9, 4)
    shift = torch.full((1, 4), 0.7)
    torch.testing.assert_close(c(x + shift), c(x), atol=1e-5, rtol=1e-5)


def test_max_and_mean_pools_both_build_and_differ():
    """Catches the pool key ignored."""
    x = _unit(8, 9, 4)
    a = _set_critic(pool="max")
    b = _set_critic(pool="mean")
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(x), b(x))


def test_unknown_pool_raises():
    with pytest.raises(ValueError, match="edge_pool"):
        SetNeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, edge_pool="sum")


def test_gradient_penalty_is_finite_and_double_backward_works():
    torch.manual_seed(0)
    c = _set_critic(k=3)
    gp = gradient_penalty(c, _unit(9, 8, 4), _unit(10, 8, 4), device=torch.device("cpu"))
    assert torch.isfinite(gp)
    gp.backward()
    # The head's bias never gets gradient from the penalty (it does not
    # affect d score / d input), on any critic. The edge MLP's first weight
    # must, since the penalty reaches neighbours through it.
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    assert first_edge.weight.grad is not None and torch.isfinite(first_edge.weight.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in c.parameters() if p.grad is not None)


def test_gradient_reaches_neighbour_rows_through_the_differences():
    """The summed-score penalty is doing real work here: row 0's score must
    have gradient with respect to its neighbours' coordinates."""
    c = _set_critic(k=7)
    x = _unit(11, 8, 4).requires_grad_(True)
    c(x)[0].backward()
    assert x.grad[1:].abs().sum() > 0


def test_exact_copies_give_finite_scores_and_gradients():
    """A copy gives a zero difference vector; nothing here divides by it."""
    x = _unit(12, 8, 4)
    x[1] = x[0]
    x.requires_grad_(True)
    s = _set_critic(k=3)(x)
    assert torch.isfinite(s).all()
    s.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_forward_refuses_a_batch_no_larger_than_k():
    with pytest.raises(ValueError, match="k"):
        _set_critic(k=5)(_unit(13, 5, 4))
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_set_critic.py -v`. Expected: `ImportError: cannot import name 'SetNeighbourhoodCritic'`.

- [ ] **Step 3: Implement** (append to `src/models/critic.py`, before `CRITIC_TYPES`)

```python
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
            raise ValueError(f"critic_edge_pool must be one of {EDGE_POOLS}, got {edge_pool!r}")
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
        return self.mlp(torch.cat([x, a.to(x.dtype)], dim=1))
```

Note on `inplace=True` LeakyReLU inside `edge`: the input to the first activation is the fresh output of a `Linear`, so in-place is safe, as in `Critic`. If the double-backward test fails with an in-place error, set `inplace=False` on both edge activations and say so in the commit.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_set_critic.py tests/test_neighbourhood_critic.py tests/test_critic.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_set_critic.py
git commit -m "feat(critic): SetNeighbourhoodCritic, one EdgeConv layer over within-batch neighbour differences"
```

---

### Task 3: factory branch and training smoke

**Files:**
- Modify: `src/models/critic.py` (`CRITIC_TYPES`, `build_critic`)
- Test: `tests/test_critic_factory.py`, `tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_critic_factory.py` (add `SetNeighbourhoodCritic` to its import):

```python
def test_neighbourhood_set_reads_edge_dim_and_pool_from_the_config():
    cfg = dict(
        BASE_CFG, critic_type="neighbourhood_set", critic_k=4,
        critic_edge_dim=32, critic_edge_pool="mean",
    )
    critic = build_critic(cfg, input_dim=12)
    assert isinstance(critic, SetNeighbourhoodCritic)
    assert (critic.k, critic.edge_dim, critic.edge_pool) == (4, 32, "mean")


def test_neighbourhood_set_defaults_edge_dim_128_and_max_pool():
    critic = build_critic(dict(BASE_CFG, critic_type="neighbourhood_set"), input_dim=12)
    assert (critic.edge_dim, critic.edge_pool) == (128, "max")
```

Amend `test_the_documented_types` to include `"neighbourhood_set"` at the end of the tuple (after `"neighbourhood_bank"` if the `v2b` plan has landed, otherwise after `"neighbourhood"`).

Append to `tests/test_train_smoke.py`:

```python
def test_set_critic_trains_and_its_checkpoint_reloads(tmp_path):
    from src.models.critic import SetNeighbourhoodCritic, build_critic

    cfg = make_config(tmp_path, "mlp")
    cfg["model"].update(critic_type="neighbourhood_set", critic_k=5, critic_edge_dim=16)
    ckpt_path, meta = train(cfg)
    for entry in meta["metrics"]:
        assert math.isfinite(entry["d_loss"]) and math.isfinite(entry["gp"])
    saved = torch.load(ckpt_path, weights_only=False)
    rebuilt = build_critic(cfg["model"], input_dim=16)
    assert isinstance(rebuilt, SetNeighbourhoodCritic)
    rebuilt.load_state_dict(saved["critic_state_dict"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_critic_factory.py tests/test_train_smoke.py -k "set" -v`. Expected: `ValueError: Unknown critic_type: 'neighbourhood_set'`.

- [ ] **Step 3: Implement**

Add `"neighbourhood_set"` to `CRITIC_TYPES` and, in `build_critic` before the `raise`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_critic_factory.py tests/test_train_smoke.py tests/test_set_critic.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_critic_factory.py tests/test_train_smoke.py
git commit -m "feat(critic): build_critic dispatches neighbourhood_set; set critic trains end to end"
```

---

### Task 4: configs, script, docs

**Files:**
- Create: `configs/nytimes/v2c.yaml`, `configs/nytimes/v2c_seed42.yaml`, `scripts/nytimes_v2c_seed42_job.sh`
- Modify: `tests/test_nytimes_configs.py`, `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md`

- [ ] **Step 1: Write the failing test** (append to `tests/test_nytimes_configs.py`)

```python
def test_v2c_is_v2_plus_the_set_critic():
    v2 = _flatten(_load("v2.yaml"))
    v2c = _flatten(_load("v2c.yaml"))
    assert v2c.pop("model.critic_type") == "neighbourhood_set"
    assert v2c.pop("model.critic_edge_dim") == 128
    assert v2c.pop("model.critic_edge_pool") == "max"
    assert v2c.pop("output_dir") == "runs/nytimes/v2c"
    v2.pop("model.critic_type")
    v2.pop("output_dir")
    assert v2c == v2


def test_v2c_seed42_is_v2c_with_an_absolute_real_path_and_its_own_output_dir():
    v2c = _flatten(_load("v2c.yaml"))
    inst = _flatten(_load("v2c_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2c_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2c.pop("output_dir")
    v2c.pop("data.real_path")
    assert inst == v2c


def test_v2c_job_script_runs_the_v2c_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2c_seed42_job.sh").read_text()
    assert "configs/nytimes/v2c_seed42.yaml" in script
    assert "runs/nytimes/v2c_seed42" in script
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_nytimes_configs.py -v`. Expected: `FileNotFoundError` on `v2c.yaml`.

- [ ] **Step 3: Write the files**

`configs/nytimes/v2c.yaml`: copy `configs/nytimes/v2.yaml`; replace the header; `output_dir: runs/nytimes/v2c`; in `model`, `critic_type: neighbourhood_set` and, directly under it, `critic_edge_dim: 128` and `critic_edge_pool: max`. (`critic_k` and `critic_distance_floor` stay as in `v2`; the floor is unused by this class and the test above expects it present.) Header:

```yaml
# NYTIMES v2c -- v2 with the feature extractor changed from a hand-chosen
# distance profile to a learned one: one EdgeConv layer over each row's 20
# within-batch neighbour differences (x_j - x_i), max-pooled, then the same
# MLP head. One change from v2:
#
#   1. model.critic_type: neighbourhood -> neighbourhood_set
#      (critic_edge_dim: 128, critic_edge_pool: max)
#
# Why: the spread of the neighbour differences is the local tangent
# structure, whose singular spectrum is what a collapsed sheet loses; this
# critic can read it directly rather than through the log-ratio profile.
# critic_distance_floor is unused by this class (selection is by ordering).
# See approach 3 in
# docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md.
```

`configs/nytimes/v2c_seed42.yaml`: same as `v2c.yaml` with `output_dir: runs/nytimes/v2c_seed42`, `real_path: /workspace/data-cache/nytimes_250k.npy`, and the `v2_seed42.yaml` instrument header with `v2` replaced by `v2c`.

`scripts/nytimes_v2c_seed42_job.sh`: copy of `scripts/nytimes_v2_seed42_job.sh` with `RUN=runs/nytimes/v2c_seed42`, `KEEP=/workspace/nytimes-v2/v2c_seed42`, config `configs/nytimes/v2c_seed42.yaml`, labels `v2c_best=` / `v2c_step30000=`. `chmod +x`.

`PROJECT_DOCUMENTATION.md`, `### critic_type`: add `neighbourhood_set` with one sentence ("`SetNeighbourhoodCritic`: one EdgeConv layer over each row's k within-batch neighbour differences, pooled, then the MLP head; it learns its own neighbourhood features and reads no distances") and the rows `| model.critic_edge_dim | 128 | Width of the edge MLP. |` and `| model.critic_edge_pool | max | max or mean over the k edges. |`.

`docs/datasets/nytimes.md` ladder row:

```markdown
| `v2c` | `v2` with a learned set critic (`critic_type: neighbourhood_set`, EdgeConv 128, max pool) | `configs/nytimes/v2c.yaml`; box instrument `configs/nytimes/v2c_seed42.yaml` | `runs/nytimes/v2c_seed42` (box: `/workspace/nytimes-v2/v2c_seed42`) | planned -- approach 3 of the neighbourhood-critic spec |
```

- [ ] **Step 4: Run the gate**

Run: `make check PYTHON=$PY`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add configs/nytimes/v2c.yaml configs/nytimes/v2c_seed42.yaml scripts/nytimes_v2c_seed42_job.sh tests/test_nytimes_configs.py PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "configs+docs(nytimes): v2c rung, the learned set critic"
```

---

### Task 5: mutation checks

- [ ] In `edge_features`, replace `x_j - x_i` with `x_j`: `test_edge_features_use_the_difference...` must FAIL. Restore.
- [ ] In `pooled`, replace the max with `e.reshape(e.shape[0], -1)[:, : self.edge_dim]` (a positional slice): `test_pooling_is_invariant_to_neighbour_order[max]` must FAIL. Restore.
- [ ] In `neighbourhood_indices`, pass a zero mask (comment out the diagonal fill in `_masked_squared_distances` temporarily): `test_indices_match_brute_force_and_exclude_self` must FAIL. Restore.
- [ ] `git status --short` empty, `make check PYTHON=$PY` green. Record all four outputs.

---

### Task 6: run `v2c` on the box

Identical to Task 10 of the `v2` plan with `v2c` names: push `nytimes-eda` (probe SSH first), `ssh tig-gpu nvidia-smi`, then

```bash
ssh tig-gpu "/opt/gpuq/venv/bin/gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-eda --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v2c_seed42_job.sh"
```

Poll with `gpuq show <id>` on a 10-minute background loop. Expected wall time about 45 minutes (the edge MLP is 512 x 20 x 512 x 128 per critic call, about 0.7 GFLOP; measure it from the job log and record it). Bring back `run_metadata.json`, `run_config.yaml`, `eda_clean/summary.json` by streamed tar with hashes verified, into `docs/results/nytimes-v2c-seed42/`. Build the claim table, then write `## v2c, measured` in `docs/datasets/nytimes.md` in the shape of `## v1, measured`, reporting the spec's bar per statistic, the `skip_share` trace, the 2x-neighbour check on the selection score, and `near_duplicate_fraction` real against fake. If the run diverges or the gradient penalty climbs above 1, rerun once with `critic_edge_pool: mean` as its own instrument config (`v2c_mean_seed42.yaml`, output dir `runs/nytimes/v2c_mean_seed42`) and report both. Commit locally.

---

## Self-review

- **Spec coverage.** EdgeConv on `[x_i, x_j - x_i]`, max pool with mean fallback, `H = critic_edge_dim` default 128: Task 2. `forward(x)` convention so the trainer is unchanged: Task 2. Extra tests table: neighbour-order invariance, translation test (with the deviation stated), both pools build and differ: Task 2. Rung `v2c`: Task 4. Summed-score penalty "doing real work": `test_gradient_reaches_neighbour_rows_through_the_differences`.
- **Placeholders.** None.
- **Type consistency.** `neighbourhood_indices(x, k)` in Tasks 1 and 2; `edge_features(x, idx)`, `pooled(x, idx)`, `edge`, `mlp`, `edge_dim`, `edge_pool` match between the class, its tests and the factory test.
