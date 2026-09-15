# Neighbourhood Critic, approach 2: bank neighbourhoods (`v2b`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train NYTimes `v2b`, whose critic profiles each row against a large detached bank (a fixed real subset for real rows, a ring of recent fakes for fake rows) instead of its 511 batch-mates.

**Architecture:** `BankNeighbourhoodCritic` in `src/models/critic.py` reuses `neighbourhood_distances(x, k, floor, bank=...)` and `profile_features` from the shared change, holds both banks as buffers (so they travel in `critic_state_dict`), and takes a `population` keyword (`"real"`, `"fake"`, `"mixed"`) that says which bank to query. The trainer passes that keyword through a small `score()` helper only when the critic declares `population_aware = True`, so `Critic` and `NeighbourhoodCritic` keep their `forward(x)`.

**Tech Stack:** Python 3, PyTorch, numpy, pytest, ruff. Local venv `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`; box `/opt/venvs/wgan-synthetic/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md`, sections "Shared" and "Approach 2".

**Prerequisite:** the shared change from `docs/superpowers/plans/2026-09-14-neighbourhood-critic-v2.md` Tasks 1 to 6 is merged on `nytimes-eda` (`build_critic`, `neighbourhood_distances`, `profile_features`, `drop_zero_rows`, `near_duplicate_fraction`, `component_energies`). Start from that commit; do not edit `NeighbourhoodCritic`.

## Global Constraints

- All of the `v2` plan's constraints hold: `Critic` untouched, float32 neighbour maths, floor on squared distances, no `torch.cdist`, explicit `git add` paths, `make check` before each commit.
- `critic_bank_size` default `16384`. Both banks are `nn.Module` buffers, float32, shape `(bank_size, D)`.
- Gradients flow through the query row only; every bank read is under the bank's `requires_grad=False` (buffers never require grad).
- Interpolates in the gradient penalty query the union of the real bank and the filled part of the fake bank (spec decision; do not re-decide).
- Until the fake bank has been written `bank_size` rows, fake rows are profiled within-batch exactly as `NeighbourhoodCritic` does, and the trainer records the step at which the bank first filled as `run_metadata.json["fake_bank_filled_at_step"]`.

## Deviations from the spec

**Real bank carve-out instead of index exclusion.** The spec says a real row that is in the bank is excluded by index. The `DataLoader` yields tensors without indices, so index exclusion would need the dataset to yield `(row, index)` and every batch site in the trainer to unpack it. Instead the trainer removes the bank rows from the training split before building the loader: a real batch row is then never in the bank by construction, and an exact copy of it in the bank is a *different* row and is kept, which is what the spec's index rule was protecting. Cost: `bank_size` of the roughly 237k training rows (7%) are seen only as neighbours, never as critic inputs. `run_metadata.json["data"]["num_train"]` records the reduced count and `["real_bank_rows"]` the bank size. The `self_index` argument of `neighbourhood_distances` stays available; this class does not need it.

## File map

| file | change |
|---|---|
| `src/models/critic.py` | `BankNeighbourhoodCritic`, `"neighbourhood_bank"` in `CRITIC_TYPES` and `build_critic` |
| `src/train/train_wgan_gp.py` | `score()` helper, `population=` on `gradient_penalty`, bank carve-out, `push_fake`, fill-step metadata |
| `tests/test_bank_critic.py` | new |
| `tests/test_critic_factory.py`, `tests/test_train_smoke.py` | extended |
| `configs/nytimes/v2b.yaml`, `configs/nytimes/v2b_seed42.yaml`, `scripts/nytimes_v2b_seed42_job.sh` | new |
| `tests/test_nytimes_configs.py` | extended |
| `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md` | documented |

---

### Task 1: `BankNeighbourhoodCritic`

**Files:**
- Modify: `src/models/critic.py`
- Test: `tests/test_bank_critic.py` (new)

**Interfaces:**
- Consumes: `neighbourhood_distances(x, k, floor, bank=None, self_index=None)`, `profile_features(r)`, `Critic`, `DEFAULT_K`, `DEFAULT_DISTANCE_FLOOR`.
- Produces: `BankNeighbourhoodCritic(input_dim, hidden_dims, k=DEFAULT_K, distance_floor=DEFAULT_DISTANCE_FLOOR, bank_size=16384, negative_slope=0.2)` with
  - class attribute `population_aware = True`
  - `set_real_bank(rows: Tensor)` (must be exactly `bank_size` rows)
  - `push_fake(rows: Tensor)` (detached ring write)
  - property `fake_bank_is_full: bool`
  - `features(x, population: str) -> (n, k)`
  - `forward(x, population: str = "fake") -> (n,)`
  - buffers `real_bank`, `fake_bank`, `fake_bank_ptr` (0-d long), `fake_bank_filled` (0-d long)
  - submodule `mlp: Critic` with input width `D + k`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bank_critic.py
"""The bank-neighbourhood critic: neighbours come from a detached bank, not
the batch. Each test names the mutation it catches."""

import numpy as np
import pytest
import torch

from src.models.critic import BankNeighbourhoodCritic, NeighbourhoodCritic


def _unit(seed, n, dim):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def _critic(bank_size=16, k=3, dim=4):
    torch.manual_seed(0)
    c = BankNeighbourhoodCritic(input_dim=dim, hidden_dims=[6], k=k, bank_size=bank_size)
    c.set_real_bank(_unit(100, bank_size, dim))
    return c


def test_declares_itself_population_aware():
    assert BankNeighbourhoodCritic.population_aware is True
    assert not getattr(NeighbourhoodCritic, "population_aware", False)


def test_one_score_per_row_for_every_population():
    c = _critic()
    x = _unit(1, 5, 4)
    for pop in ("real", "fake", "mixed"):
        assert c(x, population=pop).shape == (5,), pop


def test_real_rows_are_profiled_against_the_real_bank_not_the_batch():
    """Catches the bank accepted and ignored (within-batch fallback for real)."""
    c = _critic(bank_size=16, k=3)
    x = _unit(2, 5, 4)
    from_bank = c.features(x, population="real")
    within_batch = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3).features(x)
    assert not torch.allclose(from_bank, within_batch)
    # And it is exactly the bank profile:
    from src.models.critic import neighbourhood_distances, profile_features

    expected = profile_features(
        neighbourhood_distances(x, 3, c.distance_floor, bank=c.real_bank)
    )
    torch.testing.assert_close(from_bank, expected)


def test_set_real_bank_rejects_the_wrong_size():
    c = BankNeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, bank_size=16)
    with pytest.raises(ValueError, match="bank_size"):
        c.set_real_bank(_unit(3, 15, 4))


def test_fake_rows_fall_back_to_within_batch_until_the_bank_fills():
    """Before the ring has been written bank_size rows, fake scores must equal
    NeighbourhoodCritic's on the same weights. Catches the fallback wired to
    the wrong feature (e.g. querying a zero-filled bank)."""
    c = _critic(bank_size=16, k=3)
    ref = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3)
    ref.mlp.load_state_dict(c.mlp.state_dict())
    x = _unit(4, 8, 4)
    assert not c.fake_bank_is_full
    torch.testing.assert_close(c(x, population="fake"), ref(x))
    c.push_fake(_unit(5, 8, 4))  # 8 of 16: still not full
    assert not c.fake_bank_is_full
    torch.testing.assert_close(c(x, population="fake"), ref(x))
    c.push_fake(_unit(6, 8, 4))  # 16 of 16
    assert c.fake_bank_is_full
    assert not torch.allclose(c(x, population="fake"), ref(x))


def test_ring_overwrites_the_oldest_rows():
    """Catches a ring pointer off by one or a bank that stops accepting."""
    c = _critic(bank_size=16, k=3)
    marker = torch.full((1, 4), 0.5)
    c.push_fake(marker)                   # slot 0
    c.push_fake(_unit(7, 15, 4))          # slots 1..15 -> full, ptr wraps to 0
    assert c.fake_bank_is_full
    assert torch.equal(c.fake_bank[0], marker[0])
    c.push_fake(_unit(8, 1, 4))           # overwrites slot 0
    assert not torch.equal(c.fake_bank[0], marker[0])
    assert int(c.fake_bank_ptr) == 1


def test_push_fake_stores_detached_float32_rows():
    """Catches a bank that keeps the generator graph alive (memory) or
    silently stores fp16."""
    c = _critic(bank_size=16, k=3)
    rows = _unit(9, 4, 4).requires_grad_(True)
    c.push_fake(rows)
    assert c.fake_bank.requires_grad is False
    assert c.fake_bank.dtype == torch.float32


def test_mixed_population_queries_both_banks():
    """Interpolates in the gradient penalty query real + filled fake. Catches
    'mixed' aliased to one of the two."""
    c = _critic(bank_size=16, k=3)
    c.push_fake(_unit(10, 16, 4))
    x = _unit(11, 5, 4)
    from src.models.critic import neighbourhood_distances, profile_features

    union = torch.cat([c.real_bank, c.fake_bank])
    expected = profile_features(neighbourhood_distances(x, 3, c.distance_floor, bank=union))
    torch.testing.assert_close(c.features(x, population="mixed"), expected)


def test_mixed_before_fill_uses_only_the_filled_part_of_the_fake_bank():
    """Catches zero-initialised, never-written fake rows leaking into the
    union (a zero row is at distance 1.0 from every unit row)."""
    c = _critic(bank_size=16, k=3)
    c.push_fake(_unit(12, 4, 4))  # 4 of 16 written
    x = _unit(13, 5, 4)
    from src.models.critic import neighbourhood_distances, profile_features

    union = torch.cat([c.real_bank, c.fake_bank[:4]])
    expected = profile_features(neighbourhood_distances(x, 3, c.distance_floor, bank=union))
    torch.testing.assert_close(c.features(x, population="mixed"), expected)


def test_unknown_population_raises():
    c = _critic()
    with pytest.raises(ValueError, match="population"):
        c(_unit(14, 5, 4), population="interpolated")


def test_gradient_flows_through_the_query_only():
    """Catches a bank that is a parameter or a non-detached tensor."""
    c = _critic(bank_size=16, k=3)
    x = _unit(15, 5, 4).requires_grad_(True)
    c(x, population="real").sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert c.real_bank.grad is None


def test_state_dict_round_trip_restores_both_banks_and_the_pointer():
    """Catches banks held as plain attributes instead of buffers, which a
    resume would silently reset."""
    a = _critic(bank_size=16, k=3)
    a.push_fake(_unit(16, 10, 4))
    b = BankNeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, bank_size=16)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.real_bank, b.real_bank)
    assert torch.equal(a.fake_bank, b.fake_bank)
    assert int(b.fake_bank_ptr) == 10 and int(b.fake_bank_filled) == 10
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_bank_critic.py -v`
Expected: FAIL at import, `ImportError: cannot import name 'BankNeighbourhoodCritic'`.

- [ ] **Step 3: Implement** (append to `src/models/critic.py`, before `CRITIC_TYPES`)

```python
POPULATIONS = ("real", "fake", "mixed")


class BankNeighbourhoodCritic(nn.Module):
    """`NeighbourhoodCritic`'s features, but the neighbours of a row come
    from a large detached bank rather than its batch-mates: a fixed real
    subset for real rows, a ring of recent generator outputs for fake rows,
    and the union of both for gradient-penalty interpolates.

    Why: at batch 512 the profile is measured among 511 rows; the gate's
    statistics are 100-NN in 250k. A 16k bank moves the profile toward the
    gate's scale, and it exposes exact copies for a few percent of real rows
    (0.15% within a batch), so `near_duplicate_fraction` becomes a live
    target rather than a diagnostic.

    Gradients flow through the query row only; both banks are buffers.
    Until the fake ring has been written `bank_size` rows, fake rows are
    profiled within-batch, exactly as `NeighbourhoodCritic` does.

    `population_aware = True` tells the trainer to pass `population=`;
    the per-vector and within-batch critics do not take it.
    """

    population_aware = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        k: int = DEFAULT_K,
        distance_floor: float = DEFAULT_DISTANCE_FLOOR,
        bank_size: int = 16384,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if k < 2:
            raise ValueError(f"critic_k must be at least 2 (one ratio entry), got {k}")
        if distance_floor <= 0.0:
            raise ValueError(f"critic_distance_floor must be positive, got {distance_floor}")
        if bank_size <= k:
            raise ValueError(f"critic_bank_size must exceed critic_k={k}, got {bank_size}")
        self.k = int(k)
        self.distance_floor = float(distance_floor)
        self.bank_size = int(bank_size)
        self.mlp = Critic(
            input_dim=input_dim + self.k,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )
        self.register_buffer("real_bank", torch.zeros(self.bank_size, input_dim))
        self.register_buffer("fake_bank", torch.zeros(self.bank_size, input_dim))
        self.register_buffer("fake_bank_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("fake_bank_filled", torch.zeros((), dtype=torch.long))

    def set_real_bank(self, rows: Tensor) -> None:
        if rows.shape != self.real_bank.shape:
            raise ValueError(
                f"real bank must be exactly bank_size x input_dim = "
                f"{tuple(self.real_bank.shape)}, got {tuple(rows.shape)}"
            )
        with torch.no_grad():
            self.real_bank.copy_(rows.detach().float())

    @property
    def fake_bank_is_full(self) -> bool:
        return int(self.fake_bank_filled) >= self.bank_size

    @torch.no_grad()
    def push_fake(self, rows: Tensor) -> None:
        rows = rows.detach().float().to(self.fake_bank.device)
        n = rows.shape[0]
        ptr = int(self.fake_bank_ptr)
        idx = (ptr + torch.arange(n, device=rows.device)) % self.bank_size
        self.fake_bank[idx] = rows
        self.fake_bank_ptr.fill_((ptr + n) % self.bank_size)
        self.fake_bank_filled.fill_(min(int(self.fake_bank_filled) + n, self.bank_size))

    def _bank_for(self, population: str) -> Tensor | None:
        if population == "real":
            return self.real_bank
        if population == "fake":
            return self.fake_bank if self.fake_bank_is_full else None
        if population == "mixed":
            return torch.cat([self.real_bank, self.fake_bank[: int(self.fake_bank_filled)]])
        raise ValueError(f"population must be one of {POPULATIONS}, got {population!r}")

    def features(self, x: Tensor, population: str) -> Tensor:
        bank = self._bank_for(population)
        r = neighbourhood_distances(x, self.k, self.distance_floor, bank=bank)
        return profile_features(r)

    def forward(self, x: Tensor, population: str = "fake") -> Tensor:
        phi = self.features(x, population).to(x.dtype)
        return self.mlp(torch.cat([x, phi], dim=1))
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_bank_critic.py tests/test_neighbourhood_critic.py tests/test_critic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_bank_critic.py
git commit -m "feat(critic): BankNeighbourhoodCritic, neighbours from a real bank and a fake ring"
```

---

### Task 2: factory branch

**Files:**
- Modify: `src/models/critic.py` (`CRITIC_TYPES`, `build_critic`)
- Test: `tests/test_critic_factory.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_critic_factory.py`; add `BankNeighbourhoodCritic` to its import)

```python
def test_neighbourhood_bank_reads_bank_size_from_the_config():
    cfg = dict(BASE_CFG, critic_type="neighbourhood_bank", critic_k=4, critic_bank_size=64)
    critic = build_critic(cfg, input_dim=12)
    assert isinstance(critic, BankNeighbourhoodCritic)
    assert critic.bank_size == 64 and critic.k == 4
    assert critic.real_bank.shape == (64, 12)


def test_neighbourhood_bank_default_bank_size_is_16384():
    critic = build_critic(dict(BASE_CFG, critic_type="neighbourhood_bank"), input_dim=12)
    assert critic.bank_size == 16384
```

Change `test_the_documented_types` to `assert CRITIC_TYPES == ("per_vector", "neighbourhood", "neighbourhood_bank")` (if the `v2c` plan has already landed, the tuple has `"neighbourhood_set"` too; keep whatever order is already there and append).

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_critic_factory.py -v`
Expected: FAIL, `ValueError: Unknown critic_type: 'neighbourhood_bank'`.

- [ ] **Step 3: Implement**

```python
CRITIC_TYPES = ("per_vector", "neighbourhood", "neighbourhood_bank")
```

and in `build_critic`, before the `raise`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_critic_factory.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_critic_factory.py
git commit -m "feat(critic): build_critic dispatches neighbourhood_bank"
```

---

### Task 3: the trainer: population routing, bank carve-out, ring writes

**Files:**
- Modify: `src/train/train_wgan_gp.py` (`gradient_penalty` at `:128`; construction after `build_critic`; critic loop at `:600-625`; generator step at `:638-642`; `run_meta`)
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Produces: `score(critic: nn.Module, x: Tensor, population: str) -> Tensor` (module-level in the trainer); `gradient_penalty(critic, real, fake, device, population: str = "mixed")`; `run_metadata.json` keys `data.real_bank_rows` and `fake_bank_filled_at_step`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_train_smoke.py`)

```python
def test_bank_critic_trains_carves_the_bank_and_records_the_fill_step(tmp_path):
    """Catches: the bank not carved (num_train unchanged), the fill step
    not recorded, and the ring not surviving a state-dict round trip."""
    from src.models.critic import BankNeighbourhoodCritic, build_critic

    cfg = make_config(tmp_path, "mlp")
    cfg["model"]["critic_type"] = "neighbourhood_bank"
    cfg["model"]["critic_k"] = 5
    cfg["model"]["critic_bank_size"] = 64   # 256 synthetic rows, 20% holdout -> 205 train
    cfg["training"]["num_gen_steps"] = 4    # batch 32: the ring fills at step 2
    ckpt_path, meta = train(cfg)
    assert meta["data"]["real_bank_rows"] == 64
    assert meta["data"]["num_train"] == 205 - 64
    assert meta["fake_bank_filled_at_step"] == 2
    saved = torch.load(ckpt_path, weights_only=False)
    rebuilt = build_critic(cfg["model"], input_dim=16)
    assert isinstance(rebuilt, BankNeighbourhoodCritic)
    rebuilt.load_state_dict(saved["critic_state_dict"])
    assert rebuilt.fake_bank_is_full
    # The carved bank is a subset of the *training* split: none of its rows
    # is a holdout row. Re-derive the split the way the trainer does.
    from src.data.dataset import PreprocessConfig, build_training_data

    x_train, x_holdout, _ = build_training_data(
        descriptor_path=None, file_format="npy", descriptor_dim=16,
        holdout_fraction=0.2, preprocess_cfg=PreprocessConfig(), seed=0,
        synthetic_if_missing=True, synthetic_num_vectors=256,
    )
    bank = rebuilt.real_bank.numpy()
    d_train = ((bank[:, None, :] - x_train[None, :, :]) ** 2).sum(-1).min(1)
    assert (d_train < 1e-10).all()


def test_bank_critic_real_rows_query_the_real_bank(tmp_path, monkeypatch):
    """Catches the trainer scoring real rows without population='real'."""
    from src.models import critic as critic_mod

    seen = []
    original = critic_mod.BankNeighbourhoodCritic.forward

    def spy(self, x, population="fake"):
        seen.append(population)
        return original(self, x, population)

    monkeypatch.setattr(critic_mod.BankNeighbourhoodCritic, "forward", spy)
    cfg = make_config(tmp_path, "mlp")
    cfg["model"].update(critic_type="neighbourhood_bank", critic_k=5, critic_bank_size=64)
    cfg["training"]["num_gen_steps"] = 1
    cfg["training"]["eval_every"] = 1
    train(cfg)
    # n_critic=2: each critic step calls real, fake, mixed (the penalty); the
    # generator step calls fake once.
    assert seen == ["real", "fake", "mixed", "real", "fake", "mixed", "fake"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_train_smoke.py -k bank -v`
Expected: FAIL, `KeyError: 'real_bank_rows'` and the population sequence is all `"fake"`.

- [ ] **Step 3: Implement**

Module-level helper, next to `gradient_penalty`:

```python
def score(critic: nn.Module, x: Tensor, population: str) -> Tensor:
    """Score a batch, telling a population-aware critic which population it
    is. `Critic` and `NeighbourhoodCritic` take no such argument, so the
    keyword is passed only when the class declares `population_aware`."""
    if getattr(critic, "population_aware", False):
        return critic(x, population=population)
    return critic(x)
```

`gradient_penalty` gains `population: str = "mixed"` and calls `score(critic, interpolated, population)` instead of `critic(interpolated)`; its docstring gains: "A population-aware critic profiles interpolates against the union of its banks (`population='mixed'`)."

Construction, right after `critic = build_critic(...)` and before `dataset = NumpyTensorDataset(x_train)`:

```python
    # A bank critic profiles real rows against a fixed subset of the training
    # split. Those rows are removed from the loader so a batch row is never
    # its own neighbour (a copy of it in the bank is a different row and is
    # kept). Seeded from the run seed so a resume carves the same rows.
    real_bank_rows = 0
    if getattr(critic, "population_aware", False):
        bank_size = int(critic.bank_size)
        if bank_size >= x_train.shape[0]:
            raise ValueError(
                f"critic_bank_size={bank_size} must be below the training split "
                f"({x_train.shape[0]} rows)"
            )
        carve = np.random.default_rng(seed).permutation(x_train.shape[0])
        bank_idx = np.sort(carve[:bank_size])
        keep_idx = np.sort(carve[bank_size:])
        critic.set_real_bank(torch.from_numpy(x_train[bank_idx]).to(device))
        x_train = x_train[keep_idx]
        real_bank_rows = bank_size
```

Then `run_meta["data"]["num_train"]` already reads `x_train.shape[0]` after this point (confirm the `run_meta` dict is built *after* the carve; move the carve above it if not), and add `"real_bank_rows": real_bank_rows,` to `run_meta["data"]`.

Critic loop: `d_real = score(critic, real, "real")`, `d_fake = score(critic, fake, "fake")`, and `gp = gradient_penalty(critic, real, fake, device=device, population="mixed")`.

Generator step: `adv_loss = -score(critic, fake, "fake").mean()`, and immediately after that line:

```python
            if getattr(critic, "population_aware", False):
                was_full = critic.fake_bank_is_full
                critic.push_fake(fake)
                if critic.fake_bank_is_full and not was_full:
                    run_meta["fake_bank_filled_at_step"] = step
```

(`push_fake` detaches internally; `fake` here is the live generator output, which is what the ring should hold.)

On resume, the ring and its counters come back through `critic.load_state_dict`, and the fill step, if it happened before the resume, is not re-recorded: read it from the previous run's metadata.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_train_smoke.py tests/test_bank_critic.py tests/test_neighbourhood_critic.py -v`
Expected: all PASS, including every pre-existing smoke test (the `score` helper is a no-op for the other critics).

- [ ] **Step 5: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_train_smoke.py
git commit -m "feat(train): route real/fake/mixed populations to a bank critic; carve the real bank; ring-write fakes"
```

---

### Task 4: configs, script, docs

**Files:**
- Create: `configs/nytimes/v2b.yaml`, `configs/nytimes/v2b_seed42.yaml`, `scripts/nytimes_v2b_seed42_job.sh`
- Modify: `tests/test_nytimes_configs.py`, `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md`

- [ ] **Step 1: Write the failing test** (append to `tests/test_nytimes_configs.py`)

```python
def test_v2b_is_v2_plus_the_bank_critic():
    v2 = _flatten(_load("v2.yaml"))
    v2b = _flatten(_load("v2b.yaml"))
    assert v2b.pop("model.critic_type") == "neighbourhood_bank"
    assert v2b.pop("model.critic_bank_size") == 16384
    assert v2b.pop("output_dir") == "runs/nytimes/v2b"
    v2.pop("model.critic_type")
    v2.pop("output_dir")
    assert v2b == v2


def test_v2b_seed42_is_v2b_with_an_absolute_real_path_and_its_own_output_dir():
    v2b = _flatten(_load("v2b.yaml"))
    inst = _flatten(_load("v2b_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2b_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2b.pop("output_dir")
    v2b.pop("data.real_path")
    assert inst == v2b


def test_v2b_job_script_runs_the_v2b_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2b_seed42_job.sh").read_text()
    assert "configs/nytimes/v2b_seed42.yaml" in script
    assert "runs/nytimes/v2b_seed42" in script
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_nytimes_configs.py -v`. Expected: FAIL, `FileNotFoundError` on `v2b.yaml`.

- [ ] **Step 3: Write the files**

`configs/nytimes/v2b.yaml`: copy `configs/nytimes/v2.yaml`; replace the header; set `output_dir: runs/nytimes/v2b`; in `model`, change `critic_type: neighbourhood` to `critic_type: neighbourhood_bank` and add `critic_bank_size: 16384` directly under it. Header:

```yaml
# NYTIMES v2b -- v2 with the neighbour source changed from the batch to a
# bank: real rows are profiled against a fixed 16,384-row subset of the
# training split (carved out of the loader, so a row is never its own
# neighbour), fake rows against a ring of the last 32 generator batches,
# gradient-penalty interpolates against both. One change from v2:
#
#   1. model.critic_type: neighbourhood -> neighbourhood_bank
#      (critic_bank_size: 16384)
#
# Why: at batch 512 the profile sees 511 rows and exposes an exact copy for
# 0.15% of real rows; a 16k bank moves the profile toward the gate's 100-NN
# scale and exposes copies for a few percent, so near_duplicate_fraction is
# a live target here. See approach 2 in
# docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md.
```

`configs/nytimes/v2b_seed42.yaml`: same as `v2b.yaml` with `output_dir: runs/nytimes/v2b_seed42`, `real_path: /workspace/data-cache/nytimes_250k.npy`, and the `v2_seed42.yaml` instrument header with `v2` replaced by `v2b`.

`scripts/nytimes_v2b_seed42_job.sh`: copy of `scripts/nytimes_v2_seed42_job.sh` with `RUN=runs/nytimes/v2b_seed42`, `KEEP=/workspace/nytimes-v2/v2b_seed42`, the config `configs/nytimes/v2b_seed42.yaml`, and the synthetic labels `v2b_best=` / `v2b_step30000=`. `chmod +x`.

`PROJECT_DOCUMENTATION.md`, in the `### critic_type` section: add `neighbourhood_bank` to the list of values with one sentence ("`BankNeighbourhoodCritic`: the same features measured against a 16k detached bank, a fixed real subset carved from the training split for real rows and a ring of recent fakes for fake rows; the trainer passes `population=` to it") and the row `| model.critic_bank_size | 16384 | Rows in each bank. Carved from the training split, so num_train drops by this much. |`, plus the two metadata keys `data.real_bank_rows` and `fake_bank_filled_at_step`.

`docs/datasets/nytimes.md` ladder row:

```markdown
| `v2b` | `v2` with bank neighbourhoods (`critic_type: neighbourhood_bank`, bank 16,384) | `configs/nytimes/v2b.yaml`; box instrument `configs/nytimes/v2b_seed42.yaml` | `runs/nytimes/v2b_seed42` (box: `/workspace/nytimes-v2/v2b_seed42`) | planned -- approach 2 of the neighbourhood-critic spec |
```

- [ ] **Step 4: Run the gate**

Run: `make check PYTHON=$PY`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add configs/nytimes/v2b.yaml configs/nytimes/v2b_seed42.yaml scripts/nytimes_v2b_seed42_job.sh tests/test_nytimes_configs.py PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "configs+docs(nytimes): v2b rung, the bank-neighbourhood critic"
```

---

### Task 5: mutation checks

- [ ] Remove the carve-out (`x_train = x_train[keep_idx]`): `test_bank_critic_trains_carves_the_bank...` must FAIL on `num_train`. Restore.
- [ ] Make `_bank_for("mixed")` return `self.real_bank`: `test_mixed_population_queries_both_banks` must FAIL. Restore.
- [ ] Make `push_fake` write at `ptr` without the modulo: `test_ring_overwrites_the_oldest_rows` must FAIL (index error or stale marker). Restore.
- [ ] `git status --short` empty, `make check PYTHON=$PY` green. Record all four outputs.

---

### Task 6: run `v2b` on the box

Identical to Task 10 of the `v2` plan with `v2b` names: push `nytimes-eda` (probe SSH first), `ssh tig-gpu nvidia-smi`, then

```bash
ssh tig-gpu "/opt/gpuq/venv/bin/gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-eda --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v2b_seed42_job.sh"
```

Poll with `gpuq show <id>` on a 10-minute background loop. Expected wall time about 45 minutes (the bank GEMM is 512 x 16,384 x 256 per critic call; measure it from the job log and record it). Bring back `run_metadata.json`, `run_config.yaml`, `eda_clean/summary.json` by streamed tar with hashes verified, into `docs/results/nytimes-v2b-seed42/`. Build the claim table, then write `## v2b, measured` in `docs/datasets/nytimes.md` in the shape of `## v1, measured`, reporting the spec's bar per statistic, the `skip_share` trace, the 2x-neighbour check on the selection score, `gate_real_near_duplicate_fraction` against `gate_fake_near_duplicate_fraction` (the live target for this approach), `fake_bank_filled_at_step`, and whether the Wasserstein trace shows a sawtooth at the 32-step ring period. Commit locally.

---

## Self-review

- **Spec coverage.** Real bank fixed and seeded, stored in the checkpoint: Task 1 (buffer) + Task 3 (carve). Fake ring, within-batch fallback, fill step recorded: Tasks 1, 3. Mixed union for interpolates: Tasks 1, 3. Extra tests table: self-exclusion becomes the carve-out test (deviation recorded); fallback equality; ring overwrite; checkpoint round-trip. Rung: Task 4. Deviation from index exclusion is stated at the top with its reason.
- **Placeholders.** None.
- **Type consistency.** `population_aware`, `set_real_bank`, `push_fake`, `fake_bank_is_full`, `features(x, population)`, `forward(x, population="fake")` are used with those names in Tasks 1, 3 and the tests. `score(critic, x, population)` and `gradient_penalty(..., population="mixed")` match between Task 3's helper and its call sites.
