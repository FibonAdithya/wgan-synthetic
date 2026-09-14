# NYTimes `v2b` Bank-Neighbourhood Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build approach 2 of the neighbourhood-critic design, `BankNeighbourhoodCritic`, wire it into the trainer, ship the NYTimes `v2b` rung, train it on the box, and record whether it meets the bar.

**Architecture:** `BankNeighbourhoodCritic` subclasses the within-batch `NeighbourhoodCritic` that landed on `nytimes-eda` with `v2`, and reuses its MLP unchanged. Real rows are profiled against a fixed 16k-row subset of the training split, fake rows against a ring buffer of recent generator outputs, and a bare call (which is what `gradient_penalty` makes) against the union of the two. The trainer learns three things: to yield row indices so a real row can be excluded from its own bank slot, to route the three scoring calls by population, and to write the fake ring after every generator step. The trunk/skip balance the success bar reads is already logged per evaluation by the landed trainer.

**Tech Stack:** Python 3.12, PyTorch 2.13, numpy, pytest, ruff. Run everything with the project venv, `~/TIG/wgan-synthetic/.venv/bin/python`, from the worktree `~/.herdr/worktrees/wgan-synthetic/nytimes-v2b` on branch `nytimes-v2b`. The box is `tig-gpu` (RTX 3060), queue `gpuq` at `/opt/gpuq/venv/bin/gpuq`, project venv `/opt/venvs/wgan-synthetic`.

**Spec:** `docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md`, sections "Shared" and "Approach 2: bank neighbourhoods (`v2b`)". The plan argues from the spec; read both.

## Global Constraints

- `make check` (ruff lint, ruff format check, pytest) must stay green after every task; it is CPU-only. Run it as `PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check`. Format only the files you touched with `ruff format <file>`; never `make format` repo-wide.
- The shared change and `v2` have landed on `nytimes-eda` (`build_critic`, `neighbourhood_distances`, `profile_features`, `NeighbourhoodCritic`, `drop_zero_rows`, `near_duplicate_fraction`, per-evaluation trunk/skip energies). This branch is rebased on them. **Do not edit `NeighbourhoodCritic` or `neighbourhood_distances`** (spec, "Running the three in parallel"). The landed `neighbourhood_distances(x, k, floor, bank=None, self_index=None)` already does what the bank critic needs.
- The only edits to landed code are: adding `"neighbourhood_bank"` to `CRITIC_TYPES` and a branch to `build_critic`; updating the one landed test that pins the tuple (`tests/test_critic_factory.py::test_the_documented_types`); the trainer wiring in Task 3; and the docs.
- `gradient_penalty` in `src/train/train_wgan_gp.py` is not changed (spec, "Gradient penalty"). Its bare `critic(interpolated)` call therefore scores the `mixed` population, which is the class's default.
- Default behaviour must not move: `critic_type` defaults to `per_vector`, every existing config trains as before, and `tests/test_train_smoke.py` passes untouched. The row-index dataset is used only when the critic is the bank type.
- Checkpoint keys existing files rely on are not changed. The real-bank indices ride inside `critic_state_dict` as a persistent buffer; the bank rows and the fake ring are non-persistent and are not written.
- Numerics follow the spec and the landed function: float32 with autocast disabled, expanded squares with `clamp(min=0)`, exclusion by index, `topk(..., largest=False, sorted=True)`, floor applied as `clamp(min=floor**2)` after the mask.
- Config keys and defaults: `critic_type: neighbourhood_bank`, `critic_k: 20`, `critic_distance_floor: 0.01`, `critic_bank_size: 16384`.
- `v1.yaml`, `v1_seed42.yaml`, `v2.yaml`, `v2_seed42.yaml` are ladder rungs and are not edited. New configs only. `tests/test_nytimes_configs.py` pins `v2` against `v1`; Task 5 adds the same pins for `v2b` against `v2`.
- Every test names the mutation it catches (spec, "Tests shared by all three approaches"). A test that catches nothing is not written. After implementing, run the mutation checks listed in Tasks 2 and 3 and record them in the PR body.
- Commit after each task with explicit paths (never `git add -A` or `git commit -a`). Run `git status --short` before each commit. Commit messages end with the session's attribution lines.
- No push and no box job until Task 6. Transfers off the box are streamed with `tar` and verified by `sha256sum`; plain `scp`/`rsync` are blocked.
- Every number in the results page is labelled MEASURED (with the command that produced it) or `ESTIMATE (unverified)`.
- `$SCRATCHPAD` in Task 6 is the session's scratchpad directory (named in the system prompt); nothing written there is committed.

---

### Task 0: Rebase onto the landed shared change and reconcile names

**Done by the controller on 2026-09-14.** The branch is rebased on `nytimes-eda` at `08fc222` ("record the v2 seed-42 run"). What the plan had assumed from the spec, against what landed:

| assumed | landed | consequence |
|---|---|---|
| `neighbourhood_distances(x, k, floor)` only | `neighbourhood_distances(x, k, floor, bank=None, self_index=None)`, with tests for the bank path and `self_index` | Task 1 (adding `bank`/`exclude`) is superseded; the keyword is `self_index` |
| `NeighbourhoodCritic.__init__(input_dim, hidden_dims, negative_slope, k, distance_floor)` | `__init__(input_dim, hidden_dims, k=DEFAULT_K, distance_floor=DEFAULT_DISTANCE_FLOOR, negative_slope=0.2)` | keyword order; module constants `DEFAULT_K`, `DEFAULT_DISTANCE_FLOOR` exist |
| attribute `net` | attribute `mlp` (a `Critic`), plus a `features(x)` method; `forward` casts `phi.to(x.dtype)` | the subclass overrides `features` and `forward`, reuses `self.mlp` |
| `build_critic` only | `build_critic` plus `CRITIC_TYPES = ("per_vector", "neighbourhood")` pinned by `tests/test_critic_factory.py::test_the_documented_types` | the tuple and that test change together |
| trunk/skip balance measured post hoc | `LinearSkipGenerator.component_energies` logs `trunk_energy`, `skip_energy`, `skip_share`, `trunk_skip_abs_cos` into every `run_meta["eval"]` entry | Task 4 (a balance module) is superseded; Task 6 reads the eval entries |
| a `v2` config exists | `v2.yaml`, `v2_seed42.yaml`, `scripts/nytimes_v2_seed42_job.sh`, and `tests/test_nytimes_configs.py` pinning `v2` = `v1` + four keys | Task 5 pins `v2b` = `v2` + two keys the same way |

Nothing else to do here.

---

### Task 1: (superseded)

The landed `neighbourhood_distances` already takes `bank` and `self_index`, and `tests/test_neighbourhood_critic.py` covers: bank neighbours come from the bank; `self_index` excludes a row from a bank that contains it; `-1` means not in the bank; exclusions count toward candidate validation. The one test this plan had that those do not cover, a rolled bank that distinguishes `[row, slot]` from `[slot, row]`, moves into Task 2's test file. No implementation work.

---

### Task 2: `BankNeighbourhoodCritic`, the bank draw, the population dispatcher and the factory branch

**Files:**
- Modify: `src/models/critic.py` (append after `NeighbourhoodCritic`; extend `CRITIC_TYPES`; one branch in `build_critic`)
- Modify: `tests/test_critic_factory.py` (the tuple assertion in `test_the_documented_types`)
- Test: `tests/test_critic_bank.py` (create)

**Interfaces:**
- Consumes: `NeighbourhoodCritic` (attributes `mlp`, `k`, `distance_floor`), `neighbourhood_distances(x, k, floor, bank=, self_index=)`, `profile_features(r)`, `build_critic`, `CRITIC_TYPES`, `DEFAULT_K`, `DEFAULT_DISTANCE_FLOOR`.
- Produces:
  - `draw_real_bank_indices(num_rows: int, bank_size: int, seed: int) -> Tensor`: `(bank_size,)` long, distinct, CPU, deterministic in `seed`; raises `ValueError` if `bank_size > num_rows`.
  - `class BankNeighbourhoodCritic(NeighbourhoodCritic)` with `__init__(self, input_dim, hidden_dims, k=DEFAULT_K, distance_floor=DEFAULT_DISTANCE_FLOOR, negative_slope=0.2, bank_size=16384)`; attributes `bank_size: int`, `real_bank_indices` (persistent long buffer, `(bank_size,)`, `-1` until set), `real_bank` and `fake_bank` (non-persistent float buffers, `(bank_size, input_dim)`), `fake_rows_written: int`, `row_to_slot` (non-persistent long buffer, empty until `set_real_bank`, then `(num_train,)` with the bank slot per training row or `-1`); property `fake_bank_full: bool`; methods `set_real_bank(x_train: Tensor, indices: Tensor) -> None`, `write_fake(rows: Tensor) -> None`, `features(x, population="mixed", row_ids=None) -> (n, k)`, `forward(x, population: str = "mixed", row_ids: Tensor | None = None) -> Tensor` of shape `(n,)`. Raises `ValueError` for `bank_size <= k` and for an unknown `population`; `RuntimeError` when `population="real"`, or `population="mixed"` once the ring is full, is scored before `set_real_bank` (property `real_bank_set: bool`).
  - `score_population(critic: nn.Module, x: Tensor, population: str, row_ids: Tensor | None = None) -> Tensor`: routes to `critic(x, population=..., row_ids=...)` for the bank class and to `critic(x)` for every other critic.
  - `CRITIC_TYPES == ("per_vector", "neighbourhood", "neighbourhood_bank")`; `build_critic` reads `critic_bank_size` (default `16384`) for the new type.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_critic_bank.py`:

```python
"""Tests for the bank-neighbourhood critic (NYTimes v2b).

Kept apart from `tests/test_critic.py` and `tests/test_neighbourhood_critic.py`
so the v2b branch does not edit files the v2 branch owns. Every test names
the mutation it catches; the plan records the mutation checks run after
implementation.
"""

import math

import numpy as np
import pytest
import torch

from src.models.critic import (
    BankNeighbourhoodCritic,
    Critic,
    NeighbourhoodCritic,
    build_critic,
    draw_real_bank_indices,
    neighbourhood_distances,
    profile_features,
    score_population,
)
from src.train.train_wgan_gp import gradient_penalty


def _rows(seed: int, n: int, dim: int) -> torch.Tensor:
    """`n` random unit vectors, the geometry every NYTimes batch has."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def _brute_force(x, bank, k, floor, self_index=None):
    d = torch.cdist(x.double(), bank.double())
    if self_index is not None:
        for i, slot in enumerate(self_index.tolist()):
            if slot >= 0:
                d[i, slot] = math.inf
    r, _ = torch.topk(d, k, dim=1, largest=False, sorted=True)
    return r.clamp(min=floor).float()


# --- the landed distance function, one case its own tests do not pin ---------


def test_self_index_masks_row_then_slot_not_the_transpose():
    """Catches: the mask applied at `[slot, row]` instead of `[row, slot]`.
    The landed tests use a bank equal to the batch with `arange`, where the
    two are the same cells. Here the bank is the batch rolled by one, so
    row i's copy sits at slot (i + 1) % 3, never at slot i."""
    x = _rows(3, 3, 6)
    bank = torch.roll(x, shifts=1, dims=0)
    self_index = torch.tensor([1, -1, 0])

    r = neighbourhood_distances(x, 2, 0.01, bank=bank, self_index=self_index)

    assert r[0, 0] > 0.01, "row 0's copy at slot 1 must be masked"
    assert r[2, 0] > 0.01, "row 2's copy at slot 0 must be masked"
    assert r[1, 0] == pytest.approx(0.01), "row 1 is not excluded; it reads its copy"
    torch.testing.assert_close(
        r, _brute_force(x, bank, 2, 0.01, self_index), atol=1e-5, rtol=1e-4
    )


# --- draw_real_bank_indices --------------------------------------------------


def test_bank_draw_is_deterministic_in_the_seed_and_distinct_across_seeds():
    """Catches: the draw reading the global RNG (which the trainer has already
    consumed by an amount that depends on everything built before the critic)."""
    a = draw_real_bank_indices(100, 10, seed=3)
    b = draw_real_bank_indices(100, 10, seed=3)
    c = draw_real_bank_indices(100, 10, seed=4)

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.dtype == torch.long and a.shape == (10,)
    assert len(set(a.tolist())) == 10, "bank rows must be distinct"
    assert a.min() >= 0 and a.max() < 100


def test_bank_draw_rejects_a_bank_larger_than_the_split():
    with pytest.raises(ValueError, match="critic_bank_size"):
        draw_real_bank_indices(10, 11, seed=0)


# --- BankNeighbourhoodCritic -------------------------------------------------


def _bank_critic(seed=0, k=3, bank_size=20, hidden=(8, 4)):
    torch.manual_seed(seed)
    return BankNeighbourhoodCritic(
        input_dim=6, hidden_dims=list(hidden), k=k, distance_floor=0.01, bank_size=bank_size
    )


def _filled(critic, seed=10):
    critic.write_fake(_rows(seed, critic.bank_size, 6))
    assert critic.fake_bank_full
    return critic


def test_bank_critic_rejects_a_bank_no_larger_than_k():
    with pytest.raises(ValueError, match="critic_bank_size"):
        BankNeighbourhoodCritic(input_dim=6, hidden_dims=[8], k=5, bank_size=5)


def test_bank_critic_scores_one_per_row_in_every_population():
    """Catches: a wrong squeeze or reduction axis on any of the three paths."""
    critic = _filled(_bank_critic())
    critic.set_real_bank(_rows(1, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    x = _rows(2, 7, 6)

    for population in ("real", "fake", "mixed"):
        assert critic(x, population=population).shape == (7,), population
    assert critic(x).shape == (7,), "the bare call is the mixed population"


def test_bank_critic_rejects_an_unknown_population_and_an_unset_real_bank():
    """Catches: the `mixed` path silently profiling against an all-zero real
    bank once the ring is full (it needs no real bank only while it falls
    back to within-batch)."""
    critic = _bank_critic()
    x = _rows(3, 5, 6)

    with pytest.raises(ValueError, match="population"):
        critic(x, population="interpolated")
    with pytest.raises(RuntimeError, match="set_real_bank"):
        critic(x, population="real")
    assert critic(x, population="mixed").shape == (5,), "within-batch fallback needs no bank"

    _filled(critic)
    with pytest.raises(RuntimeError, match="set_real_bank"):
        critic(x, population="mixed")


def test_real_population_reads_the_real_bank():
    """Catches: the real path reading the fake bank, the union, or the batch.
    The expected score is recomputed by hand from the real bank alone."""
    critic = _filled(_bank_critic())
    x_train = _rows(4, 40, 6)
    critic.set_real_bank(x_train, torch.arange(20))
    x = _rows(5, 5, 6)
    r = neighbourhood_distances(x, 3, 0.01, bank=x_train[:20])
    expected = critic.mlp(torch.cat([x, profile_features(r)], dim=1))

    torch.testing.assert_close(critic(x, population="real"), expected)


def test_a_real_row_in_the_bank_does_not_see_its_own_copy():
    """Catches: `row_to_slot` built as the identity instead of through the
    drawn indices (the bank is a permutation, so slot != row), or `row_ids`
    ignored. Spec: 'with the bank equal to the batch, the smallest distance
    read is above the floor'."""
    x_train = _rows(6, 30, 6)
    critic = _bank_critic(bank_size=30)
    critic.set_real_bank(x_train, draw_real_bank_indices(30, 30, seed=1))
    batch, ids = x_train[:5], torch.arange(5)

    self_index = critic.row_to_slot[ids]
    r = neighbourhood_distances(
        batch, 3, 0.01, bank=critic.real_bank, self_index=self_index
    )
    assert r.min() > 0.1, "with its own slot masked no row may read the floor"
    assert torch.all(critic.real_bank[self_index] == batch), "slot must point at the copy"

    with_ids = critic(batch, population="real", row_ids=ids)
    without = critic(batch, population="real")
    assert not torch.allclose(with_ids, without), "row_ids must change what a bank row sees"


def test_before_the_fake_bank_fills_fake_and_mixed_scores_equal_the_within_batch_critic():
    """Spec's fallback test. Catches: the fallback wired to a bank path, or a
    feature layout that differs from approach 1's."""
    torch.manual_seed(0)
    plain = NeighbourhoodCritic(input_dim=6, hidden_dims=[8, 4], k=3, distance_floor=0.01)
    bank = _bank_critic(seed=1, bank_size=16)
    bank.load_state_dict(plain.state_dict(), strict=False)
    x = _rows(7, 6, 6)

    torch.testing.assert_close(bank(x, population="fake"), plain(x))
    torch.testing.assert_close(bank(x, population="mixed"), plain(x))

    _filled(bank)
    assert not torch.allclose(bank(x, population="fake"), plain(x)), (
        "once full, fake rows must be profiled against the bank, not the batch"
    )


def test_bank_mode_scores_do_not_depend_on_batch_mates_but_fallback_scores_do():
    """Mirror of the spec's dependence test. In bank mode each row's profile
    comes from the bank alone, so replacing row 0 must leave every other score
    unchanged; catches the bank path concatenating the batch into the bank.
    Before the fill, the within-batch fallback must show the dependence."""
    critic = _bank_critic(bank_size=16)
    x = _rows(8, 6, 6)
    y = x.clone()
    y[0] = _rows(9, 1, 6)[0]

    assert not torch.allclose(critic(x, population="fake")[1:], critic(y, population="fake")[1:])

    _filled(critic)
    torch.testing.assert_close(critic(x, population="fake")[1:], critic(y, population="fake")[1:])


def test_fake_bank_is_a_ring_that_overwrites_the_oldest_rows():
    """Spec's ring test. Catches: a pointer that never wraps, or writes that
    append past the end."""
    critic = _bank_critic(bank_size=8)
    marker = torch.full((4, 6), 7.0)

    critic.write_fake(marker)
    critic.write_fake(_rows(10, 4, 6))
    assert critic.fake_bank_full
    assert torch.equal(critic.fake_bank[:4], marker)

    critic.write_fake(_rows(11, 4, 6))
    assert not (critic.fake_bank == 7.0).any(), "the marker rows must be gone"
    assert critic.fake_rows_written == 12


def test_fake_bank_write_wraps_a_batch_across_the_end():
    """Catches: an off-by-one at the wrap (slot 8 written, slot 0 skipped)."""
    critic = _bank_critic(bank_size=8)
    critic.write_fake(_rows(12, 6, 6))
    tail = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    critic.write_fake(tail)

    assert torch.equal(critic.fake_bank[6:8], tail[:2])
    assert torch.equal(critic.fake_bank[0:2], tail[2:])
    assert critic.fake_rows_written == 10


def test_fake_bank_write_rejects_a_batch_larger_than_the_bank():
    with pytest.raises(ValueError, match="bank"):
        _bank_critic(bank_size=8).write_fake(_rows(13, 9, 6))


def test_fake_bank_write_detaches_and_does_not_hold_the_graph():
    """Catches: dropping both the `@torch.no_grad()` decorator and the
    `.detach()` (either alone suffices, so removing only one is not
    observable here), which would attach the ring to the generator's graph
    and keep every past generator step alive on the device."""
    critic = _bank_critic(bank_size=8)
    rows = _rows(14, 4, 6).requires_grad_(True)

    critic.write_fake(rows * 2.0)

    assert not critic.fake_bank.requires_grad


def test_real_bank_indices_survive_a_state_dict_round_trip_and_the_rows_do_not():
    """Spec's checkpoint test. Catches: indices registered non-persistent (a
    resume would redraw them), or the 16 MB banks leaking into every checkpoint."""
    x_train = _rows(15, 40, 6)
    a = _bank_critic(bank_size=10)
    a.set_real_bank(x_train, draw_real_bank_indices(40, 10, seed=3))
    b = _bank_critic(seed=5, bank_size=10)

    b.load_state_dict(a.state_dict())

    assert torch.equal(b.real_bank_indices, a.real_bank_indices)
    assert "real_bank" not in a.state_dict()
    assert "fake_bank" not in a.state_dict()


def test_set_real_bank_rejects_indices_of_the_wrong_shape():
    with pytest.raises(ValueError, match="bank_size"):
        _bank_critic(bank_size=10).set_real_bank(_rows(16, 40, 6), torch.arange(9))


def test_gradient_penalty_is_finite_through_the_union_bank_and_reaches_the_weights():
    """Spec's double-backward test for this class. Catches: a non-differentiable
    op on the mixed path, or `torch.cat` of the banks breaking `create_graph`."""
    critic = _filled(_bank_critic())
    critic.set_real_bank(_rows(17, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    real, fake = _rows(18, 6, 6), _rows(19, 6, 6)

    gp = gradient_penalty(critic, real, fake, device=torch.device("cpu"))
    gp.backward()

    assert torch.isfinite(gp)
    grads = [p.grad for p in critic.mlp.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --- score_population and build_critic --------------------------------------


def test_score_population_routes_only_the_bank_critic():
    """Catches: the dispatcher passing keywords a plain critic cannot take, or
    dropping them for the bank critic."""
    x = _rows(20, 5, 6)
    plain = Critic(input_dim=6, hidden_dims=[8])
    torch.testing.assert_close(score_population(plain, x, "real"), plain(x))

    bank = _filled(_bank_critic())
    bank.set_real_bank(_rows(21, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    torch.testing.assert_close(score_population(bank, x, "fake"), bank(x, population="fake"))
    assert not torch.allclose(score_population(bank, x, "fake"), bank(x, population="real"))


def test_build_critic_dispatches_neighbourhood_bank_with_its_size():
    """Catches: the factory branch missing (a config typo trains the old
    critic), or `critic_bank_size` not read."""
    cfg = {
        "critic_type": "neighbourhood_bank",
        "critic_hidden_dims": [8],
        "negative_slope": 0.2,
        "critic_k": 3,
        "critic_distance_floor": 0.02,
        "critic_bank_size": 64,
    }

    critic = build_critic(cfg, input_dim=6)

    assert isinstance(critic, BankNeighbourhoodCritic)
    assert critic.bank_size == 64 and critic.k == 3 and critic.distance_floor == 0.02
    assert critic.real_bank.shape == (64, 6)


def test_build_critic_bank_size_defaults_to_16384():
    cfg = {"critic_type": "neighbourhood_bank", "critic_hidden_dims": [8], "negative_slope": 0.2}

    assert build_critic(cfg, input_dim=6).bank_size == 16384
```

In `tests/test_critic_factory.py`, change the assertion in `test_the_documented_types` to:

```python
    assert CRITIC_TYPES == ("per_vector", "neighbourhood", "neighbourhood_bank")
```

Its docstring ("a type added to or dropped from the tuple without the docs table changing") is why it changes here together with the docs row in Task 5. `test_negative_slope_reaches_both_classes` iterates `CRITIC_TYPES` and reads `critic.mlp.net` for every non-`Critic` type; the subclass inherits `mlp`, so it needs no edit.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_critic_bank.py tests/test_critic_factory.py -v`
Expected: `test_critic_bank.py` fails at import with `ImportError: cannot import name 'BankNeighbourhoodCritic'`; `test_the_documented_types` fails on the tuple.

- [ ] **Step 3: Implement the class, the draw, the dispatcher and the factory branch**

Append to `src/models/critic.py` after `NeighbourhoodCritic` (before `CRITIC_TYPES`):

```python
def draw_real_bank_indices(num_rows: int, bank_size: int, seed: int) -> Tensor:
    """A fixed random subset of the training split, one draw per run seed.

    Seeded on its own generator, on the CPU, so the draw does not depend on
    how much of the global RNG the trainer consumed before building the
    critic. The result is stored in the checkpoint (`real_bank_indices`) so a
    resume rebuilds the same bank instead of redrawing it.
    """
    if bank_size > num_rows:
        raise ValueError(
            f"critic_bank_size={bank_size} exceeds the training split "
            f"({num_rows} rows)"
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
```

Change `CRITIC_TYPES` to:

```python
CRITIC_TYPES = ("per_vector", "neighbourhood", "neighbourhood_bank")
```

and add to `build_critic`, after the `neighbourhood` branch and before the `raise`:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_critic_bank.py tests/test_critic_factory.py tests/test_neighbourhood_critic.py tests/test_critic.py -v`
Expected: all PASS. Then `PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check` green.

- [ ] **Step 5: Mutation checks**

Run each, confirm the named test fails, restore:

1. In `features`, replace `bank=self.real_bank` with `bank=self.fake_bank` on the real path: `test_real_population_reads_the_real_bank` fails.
2. In `set_real_bank`, replace `slot[indices] = torch.arange(self.bank_size)` with `slot[: self.bank_size] = torch.arange(self.bank_size)`: `test_a_real_row_in_the_bank_does_not_see_its_own_copy` fails.
3. In `write_fake`, change `pos = self.fake_rows_written % self.bank_size` to `pos = min(self.fake_rows_written, self.bank_size - 1)`: `test_fake_bank_is_a_ring_that_overwrites_the_oldest_rows` fails.
4. Register `real_bank_indices` with `persistent=False`: `test_real_bank_indices_survive_a_state_dict_round_trip_and_the_rows_do_not` fails.
5. On the `fake` path after the fill, use `bank=torch.cat([x, self.fake_bank])`: `test_bank_mode_scores_do_not_depend_on_batch_mates_but_fallback_scores_do` fails.

- [ ] **Step 6: Commit**

```bash
ruff format src/models/critic.py tests/test_critic_bank.py tests/test_critic_factory.py
git status --short
git add src/models/critic.py tests/test_critic_bank.py tests/test_critic_factory.py
git commit -m "feat(models): BankNeighbourhoodCritic -- profiles against a real bank and a fake ring"
```

---

### Task 3: Trainer wiring

**Files:**
- Modify: `src/data/dataset.py` (add `IndexedTensorDataset` after `NumpyTensorDataset`)
- Modify: `src/train/train_wgan_gp.py` (imports; a `split_batch` helper next to `build_dataloader`; the `dataset = NumpyTensorDataset(x_train)` line; the bank setup after the resume block; the two `next(data_iter)` sites; the three scoring calls `d_real = critic(real)`, `d_fake = critic(fake)`, `adv_loss = -critic(fake).mean()`; the fake write after `scaler_g.update()`)
- Test: `tests/test_train_bank_critic.py` (create)

**Interfaces:**
- Consumes: `BankNeighbourhoodCritic`, `draw_real_bank_indices`, `score_population` (Task 2); the landed `build_critic` construction site and `from src.models.critic import DEFAULT_DISTANCE_FLOOR, build_critic`.
- Produces: `IndexedTensorDataset(x: np.ndarray)` yielding `(row, index)`; `split_batch(batch) -> (rows, ids | None)`; `run_metadata.json` keys `critic_bank_size` (int) and `fake_bank_filled_step` (int, first generator step at which the ring was full; on a resume, the first such step after the resume); checkpoints whose `critic_state_dict["real_bank_indices"]` holds the bank draw.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_bank_critic.py`:

```python
"""One training step end-to-end with `critic_type: neighbourhood_bank`.

`tests/test_train_smoke.py` is left untouched: it is the guard that plain
critics still receive plain tensor batches. This file covers what the bank
critic adds to the loop: the row-index dataset, the bank draw, the ring
write, and the resume path that must rebuild the same bank.
"""

import math

import torch

from src.models.critic import BankNeighbourhoodCritic, build_critic, draw_real_bank_indices
from src.train.train_wgan_gp import train
from tests.test_train_smoke import make_config

BANK_SIZE = 64  # batch 32, so the ring is full after exactly two generator steps


def make_bank_config(tmp_path, name="bank", seed=0):
    cfg = make_config(tmp_path, "mlp")
    cfg["seed"] = seed
    cfg["output_dir"] = str(tmp_path / name)
    cfg["model"].update(
        {
            "critic_type": "neighbourhood_bank",
            "critic_k": 5,
            "critic_distance_floor": 0.01,
            "critic_bank_size": BANK_SIZE,
        }
    )
    return cfg


def test_bank_critic_smoke_run_writes_a_checkpoint_the_factory_can_load(tmp_path):
    """Spec's end-to-end test. Catches: the trainer not using the factory, a
    state-dict key mismatch, indices never drawn (still -1), or the ring
    written once per critic step instead of once per generator step."""
    cfg = make_bank_config(tmp_path)
    ckpt_path, meta = train(cfg)

    assert ckpt_path.exists()
    ckpt = torch.load(tmp_path / "bank" / "checkpoint_step_4.pt", weights_only=False)
    critic = build_critic(cfg["model"], input_dim=16)
    assert isinstance(critic, BankNeighbourhoodCritic)
    critic.load_state_dict(ckpt["critic_state_dict"])

    indices = critic.real_bank_indices
    assert indices.shape == (BANK_SIZE,)
    assert indices.min() >= 0 and indices.max() < meta["data"]["num_train"]
    assert len(set(indices.tolist())) == BANK_SIZE
    assert "real_bank" not in ckpt["critic_state_dict"]
    assert meta["critic_bank_size"] == BANK_SIZE
    expected_fill = math.ceil(BANK_SIZE / cfg["training"]["batch_size"])
    assert meta["fake_bank_filled_step"] == expected_fill
    for entry in meta["metrics"]:
        assert math.isfinite(entry["g_loss"]) and math.isfinite(entry["d_loss"])


def test_resume_rebuilds_the_real_bank_from_the_checkpoint_not_from_the_seed(tmp_path):
    """Spec's round-trip test at trainer level. The resumed config carries a
    different seed, so a trainer that redraws on resume produces different
    indices; catches exactly that redraw. (The different seed also changes
    the train/holdout split, so the rows behind the indices differ; resuming
    under another seed is already unsupported for the data order, and this
    test compares the indices, which is what the checkpoint carries.)"""
    first = make_bank_config(tmp_path, name="first", seed=0)
    _, meta_first = train(first)
    step4 = torch.load(tmp_path / "first" / "checkpoint_step_4.pt", weights_only=False)
    drawn_at_seed_0 = step4["critic_state_dict"]["real_bank_indices"]

    second = make_bank_config(tmp_path, name="second", seed=1)
    second["training"]["num_gen_steps"] = 8
    train(second, resume=str(tmp_path / "first" / "checkpoint_step_4.pt"))
    step8 = torch.load(tmp_path / "second" / "checkpoint_step_8.pt", weights_only=False)

    assert torch.equal(step8["critic_state_dict"]["real_bank_indices"], drawn_at_seed_0)
    n_train = meta_first["data"]["num_train"]
    redraw = draw_real_bank_indices(n_train, BANK_SIZE, seed=1)
    assert not torch.equal(drawn_at_seed_0, redraw), "the mutation must be observable"
```

Both tests take pytest's `tmp_path` fixture; `train` returns `(best_checkpoint_path, run_meta)`, and `run_meta["data"]["num_train"]` is the training split's size.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_bank_critic.py -v`
Expected: FAIL. The first test fails on `meta["critic_bank_size"]` (KeyError) or earlier with the `RuntimeError` from an unset real bank; the second fails on the indices assertion.

- [ ] **Step 3: Add `IndexedTensorDataset`**

In `src/data/dataset.py`, after `NumpyTensorDataset`:

```python
class IndexedTensorDataset(NumpyTensorDataset):
    """`NumpyTensorDataset` that also yields each row's index into `x`.

    The bank-neighbourhood critic excludes a real row's own bank slot by
    index, so the trainer must know which training row each batch row is.
    The default collate turns the ints into a `(batch,)` long tensor. Used
    only when the critic is the bank type; every other critic keeps the
    plain dataset and plain tensor batches.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.x[idx], idx
```

- [ ] **Step 4: Wire the trainer**

Imports: add `IndexedTensorDataset` to the `src.data.dataset` import and extend the critic import to `from src.models.critic import (BankNeighbourhoodCritic, DEFAULT_DISTANCE_FLOOR, build_critic, draw_real_bank_indices, score_population)` (ruff's isort will set the order).

After `build_dataloader`, add:

```python
def split_batch(batch: Tensor | list | tuple) -> tuple[Tensor, Tensor | None]:
    """Rows and, from an `IndexedTensorDataset`, their training indices."""
    if isinstance(batch, (list, tuple)):
        rows, ids = batch
        return rows, ids
    return batch, None
```

Replace `dataset = NumpyTensorDataset(x_train)` with:

```python
    bank_critic = isinstance(critic, BankNeighbourhoodCritic)
    dataset = IndexedTensorDataset(x_train) if bank_critic else NumpyTensorDataset(x_train)
```

After the resume block (after `run_meta["resumed_from_step"] = start_step`, before the `real_gate` computation):

```python
    if bank_critic:
        # The real bank is a fixed subset of the training split, drawn once
        # per run seed. A resume takes the indices load_state_dict restored
        # from the checkpoint rather than redrawing, so the bank the critic
        # was trained against is the bank it continues against. The fake
        # ring is not checkpointed; it refills within bank_size / batch_size
        # generator steps, during which fake rows fall back to within-batch
        # profiles, and `fake_bank_filled_step` records when that ended.
        indices = (
            critic.real_bank_indices.cpu()
            if resume is not None
            else draw_real_bank_indices(x_train.shape[0], critic.bank_size, seed)
        )
        critic.set_real_bank(dataset.x, indices)
        run_meta["critic_bank_size"] = critic.bank_size
```

In the critic loop, after the `try/except StopIteration` that yields `real_batch`:

```python
            real_batch, real_ids = split_batch(real_batch)
            real = real_batch.to(device)
            real_ids = None if real_ids is None else real_ids.to(device)
```

and replace the two scoring lines:

```python
                d_real = score_population(critic, real, "real", row_ids=real_ids)
                d_fake = score_population(critic, fake, "fake")
```

In the generator step, after its `try/except StopIteration` and before the existing `batch_size = real_batch.shape[0]` line (a list has no `.shape`):

```python
        real_batch, _ = split_batch(real_batch)
```

and replace `adv_loss = -critic(fake).mean()` with:

```python
            adv_loss = -score_population(critic, fake, "fake").mean()
```

After `scaler_g.update()`:

```python
        if bank_critic:
            # After the score, never before: the batch just scored must not
            # find itself in the ring at the floor.
            critic.write_fake(fake.detach())
            if critic.fake_bank_full and "fake_bank_filled_step" not in run_meta:
                run_meta["fake_bank_filled_step"] = step
```

`gradient_penalty` is not touched.

- [ ] **Step 5: Run the tests to verify they pass, and the whole suite**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_bank_critic.py tests/test_train_smoke.py tests/test_dataloader_reproducibility.py -v`
Expected: all PASS. Then `PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check` green.

- [ ] **Step 6: Mutation checks**

1. Move the `critic.write_fake(...)` call into the critic loop (once per critic step): `fake_bank_filled_step` becomes 1 with `n_critic: 2`, and the first test fails on the `math.ceil` assertion.
2. In the bank setup, always call `draw_real_bank_indices(...)` regardless of `resume`: the resume test fails.
3. Pass `row_ids=None` on the real scoring call: no test in this file fails. That is expected; the exclusion itself is covered at class level by `test_a_real_row_in_the_bank_does_not_see_its_own_copy`, and the trainer's contribution (threading the ids through) is asserted by the smoke run scoring `"real"` without a `RuntimeError` and by `IndexedTensorDataset` being the dataset. Record this as the one trainer edge the tests reach only indirectly.

Restore after each.

- [ ] **Step 7: Commit**

```bash
ruff format src/data/dataset.py src/train/train_wgan_gp.py tests/test_train_bank_critic.py
git status --short
git add src/data/dataset.py src/train/train_wgan_gp.py tests/test_train_bank_critic.py
git commit -m "feat(train): route critic calls by population and keep the bank critic's real bank and fake ring"
```

---

### Task 4: (superseded)

The landed trainer builds a fixed 4,096-latent probe for every `linear_skip` run and logs `LinearSkipGenerator.component_energies` (`trunk_energy`, `skip_energy`, `skip_share`, `trunk_skip_abs_cos`) into every `run_meta["eval"]` entry (`tests/test_train_smoke.py::test_linear_skip_evals_log_the_trunk_skip_balance`). The success bar's drift clause is read from those entries in Task 6. No implementation work.

---

### Task 5: `v2b` configs, job script and documentation

**Files:**
- Create: `configs/nytimes/v2b.yaml`, `configs/nytimes/v2b_seed42.yaml`, `scripts/nytimes_v2b_seed42_job.sh`
- Modify: `tests/test_nytimes_configs.py` (append the `v2b` pins), `PROJECT_DOCUMENTATION.md` (`### critic_type`: the prose list of values and the key table), `docs/datasets/nytimes.md` (Ladder table: one `v2b` row after `v2`'s, status `planned`)

**Interfaces:**
- Consumes: `critic_type: neighbourhood_bank`, `critic_bank_size` (Task 2); `drop_zero_rows` (landed); `configs/nytimes/v2.yaml` and `v2_seed42.yaml` (landed).
- Produces: the rung's config, its box instrument, the job Task 6 submits, and the pins that keep `v2b` = `v2` + two keys.

- [ ] **Step 1: Write the failing config pins**

Append to `tests/test_nytimes_configs.py`:

```python
V2B_DELTA = {
    "model.critic_type": "neighbourhood_bank",
    "model.critic_bank_size": 16384,
}


def test_v2b_is_v2_plus_the_neighbour_source():
    """Catches any v2b key drifting from v2 beyond the bank: the spec says
    v2b against v2 is one change, the neighbour source."""
    v2 = _flatten(_load("v2.yaml"))
    v2b = _flatten(_load("v2b.yaml"))
    for key, value in V2B_DELTA.items():
        assert v2b.pop(key) == value, key
    assert v2b.pop("output_dir") == "runs/nytimes/v2b"
    v2.pop("output_dir")
    v2.pop("model.critic_type")
    assert v2b == v2


def test_v2b_seed42_is_v2b_with_an_absolute_real_path_and_its_own_output_dir():
    """Catches the instrument config drifting from v2b beyond path and output dir."""
    v2b = _flatten(_load("v2b.yaml"))
    inst = _flatten(_load("v2b_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2b_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2b.pop("output_dir")
    v2b.pop("data.real_path")
    assert inst == v2b


def test_v2b_job_script_runs_the_v2b_seed42_config():
    """Catches the job script pointing at another config or run dir."""
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2b_seed42_job.sh").read_text()
    assert "configs/nytimes/v2b_seed42.yaml" in script
    assert "runs/nytimes/v2b_seed42" in script
```

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_nytimes_configs.py -v`
Expected: the three new tests FAIL with `FileNotFoundError`.

- [ ] **Step 2: Write `configs/nytimes/v2b.yaml`**

```yaml
# NYTIMES v2b -- v2 with the critic's neighbours drawn from a bank instead of
# the batch. One change from v2, stated:
#
#   1. model.critic_type: neighbourhood -> neighbourhood_bank, with
#      critic_bank_size: 16384 new and critic_k / critic_distance_floor
#      unchanged. Real rows are profiled against a fixed 16k-row subset of
#      the training split, fake rows against a ring of the last 32 generator
#      batches, so the profile is measured at a scale closer to the gate's
#      100-NN in 250k and exact copies are visible to the critic for about
#      3% of real rows rather than 0.15%.
#
# Everything else is v2's: the linear-skip generator, drop_zero_rows: true
# (duplicates kept), gate-aware selection, 30k steps.
# See docs/superpowers/specs/2026-09-14-neighbourhood-critic-design.md,
# "Approach 2: bank neighbourhoods (v2b)".
seed: 42
device: auto
output_dir: runs/nytimes/v2b

data:
  real_path: data/nytimes_250k.npy
  format: npy
  metric: angular
  descriptor_dim: 256
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true
    drop_zero_rows: true

model:
  # 256 trunk + 256 skip. The generator splits the latent itself.
  latent_dim: 512
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: linear_skip
  skip_dim: 256
  skip_init: orthogonal
  skip_init_gain: 1.0
  critic_type: neighbourhood_bank
  critic_k: 20
  critic_distance_floor: 0.01
  critic_bank_size: 16384

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
  select_on: gate
```

If `test_v2b_is_v2_plus_the_neighbour_source` fails on a key this text does not carry, `v2.yaml` has a key the plan did not see: copy `v2.yaml` and apply only the header, `output_dir`, `critic_type` and `critic_bank_size` changes, so the pin holds.

- [ ] **Step 3: Write `configs/nytimes/v2b_seed42.yaml`**

Same file with this header and these two lines changed (`output_dir`, `real_path`), the pattern `v2_seed42.yaml` uses:

```yaml
# NYTimes v2b, run 42. Identical to configs/nytimes/v2b.yaml in every training
# hyperparameter. Two things differ, listed so a reader can confirm nothing
# else moved:
#
#   1. output_dir  runs/nytimes/v2b -> runs/nytimes/v2b_seed42
#   2. real_path   data/nytimes_250k.npy -> an absolute path (see below)
#
# Why this file exists rather than an edit to v2b.yaml: v2b is a ladder rung,
# and a rung is a historical record of a run that happened. This config is a
# measurement instrument, the same pattern as configs/nytimes/v2_seed42.yaml.
#
# Why an absolute real_path: the gpuq runner executes each job in a fresh
# detached git worktree cut from the pinned commit, and data/ is gitignored,
# so a relative data/... path does not exist at run time. The file it names
# is `python -m src.data.fetch nytimes` output at seed 42, sha256
# ba8e8d512cc465e983661cbbabba64ead8251c3f313de86c3b1c3abe56c03428, the same
# bytes as a local data/nytimes_250k.npy.
#
# The 197 exact-zero rows are dropped at load (drop_zero_rows: true, as in
# v2); the 32,145 exact duplicates are kept as part of the search target.
seed: 42
device: auto
output_dir: runs/nytimes/v2b_seed42

data:
  real_path: /workspace/data-cache/nytimes_250k.npy
```

with the rest byte-identical to `v2b.yaml` from `format:` down.

- [ ] **Step 4: Write `scripts/nytimes_v2b_seed42_job.sh`**

```bash
#!/usr/bin/env bash
# One gpuq job: train NYTimes v2b (seed 42), sample it, and measure it against
# the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v2b \
#     --lane gpu -- bash scripts/nytimes_v2b_seed42_job.sh
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v2b at the end instead, the way
# the v2 job did. The sha256 lines at the end are what the laptop-side fetch
# verifies against. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v2b_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v2b/v2b_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v2b_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v2b_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v2b_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
(cd "$KEEP" && sha256sum run_config.yaml run_metadata.json eda_clean/summary.json)
```

`chmod +x scripts/nytimes_v2b_seed42_job.sh`, and `bash -n` it.

- [ ] **Step 5: Document the type and the key**

In `PROJECT_DOCUMENTATION.md`, section `### critic_type`: the first paragraph says "Two values: `per_vector` (...) and `neighbourhood` (...)". Change it to three values and add, after the `neighbourhood` description: "and `neighbourhood_bank` (`BankNeighbourhoodCritic`: the same features, but a real row's neighbours come from a fixed `critic_bank_size`-row bank of the training split and a fake row's from a ring of the last `critic_bank_size / batch_size` generator batches; interpolated rows, which is what `gradient_penalty` scores, query the union. A real row that is itself in the bank is excluded by index, so genuine duplicates stay visible. Checkpoints store the bank's row indices, not its rows, and the fake ring refills after a resume)." In the key table, after the `model.critic_distance_floor` row, add:

```
| `model.critic_bank_size` | `16384` | `neighbourhood_bank` only: rows in each of the real bank and the fake ring. Must exceed `critic_k` and not exceed the training split. `run_metadata.json` records it as `critic_bank_size` and the first step the ring was full as `fake_bank_filled_step`. |
```

In the closing paragraph ("the per-vector and neighbourhood state dicts do not cross-load"), append: "The bank critic's state dict is the neighbourhood critic's plus `real_bank_indices`."

In `docs/datasets/nytimes.md`, add a Ladder row directly after `v2`'s:

```
| `v2b` | `v2` with the critic's neighbours drawn from a bank (`critic_type: neighbourhood_bank`, `critic_bank_size: 16384`) | `configs/nytimes/v2b.yaml`; box instrument `configs/nytimes/v2b_seed42.yaml` | `runs/nytimes/v2b_seed42` (box: `/workspace/nytimes-v2b/v2b_seed42`) | planned |
```

- [ ] **Step 6: Check, then commit**

```bash
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git status --short
git add configs/nytimes/v2b.yaml configs/nytimes/v2b_seed42.yaml scripts/nytimes_v2b_seed42_job.sh tests/test_nytimes_configs.py PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "configs+scripts(nytimes): v2b rung -- bank-neighbourhood critic, seed-42 instrument and job"
```

`tests/test_docs_references.py` runs inside `make check` and fails if the ladder row or the docs cite a path or symbol that does not exist; everything cited above exists after Tasks 2 and 5.

---

### Task 6: Train `v2b` on the box and record the result

**Files:**
- Create: `docs/results/nytimes-v2b-seed42/{eda_clean_summary.json,run_config.yaml,run_metadata.json}`
- Modify: `docs/datasets/nytimes.md` (the `v2b` Ladder row's status; a new `## v2b, measured` section after `## v2, measured`, before `## Gate`)

**Interfaces:**
- Consumes: everything above, pushed to the public fork as `nytimes-v2b`.
- Produces: the measured rung and its verdict against the spec's bar.

- [ ] **Step 1: Probe transports and push**

```bash
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com; echo "ssh rc $?"
```

`rc 1` with a "successfully authenticated" banner means SSH works: `git push origin nytimes-v2b`. A timeout (`rc 124`) or connection refusal means push over HTTPS instead: `git push https-origin nytimes-v2b`. Say which transport was used in the final report. Then:

```bash
SHA=$(git rev-parse HEAD); echo "$SHA"
```

The full 40-character SHA is what `gpuq submit --commit` needs; an abbreviated one is accepted and then fails at checkout.

- [ ] **Step 2: Preflight on the CPU lane**

```bash
ssh tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; PRE=\$(gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-v2b --lane cpu --dedupe-key nytimes-v2b-preflight-$SHA -- bash -c 'bash -n scripts/nytimes_v2b_seed42_job.sh && test -f /workspace/data-cache/nytimes_250k.npy && test -f /workspace/data-cache/nytimes_250k_l2_clean.npy && /opt/venvs/wgan-synthetic/bin/python -c \"import yaml; from src.models.critic import build_critic; c=yaml.safe_load(open(\\\"configs/nytimes/v2b_seed42.yaml\\\")); print(type(build_critic(c[\\\"model\\\"], 256)).__name__)\"'); gpuq wait \$PRE; echo preflight rc \$?; gpuq show \$PRE | grep -E '\"(exit_code|error)\"'"
```

Expected: the log prints `BankNeighbourhoodCritic` and `preflight rc 0`. The ssh banner's `bind [127.0.0.1]:8080` line is noise. If `/workspace/data-cache/` is missing, the box has been rebuilt; stop and report rather than refetching inside this task.

- [ ] **Step 3: Submit the GPU job**

```bash
ssh tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-v2b --lane gpu --timeout-s 10800 --dedupe-key nytimes-v2b-seed42-$SHA -- bash scripts/nytimes_v2b_seed42_job.sh"
```

Record the job id. Budget: `v1` took 38 minutes; the spec estimates the bank profile adds a few percent, so expect about 45 minutes. `ESTIMATE (unverified)` until the job's timestamps say otherwise.

- [ ] **Step 4: Wait, in the background, to a file**

```bash
ssh -o ServerAliveInterval=30 tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; gpuq wait --timeout 9000 --poll 60 <job-id>; echo wait rc \$?; gpuq show <job-id>" > "$SCRATCHPAD/v2b_wait.log" 2>&1
```

Run with `run_in_background` and a timeout of at least 9,600,000 ms. Read the file; expected `wait rc 0` and `"state": "done"`. On failure the `error` field carries the stderr tail: fix, commit, push, resubmit with the new SHA. If the poll itself fails three times, report "host unreachable" as distinct from "job failed"; do not keep retrying.

- [ ] **Step 5: Fetch the small artifacts by stream and verify the hashes**

The 50k sample files and checkpoints are regenerable from the checkpoint on the box and are not fetched, matching `v1` and `v2`. What comes back is the three small files plus the EDA directory.

```bash
ssh tig-gpu 'cd /workspace/nytimes-v2b/v2b_seed42 && ls -l run_config.yaml run_metadata.json eda_clean/summary.json && sha256sum run_config.yaml run_metadata.json eda_clean/summary.json' | tee "$SCRATCHPAD/v2b_remote_hashes.txt"
mkdir -p runs/nytimes/v2b_seed42
ssh tig-gpu 'tar czf - -C /workspace/nytimes-v2b/v2b_seed42 run_config.yaml run_metadata.json eda_clean' | tar xzf - -C runs/nytimes/v2b_seed42
(cd runs/nytimes/v2b_seed42 && sha256sum run_config.yaml run_metadata.json eda_clean/summary.json)
```

Compare the two hash lists line by line; every hash must match before anything else happens. Then:

```bash
mkdir -p docs/results/nytimes-v2b-seed42
cp runs/nytimes/v2b_seed42/run_config.yaml runs/nytimes/v2b_seed42/run_metadata.json docs/results/nytimes-v2b-seed42/
cp runs/nytimes/v2b_seed42/eda_clean/summary.json docs/results/nytimes-v2b-seed42/eda_clean_summary.json
```

- [ ] **Step 6: Read the result against the spec's bar**

```bash
~/TIG/wgan-synthetic/.venv/bin/python - <<'EOF'
import json
nf = json.load(open("docs/datasets/nytimes_noise_floor.json"))["zero_and_duplicate_rows_removed"]["per_seed"]
st = {e["name"]: e for e in json.load(open("docs/results/nytimes-v2b-seed42/eda_clean_summary.json"))["stats"]}
keys = [("lid_median","LID median"),("relative_contrast_median","Relative contrast"),("hubness_skew","Hubness skew"),("ivf_gini","IVF cell-balance Gini")]
for k, n in keys:
    v = [p[k] for p in nf]; lo, hi = min(v), max(v); real = st["real"][k]
    cells = [f"`{real:.4g}` ({lo:.4g} -- {hi:.4g})"]
    for s in ("v2b_best", "v2b_step30000"):
        x = st[s][k]; inside = lo <= x <= hi; pct = abs(x - real) / real * 100
        cells.append(f"`{x:.4g}` ({'in range' if inside else f'{pct:.1f}% off'})")
    print(f"| {n} | " + " | ".join(cells) + " |")
m = json.load(open("docs/results/nytimes-v2b-seed42/run_metadata.json"))
ev = m["eval"]
best = min(ev, key=lambda e: e["selection_score"])
i = ev.index(best)
neighbours = [ev[j]["selection_score"] for j in (i - 1, i + 1) if 0 <= j < len(ev)]
print("selected step", best["step"], "score", round(best["selection_score"], 4), "neighbours", [round(s, 4) for s in neighbours])
print("transient?", any(s > 2 * best["selection_score"] for s in neighbours))
print("fake_bank_filled_step", m.get("fake_bank_filled_step"), "critic_bank_size", m.get("critic_bank_size"))
print("| step | fake LID | fake RC | hubness | Gini | score | skip share | trunk | skip | real near-dup | fake near-dup |")
for e in ev:
    print(f"| {e['step']} | {e['gate_fake_lid_median']:.1f} | {e['gate_fake_relative_contrast_median']:.3f} | {e['gate_fake_hubness_skew']:.2f} | {e['gate_fake_ivf_gini']:.3f} | {e['selection_score']:.3f} | {e['skip_share']:.3f} | {e['trunk_energy']:.0f} | {e['skip_energy']:.0f} | {e.get('gate_real_near_duplicate_fraction')} | {e.get('gate_fake_near_duplicate_fraction')} |")
trunk = [e["trunk_energy"] for e in ev]
print("trunk energy monotone?", all(a <= b for a, b in zip(trunk, trunk[1:])) or all(a >= b for a, b in zip(trunk, trunk[1:])))
w = [x["wasserstein"] for x in m["metrics"]]
print("wasserstein first/last 8:", [round(v, 4) for v in w[:8]], [round(v, 4) for v in w[-8:]])
EOF
```

If an eval key above is named differently in the landed metadata (check one entry with `python -c "import json; print(sorted(json.load(open('docs/results/nytimes-v2b-seed42/run_metadata.json'))['eval'][0]))"`), use the landed name; `docs/results/nytimes-v2-seed42/run_metadata.json` shows the same layout.

The bar, from the spec: LID 55.0 to 56.9, contrast 1.265 to 1.276, hubness 2.32 to 2.78, Gini 0.78 to 0.82 on the selected checkpoint (LID and contrast may instead be within 3% of the real median); the trunk energy must not be monotone across the evaluations; and the selected step must not be a transient (`transient?` must print `False`). The ring period is 32 generator steps, inside one `log_every` window of 250, so a sawtooth at that period cannot show in the logged Wasserstein trace; say so, and report the trace's variance between consecutive logged values instead.

- [ ] **Step 7: Record on the page**

In `docs/datasets/nytimes.md`: set the `v2b` Ladder row's status to `trained -- n=1 seed, <met the bar | missed on X and Y>; see ## v2b, measured`. Add `## v2b, measured` after `## v2, measured` with: provenance (commit, job id, submitted and completed timestamps from `gpuq show`, wall time, all MEASURED); the four-statistic table from Step 6; the selected step, its neighbours' scores, and whether it is a transient; the per-evaluation table from Step 6 beside `v2`'s; `near_duplicate_fraction` real and fake per eval, with the sentence the spec requires, that the real figure is counted against the holdout only and so sits well under the corpus's 14%; `fake_bank_filled_step`; the trunk/skip monotonicity verdict against `v1` and `v2`; the ring-period note; the per-step cost against `v2` (MEASURED from wall time). End with a verdict paragraph that says plainly whether `v2b` is a rung, and if not which statistics it missed. Do not name the next rung; the spec says that decision is a human one when two approaches are in play.

- [ ] **Step 8: Commit, push, open the PR and watch it**

```bash
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git status --short
git add docs/datasets/nytimes.md docs/results/nytimes-v2b-seed42/
git commit -m "docs(nytimes): record the v2b seed-42 run"
git push origin nytimes-v2b   # or https-origin, as Step 1 found
gh pr create --base nytimes-eda --head nytimes-v2b --title "nytimes: v2b, the bank-neighbourhood critic" --body-file "$SCRATCHPAD/v2b_pr_body.md"
gh pr checks <n> --watch
```

The PR body lists the mutation checks from Tasks 2 and 3 with their outcomes, the transport used, and the claim table (each number MEASURED with its command, or `ESTIMATE (unverified)`). An empty check list means not yet started; pending is not green. Report the PR as done only after the checks report a terminal pass.

---

## Self-review

**Spec coverage.** Corpus decision (`drop_zero_rows: true`): Task 5 config, implemented by the landed shared change. Distance floor, float32, expanded squares, index exclusion, `topk` on squared distances: the landed `neighbourhood_distances`, with one extra discriminating test in Task 2. Critic interface `forward(x) -> (n,)` and the factory name `neighbourhood_bank`: Task 2. Gradient penalty unchanged and scoring the union: Global Constraints, Task 2's `mixed` default and its finite-GP test. Real bank as a fixed seeded subset of the training split stored by index, fake bank as a ring written at every generator step, within-batch fallback until first fill with the fill step recorded: Tasks 2 and 3. Union for interpolates, documented in the class docstring: Task 2. Duplication diagnostic: landed, read in Task 6 Step 6. Approach 2's four extra tests: Task 2 (self-exclusion, fallback equality, ring overwrite, indices round-trip) and Task 3 (round-trip at trainer level). Shared tests that apply to this class (shape, dependence mirror, finite GP, factory dispatch, end-to-end step with a loadable checkpoint): Tasks 2 and 3. The rung, the instrument, the job, the config pins, the results page and the ladder row: Tasks 5 and 6. Success bar including the drift and transient clauses: the landed per-evaluation energies, applied in Task 6 Step 6. Cost measured in the run: Task 6 Step 7. Staleness and the ring-period sawtooth: Task 6 Steps 6 and 7. Merge order (start after the shared change lands; do not edit `NeighbourhoodCritic`): Task 0 and Global Constraints.

**Placeholders.** None; every code step carries its code and every command its expected output. Tasks 1 and 4 are marked superseded with the landed commit's evidence rather than left as work.

**Type consistency.** `neighbourhood_distances(x, k, floor, bank=None, self_index=None)` is the landed signature and is used with those keywords in Task 2. `BankNeighbourhoodCritic(input_dim, hidden_dims, k, distance_floor, negative_slope, bank_size)` and its members `mlp`, `real_bank_indices`, `real_bank`, `fake_bank`, `fake_rows_written`, `row_to_slot`, `fake_bank_full`, `set_real_bank`, `write_fake`, `features`, `forward` match between Task 2's class, Task 2's tests, Task 3's trainer and Task 3's tests. `score_population(critic, x, population, row_ids=None)` matches between Task 2 and Task 3. `run_meta` keys `critic_bank_size` and `fake_bank_filled_step` match between Task 3's trainer and tests and Task 6's reader. The eval keys `trunk_energy`, `skip_energy`, `skip_share` in Task 6 are the landed `component_energies` names.
