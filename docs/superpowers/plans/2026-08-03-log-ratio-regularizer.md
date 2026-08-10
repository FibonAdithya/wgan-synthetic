# Local Log-Ratio Regularizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the generator loss an explicit handle on local neighbourhood geometry, by matching the mean log-ratio profile of within-batch nearest-neighbour distances between fake and real batches.

**Architecture:** A pure function computes `p_i = mean over batch of log(r_i / r_k)` for `i = 1..k-1` from a batch's within-batch neighbour distances. The real side is tracked by an EMA (the real distribution is fixed, so per-batch estimates only add gradient noise). The penalty is `alpha * ||p_fake - p_target||_1`, added to the generator loss beside the existing `distance_reg` term and logged alongside it. Off by default, so every existing variant is untouched.

**Tech Stack:** Python 3.12, PyTorch 2.x, pytest, NumPy.

## Global Constraints

- Phase 3 of 3 from `docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md`. Phase 1 (run infrastructure) is complete on this branch. Phase 2 (structured-gate generator) is a **separate plan touching `src/models/generator.py`** — do not modify that file or `tests/test_generator*.py` here.
- Baseline is **143 tests passing**. Run with the main-repo interpreter: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`. Worktrees have no `.venv`.
- **`lid_reg_alpha` defaults to `0.0`.** With it unset or zero, the generator loss must be exactly `adv_loss + distance_reg_alpha * distance_reg` as today — v0 through v3 stay bit-identical.
- `tests/test_train_smoke.py` must pass untouched.
- Do **not** touch `PROJECT_DOCUMENTATION.md` or `configs/sift_gan_v3.yaml`. Phase 2 owns both; the integrating session merges the docs and writes `configs/sift_gan_v4.yaml`.
- Everything here is CPU-testable. No task needs a GPU.

## File Structure

| File | Responsibility |
|---|---|
| `src/train/log_ratio.py` (create) | The profile function and the EMA target. Pure torch, no training-loop knowledge. |
| `src/train/train_wgan_gp.py` (modify) | Read the three config keys, add the penalty to the generator loss, log it. |
| `tests/test_log_ratio.py` (create) | Profile correctness, degenerate batches, gradients, agreement with the NumPy reference. |
| `tests/test_lid_reg_training.py` (create) | The training-loop wiring: defaults, logging, and the alpha-zero guarantee. |

**Why a new module rather than more functions in `train_wgan_gp.py`:** that file is already ~640 lines and carries the training loop, checkpointing, EMA, device claiming and collapse monitoring. The profile computation is a self-contained numeric kernel with real edge cases, and it is the piece most worth testing in isolation. `batch_pairwise_distance_mean` lives in the training module because it is four lines; this is not.

## Background the tasks assume

The eval module `src/eval/ann_difficulty.py` already computes local intrinsic dimensionality on NumPy arrays, and its docstrings characterise the two degenerate cases this plan must also handle:

- `r_1 == 0` — the query sits on an exact duplicate.
- `r_1 == r_k` — every one of the `k` neighbours ties at the same distance, so `mean(log(r_i/r_k))` is 0 and the Hill estimator divides by zero.

`survivor_mask` there drops both rather than clamping. This plan reimplements the same rule in torch. It deliberately does **not** import from `src/eval/` — that module is NumPy-only and the training path needs autograd — but the NumPy version is the reference, and Task 1 asserts the two agree.

**Why match the profile rather than LID itself:** `p` is exactly the statistic the Hill estimator reduces to a scalar (`LID = -1 / mean_i(p_i)`), so matching `p` moves LID. But `p` is bounded and smooth where LID's `-1/x` blows up as the mean log-ratio approaches zero. `p` is also a `(k-1)`-vector rather than v1_5's single scalar, so it constrains the *shape* of the neighbourhood rather than only its scale.

---

### Task 1: The log-ratio profile

**Files:**
- Create: `src/train/log_ratio.py`
- Create: `tests/test_log_ratio.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `batch_log_ratio_profile(x: Tensor, k: int, max_points: int = 0, eps: float = 1e-12) -> Optional[Tensor]` in `src.train.log_ratio`. Returns a `(k_eff - 1,)` tensor of mean log-ratios, or `None` when the batch is too small or no row survives the degenerate-case filter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_log_ratio.py`:

```python
import numpy as np
import pytest
import torch

from src.eval.ann_difficulty import knn, survivor_mask
from src.train.log_ratio import batch_log_ratio_profile


def _blob(n=128, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_profile_has_one_entry_per_neighbour_below_k():
    profile = batch_log_ratio_profile(_blob(), k=10)
    assert profile.shape == (9,)


def test_profile_entries_are_non_positive_and_rise_toward_zero():
    # r_i <= r_k by construction, so every log-ratio is <= 0, and the ratio
    # approaches 1 as i approaches k.
    profile = batch_log_ratio_profile(_blob(n=512), k=10)
    assert (profile <= 1e-6).all()
    assert (profile.diff() >= -1e-6).all()
    assert profile[-1].abs() < profile[0].abs()


def test_profile_matches_the_numpy_reference():
    # The eval module is the reference implementation. If these drift, LID
    # measured after training stops corresponding to what training optimised.
    x = _blob(n=256, d=6, seed=3)
    profile = batch_log_ratio_profile(x, k=12, max_points=0)

    dist, _, _ = knn(x.numpy().astype(np.float32), k=12)
    kept = dist[survivor_mask(dist)]
    reference = np.log(
        np.clip(kept[:, :-1] / kept[:, -1:], 1e-12, 1.0)
    ).mean(axis=0)

    assert np.allclose(profile.numpy(), reference, atol=1e-4)


def test_queries_on_a_duplicate_are_dropped_not_clamped():
    # r_1 == 0 makes log(r_1/r_k) = -inf. Dropping declines to answer;
    # clamping would invent a number.
    base = _blob(n=64, d=4, seed=5)
    x = torch.cat([base, base[:8]])
    profile = batch_log_ratio_profile(x, k=6)
    assert profile is not None
    assert torch.isfinite(profile).all()


def test_all_duplicate_batch_returns_none():
    x = torch.ones(32, 4)
    assert batch_log_ratio_profile(x, k=5) is None


def test_all_tied_batch_returns_none():
    # Rows of the identity are all sqrt(2) apart, so r_1 == r_k for every query.
    assert batch_log_ratio_profile(torch.eye(16), k=5) is None


def test_k_is_clamped_to_the_batch():
    profile = batch_log_ratio_profile(_blob(n=6, d=3), k=100)
    assert profile.shape == (4,)  # k_eff = n - 1 = 5, profile is k_eff - 1


def test_batch_too_small_returns_none():
    assert batch_log_ratio_profile(_blob(n=2, d=3), k=5) is None


def test_max_points_subsamples_without_changing_the_shape():
    torch.manual_seed(0)
    profile = batch_log_ratio_profile(_blob(n=512), k=10, max_points=64)
    assert profile.shape == (9,)


def test_gradient_flows_to_the_input():
    x = _blob(n=64, d=4, seed=7).requires_grad_(True)
    batch_log_ratio_profile(x, k=8).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gradient_is_finite_when_duplicates_are_present():
    # The trap: torch.cdist has an undefined gradient at distance zero, so a
    # collapsed generator would poison the whole backward pass. The expanded
    # -square form with a clamped sqrt keeps it finite.
    base = _blob(n=32, d=4, seed=11)
    x = torch.cat([base, base[:4]]).requires_grad_(True)
    batch_log_ratio_profile(x, k=6).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_profile_dtype_follows_the_input():
    profile = batch_log_ratio_profile(_blob().to(torch.float64), k=8)
    assert profile.dtype == torch.float64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_log_ratio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train.log_ratio'`

- [ ] **Step 3: Write the implementation**

Create `src/train/log_ratio.py`:

```python
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def batch_log_ratio_profile(
    x: Tensor,
    k: int,
    max_points: int = 0,
    eps: float = 1.0e-12,
) -> Optional[Tensor]:
    """Mean ``log(r_i / r_k)`` over a batch's within-batch neighbours.

    Returns one entry per ``i = 1 .. k-1``, or ``None`` when the batch is too
    small or every query hits a degenerate case.

    This is the sufficient statistic the Hill estimator reduces to a scalar
    (``LID = -1 / mean_i(p_i)``), so matching it moves LID -- but it is
    bounded and smooth where LID's ``-1/x`` blows up as the mean log-ratio
    approaches zero, and it is a vector, so it constrains the *shape* of the
    neighbourhood rather than only its scale.

    The distances are within-batch and therefore much larger than true k-NN
    distances in the full dataset. That bias is identical on the real and fake
    sides and cancels in the difference -- the same equal-N discipline the ANN
    report enforces.

    Two degenerate cases are dropped rather than clamped, matching
    ``src.eval.ann_difficulty.survivor_mask``: ``r_1 == 0`` (the query sits on
    a duplicate) and ``r_1 == r_k`` (every neighbour ties). Clamping would
    invent a number; dropping declines to answer.
    """
    n = x.shape[0]
    if max_points > 1 and n > max_points:
        idx = torch.randperm(n, device=x.device)[:max_points]
        x = x[idx]
        n = max_points
    k_eff = min(int(k), n - 1)
    if k_eff < 2:
        return None

    # Expanded squares with a clamped sqrt rather than torch.cdist: cdist's
    # gradient is undefined at distance zero, so a single pair of duplicate
    # rows -- exactly what a collapsing generator produces -- would poison the
    # entire backward pass, including for rows that survive the filter below.
    sq = (x * x).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = d2.clamp(min=0.0)
    # Exclude each row from its own neighbour list by index, not by dropping
    # the nearest column: an exact duplicate ties with the query at distance
    # zero and would otherwise be dropped in its place.
    d2 = d2 + torch.diag(torch.full((n,), float("inf"), device=x.device, dtype=x.dtype))

    # Select and filter on SQUARED distances, before any epsilon clamp.
    # Ordering is identical, and it keeps the degenerate tests exact: a pair of
    # duplicate rows has d2 == 0, so `r2_1 > 0` drops it. Clamping first would
    # turn that 0 into eps, the row would survive, and its log-ratio -- a large
    # negative number rather than -inf -- would quietly dominate the mean.
    r2, _ = torch.topk(d2, k_eff, dim=1, largest=False, sorted=True)
    survivors = (r2[:, 0] > 0.0) & (r2[:, 0] < r2[:, -1])
    if not bool(survivors.any()):
        return None

    # The clamp below is only gradient insurance for near-duplicates; every
    # surviving row already has a strictly positive nearest distance.
    kept = r2[survivors].clamp(min=eps).sqrt()
    ratio = (kept[:, :-1] / kept[:, -1:]).clamp(min=eps, max=1.0)
    return torch.log(ratio).mean(dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_log_ratio.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 155 tests (143 + 12).

- [ ] **Step 6: Commit**

```bash
git add src/train/log_ratio.py tests/test_log_ratio.py
git commit -m "feat(train): mean log-ratio profile of within-batch neighbours

The sufficient statistic the Hill LID estimator reduces to a scalar, so
matching it moves LID -- but bounded and smooth where LID's -1/x blows
up, and a vector rather than a scalar, so it constrains the shape of the
neighbourhood and not only its scale.

Degenerate queries (r_1 == 0, r_1 == r_k) are dropped rather than
clamped, matching ann_difficulty.survivor_mask, and the test asserts the
torch and numpy versions agree. Distances use expanded squares with a
clamped sqrt rather than cdist, whose gradient is undefined at zero --
one duplicate pair from a collapsing generator would otherwise poison
the whole backward pass."
```

---

### Task 2: The EMA target and the penalty

**Files:**
- Modify: `src/train/log_ratio.py`
- Modify: `tests/test_log_ratio.py` (append)

**Interfaces:**
- Consumes: `batch_log_ratio_profile` from Task 1.
- Produces: `class LogRatioTarget(decay: float = 0.99)` with `update(profile: Tensor) -> Tensor` and attribute `value: Optional[Tensor]`; and `log_ratio_penalty(fake, real, k, max_points, target) -> Tensor` returning a scalar (zero when either side is degenerate).

The real distribution is fixed, so estimating the target fresh from each minibatch injects noise into the gradient for nothing. An EMA over the real side costs the same and is far quieter.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_log_ratio.py`:

```python
from src.train.log_ratio import LogRatioTarget, log_ratio_penalty


def test_target_initialises_to_the_first_profile():
    target = LogRatioTarget(decay=0.9)
    first = torch.tensor([-1.0, -0.5])
    assert torch.equal(target.update(first), first)


def test_target_moves_toward_later_profiles():
    target = LogRatioTarget(decay=0.5)
    target.update(torch.tensor([0.0, 0.0]))
    updated = target.update(torch.tensor([-1.0, -1.0]))
    assert torch.allclose(updated, torch.tensor([-0.5, -0.5]))


def test_target_is_detached_from_the_graph():
    # The target is a fixed reference, not something the generator can move by
    # gradient. Keeping it attached would let the penalty be minimised by
    # dragging the target instead of the samples.
    target = LogRatioTarget()
    profile = torch.tensor([-1.0, -0.5], requires_grad=True)
    assert not target.update(profile).requires_grad


def test_target_resets_when_the_profile_length_changes():
    # k_eff shrinks on a short final batch; a stale target of the wrong length
    # would otherwise raise or broadcast silently.
    target = LogRatioTarget()
    target.update(torch.tensor([-1.0, -0.5, -0.2]))
    assert target.update(torch.tensor([-1.0, -0.5])).shape == (2,)


def test_penalty_is_near_zero_for_samples_from_one_distribution():
    torch.manual_seed(0)
    a, b = torch.randn(256, 8), torch.randn(256, 8)
    penalty = log_ratio_penalty(a, b, k=10, max_points=0, target=LogRatioTarget())
    assert penalty.item() < 0.05


def test_penalty_grows_when_local_structure_differs():
    # A 2-D blob and a 32-D blob have very different local geometry; the
    # penalty must see what mean pairwise distance alone cannot.
    torch.manual_seed(1)
    real = torch.randn(256, 32)
    same = log_ratio_penalty(
        torch.randn(256, 32), real, k=10, max_points=0, target=LogRatioTarget()
    )
    low_dim = torch.randn(256, 2)
    low_dim = torch.cat([low_dim, torch.zeros(256, 30)], dim=1)
    different = log_ratio_penalty(
        low_dim, real, k=10, max_points=0, target=LogRatioTarget()
    )
    assert different.item() > same.item() * 3.0


def test_penalty_is_zero_when_either_side_is_degenerate():
    torch.manual_seed(2)
    real = torch.randn(64, 8)
    collapsed = torch.ones(64, 8)
    penalty = log_ratio_penalty(
        collapsed, real, k=6, max_points=0, target=LogRatioTarget()
    )
    assert penalty.item() == 0.0
    assert torch.isfinite(penalty)


def test_penalty_gradient_reaches_the_fake_batch():
    torch.manual_seed(3)
    fake = torch.randn(128, 8, requires_grad=True)
    real = torch.randn(128, 8)
    log_ratio_penalty(fake, real, k=8, max_points=0, target=LogRatioTarget()).backward()
    assert fake.grad is not None
    assert fake.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_log_ratio.py -k "target or penalty" -v`
Expected: FAIL — `ImportError: cannot import name 'LogRatioTarget'`

- [ ] **Step 3: Write the implementation**

Append to `src/train/log_ratio.py`:

```python
class LogRatioTarget:
    """EMA of the real batches' log-ratio profile.

    The real distribution is fixed, so a fresh per-minibatch estimate only
    adds variance to the gradient. Averaging costs nothing and is much
    quieter. At decay 0.99 the average settles within about a hundred steps,
    which is why it is deliberately *not* checkpointed: unlike the generator
    weight EMA at decay 0.999, losing it on resume costs a brief transient
    rather than a thousand-step average.
    """

    def __init__(self, decay: float = 0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.value: Optional[Tensor] = None

    def update(self, profile: Tensor) -> Tensor:
        observed = profile.detach()
        # A short final batch clamps k_eff and shortens the profile; a stale
        # target of the wrong length would broadcast silently.
        if self.value is None or self.value.shape != observed.shape:
            self.value = observed.clone()
        else:
            self.value.mul_(self.decay).add_(observed, alpha=1.0 - self.decay)
        return self.value


def log_ratio_penalty(
    fake: Tensor,
    real: Tensor,
    k: int,
    max_points: int,
    target: LogRatioTarget,
) -> Tensor:
    """L1 gap between the fake profile and the EMA of the real one.

    Zero when either side is degenerate -- an all-duplicate or all-tied batch
    yields no usable queries, and a penalty invented from nothing would push
    the generator in an arbitrary direction at exactly the moment it is
    collapsing.
    """
    zero = torch.zeros((), device=fake.device, dtype=fake.dtype)
    fake_profile = batch_log_ratio_profile(fake, k=k, max_points=max_points)
    if fake_profile is None:
        return zero
    with torch.no_grad():
        real_profile = batch_log_ratio_profile(real, k=k, max_points=max_points)
    if real_profile is None:
        return zero
    reference = target.update(real_profile.to(fake_profile.dtype))
    if reference.shape != fake_profile.shape:
        return zero
    return (fake_profile - reference).abs().sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_log_ratio.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 163 tests.

- [ ] **Step 6: Commit**

```bash
git add src/train/log_ratio.py tests/test_log_ratio.py
git commit -m "feat(train): EMA target and L1 penalty for the log-ratio profile

The real distribution is fixed, so a fresh per-batch estimate only adds
gradient variance; an EMA costs the same and is quieter. Deliberately
not checkpointed -- at decay 0.99 it resettles in about a hundred steps,
unlike the generator weight EMA at 0.999.

The penalty returns exactly zero when either side is degenerate: a
number invented from an all-duplicate batch would push the generator in
an arbitrary direction precisely when it is collapsing."
```

---

### Task 3: Wire the penalty into the training loop

**Files:**
- Modify: `src/train/train_wgan_gp.py`
- Create: `tests/test_lid_reg_training.py`

**Interfaces:**
- Consumes: `log_ratio_penalty` and `LogRatioTarget` from Task 2.
- Produces: three `training` config keys — `lid_reg_alpha` (default `0.0`), `lid_reg_k` (default `20`), `lid_reg_max_points` (default `256`) — and a `lid_reg` entry in the logged metrics dict.

This mirrors `distance_reg_alpha` exactly, including being read with `.get()` and defaulting off.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lid_reg_training.py`:

```python
import pytest

from src.train.train_wgan_gp import train


def _config(tmp_path, **training):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 8,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 4,
            "generator_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32, "num_gen_steps": 3, "n_critic": 1,
            "lr_g": 1e-4, "lr_d": 1e-4, "betas": [0.0, 0.9], "lambda_gp": 5.0,
            "ema_decay": 0.0, "num_workers": 0, "distance_reg_alpha": 0.0,
            "distance_reg_max_points": 16, "amp": False,
            "log_every": 1, "eval_every": 100, "save_every": 3,
        },
    }
    cfg["training"].update(training)
    return cfg


def _metrics(meta):
    return [m for m in meta["metrics"] if "g_loss" in m]


def test_absent_keys_leave_the_generator_loss_unchanged(tmp_path):
    # The backward-compatibility guarantee: with lid_reg_alpha unset, g_loss
    # is exactly adv_loss, so v0-v3 are bit-identical to before this change.
    _, meta = train(_config(tmp_path))
    rows = _metrics(meta)
    assert rows
    for row in rows:
        assert row["lid_reg"] == 0.0
        assert row["g_loss"] == pytest.approx(row["adv_loss"], abs=0.0)


def test_explicit_zero_alpha_behaves_the_same(tmp_path):
    _, meta = train(_config(tmp_path, lid_reg_alpha=0.0))
    for row in _metrics(meta):
        assert row["lid_reg"] == 0.0
        assert row["g_loss"] == pytest.approx(row["adv_loss"], abs=0.0)


def test_enabled_regulariser_contributes_to_the_generator_loss(tmp_path):
    _, meta = train(
        _config(tmp_path, lid_reg_alpha=0.5, lid_reg_k=8, lid_reg_max_points=32)
    )
    rows = _metrics(meta)
    assert rows
    assert any(row["lid_reg"] > 0.0 for row in rows)
    for row in rows:
        assert row["g_loss"] == pytest.approx(
            row["adv_loss"] + 0.5 * row["lid_reg"], rel=1e-5
        )


def test_training_completes_with_the_regulariser_enabled(tmp_path):
    ckpt, meta = train(_config(tmp_path, lid_reg_alpha=0.1, lid_reg_k=8))
    assert ckpt.exists()
    assert all(row["lid_reg"] >= 0.0 for row in _metrics(meta))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_lid_reg_training.py -v`
Expected: FAIL — `KeyError: 'lid_reg'`

- [ ] **Step 3: Wire it in**

Add to the imports in `src/train/train_wgan_gp.py`:

```python
from src.train.log_ratio import LogRatioTarget, log_ratio_penalty
```

Beside the existing `distance_reg_*` reads (around line 415), add:

```python
    lid_reg_alpha = float(train_cfg.get("lid_reg_alpha", 0.0))
    lid_reg_k = int(train_cfg.get("lid_reg_k", 20))
    lid_reg_max_points = int(train_cfg.get("lid_reg_max_points", 256))
    lid_reg_target = LogRatioTarget()
```

In the generator-loss block, replace:

```python
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = adv_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
                g_loss = adv_loss
```

with:

```python
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = adv_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
                g_loss = adv_loss
            if lid_reg_alpha > 0.0:
                lid_reg = log_ratio_penalty(
                    fake,
                    real_batch.to(device),
                    k=lid_reg_k,
                    max_points=lid_reg_max_points,
                    target=lid_reg_target,
                )
                g_loss = g_loss + lid_reg_alpha * lid_reg
            else:
                lid_reg = torch.zeros((), device=device, dtype=fake.dtype)
```

and add to the logged metrics dict, next to `"distance_reg"`:

```python
                "lid_reg": float(lid_reg.item()),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_lid_reg_training.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 167 tests. `tests/test_train_smoke.py` untouched and green.

- [ ] **Step 6: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_lid_reg_training.py
git commit -m "feat(train): add the log-ratio penalty to the generator loss

Mirrors distance_reg_alpha: read with .get(), defaults to 0.0, logged
beside distance_reg. With it unset the generator loss is exactly
adv_loss, so v0-v3 are bit-identical -- asserted directly rather than
assumed."
```

---

### Task 4: Record the deferred resume gap

**Files:**
- Modify: `FOLLOWUPS.md`

`PROJECT_DOCUMENTATION.md` and `configs/sift_gan_v4.yaml` are deliberately **not** in this task — phase 2 also edits the documentation, and the v4 config is v3's config plus one key, so both belong to the session that integrates the two branches. This task records only the one gap this plan knowingly leaves.

- [ ] **Step 1: Append to `FOLLOWUPS.md`**

```markdown
### The log-ratio EMA target is not checkpointed

`LogRatioTarget` in `src/train/log_ratio.py` holds an EMA of the real
batches' log-ratio profile, and `--resume` does not restore it: a resumed
run rebuilds the average from scratch.

This is deliberate, not an oversight. At decay 0.99 the average resettles
within roughly a hundred steps, so the cost is a brief transient on a 100k
step run. It is explicitly unlike the generator weight EMA at decay 0.999,
which `save_checkpoint` does persist and where a silent restart degrades
`best_generator.pt` with no error.

If `lid_reg_decay` is ever raised toward 0.999, revisit this — the argument
above stops holding.
```

- [ ] **Step 2: Verify the suite is unaffected**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 167 tests. Documentation-only change.

- [ ] **Step 3: Commit**

```bash
git add FOLLOWUPS.md
git commit -m "docs: record that the log-ratio EMA target is not checkpointed

Deliberate -- at decay 0.99 it resettles in about a hundred steps, unlike
the generator weight EMA at 0.999 that save_checkpoint does persist."
```

---

## Verification

After Task 4:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q
```
Expected: 167 passed, up from the 143 baseline.

Then confirm the penalty sees what mean pairwise distance cannot — the reason
it exists. This constructs two sets with near-identical global distance scale
but different local geometry:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import torch
from src.train.log_ratio import LogRatioTarget, log_ratio_penalty
from src.train.train_wgan_gp import batch_pairwise_distance_mean
torch.manual_seed(0)
real = torch.randn(1024, 32)
flat = torch.cat([torch.randn(1024, 4), torch.zeros(1024, 28)], 1)
flat = flat * (real.norm(dim=1).mean() / flat.norm(dim=1).mean())
for name, x in (('real', real), ('flat', flat)):
    print(f'{name}: mean pairwise {batch_pairwise_distance_mean(x, 0):.4f}')
print('penalty real-vs-real', log_ratio_penalty(real, real, 10, 0, LogRatioTarget()).item())
print('penalty flat-vs-real', log_ratio_penalty(flat, real, 10, 0, LogRatioTarget()).item())
"
```

Expected: the two mean pairwise distances are close, while the flat-vs-real
penalty is many times the real-vs-real one. That gap is the whole argument for
this regularizer over v1_5's single scalar.

**The test thresholds are measured, not guessed.** The kernel in Task 1 was run
against the real `ann_difficulty` reference before this plan was written:

| Check | Measured | Threshold asserted |
|---|---|---|
| Agreement with the NumPy reference | max diff `2.98e-08` | `atol=1e-4` |
| L1 gap, same distribution (n=256, k=10) | `0.0074` | `< 0.05` |
| L1 gap, 2-D structure vs 32-D | `3.9998` | `> 3x` the same-distribution gap |

The separation is roughly 540x, so the `3x` assertion has enormous headroom —
if it ever fails, something is structurally wrong rather than marginally off.

## Not in this plan

- **`configs/sift_gan_v4.yaml`** — v4 is v3's config plus `lid_reg_alpha`, so it
  cannot be written until phase 2's `configs/sift_gan_v3.yaml` exists. The
  integrating session writes it. Per the spec, start `lid_reg_alpha` at v1_5's
  proven scale of `0.1`.
- **`PROJECT_DOCUMENTATION.md`** — phase 2 edits the same file. The integrating
  session merges both sections: the v4 variant row, and the three `lid_reg_*`
  keys in the training-setup section.
- **Running the v4 arm.** Needs the GPU box and phase 1's lock. Per the spec,
  v4's `lid_median` is a *fitted* number, not evidence — it trains on LID's
  sufficient statistic. Any write-up must lead with `hubness_skew`, `ivf_gini`
  and `relative_contrast`, which the penalty does not touch.
