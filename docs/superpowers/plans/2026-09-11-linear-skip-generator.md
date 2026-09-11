# Linear-Skip Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `linear_skip` generator whose Jacobian is full rank by construction, a gate-aware checkpoint selector, and a NYTimes `v1` rung that uses both, then train and measure it on the box.

**Architecture:** A new `LinearSkipGenerator` in `src/models/generator.py` splits the latent into a trunk block (today's MLP) and a skip block mapped by a bias-free `nn.Linear`; the two are summed before the trainer's existing L2 normalisation. A new `src/train/selection.py` computes the four ANN-difficulty statistics on the eval holdout and scores checkpoints on the normalised LID and contrast gaps; the trainer selects on that score when `training.select_on: gate`, and on `cov_fro` otherwise, exactly as today.

**Tech Stack:** Python 3.12, PyTorch, numpy, scikit-learn (already used by `src/eval/ann_difficulty.py`), pytest, ruff. Run everything with the project venv: `~/TIG/wgan-synthetic/.venv/bin/python` from the worktree `~/TIG/tig-worktrees/wgan-nytimes-eda`.

**Spec:** `docs/superpowers/specs/2026-09-11-linear-skip-generator-design.md`

## Global Constraints

- `make check` (ruff lint, ruff format check, pytest) must stay green after every task; it is CPU-only and takes ~30 s. Run it as `PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check`.
- Only format files you touched: `ruff format <file>`; never `make format` repo-wide.
- Default behaviour must not move: `generator_type` defaults to `mlp`, `training.select_on` defaults to `cov_fro`, and every existing config trains bit-for-bit as before.
- Checkpoints carry no `generator_type`; the architecture is rebuilt from `run_config.yaml` (AGENTS.md invariant 4). Do not change checkpoint keys that existing files rely on (`step`, `generator_weights`, `generator_state_dict`, `critic_state_dict`, `optim_g_state_dict`, `optim_d_state_dict`, `ema_params`, `ema_step`, `best_cov`).
- `v0.yaml` is a ladder rung and is not edited. New configs only.
- Commit after each task with explicit paths (never `git add -A`); commit messages end with the session's attribution lines.
- No push, no box job, until Task 6.

---

### Task 1: `LinearSkipGenerator`

**Files:**
- Modify: `src/models/generator.py` (add the class after `Generator`, before `GatedGenerator`)
- Test: `tests/test_generator.py` (append)

**Interfaces:**
- Produces: `class LinearSkipGenerator(nn.Module)` with `__init__(self, latent_dim: int, output_dim: int, hidden_dims: Iterable[int], negative_slope: float = 0.2, skip_dim: int | None = None, skip_init: str = "orthogonal", skip_init_gain: float = 1.0)`, attributes `trunk: Generator`, `skip: nn.Linear`, `skip_dim: int`, `trunk_latent_dim: int`; `forward(z: Tensor) -> Tensor` of shape `(n, output_dim)`. Raises `ValueError` if `skip_dim >= latent_dim`, if `skip_init` is not `orthogonal`/`identity`, or if `skip_init == "identity"` and `skip_dim != output_dim`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator.py`:

```python
# --- linear_skip -----------------------------------------------------------
from src.models.generator import LinearSkipGenerator  # noqa: E402


def _cov_rank(x: torch.Tensor, floor: float = 1e-6) -> int:
    x = x - x.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh(x.T @ x / (x.shape[0] - 1))
    return int((eig > floor).sum())


def test_linear_skip_output_shape():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(latent_dim=48, output_dim=32, hidden_dims=[16, 16], skip_dim=32)
    out = gen(torch.randn(64, 48))
    assert out.shape == (64, 32)
    assert gen.trunk_latent_dim == 16 and gen.skip_dim == 32


def test_linear_skip_with_trunk_zeroed_is_the_skip_map():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(latent_dim=48, output_dim=32, hidden_dims=[16, 16], skip_dim=32)
    last = gen.trunk.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    z = torch.randn(8, 48)
    expected = z[:, 16:] @ gen.skip.weight.T
    assert torch.allclose(gen(z), expected, atol=1e-6)


def test_linear_skip_output_is_full_rank_where_mlp_is_not():
    # The discriminating claim of the design: the skip path gives the output
    # distribution full rank by construction, where an MLP of the same width
    # collapses. 4096 latents, covariance rank measured against a floor.
    torch.manual_seed(0)
    n, out_dim = 4096, 64
    mlp = Generator(latent_dim=8, output_dim=out_dim, hidden_dims=[16, 16])
    skip = LinearSkipGenerator(
        latent_dim=8 + out_dim, output_dim=out_dim, hidden_dims=[16, 16], skip_dim=out_dim
    )
    with torch.no_grad():
        rank_mlp = _cov_rank(mlp(torch.randn(n, 8)))
        rank_skip = _cov_rank(skip(torch.randn(n, 8 + out_dim)))
    assert rank_mlp <= 8            # an 8-d latent cannot make more than 8 directions
    assert rank_skip == out_dim     # the skip map supplies all 64


def test_linear_skip_identity_init_is_the_identity():
    gen = LinearSkipGenerator(
        latent_dim=40, output_dim=32, hidden_dims=[16], skip_dim=32, skip_init="identity"
    )
    assert torch.equal(gen.skip.weight, torch.eye(32))


def test_linear_skip_orthogonal_init_has_orthonormal_columns():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(
        latent_dim=40, output_dim=32, hidden_dims=[16], skip_dim=32, skip_init_gain=2.0
    )
    w = gen.skip.weight
    assert torch.allclose(w.T @ w, 4.0 * torch.eye(32), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(latent_dim=32, output_dim=32, hidden_dims=[16], skip_dim=32), "skip_dim"),
        (dict(latent_dim=48, output_dim=32, hidden_dims=[16], skip_dim=16, skip_init="identity"), "identity"),
        (dict(latent_dim=48, output_dim=32, hidden_dims=[16], skip_dim=32, skip_init="nope"), "skip_init"),
    ],
)
def test_linear_skip_rejects_bad_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinearSkipGenerator(**kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator.py -q -k linear_skip`
Expected: FAIL at import, `ImportError: cannot import name 'LinearSkipGenerator'`.

- [ ] **Step 3: Implement the class**

Insert into `src/models/generator.py` directly after the `Generator` class:

```python
class LinearSkipGenerator(nn.Module):
    """An MLP trunk plus a full-rank linear map from a separate latent block.

    The latent of size `latent_dim` is split: the first `latent_dim -
    skip_dim` entries feed the trunk (a plain `Generator`), the last
    `skip_dim` entries feed a bias-free linear map `W`. Output is the sum:

        x = trunk(z[:, :t]) + W z[:, t:]        t = latent_dim - skip_dim

    Why: the trunk's Jacobian has rank about 15 on every family measured, so
    its samples lie on a low-dimensional sheet that a per-vector critic
    cannot see and that ANN-difficulty statistics read as "too easy". With
    the skip term, d x / d z_skip = W, so the output's local dimension is at
    least rank(W) whatever the trunk does; the trunk is left to supply the
    structure a Gaussian lacks. See
    docs/superpowers/specs/2026-09-11-linear-skip-generator-design.md.

    The split lives here rather than in a second latent argument so that
    every sampling site (`sample_generator`, `src.sample.generate`) keeps
    drawing `randn(n, latent_dim)` and checkpoints keep loading from
    `run_config.yaml` unchanged.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        skip_dim: int | None = None,
        skip_init: str = "orthogonal",
        skip_init_gain: float = 1.0,
    ):
        super().__init__()
        skip_dim = output_dim if skip_dim is None else int(skip_dim)
        if skip_dim <= 0 or skip_dim >= latent_dim:
            raise ValueError(
                f"skip_dim must be in (0, latent_dim); got skip_dim={skip_dim}, "
                f"latent_dim={latent_dim}"
            )
        if skip_init not in ("orthogonal", "identity"):
            raise ValueError(f"skip_init must be 'orthogonal' or 'identity', got {skip_init!r}")
        if skip_init == "identity" and skip_dim != output_dim:
            raise ValueError(
                f"identity skip_init needs skip_dim == output_dim; got {skip_dim} != {output_dim}"
            )
        self.skip_dim = skip_dim
        self.trunk_latent_dim = latent_dim - skip_dim
        self.trunk = Generator(
            latent_dim=self.trunk_latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )
        self.skip = nn.Linear(skip_dim, output_dim, bias=False)
        with torch.no_grad():
            if skip_init == "identity":
                self.skip.weight.copy_(torch.eye(output_dim))
            else:
                nn.init.orthogonal_(self.skip.weight, gain=skip_init_gain)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        t = self.trunk_latent_dim
        return self.trunk(z[:, :t]) + self.skip(z[:, t:])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator.py -q -k linear_skip`
Expected: 8 passed (5 named tests plus 3 parametrised cases).

- [ ] **Step 5: Mutation check**

Temporarily change `forward` to return `self.trunk(z[:, :t])` only. Run the same command. Expected: `test_linear_skip_with_trunk_zeroed_is_the_skip_map` and `test_linear_skip_output_is_full_rank_where_mlp_is_not` FAIL. Restore the line and re-run: 8 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
~/TIG/wgan-synthetic/.venv/bin/ruff format src/models/generator.py tests/test_generator.py
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add src/models/generator.py tests/test_generator.py
git commit -m "feat(models): LinearSkipGenerator, an MLP trunk plus a full-rank linear skip path"
```

---

### Task 2: `build_generator` dispatch for `linear_skip`

**Files:**
- Modify: `src/models/generator.py` (`build_generator`, currently ~line 372)
- Test: `tests/test_generator_factory.py` (append)

**Interfaces:**
- Consumes: `LinearSkipGenerator` from Task 1.
- Produces: `build_generator({"generator_type": "linear_skip", "latent_dim": L, "generator_hidden_dims": [...], "negative_slope": s, "skip_dim": d?, "skip_init": "orthogonal"|"identity"?, "skip_init_gain": g?}, output_dim)` returns a `LinearSkipGenerator`. Missing `skip_dim` defaults to `output_dim`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_factory.py`:

```python
from src.models.generator import LinearSkipGenerator  # noqa: E402


def test_linear_skip_defaults_skip_dim_to_output_dim():
    cfg = dict(BASE_CFG, generator_type="linear_skip", latent_dim=16 + 128)
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, LinearSkipGenerator)
    assert generator.skip_dim == 128
    assert generator.trunk_latent_dim == 16


def test_linear_skip_honours_overrides():
    cfg = dict(
        BASE_CFG,
        generator_type="linear_skip",
        latent_dim=16 + 64,
        skip_dim=64,
        skip_init="orthogonal",
        skip_init_gain=0.5,
    )
    generator = build_generator(cfg, output_dim=128)
    assert generator.skip_dim == 64
    w = generator.skip.weight
    assert torch.allclose(w.T @ w, 0.25 * torch.eye(64), atol=1e-5)


def test_linear_skip_rejects_skip_dim_at_or_above_latent_dim():
    cfg = dict(BASE_CFG, generator_type="linear_skip", latent_dim=16, skip_dim=16)
    with pytest.raises(ValueError, match="skip_dim"):
        build_generator(cfg, output_dim=128)


def test_linear_skip_rejects_identity_init_of_the_wrong_width():
    cfg = dict(
        BASE_CFG, generator_type="linear_skip", latent_dim=16 + 64, skip_dim=64, skip_init="identity"
    )
    with pytest.raises(ValueError, match="identity"):
        build_generator(cfg, output_dim=128)
```

Add `import torch` at the top of `tests/test_generator_factory.py` (it currently imports only `pytest`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -q -k linear_skip`
Expected: FAIL with `ValueError: Unknown generator_type: linear_skip`.

- [ ] **Step 3: Add the dispatch**

In `build_generator`, before the final `raise ValueError(...)`:

```python
    if kind == "linear_skip":
        return LinearSkipGenerator(
            **common,
            skip_dim=model_cfg.get("skip_dim"),
            skip_init=str(model_cfg.get("skip_init", "orthogonal")),
            skip_init_gain=float(model_cfg.get("skip_init_gain", 1.0)),
        )
```

(`common` already carries `latent_dim`, `output_dim`, `hidden_dims`, `negative_slope`; `skip_dim=None` falls through to the class default of `output_dim`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -q`
Expected: all pass (existing tests plus 4 new).

- [ ] **Step 5: Lint, format, commit**

```bash
~/TIG/wgan-synthetic/.venv/bin/ruff format src/models/generator.py tests/test_generator_factory.py
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add src/models/generator.py tests/test_generator_factory.py
git commit -m "feat(models): build_generator dispatches generator_type: linear_skip"
```

---

### Task 3: Selection module

**Files:**
- Create: `src/train/selection.py`
- Test: `tests/test_selection.py` (new)

**Interfaces:**
- Consumes: `src.eval.ann_difficulty.compute(x, *, k, k_hub, nlist, max_rows, seed, metric)` and `src.eval.ann_difficulty.summary(m)` (returns `lid_median`, `relative_contrast_median`, `hubness_skew`, `ivf_gini`, `lid_discarded_queries`).
- Produces:
  - `SELECTORS = ("cov_fro", "gate")`
  - `gate_statistics(x: np.ndarray, *, metric: str, seed: int) -> dict[str, float | int | None]` — the four statistics plus `lid_discarded_queries`, measured at `k=100, k_hub=10, nlist=256, max_rows=0` (all rows; the caller passes the holdout).
  - `selection_score(fake: Mapping[str, float | None], real: Mapping[str, float | None]) -> float` — `|lid_f - lid_r|/lid_r + |rc_f - rc_r|/rc_r`; `math.inf` when any of the four inputs is `None` or a real value is `0`.
  - `prefixed(stats: Mapping, prefix: str) -> dict[str, float | int | None]` — `{prefix + k: v}` for logging as `gate_fake_*` / `gate_real_*`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_selection.py`:

```python
import math

import numpy as np
import pytest

from src.train.selection import SELECTORS, gate_statistics, prefixed, selection_score


def test_selectors_are_the_two_documented_choices():
    assert SELECTORS == ("cov_fro", "gate")


def test_score_is_the_sum_of_normalised_lid_and_contrast_gaps():
    real = {"lid_median": 50.0, "relative_contrast_median": 2.0}
    fake = {"lid_median": 40.0, "relative_contrast_median": 2.5}
    # |40-50|/50 + |2.5-2.0|/2.0 = 0.2 + 0.25
    assert selection_score(fake, real) == pytest.approx(0.45)


def test_score_prefers_the_checkpoint_nearer_real_on_both():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27}
    near = {"lid_median": 50.0, "relative_contrast_median": 1.30}
    far = {"lid_median": 14.0, "relative_contrast_median": 2.08}
    assert selection_score(near, real) < selection_score(far, real)


def test_score_ignores_hubness_and_gini():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27, "hubness_skew": 2.5, "ivf_gini": 0.8}
    a = {"lid_median": 50.0, "relative_contrast_median": 1.30, "hubness_skew": 9.0, "ivf_gini": 0.1}
    b = {"lid_median": 50.0, "relative_contrast_median": 1.30, "hubness_skew": 2.5, "ivf_gini": 0.8}
    assert selection_score(a, real) == selection_score(b, real)


@pytest.mark.parametrize(
    "fake, real",
    [
        ({"lid_median": None, "relative_contrast_median": 1.3}, {"lid_median": 56.0, "relative_contrast_median": 1.27}),
        ({"lid_median": 50.0, "relative_contrast_median": 1.3}, {"lid_median": 0.0, "relative_contrast_median": 1.27}),
        ({"lid_median": 50.0}, {"lid_median": 56.0, "relative_contrast_median": 1.27}),
    ],
)
def test_score_is_infinite_when_a_statistic_is_unmeasurable(fake, real):
    assert selection_score(fake, real) == math.inf


def test_gate_statistics_returns_the_four_statistics_and_the_discard_count():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    stats = gate_statistics(x, metric="angular", seed=0)
    assert set(stats) == {
        "lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini", "lid_discarded_queries"
    }
    assert stats["lid_median"] > 0 and stats["relative_contrast_median"] > 1.0
    assert stats["lid_discarded_queries"] == 0


def test_gate_statistics_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    assert gate_statistics(x, metric="l2", seed=3) == gate_statistics(x, metric="l2", seed=3)


def test_prefixed_renames_every_key():
    assert prefixed({"lid_median": 1.0, "ivf_gini": 0.5}, "gate_fake_") == {
        "gate_fake_lid_median": 1.0,
        "gate_fake_ivf_gini": 0.5,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_selection.py -q`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'src.train.selection'`.

- [ ] **Step 3: Implement the module**

Create `src/train/selection.py`:

```python
"""Checkpoint selection for the trainer.

`best_generator.pt` used to be chosen by the smallest covariance Frobenius
gap (`cov_fro`). On NYTimes that chose step 1,000 of a 100,000-step run,
because the covariance gap is smallest for a tight cone, while the
statistics the project actually gates on (src/eval/ann_difficulty.py) got
worse for the rest of training. A generator built to fix local dimension
needs a selector that can see local dimension.

`gate` scores a checkpoint on the normalised gaps of the two statistics that
read local dimension -- LID median and relative contrast -- between the
fake holdout sample and the real holdout. Hubness skew and IVF Gini are
measured and logged but not scored: their noise floors are wider, and on a
near-isotropic corpus Gini mostly reflects k-means behaviour.

The holdout is smaller than the canonical N on a dataset page, so these
numbers rank checkpoints of one run against each other; they are not the
family's profile and must not be copied into a dataset page.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from src.eval import ann_difficulty

SELECTORS = ("cov_fro", "gate")

# The gate's canonical k / k_hub / nlist (gates/*.yaml). max_rows=0 means
# "all rows": the caller passes the holdout, and the holdout is the sample.
GATE_K = 100
GATE_K_HUB = 10
GATE_NLIST = 256

SCORED = ("lid_median", "relative_contrast_median")
LOGGED = ("lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini", "lid_discarded_queries")


def gate_statistics(x: np.ndarray, *, metric: str, seed: int) -> dict[str, float | int | None]:
    """The four ANN-difficulty statistics of `x`, plus the LID discard count."""
    m = ann_difficulty.compute(
        np.ascontiguousarray(x, dtype=np.float32),
        k=GATE_K,
        k_hub=GATE_K_HUB,
        nlist=GATE_NLIST,
        max_rows=0,
        seed=seed,
        metric=metric,
    )
    s = ann_difficulty.summary(m)
    return {k: s[k] for k in LOGGED}


def selection_score(
    fake: Mapping[str, float | int | None], real: Mapping[str, float | int | None]
) -> float:
    """Sum over LID and contrast of |fake - real| / real. Lower is better.

    Infinite when a statistic is missing or None on either side (every query
    discarded), or when the real value is zero: an unmeasurable checkpoint
    must never win by accident.
    """
    total = 0.0
    for key in SCORED:
        f, r = fake.get(key), real.get(key)
        if f is None or r is None or r == 0:
            return math.inf
        total += abs(float(f) - float(r)) / abs(float(r))
    return total


def prefixed(stats: Mapping[str, float | int | None], prefix: str) -> dict[str, float | int | None]:
    return {prefix + k: v for k, v in stats.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_selection.py -q`
Expected: 9 passed.

- [ ] **Step 5: Mutation check**

Change `total += abs(...)` to `total -= abs(...)`. Run the tests. Expected: `test_score_is_the_sum_of_normalised_lid_and_contrast_gaps` and `test_score_prefers_the_checkpoint_nearer_real_on_both` FAIL. Restore, re-run: 9 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
~/TIG/wgan-synthetic/.venv/bin/ruff format src/train/selection.py tests/test_selection.py
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add src/train/selection.py tests/test_selection.py
git commit -m "feat(train): gate-aware checkpoint scoring on the holdout's LID and contrast"
```

---

### Task 4: Trainer integration of `training.select_on`

**Files:**
- Modify: `src/train/train_wgan_gp.py` — `save_checkpoint` (~line 320), the resume block (~lines 501-546), the eval block (~lines 675-713)
- Test: `tests/test_train_smoke.py` (append)

**Interfaces:**
- Consumes: `gate_statistics`, `selection_score`, `prefixed`, `SELECTORS` from Task 3.
- Produces: config key `training.select_on` (`"cov_fro"` default, or `"gate"`); `run_metadata.json["eval"][i]` gains `gate_fake_<stat>` and `gate_real_<stat>` keys and `selection_score` when `select_on == "gate"`; checkpoints gain `select_on: str` and `best_score: float`; `best_cov` keeps its meaning (the running minimum `cov_fro`, still recorded under either selector). Resume raises `ValueError` if the checkpoint's `select_on` differs from the config's.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_smoke.py`:

```python
def test_select_on_defaults_to_cov_fro_and_records_it(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    ckpt_path, meta = train(cfg)
    best = torch.load(ckpt_path, weights_only=False)
    assert best["select_on"] == "cov_fro"
    assert "gate_fake_lid_median" not in meta["eval"][0]


def test_select_on_gate_logs_gate_statistics_and_picks_the_lowest_score(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["training"]["select_on"] = "gate"
    cfg["training"]["num_gen_steps"] = 6
    cfg["training"]["eval_every"] = 2
    ckpt_path, meta = train(cfg)

    evals = meta["eval"]
    assert len(evals) == 3
    for e in evals:
        for stat in ("lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini"):
            assert f"gate_fake_{stat}" in e and f"gate_real_{stat}" in e
        assert math.isfinite(e["selection_score"]) or e["selection_score"] == math.inf
    # real-side statistics are computed once and repeated, not re-drawn
    assert all(e["gate_real_lid_median"] == evals[0]["gate_real_lid_median"] for e in evals)

    best = torch.load(ckpt_path, weights_only=False)
    assert best["select_on"] == "gate"
    best_step = min(evals, key=lambda e: e["selection_score"])["step"]
    assert best["step"] == best_step
    assert best["best_score"] == pytest.approx(min(e["selection_score"] for e in evals))


def test_select_on_rejects_unknown_values(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["training"]["select_on"] = "vibes"
    with pytest.raises(ValueError, match="select_on"):
        train(cfg)


def test_resume_refuses_a_checkpoint_selected_under_a_different_selector(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["training"]["save_every"] = 2
    train(cfg)
    live_ckpt = tmp_path / "mlp" / "checkpoint_step_2.pt"
    assert live_ckpt.exists()
    cfg2 = make_config(tmp_path, "mlp")
    cfg2["training"]["select_on"] = "gate"
    cfg2["training"]["num_gen_steps"] = 8
    with pytest.raises(ValueError, match="select_on"):
        train(cfg2, resume=str(live_ckpt))
```

The smoke fixture's holdout is `256 * 0.2 = 51` rows, so `ann_difficulty.compute` clamps `k` to 50 and `nlist` to 25; that is fine for a smoke test, and `selection_score` may legitimately be `inf` if every query is discarded on a degenerate step, which the test allows.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_smoke.py -q -k "select_on or resume_refuses"`
Expected: 4 FAIL (`KeyError: 'select_on'` on the first; the gate test fails on missing keys; the rejection tests fail because nothing raises).

- [ ] **Step 3: Implement**

3a. Import, near the other `src.train` imports at the top of `src/train/train_wgan_gp.py`:

```python
from src.train.selection import (
    SELECTORS,
    gate_statistics,
    prefixed,
    selection_score,
)
```

3b. `save_checkpoint`: add two keyword parameters and two keys.

```python
    best_cov: float = float("inf"),
    select_on: str = "cov_fro",
    best_score: float = float("inf"),
) -> None:
```

and in the `ckpt` dict, after `"best_cov": float(best_cov),`:

```python
        "select_on": select_on,
        "best_score": float(best_score),
```

Extend the docstring's list of resume state with one sentence: `select_on` and `best_score` record which selector chose `best_generator.pt` and its running best, so a resume under a different selector is refused rather than silently mixing two scores.

3c. In `train`, right after `best_cov = float("inf")` (~line 501):

```python
    select_on = str(train_cfg.get("select_on", "cov_fro"))
    if select_on not in SELECTORS:
        raise ValueError(f"training.select_on must be one of {SELECTORS}, got {select_on!r}")
    best_score = float("inf")
    metric = data_cfg.get("metric", "l2")
    # Real-side gate statistics are a property of the holdout, not of the
    # step: computed once, repeated into every eval entry so each is
    # self-describing.
    real_gate = gate_statistics(x_holdout, metric=metric, seed=seed) if select_on == "gate" else None
```

Note: `train_cfg` is already bound earlier in `train` (it is used for `gpu_memory_fraction`); if the name in scope at that point differs, use `config["training"]`. `data_cfg` is bound at ~line 417.

3d. In the resume block, after `best_cov = float(ckpt.get("best_cov", float("inf")))`:

```python
        ckpt_select_on = str(ckpt.get("select_on", "cov_fro"))
        if ckpt_select_on != select_on:
            raise ValueError(
                f"{resume} was selected under select_on={ckpt_select_on!r} but the "
                f"config says {select_on!r}; a resume cannot mix two selection scores"
            )
        best_score = float(ckpt.get("best_score", float("inf")))
```

3e. In the eval block, replace

```python
                stats = tensor_stats(x_holdout, fake_holdout)
                stats.update(collapse_stats(fake_holdout))
                stats["step"] = step
                run_meta.setdefault("eval", []).append(stats)
                print(json.dumps({"eval": stats}))
                if stats["cov_fro"] < best_cov:
                    best_cov = stats["cov_fro"]
```

with

```python
                stats = tensor_stats(x_holdout, fake_holdout)
                stats.update(collapse_stats(fake_holdout))
                if select_on == "gate":
                    fake_gate = gate_statistics(fake_holdout, metric=metric, seed=seed)
                    stats.update(prefixed(fake_gate, "gate_fake_"))
                    stats.update(prefixed(real_gate, "gate_real_"))
                    stats["selection_score"] = selection_score(fake_gate, real_gate)
                stats["step"] = step
                run_meta.setdefault("eval", []).append(stats)
                print(json.dumps({"eval": stats}))
                # best_cov keeps tracking cov_fro under both selectors so the
                # two can be compared after the fact.
                improved_cov = stats["cov_fro"] < best_cov
                if improved_cov:
                    best_cov = stats["cov_fro"]
                if select_on == "gate":
                    improved = stats["selection_score"] < best_score
                    if improved:
                        best_score = stats["selection_score"]
                else:
                    improved = improved_cov
                if improved:
```

and pass `select_on=select_on, best_score=best_score` into both `save_checkpoint(...)` calls (the `best=True` one inside the eval block and the periodic `best=False` one).

`json.dumps` must not see `math.inf` as a bare float in a way that breaks: Python's `json` writes `Infinity`, which `json.load` reads back, so `run_metadata.json` stays loadable. If ruff or a reviewer objects, cast with `float(...)` only; do not replace `inf` with a sentinel number.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_smoke.py -q`
Expected: all pass, including the four new tests and the three pre-existing parametrised loops.

- [ ] **Step 5: Mutation check**

In the eval block change `improved = stats["selection_score"] < best_score` to `>`. Run `pytest tests/test_train_smoke.py -q -k gate_logs`. Expected: FAIL on `best["step"] == best_step`. Restore, re-run: pass.

- [ ] **Step 6: Lint, format, full check, commit**

```bash
~/TIG/wgan-synthetic/.venv/bin/ruff format src/train/train_wgan_gp.py tests/test_train_smoke.py
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add src/train/train_wgan_gp.py tests/test_train_smoke.py
git commit -m "feat(train): training.select_on chooses best_generator.pt by cov_fro or by the gate"
```

---

### Task 5: NYTimes `v1` configs, job script and documentation

**Files:**
- Create: `configs/nytimes/v1.yaml`, `configs/nytimes/v1_seed42.yaml`, `scripts/nytimes_v1_seed42_job.sh`
- Modify: `PROJECT_DOCUMENTATION.md` (`### generator_type` section, ~line 321; the "Generator regularizers" table, ~line 366), `docs/datasets/nytimes.md` (Ladder table)

**Interfaces:**
- Consumes: `generator_type: linear_skip` (Task 2), `training.select_on: gate` (Task 4).
- Produces: the rung's config and the box job that Task 6 submits.

- [ ] **Step 1: Write `configs/nytimes/v1.yaml`**

Copy `configs/nytimes/v0.yaml` and change exactly these lines (header comment, `output_dir`, `model.latent_dim`, three new `model` keys, one new `training` key):

```yaml
# NYTIMES v1 -- v0 plus a full-rank linear skip path in the generator and
# gate-aware checkpoint selection. One architectural delta on top of v0:
# v0's samples sit on a ~15-dimensional local sheet on every family, which
# ANN-difficulty statistics read as far too easy to search (LID 14 against
# the corpus's 56). The skip path makes the output Jacobian full rank by
# construction; the selector picks the checkpoint by the holdout's LID and
# contrast gaps instead of cov_fro, which chose an untrained step for v0.
# See docs/superpowers/specs/2026-09-11-linear-skip-generator-design.md.
seed: 42
device: auto
output_dir: runs/nytimes/v1
```

`model` block:

```yaml
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
```

`training` block: identical to v0 plus, after `save_every: 2000`:

```yaml
  select_on: gate
```

- [ ] **Step 2: Write `configs/nytimes/v1_seed42.yaml`**

Same content as `v1.yaml` with the header replaced by the instrument header used in `configs/nytimes/v0_seed42.yaml` (adapted: "Identical to configs/nytimes/v1.yaml ..."), `output_dir: runs/nytimes/v1_seed42`, and `real_path: /workspace/data-cache/nytimes_250k.npy`.

- [ ] **Step 3: Write `scripts/nytimes_v1_seed42_job.sh`**

Copy `scripts/nytimes_v0_seed42_job.sh` and change: the header comment (v1), `RUN=runs/nytimes/v1_seed42`, `KEEP=/workspace/nytimes-v1/v1_seed42`, the config path `configs/nytimes/v1_seed42.yaml`, and the sampling/measuring tail to:

```bash
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v1_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v1_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
```

Then `bash -n scripts/nytimes_v1_seed42_job.sh && chmod +x scripts/nytimes_v1_seed42_job.sh`.

- [ ] **Step 4: Verify the configs build**

```bash
~/TIG/wgan-synthetic/.venv/bin/python - <<'EOF'
import yaml
from src.models.generator import LinearSkipGenerator, build_generator
for p in ("configs/nytimes/v1.yaml", "configs/nytimes/v1_seed42.yaml"):
    c = yaml.safe_load(open(p))
    g = build_generator(c["model"], c["data"]["descriptor_dim"])
    assert isinstance(g, LinearSkipGenerator) and g.skip_dim == 256 and g.trunk_latent_dim == 256
    assert c["training"]["select_on"] == "gate" and c["training"]["num_gen_steps"] == 30000
    print(p, "ok", c["output_dir"], c["data"]["real_path"])
EOF
```

Expected: two `ok` lines.

- [ ] **Step 5: Document**

`PROJECT_DOCUMENTATION.md`, `### generator_type`: change "accepting `mlp` (default), `gated`, and `structured_gated`" to include `linear_skip`, and add one paragraph after the existing ones:

```markdown
`linear_skip` is an `mlp` trunk plus a bias-free linear map on a separate
block of the latent: `x = trunk(z[:, :t]) + W z[:, t:]` with
`t = latent_dim - skip_dim`. The output Jacobian is full rank by
construction, which is the property `mlp` lacks on every family measured
(its local dimension sits at about 15 regardless of the corpus). Keys:
`skip_dim` (default `descriptor_dim`), `skip_init` (`orthogonal` or
`identity`), `skip_init_gain`. Sampling and checkpoints are unchanged: the
split happens inside the generator. First used by `configs/nytimes/v1.yaml`.
```

Add a `### Checkpoint selection` subsection directly after the "Generator regularizers" table:

```markdown
### Checkpoint selection

| Config key | Default | Meaning |
|---|---|---|
| `training.select_on` | `cov_fro` | Which statistic chooses `best_generator.pt`. `cov_fro` is the covariance Frobenius gap on the holdout, as always. `gate` scores each evaluation by the holdout's normalised LID-median and relative-contrast gaps (`src/train/selection.py`), logs all four ANN-difficulty statistics for fake and real as `gate_fake_*` / `gate_real_*`, and records `selection_score`. |

The holdout is smaller than a family's canonical N, so `gate_*` values rank
checkpoints within one run and are not the family's profile. Checkpoints
record `select_on` and `best_score`; a resume under a different selector is
refused.
```

`docs/datasets/nytimes.md`, Ladder table: add the row

```markdown
| `v1` | + linear skip path (`generator_type: linear_skip`) and `select_on: gate` | `configs/nytimes/v1.yaml`; box instrument `configs/nytimes/v1_seed42.yaml` | `runs/nytimes/v1_seed42` | not trained |
```

- [ ] **Step 6: Check and commit**

```bash
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add configs/nytimes/v1.yaml configs/nytimes/v1_seed42.yaml scripts/nytimes_v1_seed42_job.sh PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "configs(nytimes): v1 rung -- linear-skip generator with gate-aware selection"
```

---

### Task 6: Train `v1` on the box and record the result

**Files:**
- Create: `docs/results/nytimes-v1-seed42/{eda_clean_summary.json,run_config.yaml,run_metadata.json}`
- Modify: `docs/datasets/nytimes.md` (Ladder row status; new `## v1, measured` section before `## Gate`)

**Interfaces:**
- Consumes: everything above, pushed to `origin/nytimes-eda`.

- [ ] **Step 1: Push and preflight**

```bash
git push origin nytimes-eda
SHA=$(git rev-parse HEAD)
ssh tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; PRE=\$(gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-eda --lane cpu --dedupe-key nytimes-v1-preflight-$SHA -- bash -c 'bash -n scripts/nytimes_v1_seed42_job.sh && test -f /workspace/data-cache/nytimes_250k.npy && test -f /workspace/data-cache/nytimes_250k_l2_clean.npy && /opt/venvs/wgan-synthetic/bin/python -c \"import yaml; from src.models.generator import build_generator; c=yaml.safe_load(open(\\\"configs/nytimes/v1_seed42.yaml\\\")); print(type(build_generator(c[\\\"model\\\"], 256)).__name__)\"'); gpuq wait \$PRE; echo preflight rc \$?; gpuq show \$PRE | grep -E '\"(exit_code|error)\"'"
```

Expected: `preflight rc 0`. The ssh banner prints `bind [127.0.0.1]:8080` noise; ignore it.

- [ ] **Step 2: Submit the GPU job**

```bash
ssh tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-eda --lane gpu --timeout-s 10800 --dedupe-key nytimes-v1-seed42-$SHA -- bash scripts/nytimes_v1_seed42_job.sh"
```

Record the printed job id. Under `select_on: gate` each eval adds a 12,500-row k-NN (~10 s), so expect ~45 minutes rather than v0's 37.

- [ ] **Step 3: Wait**

```bash
ssh -o ServerAliveInterval=30 tig-gpu "export PATH=/opt/gpuq/venv/bin:\$PATH; gpuq wait --timeout 9000 --poll 60 <job-id>; echo wait rc \$?; gpuq show <job-id> | grep -E '\"(state|exit_code|error)\"'"
```

Run in the background with the output redirected to a file. Expected: `wait rc 0`, `"state": "done"`. On failure, `gpuq show <job-id>` carries the stderr tail in `error`; fix, commit, push, resubmit with the new SHA.

- [ ] **Step 4: Fetch and stage the artifacts**

```bash
mkdir -p runs/nytimes/v1_seed42 docs/results/nytimes-v1-seed42
scp -r tig-gpu:/workspace/nytimes-v1/v1_seed42/{eda_clean,run_config.yaml,run_metadata.json,synthetic_50k_best.npy,synthetic_50k_step30000.npy} runs/nytimes/v1_seed42/
cp runs/nytimes/v1_seed42/run_config.yaml runs/nytimes/v1_seed42/run_metadata.json docs/results/nytimes-v1-seed42/
cp runs/nytimes/v1_seed42/eda_clean/summary.json docs/results/nytimes-v1-seed42/eda_clean_summary.json
```

- [ ] **Step 5: Read the result against the spec's success criterion**

```bash
~/TIG/wgan-synthetic/.venv/bin/python - <<'EOF'
import json
nf = json.load(open("docs/datasets/nytimes_noise_floor.json"))["zero_and_duplicate_rows_removed"]["per_seed"]
st = {e["name"]: e for e in json.load(open("docs/results/nytimes-v1-seed42/eda_clean_summary.json"))["stats"]}
keys = [("lid_median","LID median"),("relative_contrast_median","Relative contrast"),("hubness_skew","Hubness skew"),("ivf_gini","IVF cell-balance Gini")]
for k, n in keys:
    v = [p[k] for p in nf]; lo, hi = min(v), max(v); real = st["real"][k]
    cells = [f"`{real:.4g}` ({lo:.4g} -- {hi:.4g})"]
    for s in ("v1_best", "v1_step30000"):
        x = st[s][k]; inside = lo <= x <= hi; pct = abs(x - real) / real * 100
        cells.append(f"`{x:.4g}` ({'in range' if inside else f'{pct:.1f}% off'})")
    print(f"| {n} | " + " | ".join(cells) + " |")
for s in ("real", "v1_best", "v1_step30000"):
    print(s, "effective rank", round(st[s]["effective_rank"], 1), "median 5-NN", round(st[s]["median_5nn_distance"], 3))
m = json.load(open("docs/results/nytimes-v1-seed42/run_metadata.json"))
best = min(m["eval"], key=lambda e: e["selection_score"]); print("selected step", best["step"], "score", round(best["selection_score"], 4))
EOF
```

Success per the spec: LID and contrast within 3% of the real median (or inside the range); hubness and Gini read against the noise-sweep bound (4.0 and 0.72).

- [ ] **Step 6: Record on the page**

In `docs/datasets/nytimes.md`: set the `v1` ladder row's status to `trained -- n=1 seed, see ## v1, measured` (and whether it met the bar); add `## v1, measured` before `## Gate` with: the run provenance (commit, job, wall time), the table from Step 5, effective rank and 5-NN lines, which step the gate selector chose and what `cov_fro` would have chosen (`min(m["eval"], key=lambda e: e["cov_fro"])["step"]`), and a verdict paragraph that names the next rung: if LID and contrast land but hubness/Gini do not, the trunk's structure is the next lever; if it lands on the Gaussian's numbers (LID ~82, contrast ~1.16), the neighbourhood-aware critic is next. State the verdict plainly; do not hedge.

- [ ] **Step 7: Commit and push**

```bash
PATH=~/TIG/wgan-synthetic/.venv/bin:$PATH make check
git add docs/datasets/nytimes.md docs/results/nytimes-v1-seed42/
git commit -m "docs(nytimes): record the v1 seed-42 run"
git push origin nytimes-eda
```

---

## Self-review

**Spec coverage.** `linear_skip` class and keys: Task 1 and 2. Split-inside-generator rationale: Task 1 docstring. `select_on` with default `cov_fro`, `gate_*` logging, score on LID and contrast only, `best_score` and selector recorded in checkpoints, resume refusal: Task 3 and 4. Holdout caveat: Task 3 docstring and Task 5 docs. The rung and its box instrument, trained as shipped: Task 5. Final checkpoint sampled beside the selected one: Task 5 script. Tests named in the spec, including the rank-gap assertion and both mutation checks: Tasks 1, 3, 4. Success criterion applied: Task 6 Step 5.

**Placeholders.** None; every code step carries its code, every command its expected output.

**Type consistency.** `LinearSkipGenerator(latent_dim, output_dim, hidden_dims, negative_slope, skip_dim, skip_init, skip_init_gain)` is used identically in Tasks 1, 2 and 5. `gate_statistics(x, *, metric, seed)`, `selection_score(fake, real)`, `prefixed(stats, prefix)` and `SELECTORS` match between Tasks 3 and 4. `save_checkpoint` gains `select_on` and `best_score` in Task 4 and both call sites pass them. The eval keys `gate_fake_*`, `gate_real_*`, `selection_score` match between Task 4's trainer, Task 4's tests, and Task 6's reader.
