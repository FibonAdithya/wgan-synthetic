# Spherical Generator (NYTimes v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `generator_type: spherical`, a generator whose output is unit-norm by construction with a constant residual angle across samples and a residual shape conditioned on the trunk, and stage the NYTimes `v3` rung that trains it.

**Architecture:** One new class `SphericalGenerator` in `src/models/generator.py`, selected by the existing `build_generator` factory. The trunk MLP emits a hidden state `h` and a unit direction `u`; a tangent head reads the skip latent block, is modulated per channel by `gamma(h)` and `beta(h)`, and its output is projected orthogonal to `u` and normalised to `t`; the output is `cos(r) u + sin(r) t` with one learned scalar `r` inside a configured band. The trainer logs `radius` and `direction_effective_rank` per evaluation. The `v3` config is `v2c` with the generator keys swapped.

**Tech Stack:** PyTorch, PyYAML, pytest, ruff. Python 3.12. Tests run from the repo root with the venv at the main checkout: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`.

**Spec:** `docs/ai/specs/2026-09-15-spherical-generator-design.md`

## Global Constraints

- Every config key name and default is exactly as the spec's table: `skip_dim` (default `output_dim`), `tangent_hidden_dim` (`512`), `radius_init` (`0.95`), `radius_min` (`0.2`), `radius_max` (`1.5`).
- Construction validates `0 < skip_dim < latent_dim`, `0 < radius_min < radius_init < radius_max < pi/2`, `tangent_hidden_dim > 0`, and non-empty `hidden_dims`, raising `ValueError` with the key name in the message.
- `configs/nytimes/v3.yaml` is byte-for-byte `v2c.yaml` outside the `model` generator keys and `output_dir`; the pinning test states the diff.
- No change to the critic, the selector, `linear_skip`, or any config other than the two new NYTimes files.
- `make check` (ruff lint, ruff format check, pytest) passes at the end of every task. Run it as `make check` from the worktree root.
- Commit only the files each task names. Never `git add -A`.
- Work on branch `nytimes-v3`, cut from `probe/nytimes-trunk-scale` at its head so the spec and the probe results travel with the rung.
- Every new test is mutation-checked in its task: break the code as the step says, confirm the named test fails, restore, confirm it passes.

---

## File map

| file | responsibility |
|---|---|
| `src/models/generator.py` | `SphericalGenerator` class, `_unit` and `_effective_rank` helpers, factory branch |
| `src/train/train_wgan_gp.py` | build the eval probe for the new class; log its diagnostics |
| `tests/test_generator_spherical.py` | new: geometry, band, rank, conditioning, diagnostics tests |
| `tests/test_generator_factory.py` | factory branch tests |
| `tests/test_train_smoke.py` | smoke run and diagnostics-logging test |
| `configs/nytimes/v3.yaml`, `configs/nytimes/v3_seed42.yaml` | the rung and its box instrument |
| `scripts/nytimes_v3_seed42_job.sh` | the gpuq job |
| `tests/test_nytimes_configs.py` | pinning tests for the two configs and the script |
| `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md` | `generator_type` section; ladder row and model-family line |
| `docs/ai/specs/2026-09-15-spherical-generator-design.md` | two corrections (Task 1) |

---

### Task 1: `SphericalGenerator` geometry

**Files:**
- Modify: `src/models/generator.py` (add `import math` at the top; add the class after `LinearSkipGenerator`, before `GatedGenerator`, around line 128)
- Modify: `docs/ai/specs/2026-09-15-spherical-generator-design.md` (two corrections, step 1)
- Test: `tests/test_generator_spherical.py` (new)

**Interfaces:**
- Consumes: `nn.Module`, `torch`.
- Produces: `class SphericalGenerator(nn.Module)` with `__init__(latent_dim: int, output_dim: int, hidden_dims: Iterable[int], negative_slope: float = 0.2, skip_dim: int | None = None, tangent_hidden_dim: int = 512, radius_init: float = 0.95, radius_min: float = 0.2, radius_max: float = 1.5, eps: float = 1e-8)`; attributes `skip_dim: int`, `trunk_latent_dim: int`, `trunk: nn.Sequential`, `direction: nn.Linear`, `tangent_in: nn.Linear`, `gamma: nn.Linear`, `beta: nn.Linear`, `tangent_out: nn.Linear`, `radius_raw: nn.Parameter`, `radius_min: float`, `radius_max: float`; property `radius -> Tensor` (0-d); methods `tangent_raw(h: Tensor, z_skip: Tensor) -> Tensor` (pre-projection tangent output, shape `(n, output_dim)`), `components(z: Tensor) -> tuple[Tensor, Tensor]` returning `(u, t)`, `forward(z: Tensor) -> Tensor`.

- [ ] **Step 1: Correct the spec**

In `docs/ai/specs/2026-09-15-spherical-generator-design.md`:

Replace the local-rank row of the tests table:

```
| local rank | the Jacobian of one output row with respect to `z_s` has rank at least `min(skip_dim, output_dim - 1)` at init | zeroing the tangent head's skip weight |
```
with
```
| local rank | the Jacobian of one output row with respect to `z_s` has rank at least `min(skip_dim, output_dim - 2)` at init (`t` is a unit vector in the `output_dim - 1` dimensional tangent space, so its Jacobian has rank at most `output_dim - 2`) | zeroing the tangent head's skip weight |
```

Replace the band-enforcement row:

```
| band enforcement | `radius_raw = +-50` leaves `r` strictly inside `(radius_min, radius_max)` | an unclamped radius |
```
with
```
| band enforcement | `radius_raw = +-50` leaves `r` inside the closed band `[radius_min, radius_max]` (float32 sigmoid saturates to exactly 0 and 1 there), and `radius_raw = +-5` leaves it strictly inside | an unclamped radius |
```

Also in the "Data flow" section replace

```
- `direction_effective_rank`: the effective rank (the report's definition,
  `src/eval/eda/metrics.effective_rank`) of `u` over the evaluation batch,
```
with
```
- `direction_effective_rank`: the effective rank of `u` over the evaluation
  batch, `exp` of the Shannon entropy of the covariance eigenvalue ratios,
  the same formula as `src/eval/eda/metrics.effective_rank`, computed in
  torch inside the model module so `src/models` does not import `src/eval`,
```

- [ ] **Step 2: Write the failing geometry tests**

Create `tests/test_generator_spherical.py`:

```python
"""SphericalGenerator: unit-norm by construction, constant residual angle
across samples, residual shape conditioned on the trunk.

Each test names the mutation it catches. See
docs/ai/specs/2026-09-15-spherical-generator-design.md, "Tests".
"""

import math

import pytest
import torch

from src.models.generator import SphericalGenerator

LATENT, OUT, SKIP, HID, TANGENT = 8 + 32, 32, 32, [16, 16], 64


def make(**overrides):
    kwargs = dict(
        latent_dim=LATENT,
        output_dim=OUT,
        hidden_dims=HID,
        skip_dim=SKIP,
        tangent_hidden_dim=TANGENT,
    )
    kwargs.update(overrides)
    return SphericalGenerator(**kwargs)


def test_output_shape_and_split():
    torch.manual_seed(0)
    gen = make()
    out = gen(torch.randn(64, LATENT))
    assert out.shape == (64, OUT)
    assert gen.trunk_latent_dim == 8 and gen.skip_dim == 32


def test_output_is_unit_norm_by_construction():
    """Catches dropping the normalisation of `t` or of `u`."""
    torch.manual_seed(0)
    gen = make()
    out = gen(torch.randn(512, LATENT))
    norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-5)


def test_tangent_is_orthogonal_to_direction():
    """Catches removing the projection."""
    torch.manual_seed(0)
    gen = make()
    u, t = gen.components(torch.randn(512, LATENT))
    assert torch.allclose((u * t).sum(dim=1), torch.zeros(512), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(u, dim=1), torch.ones(512), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(t, dim=1), torch.ones(512), atol=1e-5)


def test_angle_from_direction_is_the_radius_on_every_row():
    """The hub guard. Catches a per-sample radius and a wrong sigmoid
    mapping: every row sits at exactly `radius_init` from `u`."""
    torch.manual_seed(0)
    gen = make(radius_init=0.95)
    z = torch.randn(512, LATENT)
    u, _ = gen.components(z)
    x = gen(z)
    angles = torch.acos((x * u).sum(dim=1).clamp(-1.0, 1.0))
    assert torch.allclose(angles, torch.full((512,), 0.95), atol=1e-5)
    assert float(angles.std()) < 1e-5


def test_radius_starts_at_radius_init():
    gen = make(radius_init=0.7, radius_min=0.1, radius_max=1.2)
    assert abs(float(gen.radius) - 0.7) < 1e-6


def test_radius_stays_in_the_band():
    """Catches an unclamped radius. sigmoid(+-50) is exactly 0 or 1 in
    float32, so the extremes land on the closed band's edges; +-5 stays
    strictly inside."""
    gen = make(radius_min=0.2, radius_max=1.5)
    for raw, strict in ((50.0, False), (-50.0, False), (5.0, True), (-5.0, True)):
        with torch.no_grad():
            gen.radius_raw.fill_(raw)
        r = float(gen.radius)
        assert 0.2 <= r <= 1.5
        if strict:
            assert 0.2 < r < 1.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(skip_dim=0), "skip_dim"),
        (dict(skip_dim=LATENT), "skip_dim"),
        (dict(hidden_dims=[]), "hidden_dims"),
        (dict(tangent_hidden_dim=0), "tangent_hidden_dim"),
        (dict(radius_min=0.0), "radius_min"),
        (dict(radius_min=0.95), "radius_min"),
        (dict(radius_init=1.5), "radius_init"),
        (dict(radius_max=math.pi / 2), "radius_max"),
    ],
)
def test_rejects_bad_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make(**kwargs)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_spherical.py -q`
Expected: FAIL at import with `ImportError: cannot import name 'SphericalGenerator'`.

- [ ] **Step 4: Implement the class**

At the top of `src/models/generator.py`, after `from __future__ import annotations`, add:

```python
import math
```

After the `LinearSkipGenerator` class (its last method is `component_energies`) and before `class GatedGenerator`, add:

```python
def _unit(x: Tensor, eps: float) -> Tensor:
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp(min=eps)


def _effective_rank(x: Tensor) -> float:
    """exp(Shannon entropy of the covariance eigenvalue ratios): the same
    definition as `src.eval.eda.metrics.effective_rank`, in torch, so this
    module does not import the report."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh(x.T @ x / max(x.shape[0] - 1, 1)).clamp(min=0.0)
    ratio = eig / eig.sum().clamp(min=1.0e-12)
    return float(torch.exp(-(ratio * torch.log(ratio + 1.0e-12)).sum()))


class SphericalGenerator(nn.Module):
    """Unit-norm output with a constant residual angle and a residual shape
    conditioned on the trunk.

        u = unit(direction(h)),  h = trunk(z[:, :t])
        v = tangent_out(act(tangent_in(z[:, t:]) * (1 + gamma(h)) + beta(h)))
        t = unit(v - (v . u) u)
        x = cos(r) u + sin(r) t,     r = radius_min + (radius_max - radius_min) sigmoid(radius_raw)

    Why: on NYTimes the linear-skip generator's hubs are rows whose residual
    share is a few percent smaller than their neighbours' (trunk-norm CV
    0.06-0.08 was enough for hubness 17-30; equalising it gave 3-6), and a
    fixed linear residual cannot reach the corpus's local dimension without
    losing its global rank. So the angle `r` is one scalar shared by every
    sample, and the tangent direction's shape follows `h` through the
    per-channel modulation while its magnitude, sin(r), does not. See
    docs/ai/specs/2026-09-15-spherical-generator-design.md.

    Output is unit-norm to float precision, so the trainer's `normalize_l2`
    and the sampler's normalisation are no-ops on it.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        skip_dim: int | None = None,
        tangent_hidden_dim: int = 512,
        radius_init: float = 0.95,
        radius_min: float = 0.2,
        radius_max: float = 1.5,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        skip_dim = output_dim if skip_dim is None else int(skip_dim)
        if skip_dim <= 0 or skip_dim >= latent_dim:
            raise ValueError(
                f"skip_dim must be in (0, latent_dim); got skip_dim={skip_dim}, "
                f"latent_dim={latent_dim}"
            )
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty: the tangent head reads h")
        if tangent_hidden_dim <= 0:
            raise ValueError(f"tangent_hidden_dim must be positive, got {tangent_hidden_dim}")
        if not 0.0 < radius_min:
            raise ValueError(f"radius_min must be positive, got {radius_min}")
        if not radius_min < radius_init:
            raise ValueError(
                f"radius_min must be below radius_init; got radius_min={radius_min}, "
                f"radius_init={radius_init}"
            )
        if not radius_init < radius_max:
            raise ValueError(
                f"radius_init must be below radius_max; got radius_init={radius_init}, "
                f"radius_max={radius_max}"
            )
        if not radius_max < math.pi / 2:
            raise ValueError(f"radius_max must be below pi/2, got {radius_max}")
        self.skip_dim = skip_dim
        self.trunk_latent_dim = latent_dim - skip_dim
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.eps = float(eps)

        dims = [self.trunk_latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.direction = nn.Linear(hidden_dims[-1], output_dim, bias=False)
        self.tangent_in = nn.Linear(skip_dim, tangent_hidden_dim)
        # Default init, not zero: the residual's dependence on h exists from
        # step one, which is what the location-dependence test measures.
        self.gamma = nn.Linear(hidden_dims[-1], tangent_hidden_dim)
        self.beta = nn.Linear(hidden_dims[-1], tangent_hidden_dim)
        self.tangent_act = nn.LeakyReLU(negative_slope=negative_slope)
        self.tangent_out = nn.Linear(tangent_hidden_dim, output_dim)
        p = (radius_init - radius_min) / (radius_max - radius_min)
        self.radius_raw = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

    @property
    def radius(self) -> Tensor:
        return self.radius_min + (self.radius_max - self.radius_min) * torch.sigmoid(
            self.radius_raw
        )

    def tangent_raw(self, h: Tensor, z_skip: Tensor) -> Tensor:
        """The tangent head's output before projection onto the tangent
        space at u. Exposed so a test can hold the modulation constant and
        confirm the head then stops depending on the trunk."""
        a = self.tangent_in(z_skip) * (1.0 + self.gamma(h)) + self.beta(h)
        return self.tangent_out(self.tangent_act(a))

    def components(self, z: Tensor) -> tuple[Tensor, Tensor]:
        t_dim = self.trunk_latent_dim
        h = self.trunk(z[:, :t_dim])
        u = _unit(self.direction(h), self.eps)
        v = self.tangent_raw(h, z[:, t_dim:])
        v = v - (v * u).sum(dim=1, keepdim=True) * u
        return u, _unit(v, self.eps)

    def forward(self, z: Tensor) -> Tensor:
        u, t = self.components(z)
        r = self.radius
        return torch.cos(r) * u + torch.sin(r) * t
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_spherical.py -q`
Expected: all PASS.

- [ ] **Step 6: Mutation-check**

Each of these in turn: apply, run the named test, confirm FAIL, revert.

1. In `components`, replace `return u, _unit(v, self.eps)` with `return u, v`. `test_output_is_unit_norm_by_construction` and `test_tangent_is_orthogonal_to_direction` fail.
2. In `components`, delete the line `v = v - (v * u).sum(dim=1, keepdim=True) * u`. `test_tangent_is_orthogonal_to_direction` fails.
3. In `forward`, replace `r = self.radius` with `r = self.radius * (1.0 + 0.05 * torch.rand(z.shape[0], 1))`. `test_angle_from_direction_is_the_radius_on_every_row` fails.
4. In `radius`, return `self.radius_min + self.radius_raw` instead. `test_radius_stays_in_the_band` fails.

- [ ] **Step 7: Lint and commit**

Run: `make check` from the worktree root. Expected: passes.

```bash
git add src/models/generator.py tests/test_generator_spherical.py docs/ai/specs/2026-09-15-spherical-generator-design.md
git commit -m "feat(models): SphericalGenerator -- constant residual angle, conditioned tangent"
```

---

### Task 2: Local rank and location dependence

**Files:**
- Test: `tests/test_generator_spherical.py` (append)

**Interfaces:**
- Consumes: `SphericalGenerator.components`, `.tangent_raw`, `.trunk`, `.gamma`, `.beta`, `.tangent_in` from Task 1.
- Produces: nothing new in `src/`; if a test fails for a code reason, fix the class in Task 1's file and say so in the commit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_spherical.py`:

```python
def _jacobian_wrt_skip(gen, z_trunk, z_skip):
    def f(zs):
        return gen(torch.cat([z_trunk, zs[None, :]], dim=1))[0]

    return torch.autograd.functional.jacobian(f, z_skip)


@pytest.mark.parametrize("skip_dim, expected_rank", [(32, OUT - 2), (8, 8)])
def test_local_rank_wrt_skip_block_is_full(skip_dim, expected_rank):
    """The Jacobian of one output row wrt the skip block has rank
    min(skip_dim, output_dim - 2): t is a unit vector in the (output_dim - 1)
    dimensional tangent space, so its Jacobian loses one more dimension.
    Catches zeroing the tangent head's skip weight."""
    torch.manual_seed(0)
    gen = make(latent_dim=8 + skip_dim, skip_dim=skip_dim)
    z_trunk = torch.randn(1, 8)
    z_skip = torch.randn(skip_dim)
    jac = _jacobian_wrt_skip(gen, z_trunk, z_skip)
    assert jac.shape == (OUT, skip_dim)
    assert int(torch.linalg.matrix_rank(jac, rtol=1e-4)) >= expected_rank


def test_tangent_head_depends_on_the_trunk_through_the_modulation():
    """Catches a modulation that is wired but inert. With gamma and beta
    live, the pre-projection tangent output for a fixed skip draw changes
    with the trunk latent; with both frozen to constants it does not."""
    torch.manual_seed(0)
    gen = make()
    z_skip = torch.randn(4, SKIP)
    h1 = gen.trunk(torch.randn(4, 8))
    h2 = gen.trunk(torch.randn(4, 8))
    assert not torch.allclose(gen.gamma(h1), gen.gamma(h2))
    assert not torch.allclose(gen.tangent_raw(h1, z_skip), gen.tangent_raw(h2, z_skip))
    with torch.no_grad():
        for layer in (gen.gamma, gen.beta):
            layer.weight.zero_()
            layer.bias.zero_()
    assert torch.allclose(
        gen.tangent_raw(h1, z_skip), gen.tangent_raw(h2, z_skip), atol=1e-6
    )
```

- [ ] **Step 2: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_spherical.py -q -k "local_rank or modulation"`
Expected: PASS. (These tests are written against Task 1's class. If the rank test reports a rank below `OUT - 2` at `skip_dim=32`, print `torch.linalg.svdvals(jac)` and check whether the two smallest singular values are the only ones under `1e-4` of the largest. If a third is, the class is wrong, not the test.)

- [ ] **Step 3: Mutation-check**

1. In `SphericalGenerator.__init__`, after `self.tangent_in = ...`, add `with torch.no_grad(): self.tangent_in.weight.zero_()`. `test_local_rank_wrt_skip_block_is_full` fails (rank 0). Revert.
2. In `tangent_raw`, replace `a = self.tangent_in(z_skip) * (1.0 + self.gamma(h)) + self.beta(h)` with `a = self.tangent_in(z_skip); self.gamma(h); self.beta(h)`. `test_tangent_head_depends_on_the_trunk_through_the_modulation` fails at the second assertion. Revert.

- [ ] **Step 4: Commit**

Run: `make check`. Expected: passes.

```bash
git add tests/test_generator_spherical.py
git commit -m "test(models): spherical generator local rank and trunk conditioning"
```

---

### Task 3: Diagnostics method

**Files:**
- Modify: `src/models/generator.py` (add a method to `SphericalGenerator`, after `forward`)
- Test: `tests/test_generator_spherical.py` (append)

**Interfaces:**
- Produces: `SphericalGenerator.diagnostics(z: Tensor) -> dict[str, float]` with keys `radius` and `direction_effective_rank`. Task 4's trainer calls it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_spherical.py`:

```python
def test_diagnostics_report_radius_and_direction_rank():
    """Catches the rank computed on the wrong tensor: with the direction
    head pinned to one vector, u is constant and its effective rank is 1
    whatever t does."""
    torch.manual_seed(0)
    gen = make(radius_init=0.8)
    z = torch.randn(2048, LATENT)
    d = gen.diagnostics(z)
    assert set(d) == {"radius", "direction_effective_rank"}
    assert abs(d["radius"] - 0.8) < 1e-6
    assert 1.0 < d["direction_effective_rank"] <= OUT
    with torch.no_grad():
        gen.direction.weight.zero_()
        gen.direction.weight[0, 0] = 1.0
    assert gen.diagnostics(z)["direction_effective_rank"] < 1.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_spherical.py -q -k diagnostics`
Expected: FAIL with `AttributeError: 'SphericalGenerator' object has no attribute 'diagnostics'`.

- [ ] **Step 3: Implement**

Add to `SphericalGenerator` after `forward`:

```python
    @torch.no_grad()
    def diagnostics(self, z: Tensor) -> dict[str, float]:
        """Per-evaluation readout: the shared angle, and the effective rank
        of the trunk's direction over `z`, which shows whether u is
        collapsing into a sheet the way linear_skip's trunk did."""
        u, _ = self.components(z)
        return {
            "radius": float(self.radius),
            "direction_effective_rank": _effective_rank(u),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_spherical.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

In `diagnostics`, replace `u, _ = self.components(z)` with `_, u = self.components(z)`. The pinned-direction assertion fails (t still varies). Revert.

- [ ] **Step 6: Commit**

Run: `make check`. Expected: passes.

```bash
git add src/models/generator.py tests/test_generator_spherical.py
git commit -m "feat(models): spherical generator diagnostics -- radius and direction rank"
```

---

### Task 4: Factory branch

**Files:**
- Modify: `src/models/generator.py` (`build_generator`, the `if kind == "linear_skip":` block near the end of the file)
- Test: `tests/test_generator_factory.py` (append)

**Interfaces:**
- Consumes: `SphericalGenerator` from Task 1.
- Produces: `build_generator({"generator_type": "spherical", ...}, output_dim)` returns a `SphericalGenerator`; keys read: `skip_dim`, `tangent_hidden_dim`, `radius_init`, `radius_min`, `radius_max`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_factory.py` (add `SphericalGenerator` to the existing `from src.models.generator import (...)` list):

```python
def test_spherical_defaults():
    cfg = dict(BASE_CFG, generator_type="spherical", latent_dim=16 + 128)
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, SphericalGenerator)
    assert generator.skip_dim == 128
    assert generator.trunk_latent_dim == 16
    assert generator.tangent_in.out_features == 512
    assert abs(float(generator.radius) - 0.95) < 1e-6
    assert generator.radius_min == 0.2 and generator.radius_max == 1.5


def test_spherical_honours_overrides():
    cfg = dict(
        BASE_CFG,
        generator_type="spherical",
        latent_dim=16 + 64,
        skip_dim=64,
        tangent_hidden_dim=96,
        radius_init=0.6,
        radius_min=0.1,
        radius_max=1.0,
    )
    generator = build_generator(cfg, output_dim=128)
    assert generator.skip_dim == 64
    assert generator.tangent_in.out_features == 96
    assert abs(float(generator.radius) - 0.6) < 1e-6
    assert generator.radius_min == 0.1 and generator.radius_max == 1.0


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(latent_dim=16, skip_dim=16), "skip_dim"),
        (dict(latent_dim=32, radius_min=0.95), "radius_min"),
        (dict(latent_dim=32, radius_init=1.5), "radius_init"),
        (dict(latent_dim=32, radius_max=1.6), "radius_max"),
    ],
)
def test_spherical_rejects_bad_keys(overrides, match):
    cfg = dict(BASE_CFG, generator_type="spherical", **overrides)
    with pytest.raises(ValueError, match=match):
        build_generator(cfg, output_dim=128)


def test_spherical_state_dict_round_trips_through_a_rebuild():
    cfg = dict(BASE_CFG, generator_type="spherical", latent_dim=16 + 32, skip_dim=32)
    torch.manual_seed(0)
    a = build_generator(cfg, output_dim=32)
    with torch.no_grad():
        a.radius_raw.fill_(1.0)
    b = build_generator(cfg, output_dim=32)
    b.load_state_dict(a.state_dict())
    z = torch.randn(8, 16 + 32)
    assert torch.allclose(a(z), b(z))
    assert float(b.radius) == float(a.radius)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -q -k spherical`
Expected: FAIL with `ValueError: Unknown generator_type: spherical` (and the import error first if `SphericalGenerator` is not yet exported; it is a top-level class, so the import works).

- [ ] **Step 3: Implement**

In `build_generator`, after the `if kind == "linear_skip": return LinearSkipGenerator(...)` block and before `raise ValueError(f"Unknown generator_type: {kind}")`, add:

```python
    if kind == "spherical":
        return SphericalGenerator(
            **common,
            skip_dim=model_cfg.get("skip_dim"),
            tangent_hidden_dim=int(model_cfg.get("tangent_hidden_dim", 512)),
            radius_init=float(model_cfg.get("radius_init", 0.95)),
            radius_min=float(model_cfg.get("radius_min", 0.2)),
            radius_max=float(model_cfg.get("radius_max", 1.5)),
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

In the new branch, change `model_cfg.get("tangent_hidden_dim", 512)` to `model_cfg.get("tangent_hidden", 512)`. `test_spherical_honours_overrides` fails. Revert.

- [ ] **Step 6: Commit**

Run: `make check`. Expected: passes.

```bash
git add src/models/generator.py tests/test_generator_factory.py
git commit -m "feat(models): build_generator knows generator_type: spherical"
```

---

### Task 5: Trainer integration and smoke test

**Files:**
- Modify: `src/train/train_wgan_gp.py:33` (import), `:639-644` (probe latents), `:800-801` (eval readout)
- Test: `tests/test_train_smoke.py` (`make_config`, the parametrize list, one new test)

**Interfaces:**
- Consumes: `SphericalGenerator.diagnostics` from Task 3.
- Produces: `run_metadata.json` `eval` entries carry `radius` and `direction_effective_rank` for a spherical run.

- [ ] **Step 1: Write the failing tests**

In `tests/test_train_smoke.py`, in `make_config`, after the `if generator_type == "linear_skip":` block add:

```python
        if generator_type == "spherical":
            cfg["model"]["skip_dim"] = 4  # latent_dim is 8: 4 trunk + 4 skip
            cfg["model"]["tangent_hidden_dim"] = 8
```

Change the parametrize list of `test_training_loop_runs` to:

```python
@pytest.mark.parametrize(
    "generator_type", ["mlp", "gated", "structured_gated", "linear_skip", "spherical"]
)
```

After `test_linear_skip_evals_log_the_trunk_skip_balance`, add:

```python
def test_spherical_evals_log_radius_and_direction_rank(tmp_path):
    """Catches the diagnostics not reaching the eval entry, or reaching it
    for the wrong generator type."""
    _, meta = train(make_config(tmp_path, "spherical"))
    assert meta["eval"]
    for e in meta["eval"]:
        assert 0.2 <= e["radius"] <= 1.5
        assert 1.0 <= e["direction_effective_rank"] <= 16.0
    _, meta_mlp = train(make_config(tmp_path, "mlp"))
    assert "radius" not in meta_mlp["eval"][0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_smoke.py -q -k "spherical"`
Expected: `test_training_loop_runs[spherical]` PASSES already (the factory works); `test_spherical_evals_log_radius_and_direction_rank` FAILS with `KeyError: 'radius'`.

- [ ] **Step 3: Implement**

Line 33 of `src/train/train_wgan_gp.py`:

```python
from src.models.generator import LinearSkipGenerator, SphericalGenerator, build_generator
```

Around lines 639-644, replace

```python
    energy_probe = None
    if isinstance(generator, LinearSkipGenerator):
```
with
```python
    energy_probe = None
    if isinstance(generator, (LinearSkipGenerator, SphericalGenerator)):
```

Around lines 800-801, replace

```python
                if energy_probe is not None:
                    stats.update(generator.component_energies(energy_probe))
```
with
```python
                if energy_probe is not None:
                    if isinstance(generator, SphericalGenerator):
                        stats.update(generator.diagnostics(energy_probe))
                    else:
                        stats.update(generator.component_energies(energy_probe))
```

Update the comment above the probe (`# Fixed latents for the linear-skip balance readout, ...`) to `# Fixed latents for the linear-skip balance readout and the spherical radius/direction-rank readout, ...`.

- [ ] **Step 4: Run to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_smoke.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

Change the isinstance tuple back to `LinearSkipGenerator` only. `test_spherical_evals_log_radius_and_direction_rank` fails with `KeyError: 'radius'`. Revert.

- [ ] **Step 6: Commit**

Run: `make check`. Expected: passes.

```bash
git add src/train/train_wgan_gp.py tests/test_train_smoke.py
git commit -m "feat(train): log the spherical generator's radius and direction rank per eval"
```

---

### Task 6: NYTimes `v3` configs, job script, pinning tests

**Files:**
- Create: `configs/nytimes/v3.yaml`, `configs/nytimes/v3_seed42.yaml`, `scripts/nytimes_v3_seed42_job.sh`
- Test: `tests/test_nytimes_configs.py` (append)

**Interfaces:**
- Consumes: the factory keys from Task 4.
- Produces: the three files above; Task 8 submits the script.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nytimes_configs.py`:

```python
V3_GENERATOR_KEYS = {
    "model.generator_type": "spherical",
    "model.tangent_hidden_dim": 512,
    "model.radius_init": 0.95,
    "model.radius_min": 0.2,
    "model.radius_max": 1.5,
}


def test_v3_is_v2c_with_the_generator_swapped():
    v2c = _flatten(_load("v2c.yaml"))
    v3 = _flatten(_load("v3.yaml"))
    for key, value in V3_GENERATOR_KEYS.items():
        assert v3.pop(key) == value, key
    assert v3.pop("output_dir") == "runs/nytimes/v3"
    assert v3["model.skip_dim"] == 256
    for key in ("model.generator_type", "model.skip_init", "model.skip_init_gain", "output_dir"):
        v2c.pop(key)
    assert v3 == v2c


def test_v3_seed42_is_v3_with_an_absolute_real_path_and_its_own_output_dir():
    v3 = _flatten(_load("v3.yaml"))
    inst = _flatten(_load("v3_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v3_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v3.pop("output_dir")
    v3.pop("data.real_path")
    assert inst == v3


def test_v3_job_script_runs_the_v3_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v3_seed42_job.sh").read_text()
    assert "configs/nytimes/v3_seed42.yaml" in script
    assert "runs/nytimes/v3_seed42" in script
    assert "/workspace/nytimes-v3/v3_seed42" in script


def test_v3_requires_amp_off():
    assert _flatten(_load("v3.yaml"))["training.amp"] is False
    assert _flatten(_load("v3_seed42.yaml"))["training.amp"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_nytimes_configs.py -q -k v3`
Expected: FAIL with `FileNotFoundError` on `v3.yaml`.

- [ ] **Step 3: Write `configs/nytimes/v3.yaml`**

```yaml
# NYTIMES v3 -- v2c with the generator changed from linear_skip to spherical.
# One change from v2c, stated:
#
#   1. model.generator_type: linear_skip -> spherical, with
#      tangent_hidden_dim: 512, radius_init: 0.95, radius_min: 0.2,
#      radius_max: 1.5 new, skip_dim unchanged at 256, and the linear_skip
#      keys skip_init / skip_init_gain removed (the new class has no W).
#
# Why: on the v2c checkpoints, hubs are rows whose residual share is a few
# percent smaller than their neighbours' (trunk-norm CV 0.06-0.08 gave
# hubness 17-30; equalising it gave 3-6), and a fixed linear residual reads
# LID 62-70 at every mix short of a sheet. The spherical generator holds the
# residual's angle constant across samples and conditions its shape on the
# trunk's hidden state. See
# docs/ai/specs/2026-09-15-spherical-generator-design.md and
# docs/results/nytimes-trunk-scale-probe/.
seed: 42
device: auto
output_dir: runs/nytimes/v3

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
  generator_type: spherical
  skip_dim: 256
  tangent_hidden_dim: 512
  radius_init: 0.95
  radius_min: 0.2
  radius_max: 1.5
  critic_type: neighbourhood_set
  critic_edge_dim: 128
  critic_edge_pool: max
  critic_k: 20
  critic_distance_floor: 0.01

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

- [ ] **Step 4: Write `configs/nytimes/v3_seed42.yaml`**

Identical to `v3.yaml` from `seed: 42` down, except the header comment, `output_dir` and `data.real_path`:

```yaml
# NYTimes v3, run 42. Identical to configs/nytimes/v3.yaml in every
# training hyperparameter. Two things differ, listed so a reader can
# confirm nothing else moved:
#
#   1. output_dir  runs/nytimes/v3 -> runs/nytimes/v3_seed42
#   2. real_path   data/nytimes_250k.npy -> an absolute path (see below)
#
# Why this file exists rather than an edit to v3.yaml: a rung is a
# historical record of a run that happened; this config is the measurement
# instrument, the same pattern as configs/nytimes/v2c_seed42.yaml.
#
# Why an absolute real_path: the gpuq runner executes each job in a fresh
# detached git worktree cut from the pinned commit, and data/ is gitignored.
# The file it names is `python -m src.data.fetch nytimes` output at seed 42,
# sha256 ba8e8d512cc465e983661cbbabba64ead8251c3f313de86c3b1c3abe56c03428,
# the same bytes as a local data/nytimes_250k.npy. The trainer drops its
# 197 exact-zero rows at load (drop_zero_rows) and keeps its duplicates.
seed: 42
device: auto
output_dir: runs/nytimes/v3_seed42

data:
  real_path: /workspace/data-cache/nytimes_250k.npy
```

followed by the rest of `v3.yaml`'s `data`, `model` and `training` blocks unchanged. Write it by copying `v3.yaml` and editing those three places, then confirm with:

Run: `diff configs/nytimes/v3.yaml configs/nytimes/v3_seed42.yaml`
Expected: only the header comment lines, the `output_dir` line and the `real_path` line differ.

- [ ] **Step 5: Write `scripts/nytimes_v3_seed42_job.sh`**

```bash
#!/usr/bin/env bash
# One gpuq job: train NYTimes v3 (seed 42), sample it, and measure it against
# the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v3 \
#     --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v3_seed42_job.sh
#
# Timeout from the v2c job: 67 minutes wall for 30,000 steps plus sampling and
# the report; the tangent head adds three small linear layers. 10,800 s is
# the same margin the v2c job had.
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v3 at the end instead, the way
# the v2c job did. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v3_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v3/v3_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v3_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v3_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v3_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
```

Then `chmod +x scripts/nytimes_v3_seed42_job.sh`.

- [ ] **Step 6: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_nytimes_configs.py -q`
Expected: all PASS.

- [ ] **Step 7: Mutation-check**

In `v3.yaml`, change `lambda_gp: 5.0` to `lambda_gp: 10.0`. `test_v3_is_v2c_with_the_generator_swapped` fails. Revert.

- [ ] **Step 8: Local end-to-end smoke of the instrument shape**

Confirm the config builds the generator the factory expects, without training:

Run:
```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import yaml; from src.models.generator import build_generator, SphericalGenerator
cfg = yaml.safe_load(open('configs/nytimes/v3.yaml'))
g = build_generator(cfg['model'], output_dim=cfg['data']['descriptor_dim'])
assert isinstance(g, SphericalGenerator) and g.skip_dim == 256 and g.trunk_latent_dim == 256
print(sum(p.numel() for p in g.parameters()), 'parameters, radius', float(g.radius))"
```
Expected: prints a parameter count and `radius 0.95`.

- [ ] **Step 9: Commit**

Run: `make check`. Expected: passes.

```bash
git add configs/nytimes/v3.yaml configs/nytimes/v3_seed42.yaml scripts/nytimes_v3_seed42_job.sh tests/test_nytimes_configs.py
git commit -m "configs+scripts(nytimes): v3 rung -- spherical generator, seed-42 instrument and job"
```

---

### Task 7: Documentation

**Files:**
- Modify: `PROJECT_DOCUMENTATION.md` (the `### generator_type` section, around lines 323-340)
- Modify: `docs/datasets/nytimes.md` (the `## Model family` line around line 195; the ladder table after the `v2c` at 100k row, around line 208)

**Interfaces:** none.

- [ ] **Step 1: Update `PROJECT_DOCUMENTATION.md`**

Replace

```
The architecture axis in the `model` config block. Four values are built:
`mlp` (default), `gated`, `structured_gated`, and `linear_skip`. It sits
underneath the variant numbering: on SIFT, v0, v1 and v1_5 all use `mlp`
and differ only in training settings.

A fifth value, `spherical`, is planned and not built. It is phase (b) of the
multi-dataset design: a generator whose output is unit-norm by construction
rather than by a normalization applied afterwards, for the four `angular`
families. Until it exists, `deep`, `glove` and `openai` all start their
ladders on `mlp` (`nytimes` moved to `linear_skip` at its v1), and any
dataset page naming `spherical` is describing the intended rung, not a
trained one.
```
with
```
The architecture axis in the `model` config block. Five values are built:
`mlp` (default), `gated`, `structured_gated`, `linear_skip` and
`spherical`. It sits underneath the variant numbering: on SIFT, v0, v1 and
v1_5 all use `mlp` and differ only in training settings.

`spherical` is phase (b) of the multi-dataset design: a generator whose
output is unit-norm by construction rather than by a normalization applied
afterwards. The trunk MLP emits a unit direction `u`; a tangent head reads
the skip block of the latent, is modulated per channel by the trunk's
hidden state, and its output is projected orthogonal to `u` and normalised
to `t`; the output is `cos(r) u + sin(r) t` with `r` one learned scalar
inside the band `[radius_min, radius_max]`, shared by every sample. Keys:
`skip_dim`, `tangent_hidden_dim`, `radius_init`, `radius_min`,
`radius_max`. It exists because on NYTimes the `linear_skip` generator's
hubs are the rows whose residual share is a few percent smaller than their
neighbours', and a fixed linear residual cannot reach the corpus's local
dimension at its global rank
(`docs/ai/specs/2026-09-15-spherical-generator-design.md`). The trainer
logs `radius` and `direction_effective_rank` per evaluation for it. Built
for NYTimes `v3`; `deep`, `glove` and `openai` still start their ladders on
`mlp`, and openai's narrow cone is not handled by it.
```

- [ ] **Step 2: Update `docs/datasets/nytimes.md`**

Replace the model-family line

```
`mlp` for `v0`, `linear_skip` for `v1`; `spherical` when phase (b) lands.
```
with
```
`mlp` for `v0`, `linear_skip` from `v1`, `spherical` from `v3`.
```

After the `v2c` at 100k steps ladder row, add:

```
| `v3` | `v2c` with the generator changed to `spherical` (`generator_type: spherical`, `tangent_hidden_dim` 512, radius band 0.2 to 1.5 from 0.95) | `configs/nytimes/v3.yaml`; box instrument `configs/nytimes/v3_seed42.yaml` | `runs/nytimes/v3_seed42` (box: `/workspace/nytimes-v3/v3_seed42`) | config written 2026-09-15 (`scripts/nytimes_v3_seed42_job.sh`); not yet trained. Why: `docs/ai/specs/2026-09-15-spherical-generator-design.md`, from the trunk-scale probes under `docs/results/nytimes-trunk-scale-probe/` |
```

- [ ] **Step 3: Run the docs gate**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py tests/test_contract.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

Run: `make check`. Expected: passes.

```bash
git add PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "docs: generator_type spherical is built; NYTimes v3 ladder row"
```

---

### Task 8: Submit the training job (orchestrator, not a subagent)

**Files:** none changed in the repo by this task.

- [ ] **Step 1: Push the branch**

Run from the worktree:
```bash
git push -u origin nytimes-v3
git rev-parse HEAD
```
Expected: the branch is on `origin` (the public fork the box clones from); note the full 40-character SHA.

- [ ] **Step 2: Submit**

Run (ssh needs the sandbox disabled; pass the full SHA):
```bash
ssh tig-gpu '/opt/gpuq/venv/bin/gpuq submit --project wgan-synthetic \
  --commit <full-sha> --branch nytimes-v3 --lane gpu --timeout-s 10800 \
  --dedupe-key nytimes-v3-seed42-<short-sha> -- bash scripts/nytimes_v3_seed42_job.sh'
```
Expected: prints a job id `wgan-synthetic-<timestamp>-<hash>`.

- [ ] **Step 3: Record**

Write the job id, the SHA and the submit time into the ladder row's status cell in `docs/datasets/nytimes.md` (replacing `not yet trained` with `submitted <date> as <job id>`), commit that one file, push. Then `gpuq wait <id>` in the background with a 4-hour timeout, output redirected to a file.

The results write-up (gate table, mechanism checks, transient test, wall time) is a separate piece of work after the job finishes and is not part of this plan.

---

## Self-review

**Spec coverage.** Trunk, tangent head with modulation, radius band, output formula: Task 1. Config keys and validation: Tasks 1 and 4. Configs, instrument, job script, pinning: Task 6. Data flow and diagnostics: Tasks 3 and 5. Numerical guards: `_unit` clamp in Task 1. Every row of the spec's test table: unit norm, orthogonality, constant angle, band (Task 1); local rank, location dependence (Task 2); factory (Task 4); config pinning (Task 6); trainer smoke (Task 5). Mechanism checks and success bar are properties of the trained run and belong to the write-up after Task 8. Documentation of the new type: Task 7.

**Placeholders.** None. Every code step carries its code.

**Type consistency.** `components` returns `(u, t)` in Tasks 1, 2 and 3. `tangent_raw(h, z_skip)` is defined in Task 1 and used in Task 2. `diagnostics` keys `radius` and `direction_effective_rank` match between Task 3, Task 5 and the docs in Task 7. Config keys match between Task 4's factory, Task 6's YAML and pinning test, and Task 7's docs.
