# Structured-Gate Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generator whose exact-zero pattern reproduces SIFT's measured support statistics — 3× over-dispersed non-zero counts and local correlation on the (4,4,8) descriptor grid — which v2's independent per-coordinate gates cannot express at any parameter setting.

**Architecture:** A new `StructuredGateGenerator` keeps v2's trunk, floored-softplus magnitude, straight-through binary-concrete gate and unit-norm contract, and adds three things: a per-vector scalar added to every gate logit (over-dispersion), a 3×3×3 convolution over the reshaped (4,4,8) logit grid with circular orientation padding (spatial/orientation coupling), and a fixed variance-normalized smoothing of the gate noise (so sampling is correlated, not merely the logits).

**Tech Stack:** Python 3.12, PyTorch 2.x, pytest.

## Global Constraints

- Phase 2 of 3 from `docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md`. Phase 1 (run infrastructure) is complete on this branch. Phase 3 (log-ratio regularizer) is a separate plan and is **not** in scope here.
- Baseline is **143 tests passing**. Run with the main-repo interpreter: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`. Worktrees have no `.venv`.
- **`GatedGenerator` (v2) must not be modified.** It is the comparison baseline for the v2-vs-v3 arm; a change to it would silently move the thing v3 is measured against.
- Every existing test must keep passing, `tests/test_generator.py` untouched.
- The dtype contract from `tests/test_generator.py` binds the new class too: a forward pass in `float16`/`bfloat16` returns that dtype, stays finite, and never yields an all-zero row.
- Layout is `(4, 4, 8)` — 4×4 spatial grid, 8 orientation bins, orientation varying fastest. Confirmed empirically: residual zero-correlation peaks at exactly `|i−j| = 8`.
- Everything here is CPU-testable. No task needs a GPU.

## File Structure

| File | Responsibility |
|---|---|
| `src/models/generator.py` (modify) | Add `StructuredGateGenerator`; register `structured_gated` in `build_generator`. |
| `tests/test_generator_structured.py` (create) | Contract, mechanism and dtype tests for the new class. |
| `tests/test_generator_factory.py` (modify) | Factory dispatch and config plumbing for the new type. |
| `configs/sift_gan_v3.yaml` (create) | The v3 variant: exactly one change from `sift_gan_v2.yaml`. |
| `PROJECT_DOCUMENTATION.md` (modify) | Variant table and `generator_type` section. |

**On duplication:** `StructuredGateGenerator` is a standalone `nn.Module`, not a subclass of `GatedGenerator`, and will repeat its validation, `_sample_gate` skeleton and unit-norm tail. This is deliberate. v2 is a frozen experimental baseline; subclassing would mean any future edit to v2 silently changes v3 and destroys the one-change-per-variant comparison the whole project rests on. The repo already works this way — `GatedGenerator` does not subclass `Generator`.

**Target numbers** (from `tools/probes/layout_probe.py` against real SIFT, for orientation — not asserted as exact values in tests):

- `exact_zero_fraction` 0.2301, `nnz` per vector mean 98.54, **std 14.45**
- Independent-gate equivalent std: `sqrt(128·p·(1−p))` = **4.76**
- Residual zero-correlation +0.317 at `|i−j| = 1`, +0.275 at `|i−j| = 8`

---

### Task 1: Core class with a per-vector sparsity level

**Files:**
- Modify: `src/models/generator.py` (add `StructuredGateGenerator` after `GatedGenerator`)
- Create: `tests/test_generator_structured.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StructuredGateGenerator(latent_dim, output_dim, hidden_dims, negative_slope=0.2, gate_temperature=0.5, logit_clamp=10.0, layout=(4,4,8), gate_kernel=3, noise_kernel_sigma=0.8, eps=1e-8)`, with attributes `trunk`, `magnitude_head`, `gate_head`, `sparsity_head`, `layout`, `gate_temperature`, `logit_clamp`, `eps`, and a `_sample_gate(logits) -> Tensor` method. Later tasks add `gate_coupling` (Task 2) and the `noise_kernel` buffer (Task 3).

This task delivers the mechanism with the largest measured effect: a scalar added to every gate logit makes `nnz` a mixture of binomials instead of a single binomial, which is the only way to reach the measured std of 14.45.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generator_structured.py`:

```python
import pytest
import torch
import torch.nn.functional as F

from src.models.generator import GatedGenerator, StructuredGateGenerator

LATENT = 16
OUTPUT = 128
HIDDEN = [32, 32]


def build(**overrides):
    kwargs = dict(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    kwargs.update(overrides)
    return StructuredGateGenerator(**kwargs)


@pytest.fixture
def gen():
    torch.manual_seed(0)
    return build()


@pytest.fixture
def out(gen):
    torch.manual_seed(1)
    return gen(torch.randn(64, LATENT))


def test_output_shape(out):
    assert out.shape == (64, OUTPUT)


def test_non_negative(out):
    assert (out >= 0).all()


def test_unit_norm(out):
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_zeros_are_exact(out):
    assert (out == 0.0).any()


def test_no_all_zero_row(out):
    assert ((out > 0).sum(dim=1) > 0).all()


def test_all_gates_closed_still_yields_a_unit_vector():
    # The fallback path: if every stochastic gate in a row closes, the row would
    # normalize to the zero vector. One coordinate is forced open instead.
    generator = build(output_dim=8, layout=(1, 1, 8))
    with torch.no_grad():
        generator.gate_head.weight.zero_()
        generator.gate_head.bias.fill_(-100.0)
        generator.sparsity_head.weight.zero_()
        generator.sparsity_head.bias.fill_(-100.0)
    torch.manual_seed(12)
    out = generator(torch.randn(32, LATENT))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Deliberately `>= 1`, not `== 1`: at a clamped logit of -10 a coordinate
    # still opens with probability about 4.5e-5, so an exact count would be
    # flaky across seeds. The contract under test is that no row is empty.
    assert ((out > 0).sum(dim=1) >= 1).all()


def test_saturated_magnitude_still_yields_unit_norm():
    # softplus underflows to exactly 0.0 below about -90 in float32, and an
    # open gate over a zero magnitude still normalizes to the zero vector.
    # Only the magnitude floor rescues this.
    generator = build()
    with torch.no_grad():
        generator.magnitude_head.weight.zero_()
        generator.magnitude_head.bias.fill_(-1000.0)
    torch.manual_seed(9)
    out = generator(torch.randn(32, LATENT))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert ((out > 0).sum(dim=1) > 0).all()


def test_gate_noise_is_kept_at_sample_time(gen):
    gen.eval()
    z = torch.randn(64, LATENT)
    with torch.no_grad():
        a, b = gen(z), gen(z)
    assert not torch.equal(a, b)


def test_seeded_forward_is_deterministic(gen):
    z = torch.randn(16, LATENT)
    torch.manual_seed(8)
    a = gen(z)
    torch.manual_seed(8)
    b = gen(z)
    assert torch.equal(a, b)


@pytest.mark.parametrize("head", ["gate_head", "magnitude_head", "sparsity_head"])
def test_every_head_receives_gradient(gen, head):
    torch.manual_seed(2)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = getattr(gen, head).weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hidden_dims": []}, "hidden dimension"),
        ({"gate_temperature": 0}, "gate_temperature"),
        ({"logit_clamp": 0}, "logit_clamp"),
        ({"eps": 0}, "eps"),
        ({"output_dim": 0}, "dimensions"),
        ({"negative_slope": -0.1}, "negative_slope"),
        ({"layout": (4, 4, 4)}, "layout"),
        ({"layout": (4, 32)}, "layout"),
        ({"gate_kernel": 2}, "gate_kernel"),
        ({"noise_kernel_sigma": 0.0}, "noise_kernel_sigma"),
    ],
)
def test_invalid_configuration_fails_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(**kwargs)


def _nnz_stats(generator, n=4096, seed=11):
    torch.manual_seed(seed)
    with torch.no_grad():
        x = generator(torch.randn(n, LATENT))
    nnz = (x > 0).sum(dim=1).float()
    p = nnz.mean().item() / OUTPUT
    binomial = (OUTPUT * p * (1.0 - p)) ** 0.5
    return nnz.std().item(), binomial


def test_sparsity_level_makes_support_size_over_dispersed():
    # The whole reason this class exists. Independent per-coordinate gates give
    # nnz ~ Binomial(128, p), std = sqrt(128 p (1-p)), *by construction*. Real
    # SIFT is 3x that (14.45 vs 4.76). A per-vector logit shift turns nnz into
    # a mixture of binomials, whose variance can reach the measured value.
    generator = build()
    with torch.no_grad():
        # Amplify the head so the spread is unambiguous at random init.
        generator.sparsity_head.weight.mul_(8.0)
    observed, binomial = _nnz_stats(generator)
    assert observed > 1.5 * binomial


def test_v2_gate_stays_near_the_binomial_baseline():
    # The contrast that makes the previous test meaningful: v2 has no per-vector
    # level, so no amount of training moves it off the binomial.
    torch.manual_seed(0)
    v2 = GatedGenerator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    observed, binomial = _nnz_stats(v2)
    assert observed < 1.5 * binomial


@pytest.mark.parametrize("dtype, atol", [
    pytest.param(torch.bfloat16, 3e-2, id="bfloat16"),
    pytest.param(torch.float16, 5e-3, id="float16"),
])
def test_low_precision_forward_preserves_dtype(dtype, atol):
    torch.manual_seed(6)
    generator = build().to(dtype)
    out = generator(torch.randn(64, LATENT, dtype=dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    norms = out.float().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_gate_preserves_input_dtype(gen, dtype):
    torch.manual_seed(5)
    logits = torch.randn(64, OUTPUT, dtype=dtype)
    gate = gen._sample_gate(logits)
    assert gate.dtype == dtype
    assert torch.isfinite(gate).all()
    assert (gate.sum(dim=1) > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -v`
Expected: FAIL — `ImportError: cannot import name 'StructuredGateGenerator'`

- [ ] **Step 3: Write the implementation**

Add to `src/models/generator.py`, after `GatedGenerator` and before `build_generator`. Add `Sequence` to the `typing` import at the top of the file if absent.

```python
class StructuredGateGenerator(nn.Module):
    """Gated generator whose support statistics match SIFT's measured shape.

    `GatedGenerator` samples every coordinate's gate independently at a rate
    the trunk controls, so its non-zero count is Binomial(d, p) with standard
    deviation sqrt(d p (1-p)) -- about 4.76 at d=128, p=0.77. Real SIFT
    measures 14.45. That is an expressiveness ceiling, not a tuning problem:
    no parameter setting of independent gates reaches it.

    Three additions lift it, each targeting a measured property of the real
    descriptors (see tools/probes/):

    1. A per-vector scalar added to every gate logit, making the non-zero
       count a mixture of binomials whose variance the trunk can learn.
    2. A convolution over the logits reshaped to the (4,4,8) descriptor grid,
       producing the measured local correlation (Task 2).
    3. Smoothing of the gate *noise* with a fixed kernel, so sampling is
       correlated and not merely the logits (Task 3).

    Deliberately not a subclass of GatedGenerator: v2 is the frozen baseline
    this variant is measured against, and inheritance would let a change to
    it silently move the comparison.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        gate_temperature: float = 0.5,
        logit_clamp: float = 10.0,
        layout: Sequence[int] = (4, 4, 8),
        gate_kernel: int = 3,
        noise_kernel_sigma: float = 0.8,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        layout = tuple(int(v) for v in layout)
        if latent_dim <= 0 or output_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("model dimensions must be greater than zero")
        if not hidden_dims:
            raise ValueError("StructuredGateGenerator requires at least one hidden dimension")
        if negative_slope < 0:
            raise ValueError("negative_slope must not be negative")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be greater than zero")
        if logit_clamp <= 0:
            raise ValueError("logit_clamp must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")
        if len(layout) != 3 or any(v <= 0 for v in layout):
            raise ValueError(f"layout must be three positive dimensions, got {layout}")
        if layout[0] * layout[1] * layout[2] != output_dim:
            raise ValueError(
                f"layout {layout} has {layout[0] * layout[1] * layout[2]} cells "
                f"but output_dim is {output_dim}"
            )
        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError(f"gate_kernel must be a positive odd number, got {gate_kernel}")
        if noise_kernel_sigma <= 0:
            raise ValueError("noise_kernel_sigma must be greater than zero")

        dims: List[int] = [latent_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.magnitude_head = nn.Linear(dims[-1], output_dim)
        self.gate_head = nn.Linear(dims[-1], output_dim)
        # One scalar per vector, broadcast across all coordinates. This is the
        # entire over-dispersion mechanism.
        self.sparsity_head = nn.Linear(dims[-1], 1)
        self.layout = layout
        self.gate_kernel = int(gate_kernel)
        self.noise_kernel_sigma = float(noise_kernel_sigma)
        self.gate_temperature = float(gate_temperature)
        self.logit_clamp = float(logit_clamp)
        self.eps = float(eps)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a hard binary-concrete gate, returned in ``logits.dtype``.

        Draw in float32 under AMP: eps=1e-8 rounds to zero in float16, which
        would leave log(0) capable of poisoning gate gradients.
        """
        sample_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        u = torch.rand_like(sample_logits).clamp(self.eps, 1.0 - self.eps)
        logistic = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((sample_logits + logistic) / self.gate_temperature)
        hard = (soft > 0.5).to(soft.dtype)
        # Preserve the unit-norm contract even if every gate in a row closes.
        empty = hard.sum(dim=1, keepdim=True) == 0
        fallback = F.one_hot(
            sample_logits.argmax(dim=1), sample_logits.shape[1]
        ).to(hard.dtype)
        hard = torch.where(empty, fallback, hard)
        return (hard + soft - soft.detach()).to(logits.dtype)

    def _gate_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.gate_head(h) + self.sparsity_head(h)
        return self.logit_clamp * torch.tanh(logits / self.logit_clamp)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        # Floor the magnitude above `eps` so the unit-norm contract holds under
        # divergence: softplus saturates to exactly 0.0 below about -90 in
        # float32, and an open gate over a zero magnitude still normalizes to
        # the zero vector. Exact zeros come from the gate, not from here.
        magnitude = F.softplus(self.magnitude_head(h)).clamp(min=self.eps * 100.0)
        x = self._sample_gate(self._gate_logits(h)) * magnitude
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=self.eps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -v`
Expected: PASS (28 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 171 tests (143 + 28). `tests/test_generator.py` untouched and green.

- [ ] **Step 6: Commit**

```bash
git add src/models/generator.py tests/test_generator_structured.py
git commit -m "feat(models): structured gate generator with a per-vector sparsity level

Independent per-coordinate gates give nnz ~ Binomial(128, p), std about
4.76 by construction. Real SIFT measures 14.45 -- an expressiveness
ceiling no parameter setting of v2 can clear. A per-vector scalar added
to every gate logit makes nnz a mixture of binomials instead.

Standalone rather than a GatedGenerator subclass: v2 is the frozen
baseline this variant is measured against."
```

---

### Task 2: Neighbourhood-coupled gate logits

**Files:**
- Modify: `src/models/generator.py` (`StructuredGateGenerator.__init__` and `_gate_logits`)
- Modify: `tests/test_generator_structured.py` (append)

**Interfaces:**
- Consumes: `StructuredGateGenerator` from Task 1.
- Produces: attribute `gate_coupling` (an `nn.Conv3d(1, 1, gate_kernel, bias=False)`, identity-initialised) and a `_couple(logits) -> Tensor` method.

The measured correlation structure is local smoothness on the (4,4,8) grid: +0.317 between adjacent orientation bins and +0.275 at offset 8 (the same orientation bin in the neighbouring spatial cell). Orientation is **circular** — bin 7 neighbours bin 0, because a gradient direction between two bins deposits in both. Spatial cells are not periodic, so the grid edge replicates.

**Identity initialisation matters.** A randomly-initialised conv would scramble the logits at step 0 and make v3 start from a different place than v2 for reasons unrelated to the mechanism. Initialised to identity (centre weight 1, rest 0), the class begins behaving exactly as Task 1 left it and learns coupling from there.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_structured.py`:

```python
def test_coupling_is_identity_at_initialisation():
    # v3 must start where v2 starts; a random conv would scramble the logits at
    # step 0 for reasons unrelated to the mechanism under test.
    generator = build()
    torch.manual_seed(3)
    logits = torch.randn(8, OUTPUT)
    assert torch.allclose(generator._couple(logits), logits, atol=1e-6)


def test_orientation_axis_wraps_circularly():
    # A gradient direction falling between bins 7 and 0 deposits in both, so
    # the last orientation bin must neighbour the first.
    generator = build()
    rows, cols, orient = generator.layout
    with torch.no_grad():
        generator.gate_coupling.weight.zero_()
        # Pick up only the neighbour one step *back* along orientation.
        generator.gate_coupling.weight[0, 0, 1, 1, 0] = 1.0
    logits = torch.zeros(1, OUTPUT)
    logits[0, orient - 1] = 5.0  # last orientation bin of the first cell
    coupled = generator._couple(logits).reshape(rows, cols, orient)
    assert coupled[0, 0, 0].item() == pytest.approx(5.0)


def test_spatial_edge_replicates_rather_than_wrapping():
    # The 4x4 grid is not periodic: cell (0,0) and cell (3,3) are opposite
    # corners of the patch, not neighbours.
    generator = build()
    rows, cols, orient = generator.layout
    with torch.no_grad():
        generator.gate_coupling.weight.zero_()
        generator.gate_coupling.weight[0, 0, 0, 1, 1] = 1.0  # one step back in rows
    logits = torch.zeros(1, OUTPUT)
    logits[0, ((rows - 1) * cols + 0) * orient + 0] = 7.0  # last row, first cell
    coupled = generator._couple(logits).reshape(rows, cols, orient)
    # Row 0 pulls from replicated row 0, not from wrapped row 3.
    assert coupled[0, 0, 0].item() == pytest.approx(0.0)


def test_coupling_receives_gradient(gen):
    torch.manual_seed(4)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.gate_coupling.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_coupling_preserves_shape_and_dtype(gen):
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(4, OUTPUT, dtype=dtype)
        coupled = gen.to(dtype)._couple(logits)
        assert coupled.shape == (4, OUTPUT)
        assert coupled.dtype == dtype
    gen.to(torch.float32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -k couple -v`
Expected: FAIL — `AttributeError: 'StructuredGateGenerator' object has no attribute '_couple'`

- [ ] **Step 3: Add the coupling**

In `__init__`, after `self.sparsity_head = ...`, insert:

```python
        # Local coupling over the (row, col, orientation) grid. Identity-init
        # so the module starts as an uncoupled gate and learns structure.
        self.gate_coupling = nn.Conv3d(1, 1, kernel_size=gate_kernel, bias=False)
        with torch.no_grad():
            self.gate_coupling.weight.zero_()
            centre = gate_kernel // 2
            self.gate_coupling.weight[0, 0, centre, centre, centre] = 1.0
```

Add the `_couple` method just above `_gate_logits`:

```python
    def _couple(self, logits: torch.Tensor) -> torch.Tensor:
        """Mix each gate logit with its neighbours on the descriptor grid.

        Orientation is padded **circularly** -- bin 7 neighbours bin 0, since a
        gradient direction between two bins deposits in both. The 4x4 spatial
        grid is not periodic, so its edges replicate: opposite corners of the
        patch are not neighbours.
        """
        batch = logits.shape[0]
        rows, cols, orient = self.layout
        pad = self.gate_kernel // 2
        grid = logits.reshape(batch, 1, rows, cols, orient)
        # F.pad's tuple runs last-dim-first: (W, W, H, H, D, D).
        grid = F.pad(grid, (pad, pad, 0, 0, 0, 0), mode="circular")
        grid = F.pad(grid, (0, 0, pad, pad, pad, pad), mode="replicate")
        return self.gate_coupling(grid).reshape(batch, -1)
```

and change `_gate_logits` to route through it:

```python
    def _gate_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self._couple(self.gate_head(h)) + self.sparsity_head(h)
        return self.logit_clamp * torch.tanh(logits / self.logit_clamp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -v`
Expected: PASS (33 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 176 tests.

- [ ] **Step 6: Commit**

```bash
git add src/models/generator.py tests/test_generator_structured.py
git commit -m "feat(models): couple gate logits over the (4,4,8) descriptor grid

Measured residual zero-correlation peaks at |i-j| = 1 (+0.32, adjacent
orientation bins) and |i-j| = 8 (+0.27, same bin in the neighbouring
spatial cell). A 3x3x3 conv with circular orientation padding and
replicated spatial edges reproduces both.

Identity-initialised so v3 starts exactly where v2 does and learns the
coupling, rather than scrambling the logits at step 0."
```

---

### Task 3: Correlated gate noise

**Files:**
- Modify: `src/models/generator.py` (`StructuredGateGenerator.__init__` and `_sample_gate`)
- Modify: `tests/test_generator_structured.py` (append)

**Interfaces:**
- Consumes: `StructuredGateGenerator` from Tasks 1-2.
- Produces: a registered buffer `noise_kernel` of shape `(1, 1, k, k, k)`, and a static method `_gaussian_kernel(size, sigma) -> Tensor`.

Correlated logits with independent noise still sample near-independently — the correlation has to be in the sampling, not only in the mean. This smooths the logistic noise on the same grid before the binary-concrete transform.

**The kernel is fixed, not learned.** A learned kernel applied to noise could be driven toward zero, killing gate stochasticity and collapsing the support distribution — the exact failure this class exists to prevent. Fixing it keeps injected correlation structural rather than something training can optimise away. It is normalised so its squared weights sum to 1, which preserves the variance of i.i.d. input, so the gate's effective temperature does not shift.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_structured.py`:

```python
def test_noise_kernel_is_variance_preserving():
    # Smoothing must not change the noise scale, or it silently shifts the
    # gate's effective temperature.
    generator = build()
    assert generator.noise_kernel.pow(2).sum().item() == pytest.approx(1.0, abs=1e-6)


def test_noise_kernel_is_not_a_learnable_parameter():
    # A learned noise kernel could be driven to zero, killing gate
    # stochasticity and collapsing the support distribution.
    generator = build()
    assert "noise_kernel" in dict(generator.named_buffers())
    assert "noise_kernel" not in dict(generator.named_parameters())


def _adjacent_vs_distant_gate_correlation(generator, n=8192, seed=21):
    """Correlation between neighbouring vs far-apart gates, logits held flat.

    Takes any generator exposing `_sample_gate`, so v2 and v3 can be compared
    on identical terms.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        gate = generator._sample_gate(torch.zeros(n, OUTPUT))
    g = gate - gate.mean(dim=0, keepdim=True)
    sd = g.std(dim=0, keepdim=True).clamp(min=1e-8)
    g = g / sd
    corr = (g.T @ g) / n
    idx = torch.arange(OUTPUT)
    sep = (idx[:, None] - idx[None, :]).abs()
    off_diagonal = ~torch.eye(OUTPUT, dtype=torch.bool)
    adjacent = corr[(sep == 1) & off_diagonal].mean().item()
    distant = corr[(sep >= 16) & off_diagonal].mean().item()
    return adjacent, distant


def test_gate_noise_is_spatially_correlated():
    # With logits flat at zero the gate is pure noise, so any correlation
    # between neighbouring coordinates comes from the smoothing.
    adjacent, distant = _adjacent_vs_distant_gate_correlation(build())
    assert adjacent > 0.05
    assert adjacent > distant + 0.04


def test_v2_gate_noise_is_uncorrelated():
    # The contrast: v2 samples every coordinate independently, so neighbours
    # are no more alike than distant coordinates.
    torch.manual_seed(0)
    v2 = GatedGenerator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    adjacent, distant = _adjacent_vs_distant_gate_correlation(v2)
    assert abs(adjacent - distant) < 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -k noise -v`
Expected: FAIL — `AttributeError: 'StructuredGateGenerator' object has no attribute 'noise_kernel'`

- [ ] **Step 3: Add the correlated noise**

Add this static method to the class, above `_sample_gate`:

```python
    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Separable 3-D Gaussian, normalised so smoothing preserves variance.

        Scaling by the L2 norm rather than the sum is deliberate: convolving
        i.i.d. unit-variance noise with weights w gives variance sum(w^2), so
        an L2-normalised kernel leaves the noise scale -- and therefore the
        gate's effective temperature -- unchanged.
        """
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        line = torch.exp(-(coords**2) / (2.0 * sigma**2))
        kernel = line[:, None, None] * line[None, :, None] * line[None, None, :]
        kernel = kernel / torch.linalg.vector_norm(kernel)
        return kernel.reshape(1, 1, size, size, size)
```

In `__init__`, after the `gate_coupling` block, insert:

```python
        # Fixed, not learned: a trainable noise kernel could be driven toward
        # zero, removing gate stochasticity and collapsing the support
        # distribution -- the failure this class exists to prevent.
        self.register_buffer(
            "noise_kernel", self._gaussian_kernel(gate_kernel, noise_kernel_sigma)
        )
```

Add a smoothing helper just below `_couple`:

```python
    def _smooth_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Correlate i.i.d. noise over the descriptor grid.

        Correlated logits with independent noise still sample near
        independently: the correlation has to be in the draw, not only in the
        mean.
        """
        batch = noise.shape[0]
        rows, cols, orient = self.layout
        pad = self.gate_kernel // 2
        grid = noise.reshape(batch, 1, rows, cols, orient)
        grid = F.pad(grid, (pad, pad, 0, 0, 0, 0), mode="circular")
        grid = F.pad(grid, (0, 0, pad, pad, pad, pad), mode="replicate")
        kernel = self.noise_kernel.to(grid.dtype)
        return F.conv3d(grid, kernel).reshape(batch, -1)
```

In `_sample_gate`, replace the single `logistic = ...` line with:

```python
        logistic = self._smooth_noise(torch.log(u) - torch.log1p(-u))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_structured.py -v`
Expected: PASS (37 tests)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 180 tests.

- [ ] **Step 6: Commit**

```bash
git add src/models/generator.py tests/test_generator_structured.py
git commit -m "feat(models): correlate the gate noise on the descriptor grid

Correlated logits with independent noise still sample near
independently, so the smoothing has to apply to the draw. The kernel is
a registered buffer, not a parameter: a learnable one could be driven to
zero, removing gate stochasticity and collapsing the support
distribution. L2-normalised so it preserves noise variance and does not
shift the gate's effective temperature."
```

---

### Task 4: Wire up the variant

**Files:**
- Modify: `src/models/generator.py` (`build_generator`)
- Modify: `tests/test_generator_factory.py` (append)
- Create: `configs/sift_gan_v3.yaml`
- Modify: `PROJECT_DOCUMENTATION.md`

**Interfaces:**
- Consumes: `StructuredGateGenerator` from Tasks 1-3.
- Produces: `build_generator` accepting `generator_type: "structured_gated"` with optional `layout`, `gate_kernel`, `noise_kernel_sigma` keys.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_factory.py`, and add `StructuredGateGenerator` to the import at the top:

```python
def test_structured_gated():
    cfg = dict(BASE_CFG, generator_type="structured_gated")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, StructuredGateGenerator)
    assert generator.layout == (4, 4, 8)
    assert generator.gate_kernel == 3
    assert generator.gate_temperature == 0.5
    assert generator.logit_clamp == 10.0


def test_structured_gated_honours_overrides():
    cfg = dict(
        BASE_CFG,
        generator_type="structured_gated",
        layout=[2, 4, 8],
        gate_kernel=1,
        noise_kernel_sigma=1.5,
        logit_clamp=4.0,
    )
    generator = build_generator(cfg, output_dim=64)
    assert generator.layout == (2, 4, 8)
    assert generator.gate_kernel == 1
    assert generator.noise_kernel_sigma == 1.5
    assert generator.logit_clamp == 4.0


def test_structured_gated_rejects_a_layout_that_does_not_match_output_dim():
    cfg = dict(BASE_CFG, generator_type="structured_gated", layout=[4, 4, 8])
    with pytest.raises(ValueError, match="layout"):
        build_generator(cfg, output_dim=64)


def test_structured_and_gated_checkpoints_do_not_interchange():
    structured = build_generator(
        dict(BASE_CFG, generator_type="structured_gated"), output_dim=128
    )
    gated = build_generator(dict(BASE_CFG, generator_type="gated"), output_dim=128)
    with pytest.raises(RuntimeError):
        gated.load_state_dict(structured.state_dict())
    with pytest.raises(RuntimeError):
        structured.load_state_dict(gated.state_dict())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -v`
Expected: FAIL — `ImportError`, then `ValueError: Unknown generator_type: structured_gated`

- [ ] **Step 3: Register the type in `build_generator`**

Insert before the final `raise`:

```python
    if kind == "structured_gated":
        return StructuredGateGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
            layout=tuple(model_cfg.get("layout", (4, 4, 8))),
            gate_kernel=int(model_cfg.get("gate_kernel", 3)),
            noise_kernel_sigma=float(model_cfg.get("noise_kernel_sigma", 0.8)),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -v`
Expected: PASS (13 tests, up from 9)

- [ ] **Step 5: Create `configs/sift_gan_v3.yaml`**

Read `configs/sift_gan_v2.yaml` first and copy it exactly, changing **only** `generator_type` and adding the three new keys, so v3 is one architectural change from v2 and nothing else. Keep whatever `logit_clamp`, hyperparameters and step count v2 uses. Head the file with:

```yaml
# Variant v3: v2's gated generator with structured support statistics.
# Exactly one change from configs/sift_gan_v2.yaml -- generator_type -- plus
# the three keys that change introduces. v2's gates are independent per
# coordinate, so its non-zero count is Binomial(128, p) with std about 4.76;
# real SIFT measures 14.45. See
# docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md
```

with the model block gaining:

```yaml
  generator_type: structured_gated
  layout: [4, 4, 8]
  gate_kernel: 3
  noise_kernel_sigma: 0.8
```

Note `device:` — phase 1 made training reject `device: auto` when CUDA is present and `CUDA_VISIBLE_DEVICES` is unset. Set `device: cuda:0` in this config rather than copying v2's `auto`, so the v3 arm launches without an exported environment variable.

- [ ] **Step 6: Verify the config builds a generator**

Run:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import yaml
from src.models.generator import build_generator
cfg = yaml.safe_load(open('configs/sift_gan_v3.yaml'))
g = build_generator(cfg['model'], output_dim=cfg['data']['descriptor_dim'])
print(type(g).__name__, g.layout, g.gate_kernel, g.noise_kernel_sigma)
"
```
Expected: `StructuredGateGenerator (4, 4, 8) 3 0.8`

- [ ] **Step 7: Update `PROJECT_DOCUMENTATION.md`**

Add a `v3` row to the variant table:

```markdown
| `v3` | + structured gate (`generator_type: structured_gated`) | `configs/sift_gan_v3.yaml` | *(untrained)* |
```

Extend the `generator_type` section to list `structured_gated` alongside `mlp` and `gated`, and add after the "Why v2 exists" subsection:

```markdown
### Why v3 exists

v2 produces exact zeros but gets their *distribution* wrong. Its gates are
sampled independently per coordinate, so the non-zero count per vector is
Binomial(128, p) with standard deviation about 4.76. Real SIFT measures
14.45 -- three times as variable -- and its zero pattern is locally
correlated: +0.32 between adjacent orientation bins and +0.27 between the
same bin in neighbouring spatial cells. Neither is reachable by tuning v2.

v3 adds a per-vector sparsity level (over-dispersion), a 3x3x3 convolution
over the (4,4,8) descriptor grid with circular orientation padding (local
correlation), and fixed smoothing of the gate noise so sampling is
correlated too. Measurements are in `tools/probes/`; the design is in
`docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md`.
```

- [ ] **Step 8: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 184 tests.

- [ ] **Step 9: Commit**

```bash
git add src/models/generator.py tests/test_generator_factory.py configs/sift_gan_v3.yaml PROJECT_DOCUMENTATION.md
git commit -m "feat(configs): register structured_gated and add the v3 variant

One architectural change from v2. Sets device: cuda:0 rather than auto,
since training now refuses to guess a card on a shared box."
```

---

## Verification

After Task 4:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q
```
Expected: 184 passed, up from the 143 baseline.

Then confirm the mechanism end to end on real scale — the numbers the design is accountable for. This runs on CPU and needs no training:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import torch, yaml
from src.models.generator import build_generator
cfg = yaml.safe_load(open('configs/sift_gan_v3.yaml'))
torch.manual_seed(0)
g = build_generator(cfg['model'], output_dim=128)
with torch.no_grad():
    x = g(torch.randn(20000, cfg['model']['latent_dim']))
nnz = (x > 0).sum(1).float()
p = nnz.mean().item() / 128
print(f'zero frac {1 - p:.4f}  nnz mean {nnz.mean():.2f}  std {nnz.std():.2f}')
print(f'binomial-equivalent std {(128 * p * (1 - p)) ** 0.5:.2f}')
print('real SIFT: zero frac 0.2301, nnz mean 98.54, std 14.45')
"
```

At random init the numbers will not match real SIFT — the point is only that
`nnz std` is not pinned to the binomial value the way v2's is. Matching 14.45
is what training is for, and is the v3 arm's success criterion.

## Not in this plan

- **The log-ratio regularizer** — phase 3, its own plan. Inert at
  `lid_reg_alpha: 0.0`, so it can land while the v3 arm is running.
- **Running the v3 arm.** Requires the GPU box and the phase 1 lock. The
  success gates are in the spec: `lid_median` in 16.85-18.63 and
  `exact_zero_fraction` in 0.21-0.25, with `nnz` std, the correlation profile,
  `hubness_skew` and `ivf_gini` as independent evidence.
- **`compare_variants` plumbing.** v3 needs no new report flags; the existing
  ANN panels already measure everything the arm is judged on.
