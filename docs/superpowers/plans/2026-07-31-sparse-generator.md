# Sparse Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SparseGenerator` that emits non-negative vectors with a learnable point mass at zero, so synthetic descriptors match real SIFT's support instead of smearing mass across zero.

**Architecture:** A shared MLP trunk feeds two heads — a softplus magnitude head and a binary-concrete gate head with a hard-forward straight-through estimator. Their product is L2-normalized inside `forward`. The new class sits alongside the untouched `Generator`, selected by a `generator_type` config key through a shared factory.

**Tech Stack:** Python 3.12, PyTorch 2.13, numpy 2.5, pytest (to be added), PyYAML.

**Spec:** `docs/superpowers/specs/2026-07-31-sparse-generator-design.md`

## Global Constraints

- Python interpreter is at `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`. It lives in the main checkout, not this worktree. Always invoke it by that absolute path. Do not create a second venv.
- CUDA is unavailable on this machine (`torch.cuda.is_available()` is `False`). All tests must be CPU-only and fast.
- This worktree is on `4a373c2`. Every line number below refers to that revision. PR #1 is still open; do not rebase as part of this plan.
- `Generator` in `src/models/generator.py` must not change behaviour. Seven existing `configs/bench_*.yaml` files carry no `generator_type` key and must keep working untouched.
- Existing normalization call sites (`train_wgan_gp.py:134`, `:239`, `:267`, `generate.py:66-67`) are deliberately left in place. `normalize_l2` is idempotent, so it is a no-op on `SparseGenerator` output. Do not remove them.
- Default hyperparameters, exact values: `gate_temperature=0.5`, `logit_clamp=10.0`, `eps=1e-8`, `negative_slope=0.2`.
- Run tests with `cd` to the worktree root so `src` is importable, or rely on the `pytest.ini` added in Task 1.

---

### Task 1: Test scaffolding and `SparseGenerator`

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Modify: `src/models/generator.py` (append after line 27)
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SparseGenerator(latent_dim: int, output_dim: int, hidden_dims: Iterable[int], negative_slope: float = 0.2, gate_temperature: float = 0.5, logit_clamp: float = 10.0, eps: float = 1e-8)`. Its `forward(z: Tensor) -> Tensor` maps `(B, latent_dim)` to `(B, output_dim)`, non-negative and unit-norm. Public attributes used by later tasks and tests: `.trunk`, `.magnitude_head`, `.gate_head`, `.gate_temperature`, `.logit_clamp`, `.eps`.

- [ ] **Step 1: Add pytest to requirements**

Append one line to `requirements.txt`, which currently reads `numpy / torch / pyyaml / tqdm / scikit-learn`:

```
pytest
```

- [ ] **Step 2: Install it**

Run:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pip install pytest
```

Expected: pytest installs. Confirm with `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest --version`.

- [ ] **Step 3: Create pytest.ini so `src` is importable**

Create `pytest.ini` at the worktree root:

```ini
[pytest]
testpaths = tests
addopts = -q
```

`rootdir` insertion puts the worktree root on `sys.path`, which is what makes `from src.models.generator import ...` resolve.

- [ ] **Step 4: Write the failing tests**

Create `tests/test_generator.py`:

```python
import pytest
import torch

from src.models.generator import Generator, SparseGenerator

LATENT = 16
OUTPUT = 128
HIDDEN = [32, 32]


def build(**overrides):
    kwargs = dict(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    kwargs.update(overrides)
    return SparseGenerator(**kwargs)


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
    # Exact equality is the point: the gate is hard, not merely small.
    assert (out == 0.0).any()


def test_no_all_zero_row(out):
    # A row with every gate off normalizes to the zero vector, silently
    # violating unit norm. Guard against it.
    assert ((out > 0).sum(dim=1) > 0).all()


def test_gate_head_receives_gradient(gen):
    torch.manual_seed(2)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.gate_head.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0, "straight-through estimator is not passing gradient"


def test_magnitude_head_receives_gradient(gen):
    torch.manual_seed(3)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.magnitude_head.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_gate_noise_is_kept_at_sample_time(gen):
    # Documents a deliberate design decision: gate noise is NOT disabled in
    # eval mode, so the sampled distribution matches what the critic trained
    # against. sample_generator() calls generator.eval(), so this matters.
    gen.eval()
    z = torch.randn(64, LATENT)
    with torch.no_grad():
        a, b = gen(z), gen(z)
    assert not torch.equal(a, b)


def test_existing_generator_is_unchanged():
    torch.manual_seed(4)
    g = Generator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    x = g(torch.randn(8, LATENT))
    assert x.shape == (8, OUTPUT)
    assert (x < 0).any(), "plain Generator should still be unconstrained"
```

- [ ] **Step 5: Run tests to verify they fail**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator.py -v
```

Expected: collection error — `ImportError: cannot import name 'SparseGenerator'`.

- [ ] **Step 6: Implement SparseGenerator**

Append to `src/models/generator.py` (after the existing `Generator`, line 27). Add `import torch.nn.functional as F` to the imports at the top:

```python
class SparseGenerator(nn.Module):
    """Generator whose output is non-negative with a learnable point mass at zero.

    A shared trunk feeds a softplus magnitude head and a binary-concrete gate
    head. The gate is hard in the forward pass so the critic sees exact zeros,
    with a straight-through estimator carrying gradient via the soft path.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        gate_temperature: float = 0.5,
        logit_clamp: float = 10.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        dims: List[int] = [latent_dim, *list(hidden_dims)]
        layers = []
        # Unlike Generator, the trunk activates after EVERY layer: its output is
        # a shared feature, not the model output. Without this the two heads
        # collapse into two linear maps of the same pre-activation.
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.magnitude_head = nn.Linear(dims[-1], output_dim)
        self.gate_head = nn.Linear(dims[-1], output_dim)
        self.gate_temperature = float(gate_temperature)
        self.logit_clamp = float(logit_clamp)
        self.eps = float(eps)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(logits).clamp(self.eps, 1.0 - self.eps)
        logistic = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((logits + logistic) / self.gate_temperature)
        hard = (soft > 0.5).to(soft.dtype)
        # Hard forward, soft gradient.
        return hard + soft - soft.detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        m = F.softplus(self.magnitude_head(h))
        # Smooth bound, NOT torch.clamp: clamp has zero gradient outside its
        # range, so a saturated logit would freeze permanently -- exactly the
        # dead-gate failure this bound exists to prevent.
        logits = self.logit_clamp * torch.tanh(self.gate_head(h) / self.logit_clamp)
        x = self._sample_gate(logits) * m
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=self.eps)
```

- [ ] **Step 7: Run tests to verify they pass**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator.py -v
```

Expected: 9 passed.

If `test_zeros_are_exact` fails, the gate is producing all-ones at initialization — check that `hard` uses `> 0.5` on the noisy sigmoid, not on the raw logits.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini src/models/generator.py tests/test_generator.py
git commit -m "feat(models): add SparseGenerator with gated non-negative output

Shared trunk feeds a softplus magnitude head and a binary-concrete gate
head. Hard forward so the critic sees exact zeros; straight-through
estimator carries gradient. Adds pytest and the first test suite.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Generator factory and wiring

**Files:**
- Modify: `src/models/generator.py` (append `build_generator`)
- Modify: `src/train/train_wgan_gp.py:24` (import), `:96` and `:118` (type hints), `:166-171` (construction)
- Modify: `src/sample/generate.py:14` (import), `:48-53` (construction)
- Test: `tests/test_generator_factory.py`

**Interfaces:**
- Consumes: `SparseGenerator` and `Generator` from Task 1.
- Produces: `build_generator(model_cfg: dict, output_dim: int) -> nn.Module`. Reads `generator_type` from `model_cfg`, defaulting to `"mlp"`. Used by both `train_wgan_gp.py` and `generate.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generator_factory.py`:

```python
import pytest

from src.models.generator import Generator, SparseGenerator, build_generator

BASE_CFG = {
    "latent_dim": 16,
    "generator_hidden_dims": [32, 32],
    "negative_slope": 0.2,
}


def test_missing_generator_type_defaults_to_mlp():
    # Backward-compat guarantee: the seven existing bench_*.yaml configs
    # carry no generator_type key and must keep resolving to Generator.
    g = build_generator(dict(BASE_CFG), output_dim=128)
    assert isinstance(g, Generator)


def test_explicit_mlp():
    cfg = dict(BASE_CFG, generator_type="mlp")
    assert isinstance(build_generator(cfg, output_dim=128), Generator)


def test_sparse():
    cfg = dict(BASE_CFG, generator_type="sparse")
    g = build_generator(cfg, output_dim=128)
    assert isinstance(g, SparseGenerator)
    assert g.gate_temperature == 0.5
    assert g.logit_clamp == 10.0


def test_sparse_honours_overrides():
    cfg = dict(BASE_CFG, generator_type="sparse", gate_temperature=0.25, logit_clamp=4.0)
    g = build_generator(cfg, output_dim=128)
    assert g.gate_temperature == 0.25
    assert g.logit_clamp == 4.0


def test_unknown_type_raises():
    cfg = dict(BASE_CFG, generator_type="nope")
    with pytest.raises(ValueError, match="nope"):
        build_generator(cfg, output_dim=128)


def test_output_dim_is_respected():
    g = build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=64)
    assert g.magnitude_head.out_features == 64
    assert g.gate_head.out_features == 64


def test_checkpoint_mismatch_fails_loudly():
    # A sparse checkpoint loaded into an mlp Generator (or vice versa) must
    # raise, not silently produce garbage. generate.py pairs them via the
    # config's generator_type; this is the backstop when they disagree.
    sparse = build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=128)
    mlp = build_generator(dict(BASE_CFG, generator_type="mlp"), output_dim=128)
    with pytest.raises(RuntimeError):
        mlp.load_state_dict(sparse.state_dict())
    with pytest.raises(RuntimeError):
        sparse.load_state_dict(mlp.state_dict())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -v
```

Expected: collection error — `ImportError: cannot import name 'build_generator'`.

- [ ] **Step 3: Implement the factory**

Append to `src/models/generator.py`. Add `from typing import Dict` to the existing typing import line:

```python
def build_generator(model_cfg: Dict, output_dim: int) -> nn.Module:
    """Construct the generator named by model_cfg['generator_type'].

    Defaults to 'mlp' so configs written before SparseGenerator existed keep
    working unchanged.
    """
    kind = model_cfg.get("generator_type", "mlp")
    common = dict(
        latent_dim=int(model_cfg["latent_dim"]),
        output_dim=output_dim,
        hidden_dims=model_cfg["generator_hidden_dims"],
        negative_slope=float(model_cfg["negative_slope"]),
    )
    if kind == "mlp":
        return Generator(**common)
    if kind == "sparse":
        return SparseGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
        )
    raise ValueError(f"Unknown generator_type: {kind}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_generator_factory.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Wire the factory into the trainer**

In `src/train/train_wgan_gp.py`, change the import on line 24 from:

```python
from src.models.generator import Generator
```

to:

```python
from src.models.generator import build_generator
```

Replace lines 166-171:

```python
    generator = Generator(
        latent_dim=latent_dim,
        output_dim=descriptor_dim,
        hidden_dims=model_cfg["generator_hidden_dims"],
        negative_slope=float(model_cfg["negative_slope"]),
    ).to(device)
```

with:

```python
    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
```

Note `latent_dim` (line 163) is still used later at lines 260 and 303, so leave it defined.

Widen two now-inaccurate type hints, since `generator` may be either class. Line 96, in `save_checkpoint`:

```python
    generator: nn.Module,
```

Line 118, in `sample_generator`:

```python
    generator: nn.Module,
```

Add `from torch import nn` to the imports (the file currently imports only `Tensor` from `torch` on line 14).

- [ ] **Step 6: Wire the factory into the sampler**

In `src/sample/generate.py`, change the import on line 14 from:

```python
from src.models.generator import Generator
```

to:

```python
from src.models.generator import build_generator
```

Replace lines 48-53:

```python
    generator = Generator(
        latent_dim=int(model_cfg["latent_dim"]),
        output_dim=int(data_cfg["descriptor_dim"]),
        hidden_dims=model_cfg["generator_hidden_dims"],
        negative_slope=float(model_cfg["negative_slope"]),
    ).to(device)
```

with:

```python
    generator = build_generator(
        model_cfg, output_dim=int(data_cfg["descriptor_dim"])
    ).to(device)
```

- [ ] **Step 7: Verify both entry points still import**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c \
  "import src.train.train_wgan_gp, src.sample.generate; print('imports ok')"
```

Expected: `imports ok`. A `NameError` here means a leftover reference to the removed `Generator` import.

- [ ] **Step 8: Run the full suite**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -v
```

Expected: 16 passed.

- [ ] **Step 9: Commit**

```bash
git add src/models/generator.py src/train/train_wgan_gp.py src/sample/generate.py tests/test_generator_factory.py
git commit -m "feat(models): select generator via build_generator factory

Replaces duplicated construction in train_wgan_gp.py and generate.py with
one factory reading model.generator_type. Defaults to 'mlp' so existing
configs are unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Support metrics

**Files:**
- Modify: `src/train/train_wgan_gp.py:65-76` (`tensor_stats`)
- Test: `tests/test_tensor_stats.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tensor_stats(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]` gains four keys: `zero_fraction_gap`, `negative_fraction`, `per_dim_zero_rate_l1`, `nnz_std_gap`. The three existing keys `mean_l2`, `var_l2`, `cov_fro` keep their current meaning and values.

Why this task exists: the three existing keys are second-moment quantities that were all close in the 100k run while the support was completely wrong. They cannot see this defect.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tensor_stats.py`:

```python
import numpy as np

from src.train.train_wgan_gp import tensor_stats


def test_existing_keys_are_preserved():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8))
    fake = rng.normal(size=(200, 8))
    stats = tensor_stats(real, fake)
    for key in ("mean_l2", "var_l2", "cov_fro"):
        assert key in stats


def test_identical_inputs_give_zero_gaps():
    rng = np.random.default_rng(1)
    x = np.abs(rng.normal(size=(200, 8)))
    x[x < 0.5] = 0.0
    stats = tensor_stats(x, x.copy())
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["per_dim_zero_rate_l1"] == 0.0
    assert stats["nnz_std_gap"] == 0.0
    assert stats["negative_fraction"] == 0.0


def test_negative_fraction_measures_fake_only():
    real = np.ones((10, 4))
    fake = -np.ones((10, 4))
    stats = tensor_stats(real, fake)
    assert stats["negative_fraction"] == 1.0


def test_zero_fraction_gap():
    real = np.zeros((10, 4))          # 100% zeros
    fake = np.ones((10, 4))           # 0% zeros
    stats = tensor_stats(real, fake)
    assert stats["zero_fraction_gap"] == 1.0


def test_per_dim_zero_rate_catches_misplaced_sparsity():
    # Both arrays are 50% zero overall, so zero_fraction_gap is blind here.
    # real zeros the first two columns; fake zeros the last two.
    real = np.ones((10, 4))
    real[:, :2] = 0.0
    fake = np.ones((10, 4))
    fake[:, 2:] = 0.0
    stats = tensor_stats(real, fake)
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["per_dim_zero_rate_l1"] == 1.0


def test_nnz_std_gap_catches_uncorrelated_sparsity():
    # real: every row has exactly 2 non-zeros -> nnz std 0.
    # fake: half the rows are full, half are empty -> same 50% overall
    # zero rate, but nnz std 2.0.
    real = np.zeros((10, 4))
    real[:, :2] = 1.0
    fake = np.zeros((10, 4))
    fake[:5, :] = 1.0
    stats = tensor_stats(real, fake)
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["nnz_std_gap"] == 2.0


def test_all_values_are_plain_floats():
    # run_metadata.json is written with json.dump; numpy scalars are not
    # JSON-serializable.
    rng = np.random.default_rng(2)
    stats = tensor_stats(rng.normal(size=(50, 4)), rng.normal(size=(50, 4)))
    assert all(type(v) is float for v in stats.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_tensor_stats.py -v
```

Expected: `test_existing_keys_are_preserved` passes; the six others fail with `KeyError`.

- [ ] **Step 3: Extend tensor_stats**

Replace `src/train/train_wgan_gp.py:65-76` in full:

```python
def tensor_stats(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    real_mean = real.mean(axis=0)
    fake_mean = fake.mean(axis=0)
    real_var = real.var(axis=0)
    fake_var = fake.var(axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)

    # Support metrics. The three moment statistics above were all close in the
    # 100k run while negative_fraction was ~0.10 and exact_zero_fraction was
    # 0.0 against a real value of 0.2298 -- they cannot see support defects.
    real_zero = real == 0.0
    fake_zero = fake == 0.0
    real_nnz = (~real_zero).sum(axis=1)
    fake_nnz = (~fake_zero).sum(axis=1)

    return {
        "mean_l2": float(np.linalg.norm(real_mean - fake_mean)),
        "var_l2": float(np.linalg.norm(real_var - fake_var)),
        "cov_fro": float(np.linalg.norm(cov_real - cov_fake, ord="fro")),
        "zero_fraction_gap": float(abs(fake_zero.mean() - real_zero.mean())),
        "negative_fraction": float((fake < 0).mean()),
        # Is sparsity in the right PLACES, not merely at the right rate?
        "per_dim_zero_rate_l1": float(
            np.abs(fake_zero.mean(axis=0) - real_zero.mean(axis=0)).mean()
        ),
        # Correlation probe: independent Bernoulli(p) gates over d dims give
        # nnz std sqrt(d*p*(1-p)); correlated block sparsity gives much more.
        "nnz_std_gap": float(abs(fake_nnz.std() - real_nnz.std())),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_tensor_stats.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run the full suite**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -v
```

Expected: 23 passed.

- [ ] **Step 6: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_tensor_stats.py
git commit -m "feat(train): add support metrics to tensor_stats

zero_fraction_gap, negative_fraction, per_dim_zero_rate_l1 and nnz_std_gap.
The existing moment statistics stayed close through 100k steps while the
output support was entirely wrong; these make the defect visible.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Config and end-to-end smoke test

**Files:**
- Create: `configs/bench_sparse.yaml`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `build_generator` (Task 2), extended `tensor_stats` (Task 3).
- Produces: a runnable config for the real benchmark, and proof the training loop survives a `SparseGenerator` end to end.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_train_smoke.py`. It drives the real `train()` on Gaussian synthetic data via `synthetic_if_missing`, so it needs no dataset on disk:

```python
import math

import pytest

from src.train.train_wgan_gp import train


def make_config(tmp_path, generator_type):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / generator_type),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 16,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 8,
            "generator_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 4,
            "n_critic": 2,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "num_workers": 0,
            "distance_reg_alpha": 0.1,
            "distance_reg_max_points": 16,
            "amp": False,
            "log_every": 1,
            "eval_every": 2,
            "save_every": 4,
        },
    }
    if generator_type is not None:
        cfg["model"]["generator_type"] = generator_type
    return cfg


@pytest.mark.parametrize("generator_type", ["mlp", "sparse"])
def test_training_loop_runs(tmp_path, generator_type):
    # Note: the synthetic fallback generates Gaussian data, which HAS
    # negatives. This verifies the loop runs end to end -- not that the
    # sparse generator learns anything sensible.
    ckpt_path, meta = train(make_config(tmp_path, generator_type))
    assert ckpt_path.exists()
    assert meta["metrics"], "no metrics were logged"
    for entry in meta["metrics"]:
        assert math.isfinite(entry["g_loss"]), f"non-finite g_loss: {entry}"
        assert math.isfinite(entry["d_loss"]), f"non-finite d_loss: {entry}"


def test_sparse_eval_reports_zero_negatives(tmp_path):
    _, meta = train(make_config(tmp_path, "sparse"))
    evals = meta["eval"]
    assert evals, "no eval entries were recorded"
    for entry in evals:
        # Structural guarantee -- must hold from the very first eval.
        assert entry["negative_fraction"] == 0.0


def test_mlp_config_without_generator_type_still_trains(tmp_path):
    ckpt_path, _ = train(make_config(tmp_path, None))
    assert ckpt_path.exists()
```

- [ ] **Step 2: Run the smoke test**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_train_smoke.py -v
```

Expected: all 4 pass, given Tasks 2 and 3 are complete. If `test_sparse_eval_reports_zero_negatives` fails, the trainer is not routing through `build_generator` — recheck Task 2 Step 5.

If `meta["eval"]` raises `KeyError`, note that `run_meta.setdefault("eval", [])` only runs when an eval fires; `eval_every: 2` with `num_gen_steps: 4` guarantees two evals, so a `KeyError` means evals are not firing at all.

- [ ] **Step 3: Create the benchmark config**

Create `configs/bench_sparse.yaml`, cloned from `bench_100k_improved.yaml` with the generator keys added, so the sparse arm is directly comparable against the existing 100k results:

```yaml
# Sparse generator at the 100k-step horizon. Identical to bench_100k_improved.yaml
# except for the generator_type / gate keys, so any difference in the compare_100k
# summary is attributable to the generator architecture alone.
seed: 42
device: auto
output_dir: runs/x100k_sparse

data:
  real_path: data/sift_base.npy
  format: npy
  descriptor_dim: 128
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: sparse
  gate_temperature: 0.5
  logit_clamp: 10.0

training:
  batch_size: 512
  num_gen_steps: 100000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  num_workers: 0
  distance_reg_alpha: 0.1
  distance_reg_max_points: 256
  amp: false
  log_every: 500
  eval_every: 2000
  save_every: 5000
```

Note: `bench_100k_improved.yaml` also carries `ema_decay: 0.999`. It is omitted here because `4a373c2` has no EMA support and would ignore it. Add it when rebasing onto the merged PR #1.

- [ ] **Step 4: Verify the config parses and builds the right generator**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import yaml
from src.models.generator import build_generator, SparseGenerator
cfg = yaml.safe_load(open('configs/bench_sparse.yaml'))
g = build_generator(cfg['model'], output_dim=cfg['data']['descriptor_dim'])
assert isinstance(g, SparseGenerator), type(g)
print('bench_sparse builds', type(g).__name__, 'params:',
      sum(p.numel() for p in g.parameters()))
"
```

Expected: prints `bench_sparse builds SparseGenerator params: <N>`.

- [ ] **Step 5: Verify every existing config still builds**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import glob, yaml
from src.models.generator import build_generator
for path in sorted(glob.glob('configs/*.yaml')):
    cfg = yaml.safe_load(open(path))
    g = build_generator(cfg['model'], output_dim=cfg['data']['descriptor_dim'])
    print(f'{path}: {type(g).__name__}')
"
```

Expected: every pre-existing config prints `Generator`; only `bench_sparse.yaml` prints `SparseGenerator`. Note that at `4a373c2` the `configs/` directory may contain only `bench_sparse.yaml` plus whatever was committed at the initial commit — the seven `bench_*.yaml` files listed in the original git status are uncommitted in the MAIN checkout and are not present here. If a config is missing, that is expected, not a failure.

- [ ] **Step 6: Run the full suite**

Run:

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator && \
  /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -v
```

Expected: 27 passed.

- [ ] **Step 7: Commit**

```bash
git add configs/bench_sparse.yaml tests/test_train_smoke.py
git commit -m "feat(configs): add bench_sparse and end-to-end training smoke test

Smoke test drives the real train() over Gaussian synthetic data for both
generator types, asserting finite losses and structurally zero negatives
in the sparse arm's eval output.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## After the plan

These are not tasks — they need the real dataset and a GPU, neither of which is available in this worktree.

1. **Measure the real-data reference values.** `data/sift_base.npy` is not present here. On the training machine, compute the uniform-sparsity reference for success criterion 3, and the real `nnz` standard deviation for interpreting `nnz_std_gap`:

   ```python
   import numpy as np
   from src.data.sift1m_dataset import apply_preprocess  # or reuse the eval path
   real_zero = (real_l2 == 0.0)
   print("uniform_ref:", np.abs(real_zero.mean() - real_zero.mean(0)).mean())
   print("real nnz std:", (~real_zero).sum(1).std())
   print("independent-Bernoulli reference:", (128 * 0.2298 * 0.7702) ** 0.5)
   ```

2. **Train `bench_sparse.yaml`** and compare against `improved_100k` in the existing `compare_100k` harness.

3. **Check success criteria** from the spec: `negative_fraction` exactly 0.0; `exact_zero_fraction` within 0.02 of 0.2298; `per_dim_zero_rate_l1` below half `uniform_ref`; `median_5nn_distance` and `effective_rank` no worse than 0.5131 and 27.56.

4. **Rebase onto merged PR #1**, then add `ema_decay: 0.999` to `bench_sparse.yaml` and confirm EMA's parameter copying handles the two-head module (it iterates named parameters, so it should, but it has never seen this module).
