# Sparse Generator Design

**Date:** 2026-07-31
**Branch:** `worktree-sparse-generator`
**Status:** Design approved, pending implementation plan

## Problem

The generator produces vectors that are structurally incapable of matching the SIFT
distribution. Evidence from `runs/eda/compare_100k.summary.json` (50k vectors per
arm, L2-normalized, seed 42):

| Metric | real | baseline_30k | ema_only_100k | improved_100k |
|---|---|---|---|---|
| `value_min` | 0.0 | -0.1664 | -0.0978 | -0.1352 |
| `negative_fraction` | 0.0 | 0.1190 | 0.0991 | 0.0982 |
| `exact_zero_fraction` | 0.2298 | 0.0 | 0.0 | 0.0 |
| `value_mean` | 0.05233 | 0.05264 | 0.05246 | 0.05242 |
| `value_std` | 0.07123 | 0.07101 | 0.07113 | 0.07117 |
| `median_5nn_distance` | 0.5153 | 0.5101 | 0.5144 | 0.5131 |
| `effective_rank` | 27.99 | 26.96 | 27.39 | 27.56 |

Real SIFT lives on a **non-negative, sparse** slice of the unit sphere. Raw
descriptors are uint8 histogram bins; empty bins survive L2 normalization as exact
zeros, giving a genuine point mass at 0 covering 23% of all coordinates.

All three trained arms produce dense vectors with ~10% negative coordinates and
**exactly zero** exact zeros. Note that `value_mean` and `value_std` match real to
within 0.3% — the model matches the marginal's first two moments by smearing mass
across zero, which is the only move its architecture permits.

### Root cause

`Generator` (`src/models/generator.py:26`) ends in a bare `nn.Linear`, so its
pre-normalization output is unconstrained in R^128. `normalize_l2`
(`src/train/train_wgan_gp.py:79`) is sign-preserving. There is no mechanism by
which the model can emit a negative-free vector, let alone an exact zero.

This is a structural defect, not a training failure. More steps will not fix it —
`ema_only_100k` and `improved_100k` ran 100k generator steps and still show 0.0
exact zeros.

### Scope note

The geometry is already close: `median_5nn_distance` is within 0.4% and
`effective_rank` within 1.5%. This defect is one of support and realism, and is not
currently known to affect retrieval difficulty. The goal is full distributional
fidelity — synthetic instances indistinguishable from real SIFT — so it is in scope
regardless.

## Goal

A generator whose output support matches real SIFT: non-negative by construction,
with a learnable point mass at zero whose rate and per-dimension placement the
critic can push on.

## Non-goals

Modeling the quantization lattice. Real values after L2 normalization sit on a
discrete lattice `k/||v||` derived from integer bin counts. Capturing this requires
dequantized training, which perturbs the real-side distribution and changes what
`exact_zero_fraction` means for real data. Deferred: the current EDA harness does
not measure lattice structure, so the fidelity gain is unobservable today. Revisit
after adding a lattice metric.

## Design

### Approach selection

Three approaches were considered.

**A. Non-negative output head.** `trunk -> Linear -> ReLU -> normalize`. One line.
Fixes `negative_fraction` exactly and yields exact zeros for free, but the zero
*rate* is emergent with no lever to steer it toward 0.2298. Softplus and `abs`
variants fail outright — neither ever emits an exact zero. Rejected: trades an
unprincipled negative tail for an uncontrolled zero rate.

**B. Factorized gate x magnitude.** Selected. See below.

**C. Raw-domain modeling with dequantization.** Train on raw uint8 with uniform
dequantization, softplus head, round/clamp at sample time. The only option that
also captures quantization. Deferred per non-goals: it changes the real-side
distribution and introduces a train/sample mismatch the critic never supervises,
in exchange for fidelity no current metric can see.

### Architecture

New class `SparseGenerator` in `src/models/generator.py`. The existing `Generator`
is left untouched.

```python
class SparseGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims,
                 negative_slope=0.2, gate_temperature=0.5,
                 logit_clamp=10.0, eps=1e-8):
        # trunk: latent -> hidden_dims, LeakyReLU after EVERY layer
        self.trunk = ...
        self.magnitude_head = nn.Linear(hidden_dims[-1], output_dim)
        self.gate_head      = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.trunk(z)
        m = F.softplus(self.magnitude_head(h))
        raw = self.gate_head(h)
        # smooth bound to (-logit_clamp, +logit_clamp); see "bounding the logits"
        logits = self.logit_clamp * torch.tanh(raw / self.logit_clamp)
        g = self._sample_gate(logits)
        x = g * m
        return x / x.norm(dim=1, keepdim=True).clamp(min=self.eps)
```

The trunk shape differs from `Generator`, which has no activation after its final
layer (`generator.py:19-23`) because that layer *is* its output. Here the trunk
produces a shared feature, so it needs an activation on its last layer — otherwise
the two heads degenerate into two linear maps of the same pre-activation.

Non-negativity is structural: `softplus` is strictly positive, the gate is in
{0, 1}, and normalization by a positive norm preserves sign.

### Gate mechanism

Binary concrete (Gumbel-sigmoid) with a straight-through estimator:

```python
def _sample_gate(self, logits):
    u = torch.rand_like(logits).clamp(self.eps, 1 - self.eps)
    logistic = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((logits + logistic) / self.gate_temperature)
    hard = (soft > 0.5).to(soft.dtype)
    return hard + soft - soft.detach()   # hard forward, soft gradient
```

Three decisions, stated explicitly so they are not silently reverted:

**Hard forward, not soft.** The critic must see exact zeros, or it can apply no
pressure to the zero atom and the current failure reproduces in a more elaborate
form. The forward pass commits to a binary gate.

**Noise is kept at sample time.** Taking `gate = (logits > 0)` at inference would
make generation deterministic given `z`, but the sampled distribution would then
differ from the one the critic was trained against and the gate rate would shift.
Train/sample consistency wins.

**Fixed temperature, no annealing.** Since the forward pass is hard, temperature
only controls gradient sharpness. A schedule is an extra tunable before there is
evidence one is needed. Default `gate_temperature=0.5`, configurable.

### Bounding the logits

Gate logits are bounded smoothly via `logit_clamp * tanh(raw / logit_clamp)` rather
than `torch.clamp`. This matters: `torch.clamp` has zero gradient outside its range,
so a logit reaching the bound would receive no gradient and freeze permanently —
causing precisely the dead-gate failure the bound exists to prevent. `tanh` is
gradient-preserving everywhere while still bounding the range.

`logit_clamp` interacts with `gate_temperature`: the gate saturates on
`logits / gate_temperature`, so the effective saturation scale is
`logit_clamp / gate_temperature`. At the defaults (10.0 and 0.5) that is 20, deep in
saturation, meaning a bounded logit still produces a near-deterministic gate with a
very small but non-zero gradient. If dead coordinates appear in practice, lower
`logit_clamp` rather than raising the temperature — it bounds saturation directly
without changing the estimator's bias.

### Sparsity structure

SIFT's 128 dimensions are 16 spatial cells x 8 orientation bins, and sparsity is
spatially correlated — a flat gradient cell empties all 8 of its bins together. A
per-coordinate independent gate cannot represent this.

All 128 gate logits are produced from one trunk activation conditioned on a single
`z`, so the correlation structure is learnable without further architecture. If the
flat version underfits, the extension is to reshape the heads to `(16, 8)` and add
a per-cell gate multiplying the per-bin gate. Not implemented initially.

### Known risk: unsupervised magnitudes

When `g = 0`, the product rule gives `dL/dm = g * dL/dx = 0`. Magnitudes of
gated-off coordinates receive no gradient and are free to drift. If a gate later
flips on for a nearby `z`, the exposed magnitude was never directly supervised.

Judged acceptable: `m` is a smooth function of `h`, so a coordinate that is on for
most of latent space is well-trained and neighbouring `z` extrapolates sensibly.
The mitigation is a metric, not an architectural patch — `per_dim_zero_rate_l1`
(below) penalises a coordinate whose gate freezes off, since its fake zero rate
goes to exactly 1.0. The smooth logit bound is cheap insurance preventing a logit
from running to negative infinity and freezing a coordinate dead.

### Known risk: all-zero vectors

If every gate is off, `x = 0` and normalization cannot produce a unit vector — the
result is a zero vector, silently violating the unit-norm invariant. Real SIFT
averages ~98 non-zero coordinates out of 128, so this should not occur, but it is a
silent corruption if it does. Covered by an assertion in the test suite rather than
left as an assumption.

## Integration

### Factory

```python
def build_generator(model_cfg: dict, output_dim: int) -> nn.Module:
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

Defaulting `generator_type` to `"mlp"` keeps all seven existing `configs/bench_*.yaml`
working untouched — none carry the key.

This replaces the duplicated construction at `train_wgan_gp.py:166` and
`generate.py:48`, currently two hand-maintained copies of the same argument list.

### Call sites unchanged

`SparseGenerator.forward` returns a unit-norm vector, and `normalize_l2` is
idempotent — dividing a unit vector by 1.0 is a no-op up to float error. So
`train_wgan_gp.py:239`, `:267`, `:134` and `generate.py:66` keep working as written
for both generator types, and are deliberately left as-is. Stripping the redundant
normalization would make correctness depend on which class is plugged in.

### Effect on the critic

Favourable. Fakes carry exact zeros and no negatives from step 0, so the critic has
no trivial "does this contain negatives" separator to exploit early in training —
as it would under approach A. It must discriminate on the sparsity pattern and
magnitude distribution, which is the intended signal.

Gradient penalty is unaffected. `alpha * real + (1 - alpha) * fake`
(`train_wgan_gp.py:47`) yields interpolates that are non-negative but neither sparse
nor unit-norm, which was already true today. GP constrains the critic's Lipschitz
constant and does not depend on endpoint support.

### Checkpoints

`SparseGenerator` has different `state_dict` keys, so its checkpoints will not load
into `Generator` or vice versa. The factory reads `generator_type` from the same
config passed to `generate.py --config`, keeping them paired. A mismatched pair
fails loudly on the `load_state_dict` key check rather than silently producing
garbage.

### Config

`configs/bench_sparse.yaml`, cloned from `bench_100k_improved.yaml` with
`generator_type: sparse` added, so the sparse arm is directly comparable against
existing 100k results in the same `compare_100k` harness.

## Metrics

`tensor_stats` (`train_wgan_gp.py:65`) reports `mean_l2`, `var_l2` and `cov_fro` —
all second-moment quantities that were already close in the 100k results while the
support was entirely wrong. They cannot see this defect. Four additions, computable
from the sampled arrays already materialised at `train_wgan_gp.py:134`, with no
generator introspection:

```python
real_zero = (real == 0.0)
fake_zero = (fake == 0.0)
{
  "zero_fraction_gap":    abs(fake_zero.mean() - real_zero.mean()),
  "negative_fraction":    (fake < 0).mean(),
  "per_dim_zero_rate_l1": np.abs(fake_zero.mean(0) - real_zero.mean(0)).mean(),
  "nnz_std_gap":          abs((~fake_zero).sum(1).std() - (~real_zero).sum(1).std()),
}
```

**`per_dim_zero_rate_l1`** asks whether sparsity is in the right *places*, not
merely at the right rate. SIFT's 4x4 spatial grid makes edge cells systematically
emptier than centre cells, so real per-dimension zero rates are structured. A model
hitting 0.2298 overall by spreading zeros uniformly scores perfectly on
`zero_fraction_gap` and fails here.

**`nnz_std_gap`** is the correlation probe. Under independent Bernoulli(0.23) gates,
the per-vector non-zero count would have standard deviation
`sqrt(128 * 0.23 * 0.77) ~= 4.8`. Real SIFT is expected well above this, because
empty gradient cells zero all 8 of their orientation bins together. One scalar
indicating whether the gate head learned block structure or only the marginal rate.
The real value is to be measured during implementation, not assumed.

Added to the existing eval block, all O(n*d).

## Testing

`pytest` is not in `requirements.txt` and must be added. Tests live in
`tests/test_generator.py`, CPU-only and fast.

**Invariants**
- `(x >= 0).all()` — non-negativity holds structurally
- norms are 1.0 within tolerance
- `(x == 0).any(dim=1).all()` using exact equality, confirming the gate is genuinely
  hard rather than merely small
- no all-zero row (the silent-corruption guard above)

**Gradient flow**
- after `forward(z).sum().backward()`, `gate_head.weight.grad` exists and is
  non-zero — catches a miswired straight-through estimator, which otherwise fails
  invisibly by just training badly
- same for `magnitude_head.weight.grad`

**Documented behaviour**
- two `forward` calls on identical `z` produce different gate patterns, pinning the
  "noise kept at sample time" decision

**Factory**
- `generator_type` absent resolves to `Generator` (backward-compat guarantee for the
  seven existing configs)
- `"sparse"` resolves to `SparseGenerator`; an unknown value raises

**Smoke test**
- a handful of training steps via `synthetic_if_missing: true`, asserting no NaN in
  the losses. The synthetic fallback generates Gaussian data
  (`sift1m_dataset.py:167`), which has negatives, so this verifies only that the
  loop runs end to end — not that anything is learned.

## Success criteria

Against a `bench_sparse` run compared to real in the `compare_100k` harness:

1. `negative_fraction` is exactly 0.0 (structural, must hold from step 0)
2. `exact_zero_fraction` within 0.02 of 0.2298
3. `per_dim_zero_rate_l1` below half the uniform-sparsity reference. That reference
   is computable from real data alone — it is the score a model would get by
   placing the correct overall zero rate uniformly across dimensions:

   ```python
   real_zero = (real == 0.0)
   uniform_ref = np.abs(real_zero.mean() - real_zero.mean(0)).mean()
   ```

   Compute `uniform_ref` once and record it in the run log, so the criterion is a
   fixed number rather than a judgement call.
4. `median_5nn_distance` and `effective_rank` no worse than `improved_100k`
   (0.5131 and 27.56) — the support fix must not cost geometry

Criterion 4 is the guard against trading one form of unrealism for another.

## Open items

- Branch base: this worktree is on `4a373c2`. PR #1
  (`experiment/wgan-improvements`) is still open. Verified against the code at
  `4a373c2`: `distance_reg_alpha` and the `num_workers` config key are already
  present; what the PR adds and this worktree lacks is **generator EMA and the
  collapse monitor** (`ema_decay`, `collapse_stats`, `fake_std` are all absent).
  All `train_wgan_gp.py` line references above are to the `4a373c2` version and
  will shift. Rebase once PR #1 merges.
- Best-checkpoint selection is deliberately left on `cov_fro`
  (`train_wgan_gp.py:309`), unchanged, so sparse runs stay comparable with existing
  ones. Note the consequence: selection ignores the new support metrics, so
  `best_generator.pt` could in principle be a checkpoint with good covariance and
  poor sparsity. The new metrics are recorded in `run_metadata.json` per eval, so
  this is visible after the fact. Revisit only if it actually occurs.
- Real-data `nnz` standard deviation is unmeasured — `data/sift_base.npy` is not
  present in this checkout. Measure before interpreting `nnz_std_gap`.
