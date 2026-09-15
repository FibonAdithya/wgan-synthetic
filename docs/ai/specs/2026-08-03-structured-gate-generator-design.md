> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# Structured-Gate Generator and Local Log-Ratio Regularizer

**Date:** 2026-08-03
**Branch:** `worktree-gan+next-iteration`
**Status:** Design approved, pending implementation plan

## Problem

The ANN-difficulty panels merged in PR #4 produced the sharpest result the
project has. Measured at equal N (20k rows, k=100, L2-normalized):

| series | LID | rel. contrast | hubness | IVF gini | zero frac | eff. rank |
|---|---|---|---|---|---|---|
| **real** | **17.74** | **2.267** | **1.884** | **0.304** | **0.230** | **27.99** |
| v1_5 (mlp) | 17.39 | 2.261 | 2.128 | 0.313 | 0.000 | 27.53 |
| v2 (gated) | 14.81 | 2.507 | 1.747 | 0.296 | 0.153 | 27.64 |

Neither variant is usable as a benchmark stand-in. v1_5 matches LID to within
2% but emits no exact zeros at all. v2 is the only variant producing sparsity,
but undershoots it (0.153 vs 0.230) *and* is 17% short on LID. All three have
near-identical effective rank, so the global linear structure is right while
the local structure is not — precisely what the pre-existing metrics could not
see.

There is also a structural reason the objective cannot fix this on its own.
LID is a **local** property, but the critic is pointwise (`128 → 1`) and judges
each vector in isolation, and the v1_5 distance regularizer matches a single
global scalar (mean pairwise distance). A generator can satisfy both while
getting neighbourhood geometry wrong. Nothing in the current training signal
depends on local structure, so nothing pushes toward it.

## Evidence

Three read-only probes were run against the real `sift_base.npy`
(1M x 128, float32) on the GPU box, sampling 200k rows at seed 0. Probe
scripts are reproduced in Appendix A.

### Support size is 3x over-dispersed

```
exact_zero_fraction = 0.2301          (matches the documented 0.2298)
nnz per vector: mean 98.54, std 14.45, min 35, max 128
independent-gate equivalent std = sqrt(128 * p * (1-p)) = 4.76
```

v2's gates are independent per-coordinate Bernoullis, so its non-zero count is
Binomial(128, p) with std ~4.76 **by construction, at any parameter setting**.
Real SIFT is three times more variable. This is exactly the `nnz_std_gap: 6.83`
logged at the end of the v2 run, and it is not a tuning problem — it is an
expressiveness ceiling.

### Cell structure does not exist

The initial hypothesis was that whole 4x4 spatial cells go empty together, and
that gates should therefore be grouped in blocks of 8. The data refutes it:

```
whole-cell-empty rate (observed)    = 0.00035
whole-cell-empty rate (independent) = 0.00001
empty cells per vector: mean 0.006, max 4
```

Entire cells empty roughly 3 times in 10,000. More decisively, after removing
the leading eigenvector of the zero-indicator correlation matrix (the global
sparsity level), the same-cell residual correlation is **+0.0002** against an
all-pairs baseline of −0.0364. The apparent cell effect was entirely the global
level showing through. A block-structured cell gate would target a non-event.

### The real structure is local smoothness on the (4,4,8) grid

Residual correlation of the zero indicator (global level removed), by
dimension separation:

| \|i−j\| | 1 | 2 | 3–5 | 6 | 7 | 8 | 9 | 10 | >=16 |
|---|---|---|---|---|---|---|---|---|---|
| corr | **+0.317** | +0.075 | ~0 | +0.056 | +0.198 | **+0.275** | +0.163 | +0.050 | +0.059 |

Within a cell, by circular orientation distance: gap 1 → **+0.343**, gap 2 →
+0.081, gap 3 → −0.008, gap 4 → −0.015.

Two peaks, both interpretable:

- **Adjacent orientation bins** (+0.34, gone by gap 3). A gradient direction
  falling between two bins deposits in both — histogram smoothness.
- **Offset exactly 8** (+0.275, flanked by 7 and 9). The same orientation bin
  in the neighbouring spatial cell — spatial smoothness.

The peak sitting precisely at 8 independently confirms the layout is 4x4x8
with orientation varying fastest. The earlier "groups of 4" signal was an
artifact of adjacency, not blocks: at `|i−j| = 1`, same-group-of-4 scores
+0.344 and boundary-crossing scores +0.234 — both large.

The correlation is not low-rank-dominated (top eigenvalues 15.9, 10.2, 9.9,
then 4.0; rank 8 captures only 41% of correlation mass), so a small factor
model would not capture it either. The structure is local, not global.

## Design

### Generator: `StructuredGateGenerator` (`generator_type: structured_gated`)

Keeps v2's trunk, floored-softplus magnitude head, straight-through gradients
and unit-norm contract. Three additions, each targeting a measured defect:

1. **Per-vector sparsity level.** A scalar head `b(z)` added to every gate
   logit. Under v2, `p` is effectively fixed across vectors; a per-vector shift
   makes `nnz` a mixture of binomials, whose variance can reach the measured
   14.45. This is the minimum change that makes the target attainable.

2. **Neighbourhood-coupled logits.** Reshape the 128 gate logits to `(4,4,8)`
   and apply a single-channel 3-D convolution with a `3x3x3` kernel over
   `(row, col, orientation)`: **circular** padding along orientation (bin 7
   neighbours bin 0) and replicate padding across the 4x4 spatial grid. This
   produces correlation at `|i−j| = 1` and `= 8` by construction. The kernel is
   learned.

3. **Correlated gate noise.** The independence problem lives in the sampling,
   not only the logits: correlated logits with independent noise still sample
   near-independently. Draw white logistic noise on the `(4,4,8)` grid and
   smooth it before the binary-concrete transform.

   The noise kernel is **fixed and variance-normalized**, deliberately *not*
   the learned logit kernel of (2). A learned kernel applied to the noise would
   let the generator shrink it toward zero, killing gate stochasticity and
   collapsing the support distribution — the exact failure this design exists
   to prevent. Fixing the noise kernel keeps the injected correlation a
   structural property rather than something training can optimise away. Its
   width is set from the measured profile (+0.32 at separation 1, +0.27 at 8)
   and is a config constant, not a learned parameter.

The all-gates-off fallback reverts to v2's rare-event handling — there is no
cell gate that can zero a vector wholesale, so the pathological case is again
vanishingly rare at d=128.

New `model` config keys: `generator_type: structured_gated`, `layout: [4, 4, 8]`,
`gate_kernel: 3` (the learned logit kernel), `noise_kernel_sigma` (the fixed
noise smoothing width). `layout` is validated against `output_dim` at
construction: `prod(layout)` must equal `output_dim`.

### Critic: unchanged (rejected — see below)

The critic stays the existing pointwise `Critic`. An earlier draft of this spec
added a neighbour-aware critic; it was measured and cut. The negative result is
recorded below so it is not re-proposed.

### Generator regularizer: local log-ratio profile (the hedge)

v3 bets that the right architecture reaches the right local geometry without
being driven there. That bet is unhedged: with the neighbour-aware critic cut,
nothing in the objective sees local structure. **v4 hedges it** with an explicit
penalty, following the `distance_reg_alpha` pattern v1_5 already established.

**What is matched.** Within a minibatch, compute each point's `k` nearest
neighbours *among the other points in that batch*, giving sorted distances
`r_1 <= ... <= r_k`. The matched quantity is the mean log-ratio profile

```
    p_i = mean_over_batch[ log(r_i / r_k) ]  ,  i = 1 .. k-1
```

and the penalty is `L_lid = alpha * || p_fake - p_real ||_1`, added to the
generator loss.

**Why the log-ratio profile rather than LID itself.** `p` is exactly the
sufficient statistic the Hill estimator reduces to a scalar — `LID = -1 /
mean_i(p_i)` — so matching `p` moves LID directly. But `p` is bounded and
smooth, whereas LID's `-1/x` blows up as the mean log-ratio approaches zero.
Matching `p` also gives a `(k-1)`-vector target rather than v1_5's single
scalar, so it constrains the *shape* of the local neighbourhood, not just its
scale.

**Degenerate batches.** The failure modes are already characterised in
`src/eval/ann_difficulty.py`: `r_1 = 0` (duplicate samples, entirely plausible
under mode collapse) and `r_1 = r_k` (all neighbours tied). Rows failing
`survivor_mask`'s condition are dropped from the batch mean, and the penalty
contributes zero when no rows survive. The eval module's numpy implementation
is the reference for this logic; training needs a torch reimplementation
alongside it, not an import.

**Real-side target uses an EMA.** The real distribution is fixed, so estimating
`p_real` fresh from each minibatch only injects noise into the gradient.
Maintain an EMA of the real profile (decay 0.99) and use it as the target. Same
cost, lower variance.

**Batch-relative, not absolute.** `r_i` here are within-batch distances at
batch 512, far larger than true k-NN distances in a 1M set. That bias is
identical on both sides and cancels — the same equal-N discipline the ANN
panels already enforce.

Cost is one extra `512 x 512` distance matrix per generator step. Negligible.

New `training` config keys: `lid_reg_alpha` (default `0.0`, i.e. off — v3 runs
with it disabled), `lid_reg_k` (default 20), `lid_reg_max_points` (default 256,
mirroring `distance_reg_max_points`).

### Code seams

Two focused changes, following existing patterns:

- `src/models/generator.py` — add `StructuredGateGenerator`, register
  `structured_gated` in `build_generator`. `tests/test_generator_factory.py`
  already covers the dispatch.
- `src/train/train_wgan_gp.py` — add the log-ratio penalty to the generator
  loss beside the existing `distance_reg` term, and log it under `lid_reg` in
  the same metrics dict. Off by default, so v0-v3 behaviour is bit-identical.

`generate.py` and `evaluate_distribution.py` build only the generator, and
`critic.py` is untouched.

## Rejected: the neighbour-aware critic

### What was proposed

LID is a **local** property, but the critic is pointwise (`128 -> 1`) and judges
each vector in isolation, so nothing in the objective depends on local
structure. The proposal was to append, to the critic's input, each vector's
sorted distances to the `k` nearest members of a fixed reference buffer `R` of
real vectors — giving the adversarial signal a handle on local geometry while
keeping `D(x)` a function of `x` alone, so the gradient penalty stays
well-defined.

The buffer size (`reference_size: 2048`) and depth (`neighbour_k: 5`) were
guesses. Grounding them empirically killed the mechanism.

### Measurement

Queries were 4000 held-out real vectors versus 4000 vectors from the actual v2
run (`x100k_sparse_clamp4.npy`), scored by AUC from an L2-normalized logistic
probe. `tools/probes/reference_probe.py`, `critic_control.py`, `anchor_probe.py`.

| features | AUC (real vs v2) |
|---|---|
| **raw 128 dims — what the critic already sees** | **0.6291** |
| random buffer 2048, k=5 | 0.5329 |
| kmeans anchors 2048, k=5 | 0.5555 |
| random buffer 200000, k=5 | 0.6155 |
| raw + random 2048 | **0.6097** |
| raw + kmeans 2048 | 0.6344 |
| raw + random 65536 | 0.6476 |
| raw + random 200000 | 0.6824 |

### Why it was cut

1. **The proposed size was worse than nothing.** At `reference_size: 2048`,
   raw + neighbour features (0.6097) scored *below* raw alone (0.6291). The
   features are noise at that scale and dilute the signal the critic already
   has.

2. **Distance concentration caps the mechanism.** Median distance to the
   nearest reference point is 0.5354 at M=2048 and 0.4239 at M=200000 — a
   hundredfold increase in buffer size moves it 21%. For scale, the median
   5-NN distance within the *full 1M real set* is 0.5153. In 128 dimensions,
   distance-to-nearest-of-M is nearly independent of M, so a reference buffer
   cannot resolve local structure at any affordable size.

3. **The only setting with a real gain is unaffordable.** M=200000 buys +0.053
   AUC over raw, at 102M distance evaluations per critic call.

4. **The probe is a lower bound that flatters the mechanism.** These are linear
   probes; the real critic is a deep MLP that extracts more from the raw 128
   dims than 0.6291, which shrinks the neighbour features' marginal value
   further.

K-means anchors did work as hypothesised — 2048 centroids matched roughly 50-65k
random points — but the resulting gain over raw alone (+0.005) is within noise.

### A defect worth remembering

The design drew `R` from the training split, while real batches also come from
the training split. Batch points therefore land *in* `R` and score distance
exactly 0 — a free "this is real" tell the generator can never reproduce. At
batch 512 the expected rate is `512 * M / 950000`: **1.1 points per batch at
M=2048**, 8.8 at M=16384. Any future variant of this idea must draw the buffer
from the holdout split.

### What this leaves open

Cutting this leaves the original structural gap: **no part of the adversarial
objective sees local structure.** The gap is now addressed outside the critic,
by the log-ratio regularizer above — v3 tests the unhedged architectural bet,
v4 hedges it. What is *not* recovered is a local-structure signal inside the
adversarial game itself; the regularizer is an explicit penalty bolted beside
it, with the evidential cost described under Success criteria.

## GPU isolation and durability

The GPU box (`tig-gpu`) was probed directly. Findings that shape the plan:

- **One GPU.** A single RTX 4060, 8 GB, compute capability 8.9, torch
  2.12.0+cu130. Concurrent 100k-step runs must serialise; there is no second
  card to pin to.
- **Memory is not the binding constraint; throughput is.** Generator
  `[512,1024,1024]` and critic `[1024,512,256]` at batch 512 x 128 dims stay
  well under 2 GB even with the gradient penalty's double backward. Two runs
  would fit but each would run at roughly half speed.
- **Other agents share the box.** `/workspace` holds `wgan-sparse`
  (`worktree-sparse-generator`), `wgan-sparse-v2` (`sparse-magnitude-floor`),
  `ann-eda`, and a `wgan-synthetic` checkout stranded on
  `experiment/wgan-improvements` at `849981a` — roughly 15 commits behind
  merged `main`. Nothing was running at probe time, but the contention is real
  and recurring.
- **`workspace_is_volume` is `false`.** Nothing on the instance survives
  recycle or destroy: not the runs, not the checkpoints, not the 512 MB
  `sift_base.npy`. Disk itself is fine (112 GB free of 130 GB).

### Device claiming

Collapse the three duplicated `get_device` copies (`train_wgan_gp.py:34`,
`generate.py:14`, `evaluate_distribution.py:35`) into one shared
`src/train/device.py` helper — a targeted dedupe of code being touched anyway.
Today all three resolve `auto` to a bare `torch.device("cuda")`, i.e. `cuda:0`,
with no index, cap or coordination.

New behaviour: in **training**, `auto` is an error when CUDA is present but no
device was explicitly chosen and `CUDA_VISIBLE_DEVICES` is unset. Sampling and
eval keep today's permissive `auto` — they are short and read-only. Requiring
the long-running, resource-hungry path to make an explicit claim is the point.

### Lock

Exclusive by default; sharing is an explicit opt-in.

Key the lock on the GPU **UUID** from `torch.cuda.get_device_properties(...)`,
not on `cuda:N`. Two processes with different `CUDA_VISIBLE_DEVICES` mappings
both see their card as index 0, so an index-keyed lock would let them take
different locks for the same physical GPU — isolation that looks correct and
is not. Moot at one GPU, correct if a second appears.

Acquire with `fcntl.flock(LOCK_EX | LOCK_NB)` on `/tmp/wgan-gpu-<uuid>.lock`,
writing PID, run directory and start time into the file so a refusal can name
the holder. **On contention, wait with a configurable timeout, then give up** —
a queued arm can start itself overnight without supervision, and a stuck holder
still surfaces rather than blocking forever.

Limitation to state plainly: `flock` is advisory and host-local. It coordinates
cooperating processes on one machine and does nothing across hosts or against a
non-cooperating process. That matches the actual threat — other agents running
this same codebase.

Cap memory regardless via `torch.cuda.set_per_process_memory_fraction()` from
`training.gpu_memory_fraction` (default 0.9), so a bypassed lock degrades a run
instead of taking the card down. Log free/total memory and other compute
processes into `run_metadata.json` at start.

### Durability

`save_checkpoint` already persists `step`, generator, critic and **both
optimizer states** — everything a resume needs. There is simply no code path
that loads them back. Add `--resume <checkpoint>`, plus one format gap: persist
the **EMA shadow** (with `ema_decay: 0.999`, a resume that loses it silently
restarts the average).

Because the box is ephemeral, resume only helps if checkpoints leave it:

- Push code to git rather than editing on the box; refresh its stale checkout
  to merged `main` before anything runs.
- Sync each run's `summary.json`, logs and `run_metadata.json` off-box on the
  `eval_every` cadence — small files, cheap.
- Pull `best_generator.pt` back after each arm completes, not at campaign end.
- Treat `sift_base.npy` as reproducible, not precious —
  `data/sample_sift1m_100k.sh` re-fetches it.

## Implementation sequencing

This spec covers three separable bodies of work: the generator change, the
log-ratio regularizer, and the run infrastructure (device claiming, lock,
resume, off-box sync). They share no code and can be reviewed independently.
The regularizer is off by default, so it can land before v3 finishes without
affecting it.

The infrastructure lands **first**, as its own reviewable chunk. It is a
prerequisite for spending GPU time safely on a shared, ephemeral box, and it is
testable without a GPU — the lock, the device-resolution rules and the resume
path all exercise on CPU. The generator lands next, gated on a green infra
chunk, so v3 can start. The regularizer lands last and in parallel with v3's
run — it is inert at `lid_reg_alpha: 0.0` and cannot perturb an arm in flight.

## Experiment plan

Serialised on the single GPU, in dependency order:

| Arm | Config | Isolates | Runs after |
|---|---|---|---|
| **v3** | structured gate, `lid_reg_alpha: 0.0` | Does correlated, over-dispersed support fix sparsity — and does LID follow on its own? | — |
| **v4** | v3 + `lid_reg_alpha > 0` | If LID does not follow, can it be driven there — and does the rest of the local structure come with it? | v3 |
| *reserve* | seed repeat of whichever arm lands, or the sub-mechanism ablation v3 implicates | — | v4 |

Each arm is 100k generator steps at the v2 hyperparameters. v3 is exactly one
config change from v2 and v4 is exactly one from v3, so the
one-change-per-variant convention holds and v3-vs-v4 attributes cleanly to the
regularizer.

**v4 runs regardless of v3's outcome**, because it answers a different question
in each case. If v3 misses LID, v4 asks whether the metric is reachable at all.
If v3 *hits* LID unaided, v4 becomes the more interesting arm: it tests whether
the explicit penalty adds anything beyond what the architecture already
achieved, and a null result there is a strong endorsement of v3.

The reserve stays unassigned deliberately. v3 bundles two sub-mechanisms — the
per-vector sparsity level and the neighbourhood coupling — and if the result
lands between success and failure, the useful third arm is whichever one the
data implicates. Committing now would be guessing.

Because only one GPU is available, arms run serially. Two committed 100k arms
plus a reserve sit inside the 3-5 run budget.

## Success criteria

**Gates.** Measured through the merged ANN panels at equal N (20k rows, k=100):

- `lid_median` within 5% of real 17.74, i.e. **16.85–18.63**
- `exact_zero_fraction` in **0.21–0.25**

**Independent evidence** — reported, not gated:

- `nnz` std approaching the measured **14.45** (v2 sits near the binomial 4.76)
- residual zero-correlation ~**+0.32** at `|i−j| = 1` and ~**+0.27** at offset 8
- `hubness_skew` (real 1.884), `ivf_gini` (real 0.304), `relative_contrast`
  (real 2.267)

The correlation-profile targets are new and specific to this design. They are
the strongest available check that the generator reproduces SIFT's structure
rather than merely hitting a scalar.

### The two arms are not judged the same way

This matters, and it is the price of hedging.

**v3 trains on none of the above.** If it hits the LID gate, LID is genuine
independent evidence and the result is strong.

**v4 trains on LID's sufficient statistic.** Its `lid_median` is a *fitted*
number, not evidence — hitting the gate only demonstrates that the regularizer
works, which is close to tautological. So for v4 the gate above is necessary
but says almost nothing on its own, and the real question is what comes with
it: does driving the log-ratio profile to the real one also pull
`hubness_skew`, `ivf_gini` and `relative_contrast` toward real, or does the
model satisfy the penalty while the rest of the local structure stays wrong?

Those three are untouched by the regularizer and remain independent under both
arms, as do all the support statistics. Any write-up of a v4 result must lead
with them rather than with LID.

## Testing

Following `tests/test_generator.py` conventions:

- Unit-norm contract holds, including on the magnitude-floor and all-gates-off
  paths.
- Output is non-negative and does produce exact zeros.
- `layout` validation rejects an `output_dim` incompatible with `(4,4,8)`.
- The orientation axis wraps circularly: a kernel centred at bin 7 touches
  bin 0.
- Gate sampling stays stochastic in `eval()` — v2's documented contract.
- Determinism under a seeded global RNG.
- Statistical test: on random weights, the structured gate produces `nnz` std
  materially above the binomial-independent baseline. This is the single
  property the whole design exists to deliver, so it is tested directly rather
  than only observed after training.

For the log-ratio regularizer:

- The penalty is zero when fake and real batches are drawn from the same
  distribution, and grows as they diverge.
- Gradients flow to the generator through the within-batch neighbour distances.
- Degenerate batches are handled: an all-duplicate batch (`r_1 = 0`) and an
  all-tied batch (`r_1 = r_k`) both yield a finite penalty and finite
  gradients, never `inf` or `nan`. This mirrors the cases already covered in
  `tests/test_ann_difficulty.py`.
- With `lid_reg_alpha: 0.0` the generator loss is bit-identical to the current
  one, so v0-v3 are unaffected.
- The torch implementation agrees with `ann_difficulty`'s numpy reference on
  the same input, within float tolerance. This keeps the two from drifting.

The critic is unchanged, so its existing coverage stands as is.

For device handling:

- `auto` raises in training when CUDA is present and no explicit claim was
  made; still resolves permissively in sampling and eval.
- Lock acquisition, timeout, and holder reporting.

## Risks

- **The gate mechanism may fix sparsity without moving LID.** v3 rests entirely
  on the architecture reaching the right geometry without being driven there.
  This is now hedged rather than fatal: v4 exists precisely for this outcome.
  v3 still runs first, so the unhedged answer is on record before the
  regularizer muddies it.
- **The regularizer may satisfy the profile without fixing the geometry.**
  Matching a `(k-1)`-vector of mean log-ratios is a much weaker constraint than
  matching the local geometry itself; a model can hit it while `hubness_skew`
  and `ivf_gini` stay wrong. This is the specific thing v4's read-out is
  designed to detect, and a v4 that hits LID while missing everything else is
  an informative negative, not a failure of the experiment.
- **The regularizer may destabilise training.** Within-batch neighbour
  distances give a noisier gradient than v1_5's single scalar, and the penalty
  competes with the adversarial loss. The real-side EMA reduces variance;
  beyond that, `lid_reg_alpha` starts at v1_5's proven scale (0.1) and the
  logged `lid_reg` term is watched alongside `wasserstein` for divergence.
- **Correlated noise may reduce effective gate entropy**, collapsing support
  diversity. Fixing the noise kernel (above) removes the worst version — the
  generator cannot learn its way to zero noise — but smoothing still lowers the
  effective number of independent gates. The `nnz` std check catches this at
  smoke scale, before a long run.
- **The layout confirmation rests on one dataset.** The offset-8 peak is strong
  evidence for 4x4x8 on `sift_base.npy`, but the design should not be assumed
  transferable to another descriptor set without re-running Appendix A.

## Appendix A: probe scripts

Six read-only scripts, committed under `tools/probes/` so the evidence above is
reproducible rather than merely reported. All were run against
`/workspace/wgan-synthetic/data/sift_base.npy` at seed 0 — the structure probes
on 200k rows, the critic probes on 4000 queries per side. They hard-code that
path; repoint the constants at any local copy to re-run.

**Structure probes** (drove the generator design):

1. `layout_probe.py` — overall zero fraction, `nnz` mean/std vs the
   binomial-independent baseline, mean zero-indicator correlation by pair class
   (same cell, same orientation, groups of 4/16, adjacent vs distant cells),
   and observed vs independent whole-cell-empty rates.
2. `rank_probe.py` — eigenvalue spectrum of the 128x128 zero-indicator
   correlation matrix, and residual correlation by pair class after removing
   the leading eigenvector.
3. `distance_probe.py` — residual correlation vs `|i−j|`, the group-of-4
   boundary test at `|i−j| = 1`, and within-cell residual correlation vs
   circular orientation distance.

**Critic probes** (drove the rejection):

4. `reference_probe.py` — expected zero-distance leak per batch, feature
   stability across an independent buffer resample, and real-vs-v2 AUC by
   buffer size and neighbour depth.
5. `critic_control.py` — the decisive control: AUC from the raw 128 dims for
   comparison, buffer sizes to 200k, and median nearest-reference distance vs
   buffer size (the distance-concentration evidence).
6. `anchor_probe.py` — k-means centroids versus random samples at equal buffer
   size. Note: the `kmeans 16384` row was never obtained; the run was killed
   after its ssh connection dropped, and the cheaper rows had already settled
   the decision.
