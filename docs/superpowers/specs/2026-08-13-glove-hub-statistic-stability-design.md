# Which hub statistic GloVe can be gated on

Date: 2026-08-13
Status: design, approved for planning
Base: `glove-gan-v1` at `48d3764` (PR #46)

## Problem

GloVe `v0` is trained at n=5 seeds and misses all four gate statistics, every
one in the direction that makes the synthetic set easier to search. The next
rung is the obvious next piece of work. It cannot be judged.

`docs/datasets/glove.md#structure` names hubness skew as the statistic this
family is most likely to fail and the most informative one when it does.
`docs/datasets/glove_noise_floor.json` then shows that at the locked canonical
N it does not measure the corpus, it measures the draw: eight 20,000-row draws
of the **real** corpus, changing nothing but which rows were drawn, span
3.4630–8.3308, a range of 108.2% of the mean, against 0.50% for LID median and
3.68% for IVF Gini.

Issue #29 records this and lists four fixes — raise N for the hubness pass,
average over draws, replace the statistic, or gate on three — and says each
touches something `AGENTS.md` reserves for a human. It has stayed open because
choosing between them from argument alone is guesswork: nobody has measured
whether hubness skew stabilises at larger N, and nobody has measured whether a
different hub statistic would be stabler.

So the work that unblocks `v1` is not a rung. It is the measurement that says
which statistic, at which N, can carry a hubness band at all.

`v1` is designed after this lands, against a gate that can judge it.

## Non-goals

- **Designing or training `v1`.** That is the work this unblocks. Choosing the
  rung now would mean choosing it against a gate that cannot read the family's
  characteristic property.
- **Setting any band value in `gates/glove.yaml`.** The gate file's header and
  `docs/datasets/glove.md#gate` both reserve band-setting for a human working
  from a trained ladder, and GloVe has one rung. Every band stays null. This
  change makes a statistic *gateable*; it does not gate it.
- **Rewriting the other five families' gates.** If a candidate statistic wins,
  it is added to all six gate files as a null band so the schema stays uniform,
  and nothing else about those families changes. Their own stability is
  unmeasured and stays that way; #29's "scope beyond GloVe" note remains open
  for `nytimes` and `openai`.
- **Re-measuring anything under angular distance.** Phase (c) is not this
  change. This sweep runs under L2 like every committed figure it has to be
  comparable with.
- **Making the torch backend the default.** Every number in `docs/datasets/`
  was produced by the sklearn path. The new backend is opt-in.

## Design

### The pre-registered rule

Fixed here, before the sweep runs, and not revisable after seeing numbers. A
hub statistic qualifies to carry a band at a given N when both hold:

1. **Stable.** Its real-side `range_pct_of_mean` across 16 draws is ≤ **10.0**.
2. **Discriminating.** `|mean(real draws) − mean(v0 seeds)|` is ≥ **1.0 ×**
   the real-side range (`max − min`).

Both constants come from precedent in the tree rather than from taste. GloVe's
three usable statistics sit at 0.32–3.68% range-of-mean against hubness skew's
108.2%, so a bar at 10% separates what the project already treats as gateable
from what it does not. `docs/datasets/sift.md` already bolds "noise exceeds
signal" below 1×, and `docs/datasets/glove.md#noise-floor` already computes
hubness's real-side ratio as 0.81× and 0.58× under exactly this arithmetic.

Both quantities are computed by `src/eval/noise_floor.py::summarize_spread`,
which is reused rather than reimplemented, so `range_pct_of_mean` here means
what it means in `glove_noise_floor.json`.

**The rule is evaluated on GloVe.** DEEP has no synthetic series in this study
and so contributes condition 1 only: whether a statistic's instability is a
property of the statistic or of a high-hubness corpus. It is evidence about
generality, not a second vote on GloVe's gate.

**Tie-break, also pre-registered.** Take the cheapest qualifying cell. If any
hub statistic qualifies at the locked N=20,000, adopt it and leave canonical
conditions alone. Only if none qualifies at 20,000 does a larger N come into
play. If nothing qualifies at any N, the answer is #29's fourth option.

A sweep that returns "none of the three qualify" is a result, not a failure.

### What is measured

Six statistics per draw, all off one k-NN pass, so the extra columns cost
nothing beyond the neighbour search that was happening anyway:

| | |
|---|---|
| Incumbents, as control | `lid_median`, `relative_contrast_median`, `hubness_skew`, `ivf_gini` |
| Candidates | `hubness_gini`, `hub_share_top1pct` |

`hubness_gini` is the Gini coefficient of the k-occurrence counts. It reuses
the `gini()` helper `ivf_gini` already uses, is bounded in [0, 1), and is not a
third moment, so no handful of tail hubs can set it.

`hub_share_top1pct` is the fraction of all neighbour slots taken by the top 1%
of points — the measure #29 names first. It reads directly as how much of the
neighbour traffic funnels into hubs, and is bounded in [0.01, 1].

Carrying the four incumbents along re-measures the committed eight-draw table
at 16 draws and at larger N for free.

### The grid

| axis | values |
|---|---|
| corpus | `glove_1m.npy`, `deep_1m.npy` |
| N | 20,000 / 50,000 / 100,000 / 250,000 |
| draws per cell | 16 |
| preprocess | `l2`, applied identically to every series |

**The normalisation is a condition, not a detail.** Every figure committed
under `docs/datasets/` for these families was measured at `preprocess: l2`,
and a measurement taken at any other setting is not comparable with them. It
also has to be applied to *every* series or the study measures the wrong
thing: the real corpora are stored raw — `glove_250k.npy`'s row norms span
2.1658 to 11.3325 — while generator samples come out at exactly 1.0. Comparing
those two as stored would put a units mismatch into condition 2 and read it as
a generator deficit.

This was found the hard way. The first provenance run measured the vectors as
stored and returned a hubness skew of 37.0 against the committed 4.4976, which
is what the provenance cell exists to catch.

Two corpora, because one cannot separate "this statistic is unstable" from
"this corpus is pathological". GloVe's hubness skew is ~4.5; DEEP's is 1.94, a
much less skewed k-occurrence distribution. That is the difference between
"replace the statistic everywhere" and "raise N for GloVe", and it is the whole
reason to pay for a second corpus.

Draws come from the 1M files rather than `glove_250k.npy`, because subsample
spread only exists when the pool exceeds N: 16 "draws" of 250,000 rows from a
250,000-row file are one draw repeated, with a spread of zero by construction.

**One extra cell, for provenance.** N=20,000 from `glove_250k.npy`, the pool
and size the committed eight draws used — and at eight draws, not sixteen, so
the comparison is like-for-like and the draws stay disjoint inside a 250,000
row pool. Its numbers must land inside `glove_noise_floor.json`'s ranges
before anything else in the sweep is believed. It doubles as a measurement of
whether the pool change matters, and as the only check that the torch backend
agrees with the sklearn path on real data at scale — the unit tests can only
compare the two on CPU, since the development machine has no card.

Sixteen draws rather than eight because the card makes it cheap and because
#29's own complaint is that eight is thin for estimating a spread.

**The large-N cells measure overlapping draws, and that biases them.** Sixteen
draws fit inside a 1M pool without overlap up to N=62,500: at 20,000 and 50,000
the draws are disjoint, and the harness allocates them that way. At 100,000 the
sixteen draws need 1.6M rows from a 1M pool, and at 250,000 they need 4M — so
they overlap heavily, share rows, and their spread is therefore a **lower
bound** on the true subsample spread. This biases the sweep toward finding
statistics stable at large N, which is the direction that would wrongly favour
#29's "raise N" fix. GloVe's upstream file is ~1.18M vectors, so a larger pool
is not available.

Pre-registered consequence: the harness records the pool-to-N ratio and a
`draws_disjoint` flag per cell, and a statistic that qualifies **only** at
N=100,000 or 250,000 is reported as `provisional` rather than `qualified`. A
provisional result is not sufficient to move canonical N; it is grounds for
saying so and stopping.

### `v0` has to be re-sampled

Condition 2 needs `v0` measured at each N. `v0`'s committed samples are 50,000
vectors per seed, which cannot support N=100,000 or 250,000.

So the study re-samples all five `v0` checkpoints to 250,000 vectors at the
fixed sampling seed of 42 — a generator forward pass, minutes on the card. This
is a **new measurement of `v0`**, not the one behind
`docs/datasets/glove_v0_noise_floor.json`, and the write-up says so rather than
splicing the two into one table. At N=20,000 the two should agree closely, and
a note recording whether they do is part of the result.

### `src/eval/ann_difficulty.py`: a torch neighbour backend

`knn(x, k, backend="sklearn")` gains a `"torch"` backend. The default is
unchanged, so every existing committed number keeps its provenance and every
existing caller keeps its behaviour.

The self-exclusion logic is extracted into one `_exclude_self()` helper that
both backends call. That logic is the subtle part of the current function — its
docstring explains that exact duplicate rows tie with the query at distance 0
and sklearn does not promise the query sorts first, so dropping the first
column can silently leave a point in its own neighbour list, dragging its
k-occurrence up and its LID down. Two copies of that reasoning would drift.

The torch path moves the corpus to CUDA when available and falls back to CPU
torch when not, chunks rows at a configurable width (1024 by default — a
1024 × 250,000 float32 block is 1 GB, comfortable beside a 100 MB corpus on an
8 GB card), and runs `cdist` + `topk` per chunk.

`ann_metrics` already casts to float32 before calling `knn`, so the two
backends differ in accumulation, not in input precision. sklearn's brute
euclidean uses the `‖x‖² + ‖y‖² − 2xy` expansion, which is the numerically
worse form; the torch numbers may well be the better ones. The equivalence test
below is what decides whether the difference is defensible, not this argument.

### `src/eval/ann_difficulty.py`: the two candidate statistics

`hubness_gini(counts)` delegates to `gini()`, which sorts internally, so
unsorted k-occurrence counts are fine to pass.

`hub_share_top1pct(counts)` sums the largest `ceil(0.01 n)` counts over the
total.

Both land in this change. **Neither is added to `summary()` yet.** Adding a key
to `summary()` changes `summary.json` for every family and every report, and
should happen for the statistic that wins, in phase 2, and not for the two that
were tried.

### `src/eval/hub_stability.py`: the sweep harness

A CLI in the shape of `src/eval/noise_floor.py`: importable functions with a
thin `main()`, no plotly, no dependency on `src.eval.eda`.

    python -m src.eval.hub_stability \
        --real-path data/glove_1m.npy \
        --synthetic-path v0_seed42=runs/glove/v0_seed42/samples_250k.npy \
        ... \
        --n 20000 --n 50000 --n 100000 --n 250000 \
        --draws 16 --k 100 --k-hub 10 --nlist 256 \
        --backend torch \
        --output docs/datasets/glove_hub_stability.json

For each (series, N, draw) it subsamples at `seed + draw_index`, runs one k-NN
pass, and computes all six statistics from it. The output JSON holds:

- every raw per-draw value, so the arithmetic is checkable without the box;
- the `summarize_spread` block per (series, N, statistic);
- the pool size, pool-to-N ratio and `draws_disjoint` flag per cell;
- **the qualification verdict per (N, statistic), computed by the tool** under
  the rule above — `qualified`, `provisional` or `rejected`.

One invocation per corpus, writing `docs/datasets/glove_hub_stability.json` and
`docs/datasets/deep_hub_stability.json`.

The verdict is in the artifact rather than in a reader's summary of a table,
because the point of pre-registering the rule is that applying it is mechanical.

### Phase 2: applying the winner

Named now, all three shapes, so the implementation plan never improvises.

1. **A candidate qualifies at N=20,000.** Add it to `summary()`; add a null
   band for it to all six gate files; keep `hubness_skew` reported as a
   diagnostic, because withdrawing a measurement is not the same as
   withdrawing a band; rewrite `docs/datasets/glove.md`'s two hubness sections
   and the `hubness_skew` comment in `gates/glove.yaml`; close #29. Canonical
   conditions are untouched. This is the cheap branch.
2. **Nothing qualifies at 20,000; something qualifies higher.** Split the gate
   file's `canonical` block into `n` and `n_hub`; add `--ann-hub-max-rows` to
   `eda_report`; teach `check_gate.check_conditions` to compare hub statistics
   against the hub conditions and the rest against the main ones; plus all of
   (1)'s documentation work. This touches `AGENTS.md` invariant 3 and is the
   expensive branch.
3. **Nothing qualifies at any N.** `gates/glove.yaml` and
   `docs/datasets/glove.md` record a three-statistic gate as a measured
   decision with the table behind it; #29 closes as measured-and-accepted.

Which shape runs is decided by the committed JSON, and the pivot is a human
decision point in the plan, not an automatic branch.

### Tests

- **Backend equivalence**, the one that has to be genuinely convincing. On a
  fixture: torch and sklearn return identical neighbour indices except where
  the distance gap to the next candidate falls inside tolerance; distances
  agree to a stated relative tolerance; and all four incumbent summary
  statistics agree to a stated tolerance. Runs on CPU torch, so CI covers it
  without a card.
- **Draw determinism.** The same seed yields the same subsamples and the same
  per-draw values.
- **Draw allocation.** Draws are disjoint when `draws × N ≤ pool`, and the
  `draws_disjoint` flag reports it honestly when they are not.
- **The two new statistics**, against constructed count vectors with known
  answers: uniform counts give Gini 0 and a top-1% share of 0.01; a single
  point holding every slot gives Gini →(n−1)/n and a share →1.0.
- **Rule evaluation**, including the exact boundaries — a cell at exactly
  10.0% range-of-mean and exactly 1.0× separation qualifies, since both bounds
  are inclusive.
- **Harness smoke test** on a small random array, one N, three draws.

## What the provenance cell found

Recorded here because it changes how the committed floor should be read, and
because the strict form of the check did not pass.

**Three of four statistics reproduced; LID median missed by 0.28%.** Against
`glove_noise_floor.json`, relative contrast (−1.46 SE), hubness skew (+0.42 SE)
and IVF Gini (−0.36 SE) all landed inside the committed ranges. LID median came
in at 35.2206 against a committed mean of 35.1238 and a committed max of
35.2086 — outside by 0.012, which is 3.06 standard errors and 0.28% of the
mean.

**It is not the torch backend.** The same cell was re-run with
`--backend sklearn` and the two agree to 0.000% on every one of the six
statistics; `hub_share_top1pct` is bit-identical across all eight draws, and
the worst per-draw disagreement anywhere is 8.7e-05 on LID median. Both
artifacts are committed (`glove_hub_stability_provenance.json`,
`glove_hub_stability_sklearn_control.json`). This is also the only check that
the GPU path agrees with sklearn on real data at scale — the unit tests can
only compare the two on CPU, since the development machine has no card.

**It is the draws, and the cause is a two-stage subsample.** `eda_report`
reduces a corpus in two steps: `subsample` to `--max-vectors` (default 50,000,
seed 42) in `eda/series.py`, and then `_subsample` again to `--ann-max-rows`
(default 20,000) inside `ann_difficulty.compute`. Every draw behind
`glove_noise_floor.json` is therefore 20,000 rows taken from the *same*
50,000-row slice of the corpus — 40% of that slice per draw, all eight
confined to it. This sweep draws from the whole file.

So the two numbers estimate different things. The committed mean estimates LID
for one particular 50,000-row subset; this sweep's estimates it for the corpus.
A 0.28% gap between those is unremarkable, and no verdict in this study turns
on it: LID's own draw-to-draw range here is 0.63%, and `v0` sits at 16.4
against real's 35.2, a factor of two away.

**The consequence for issue #29 is the part worth keeping.** #29's headline —
hubness skew's range being 108% of its mean — was measured in the overlapping,
single-slice regime described above, which understates spread for exactly the
reason `allocate_draws` flags. Measured with eight disjoint draws from the full
250,000, the range is 95.62%. The qualitative claim survives; it is now
measured on a wider pool with independent draws, and the sweep proper measures
it wider still.

## Success criteria

- The provenance cell at N=20,000 from `glove_250k.npy` reproduces
  `glove_noise_floor.json`'s ranges. If it does not, the sweep stops and the
  discrepancy is the finding. **Outcome: three of four reproduced, LID median
  missed by 0.28%, diagnosed as a two-stage-subsample difference rather than a
  code defect — see `## What the provenance cell found`.**
- `docs/datasets/glove_hub_stability.json` and
  `docs/datasets/deep_hub_stability.json` are committed, hold every raw draw,
  and their verdicts can be recomputed from their own contents.
- `docs/datasets/glove.md` gains a section reporting the sweep, and its two
  existing hubness sections are corrected to match rather than left to
  contradict it.
- One of the three phase-2 shapes is implemented and #29 is closed.
- `make check` passes and every band in `gates/glove.yaml` is still null.

## Cost

No training. Two GPU-lane jobs:

1. Re-sample five `v0` checkpoints to 250,000 vectors each — minutes.
2. The sweep — two corpora × four N × 16 draws, plus the provenance cell.
   Dominated by the N=250,000 cells; under two hours in total.

If the torch backend cannot be defended against sklearn, the fallback is the
same sweep on the `cpu` lane at 20,000 / 50,000 / 100,000, roughly four hours,
which does not block anyone's training. Slower, not blocked.
