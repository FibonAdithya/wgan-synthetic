# GloVe v0 baseline as a five-seed sweep

Date: 2026-08-10
Status: design, approved for planning
Base: `origin/main` at `2bd6274`

## Problem

GloVe has a measured real profile and no trained model. `docs/datasets/glove.md`
lists `v0` with an em-dash for its run and "not trained" for its status. The
next rung cannot be built on that: the ladder invariant is that each rung
differs from the one below by exactly one config key, which only buys
attribution if there is a run below to attribute against.

The obvious move — train `v0` once, as `deep` did — reproduces a defect that
family's own page documents. `docs/datasets/deep.md` reports two accidental
draws of its three rungs and finds that `v0`'s IVF Gini gap moved tenfold and
`v2`'s hubness gap halved under a change that should barely have bound. Its
conclusion is that the Gini column "should not be used to rank rungs at all",
and that setting bands "needs a real seed sweep". `docs/datasets/sift.md` says
the same thing with a number: three to five seeds are needed before any of its
noise-floor figures justifies a band.

GloVe has a sharper version of the problem, because its real-side noise floor
is already measured. Eight 20,000-row draws of the real corpus move hubness
skew across 3.46–8.33, 108% of the mean, against 0.50% for LID median. The
statistic the family's structural section names as the one GloVe is most likely
to fail, and most informative when it does, cannot be read at the canonical N.
A single `v0` run would tell us nothing about whether the remaining three
statistics survive on the synthetic side either.

So the first GloVe training work is not a rung. It is the measurement that says
which statistics this family can be judged on at all.

## Non-goals

- **Choosing or training the `v1` delta.** That decision is what this evidence
  feeds. Picking it now would mean picking it from prediction, which is the
  thing the sweep exists to avoid.
- **Setting any band in `gates/glove.yaml`.** `AGENTS.md` invariant 1 and the
  gate file's own header reserve band-setting for a human working from a full
  ladder. One rung is not a ladder. Every band stays null.
- **Editing `configs/glove/v0.yaml`.** It defines the rung. Changing what it
  points at would redefine what GloVe `v0` reproduces.
- **Phase (b) and phase (c).** The spherical generator is not built, and the
  angular re-measure moves two of the four statistics by a known argument. This
  sweep runs under L2 like every other family's.
- **Backfilling SIFT's noise floor** through the new module. Worth doing; not
  this change.

## Design

### The five runs

Seeds 42–46, one instrument config each, at
`configs/glove/v0_seed{42,43,44,45,46}.yaml`. Each differs from
`configs/glove/v0.yaml` in exactly three ways, listed in its header the way
`configs/sift/noisefloor_a.yaml` lists its own:

1. `seed` — the whole measurement.
2. `output_dir` — `runs/glove/v0_seed<N>`.
3. `real_path` — absolute, not `data/glove_250k.npy`.

The third needs its reason recorded, because it is not obvious and it makes
these files box-specific: the `gpuq` runner executes each job in a fresh
detached worktree cut from the pinned commit, and `data/` is gitignored, so a
relative path does not resolve at run time. That is acceptable in a measurement
instrument and would not be in a rung.

`configs/glove/v0.yaml` is never run and is not edited. It defines the
configuration; the five runs are five draws of it. Everything except the seed —
architecture, optimizer, corpus, preprocessing, step count — is identical
across all five and identical to the rung.

`latent_dim` stays at `128` over `descriptor_dim: 100`, and gains a comment
saying why. The value was inherited from `configs/sift/v1.yaml`, where 128 is
SIFT's descriptor dimension; `configs/deep/v0.yaml` corrected the same
inheritance down to its own 96 and called the inherited value a coincidence of
that family. That correction does not transfer. Deep's measured effective rank
is 65 of 96, so 96 was already generous; GloVe's is 94.6 of 100, so a latent at
or below the corpus rank would impose a bottleneck the corpus does not have.
Without the comment this reads as an oversight and invites a later "fix".

### Measurement

Each run's `best_generator.pt` is sampled at a **fixed** `--seed 42` for 50,000
vectors, matching the vector count in the committed real profile. Only the
training seed varies; the sampling seed is held so it cannot contribute to the
spread.

All five sample files then go through **one** `eda_report` invocation, using
its repeatable `--synthetic-path LABEL=PATH`, against `glove_250k.npy` at the
locked conditions: `--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10`,
`preprocess: l2`, `seed: 42`, `nlist: 256`. One invocation rather than five
makes the conditions identical by construction instead of by care, and puts all
six series — real plus five seeds — in a single `summary.json`.

### `src/eval/noise_floor.py`

The arithmetic that turns that `summary.json` into a floor, as a module rather
than a script. Both existing noise floors in this repo were computed by
throwaway scripts: `docs/datasets/glove.md` concedes in print that its angular
table "came from a one-off script that is not in this tree, so they cannot be
reproduced from a pinned commit", and SIFT's floor has the same hole. This is
the third time the same arithmetic is needed and the first time it is needed
again within the same ladder, so it stops being a script here.

Pure functions over the loaded dict — no GPU, no corpus, no file size that
matters:

- `summarize_spread(values)` returns mean, std, min, max, `range_pct_of_mean`
  and `cv_pct`, emitting the **same key names** as the committed
  `docs/datasets/glove_noise_floor.json`. The real-side and synthetic-side
  floors are then readable against each other without a translation step.
- Per statistic, two further fields the real-side file has no use for: distance
  from the real value, and that distance **in units of the seed-to-seed
  spread**. This is the column `docs/datasets/sift.md` makes decisive — it is
  what says whether a future `v1` improvement could be distinguished from a
  reseed at all.
- A CLI writing `docs/datasets/glove_v0_noise_floor.json`.

`tests/test_noise_floor.py` covers the spread arithmetic against hand-computed
inputs, the units-of-spread column, an error when a named series is absent from
the summary, and an error on a single-seed input, which has no spread to report.

The edge case worth naming: when every seed lands on the same value the spread
is zero and the units-of-spread divisor is zero with it. That case emits
`null`, not `inf` and not a crash. JSON has no infinity, and a reader who meets
`inf` there reads "infinitely well separated" when the truth is that the
separation is unmeasurable from this sample.

### What lands in the docs

- `docs/datasets/glove.md`: the `v0` ladder row filled with **the mean across
  five seeds and the min–max range** rather than a single number, listing the
  five run directories; and a `## Noise floor` section mirroring SIFT's,
  including the units-of-spread column. Range rather than standard deviation,
  because that is what `docs/datasets/glove_noise_floor.json` already reports
  on the real side and the two tables are meant to be read together.
- `docs/datasets/glove_v0_noise_floor.json`, committed, so the table is
  checkable without the training box — the standard the page already sets for
  its other two tables.
- `gates/glove.yaml`: bands unchanged and still null. If a second statistic
  proves noise-dominated on the synthetic side, it gains a warning comment
  beside the existing hubness one.

### The result this may produce

SIFT's floor found relative contrast and IVF Gini moved further under a reseed
than the generator's entire deviation from real. GloVe's real-side floor
already rules out hubness skew. If contrast is also noise-dominated here, the
honest finding is that **LID median is the only statistic GloVe can be gated
on**, and the page should say so.

That is a legitimate outcome of the measurement rather than a failure of it,
and it is worth writing down in advance — a result that narrows the gate is
easy to mistake for a result that went wrong, and the temptation on seeing it
is to widen bands until something passes.

## Success criteria

1. Five 30k-step `v0` runs complete, differing only in training seed.
2. One `summary.json` holds real plus all five synthetic series, measured
   together at the locked conditions.
3. `src/eval/noise_floor.py` and its tests are in the tree and green, and the
   committed JSON is reproducible from a pinned commit by running the module.
4. `docs/datasets/glove.md` reports `v0` as mean ± spread and carries a
   `## Noise floor` section with the units-of-spread column.
5. A stated, evidence-backed answer to which of the four statistics GloVe can
   be gated on.
6. `gates/glove.yaml` bands are still null.

## Cost

Five 30k-step runs. The comparable runs are ~34 minutes (SIFT) and ~35 minutes
(deep) on one RTX 4060; GloVe is 100-dimensional over 250,000 rows, and step
count rather than corpus size dominates, so the same order. Roughly three
GPU-hours in total.

Those three hours are **serial, not parallel**. The box has one RTX 4060 and
`gpuq`'s GPU lane is serialized box-wide, so the five runs queue behind one
another and behind anyone else's training. The sweep is independent work that
*could* fan out, and does not, which is the reason it costs an afternoon of
wall-clock rather than 35 minutes.

Two `gpuq` constraints shape how the jobs are submitted, and both silently
waste a run when missed. Artifacts may not be declared under `runs/`, because
that path is gitignored and the runner collects with `git add`, which errors on
an ignored path and marks the job failed with `exit_code: 0` — training
succeeds and the output is discarded. And each job executes in a fresh detached
worktree containing no gitignored files, so the corpus must be linked into
place by the job's own command. Both are handled by copying outputs to
`/workspace/` from inside the job command and declaring no artifacts.

The runner fetches from `origin`, so the branch must be pushed before any job
pinned to its commit can run.
