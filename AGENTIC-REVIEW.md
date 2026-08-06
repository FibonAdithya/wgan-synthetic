# wgan-synthetic: agentic-engineering review

Date: 2026-08-05
Reviewed at: `47a7829` on `infra/parallel-dataset-runner`
Reviewer: agent, reading the repo cold. Tests were run; no training was run.

Written to be handed to Dan. It is a review of *agent-readiness*, not of the
research — the modelling choices are outside what a cold read can judge.

## Note for later readers: this was reviewed against a stale base

Added after the fact. `47a7829` on `infra/parallel-dataset-runner` is 19
commits behind `main`, so the review's counts and file inventory describe that
commit, not the project. Specifically:

- The suite is **211 tests** on `main`, not the 139 counted here.
- `tests/conftest.py` **exists** on `main` — 105 lines, added by
  "refactor(tests): promote run-dir fixtures to conftest". The review's premise
  that there is no shared test fixture module is false there.
- `main` also carries the descriptor-glyph work (`src/eval/descriptor_glyph.py`,
  `src/eval/plot_descriptor_grid.py` and their tests) and three `FOLLOWUPS.md`
  entries the review never saw.

The substantive findings below were re-checked against `main` and still hold:
no `AGENTS.md`, no CI, a gate that is documented but not executable, an
untested sampling path, and hard-coded run directories. Treat the numbers as a
snapshot; treat the conclusions as current.

## Summary

This repo is already doing most of what the imported AI-first workflow document
recommends for documentation and planning, and in places doing it better than
that document describes. That document is vendored at
`docs/ai-first-development-workflow.md`, and every reference to "the imported
document" below means that file. The gaps are concentrated almost entirely in
**executable feedback**: there is a great deal of prose telling a careful
reader what correct means, and almost nothing a machine can run to find out.

That matters more here than in a typical repo, because the next step on the
roadmap is six agents training six dataset families in parallel. An agent that
cannot check its own work will produce six ladders nobody can trust.

## What is genuinely strong

Worth stating plainly, both because it's true and because the recommendations
below are small next to it.

**The source-of-truth hierarchy is explicit and enforced socially.**
`README.md` names the documentation map; `docs/superpowers/README.md` states
that AI-generated specs and plans are *not* authoritative and that
`PROJECT_DOCUMENTATION.md` wins on conflict. §4 of the imported document spends
a page arguing for exactly this, and describes it as the thing teams get wrong.
It's already here.

**`FOLLOWUPS.md` is the best artifact in the repo.** Every entry has a location,
a reproduction, a consequence, and a proposed fix, and several explicitly say
"nothing regresses, but a reader would be surprised." That is the evidence
format §8 of the imported document asks AI reviewers to produce — being
maintained by hand, well, already.

**The gate/diagnostic distinction is a research-grade acceptance criterion.**
Deciding in advance that ANN-difficulty parity is the gate and that
`mmd_rbf`/`cov_fro`/`pairwise_hist_l1` are diagnostics — *with the argument for
why* (the metrics measure the bulk of the distance distribution; difficulty
lives in the far-left tail; no symmetric two-sample statistic constrains
hubness) — is the strongest thing in the repo. Most ML projects never write
down what would count as failure.

**The ladder discipline** — one config change per rung, variant numbers
comparable only within a family — is a real experimental design, not a naming
convention.

**139 tests pass in seconds.** Specs and plans preserve reasoning that isn't
recoverable from diffs.

## Gaps, in priority order

### 1. There is no `AGENTS.md` or `CLAUDE.md`. Anywhere.

The single biggest gap, and the cheapest to close.

All the context an agent needs exists — but it is spread across `README.md`,
`PROJECT_DOCUMENTATION.md`, six dataset pages, `data/README.md`, and
`FOLLOWUPS.md`, with nothing at the entry point saying which to read, in what
order, or what the non-negotiables are. Every session rediscovers it, and a
session that doesn't will get things wrong.

This repo has an unusually high density of **invariants that are silent until
violated**:

- ANN-difficulty is the gate; the distributional metrics are diagnostics. An
  agent that reports "MMD improved, looks good" has misunderstood the project.
- Variant numbers are per-family. SIFT `v2` and a future GIST `v2` are
  unrelated. Cross-family comparison is meaningless.
- Canonical N and k are locked per dataset, and the statistics are
  self-queried subsample figures **not comparable with published benchmarks**.
  An agent will be tempted to compare against literature values.
- A checkpoint is only loadable beside its `run_config.yaml` —
  `generator_type` is not recorded in the checkpoint.
- `data/sift_base.npy` (what the SIFT checkpoints trained on) is a different
  corpus from `data/sift_250k.npy` (what the fetcher produces).

Every one of those is currently a prose sentence somewhere. An agent that
misses any of them produces confidently wrong results that look fine.

**Fix:** one `AGENTS.md` at root. A router, not an encyclopedia — the good
material already exists and should be linked, not restated. Purpose, the
source-of-truth order, the five invariants above, the commands that define a
valid change, and what requires a human. Half a day, and it makes every
subsequent agent session better.

### 2. No executable definition of "a valid change"

There is no `Makefile`, no `pyproject.toml`, no lint, format, or type
configuration, and **no CI at all** (no `.github/`). `pytest.ini` exists and the
suite passes, but an agent has to already know to run it, and nothing else is
checked by anything.

The binding constraint on AI-assisted work is verification capacity rather than
generation capacity: an agent can produce far more change than anything here
can currently check. This repo has one verification layer (tests, manually
invoked) and no gate. That is the argument the imported document makes for
cheap deterministic feedback in its §6, and it is the reason this gap ranks
above the rest of the tooling.

**Fix:** a `Makefile` with `check` running format/lint/tests, wired into a
single GitHub Actions workflow. Then name it in `AGENTS.md` as the definition of
done. Note that `|| true` is explicitly called out as an anti-pattern in the
imported doc's §6 — worth getting right the first time, because it is much
harder to re-tighten later.

### 3. Determinism is not enforced, and this project's central claim depends on it

`set_seed` (`src/train/train_wgan_gp.py:27`) seeds `random`, `numpy`,
`torch`, and `torch.cuda`. It does **not** set
`torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.deterministic`.

Separately, the `DataLoader` (`src/train/train_wgan_gp.py:341`) uses
`shuffle=True` with configurable `num_workers` and passes neither a
`generator=` nor a `worker_init_fn`. Shuffle order is therefore not reproducible
across worker counts, so the same config on the same seed can produce different
runs on two machines configured differently.

Why this matters more here than elsewhere: **the ladder's validity rests on
attribution.** The design states each rung is exactly one config change from the
one above, so a difference visible in an overlay attributes to a single cause.
That inference is only sound if run-to-run variation is small relative to the
effect being attributed — and nobody has measured whether it is.

**The missing experiment is more valuable than any tooling in this document:**
train `configs/sift/v0.yaml` twice with different seeds, run the gate on both,
and record the spread on all four statistics. That number is the noise floor.
Without it, no gate band is interpretable and no ladder comparison is safe. It
should be measured before six agents start producing five more ladders.

Recommend recording it per family in each `docs/datasets/` page next to the gate
bands, and treating any band tighter than the noise floor as unset.

### 4. The gate is documented in prose but is not executable

The gate is four statistics with per-family relative bands recorded in
`docs/datasets/*.md`. There is no command that reads a run and returns
pass/fail. An agent that trains a rung cannot determine whether it succeeded
without a human reading an HTML report.

This is the highest-leverage change for the parallel-agent plan specifically.
Six agents each need a machine-readable verdict, for exactly the reason §8 of
the imported document gives for review verdicts: a human reading prose does not
scale, and grepping prose for a pass string is fragile.

**Fix:** move the bands out of prose into per-dataset config (or a small
`gates/<dataset>.yaml`), and add `python -m src.eval.check_gate --run <dir>`
emitting JSON — the four statistics, the bands, per-statistic pass/fail, and a
non-zero exit on failure. The dataset pages then reference the machine-readable
source rather than duplicating it.

This also closes the imported doc's §5 point about acceptance criteria being an
executable contract. This repo has unusually well-specified acceptance criteria;
they're just not runnable yet.

### 5. Test coverage is uneven along the axis that matters

139 tests is real coverage, but the gaps are not random:

| Module | LOC | Tests |
|---|---|---|
| `src/sample/generate.py` | 86 | **none** |
| `src/eval/evaluate_file_to_file.py` | 100 | **none** |
| `src/eval/plot_*.py` (3 files) | 522 | **none** |
| `src/models/critic.py` | 26 | **none direct** |

`src/sample/generate.py` is untested and it is the module that produces the
project's primary deliverable. The plotting modules matter less — visual output,
low blast radius — but sampling correctness is load-bearing.

`eda_report.py`, formerly 1017 lines and the file most likely to be edited by
an agent and the hardest for one to hold in context, has since been split
into the `src/eval/eda/` package (`894e2d3..c88ded8`, 2026-08-05, see
`docs/superpowers/specs/2026-08-05-eda-report-split-design.md`); the
entrypoint module is now 19 lines.

### 6. A fresh clone cannot reproduce the headline comparison

`runs/` is gitignored, and `compare_variants.py` hard-codes historical run
directories (`long_baseline`, `x100k_improved`, `x100k_sparse_clamp4`, …) that
do not exist in a fresh checkout — noted already in `FOLLOWUPS.md`. Combined
with the known `sift_base.npy` / fetcher mismatch, the documented quick start
fails on a clean machine.

Human readers work around this. **Autonomous agents do not** — a failing quick
start is where an agent burns a session or, worse, silently trains against the
wrong corpus. The two SIFT config issues in `FOLLOWUPS.md` are correctly
identified as needing a decision rather than a patch; that decision is now
blocking agent-run onboarding and is worth making before the parallel work
starts.

### 7. Local worktree and branch sprawl

Six worktrees under `.claude/worktrees/`, all at `6160643`, plus nine-plus
branches and a stray worktree in `/tmp`. The `2026-08-05-parallel-dataset-infra`
spec correctly diagnoses this pattern on the GPU box (10.8GB across four
checkouts, three on stale branches) and designs against it there. Nothing yet
addresses the same pattern locally.

The imported doc's "give every state one writer" (§9) is the relevant principle.
Worth a cleanup convention before six agents start, not after.

## What I would do, in order

1. **`AGENTS.md`.** Half a day. Router plus the five silent invariants.
2. **Measure the seed-to-seed noise floor on SIFT `v0`.** One GPU day, mostly
   waiting. Blocks meaningful interpretation of every gate band.
3. **`make check` + one CI workflow.** Half a day.
4. **`check_gate` with JSON output and an exit code.** A day or two. Prerequisite
   for trustworthy parallel agent work.
5. **Resolve the two SIFT config follow-ups** so the quick start works on a
   clean machine.
6. **Tests for `src/sample/generate.py`.**
7. Determinism flags — but see the caveat below.

Items 1–4 are what turns this from a well-documented repo into one an agent can
work in unsupervised. None of them is large.

## One caveat on the determinism recommendation

`torch.use_deterministic_algorithms(True)` has a real throughput cost and some
CUDA kernels have no deterministic implementation, so it can hard-error rather
than degrade. On a single shared RTX 4060 that cost is not free.

The recommendation is therefore **measure the noise floor first** (item 2) and
only enforce determinism if the spread turns out to be large enough to threaten
ladder attribution. Enforcing it blindly could slow every training run for no
gain. The `DataLoader` `generator=`/`worker_init_fn` fix is cheap and worth
doing regardless.

## Open questions for Dan

- Was the seed-to-seed spread ever measured on SIFT? If it was and I missed it,
  items 2 and 7 collapse.
- Are the per-family gate bands intended to stay in the dataset pages, or is
  moving them to config acceptable? Item 4's design depends on the answer.
- Is CI absent by decision (GPU-dependent tests, cost) or just not done yet? The
  139 tests appear to run CPU-only in seconds, which suggests CI is cheap here.
