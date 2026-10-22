# scout_kv v1: a generator that knows body from tail

Design, 2026-10-22. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built. Scope: the `v1`
rung of the `scout_kv` family. It builds on
`docs/ai/specs/2026-10-18-kv-cache-scout-design.md`, which defines the
family, its data split, its synthetic corpus and its gate. Read that first.

## Why

`v0` is a pooled generator: one unconditional WGAN-GP per global layer,
trained on every non-sink position of books 0-11, so keys and queries are
made from the same distribution of `h`. It fails four of the five gate
bands, all in the "too easy to search" direction (MEASURED,
`python -m src.eval.kv_gate check --records docs/results/scout-kv/v0/best.jsonl`,
run 2026-10-22):

| statistic | v0 best | v0 final | band |
|---|---|---|---|
| `hnsw_reach_k10` | 0.875 | 0.875 | 0.52-0.69 |
| `hnsw_recall_k10_ef176` | 0.9274 | 0.9309 | 0.78-0.88 |
| `ivf_ratio_k128` | 0.6841 | 0.6983 | 0.90-1.15 |
| `ivf_recall_k128_np24` | 0.8246 (pass) | 0.8246 (pass) | 0.80-0.865 |
| `hnsw_recall_k10_ef176_id` | 0.9757 | 0.9755 | >= 0.99 |

The real reference is not pooled. Its keys come from positions 1-31,743 of
each reference book and its queries (real-style and key-side) from the last
1,024 positions. Two controls located the difficulty in that split:

- **R1** (`docs/results/scout-kv/controls/`, PR #66, merged at b4deef6):
  real `h` with keys and queries drawn from one pool, no position, is outside
  the real range in v0's direction at seeds 0-2. For example the L43 IVF
  ratio is 1.12-1.15 pooled against 2.97-3.48 real. Much of v0's "too easy"
  verdict is the missing position split, not a bad fit to `h`.
- **Held-out-book control** (branch `scout-kv-cross-book`, commit 2590e70,
  no PR; `docs/results/scout-kv/controls/README.md` section "R6a/R6/R7"):
  keys exactly the reference's, queries from the tails of eight books from
  0-11. The positive control passes and the held-out tails flatter in 0 of 4
  draws: tail queries from books that are not in the key set are as hard as
  the reference. The difficulty comes from where in a book a query sits, not
  from sharing a book with its keys.

So the owner chose (2026-10-22) a position-aware generator as the `v1`
target, in its simplest form: a binary body/tail flag.

## What v1 is

The same per-layer setup as v0 (12 generators, one per global layer, each
producing the 5,120-dim attention input `h`, trained on books 0-11, keys and
queries made afterwards with the model's own `W_k`, `W_q`), with one change:
every training row and every sample carries a flag.

| flag | positions (T = 32,768, sink = position 0) | training rows per layer | gate use |
|---|---|---|---|
| body | 1-31,743 | 12 x 31,743 = 380,916 | keys |
| tail | 31,744-32,767 (the last 1,024) | 12 x 1,024 = 12,288 | real-style queries and key-side baseline queries |

The windows are exactly the reference corpus's (`src/kv/corpus.py`,
`mixed_real_corpus`: keys from `[n_sink, T - n_tail)`, queries from
`[T - n_tail, T)`, `n_tail` 1,024). The owner considered widening the tail
window for training to get more tail rows and decided against it: 12,288
rows per layer is accepted.

The rung's delta, for the ladder: **v0 + a body/tail position flag on
generator and critic**. Everything else starts from
`configs/scout_kv/v0/L{LL}.yaml` (architecture, 30,000 generator steps,
`n_critic` 3, learning rates, `lambda_gp` 5, EMA 0.999, preprocessing
`center: true`, `output_normalize: false`, seed 42), so that the flag is
the only difference from v0.

### Decided

- **Owner:** binary flag, not continuous position. It adds exactly the
  ingredient the controls found missing; the gate cannot see finer position,
  because reference keys are drawn evenly over the whole body.
- **Owner:** tail window = the last 1,024 positions, not widened.
- **My call, accepted by the owner:** one generator per layer that takes the
  flag (not two unconditional generators per layer). The 12,288 tail rows
  are too few on their own for a 5,120-dim WGAN; a shared generator lets the
  body rows shape the shared structure.
- **My call, accepted:** the critic sees the flag too, and tail rows are
  oversampled in each batch so the tail class is not ~3% of every batch.
- **My call, accepted:** synthetic corpora for the gate draw keys from body
  samples and both query sets from tail samples, with the same counts as v0
  (31,736 keys; 1,024 `h` for real-style queries, 5 query heads each; 1,024
  `h` for key-side baseline queries).

### Not in v1

- Continuous position, or any position finer than body/tail. If v1 moves the
  statistics but does not reach the bands, continuous position is a
  candidate for a later rung, not for v1.
- Values. The gate searches queries against keys; values never enter it.
- Anything past 32,768 tokens.
- Changing the gate, its bands or the reference. Bands are a human decision
  (`AGENTS.md`, "What requires a human"), and v1 is judged on the bands as
  set at a1c9870.

## What has to change in the code

Found by reading `origin/main` at b4deef6. Items here are requirements; how to
meet them is left to the plan unless stated.

1. **Loader.** `load_kv_capture` (`src/data/kv_capture.py`) drops the sink
   and stacks books, returning no positions. v1 needs each row's flag (or
   position, from which the flag follows). v0 configs must load exactly as
   before.
2. **Holdout.** v0 holds out 2% of rows at random (`holdout_fraction: 0.02`,
   7,864 rows). At that rate the tail class would get about 245 holdout rows
   (ESTIMATE, 0.02 x 12,288), too few for the selector's k = 100 statistics.
   The holdout must contain enough of each class for whatever the selector
   measures on it. See item 4.
3. **Generator and critic conditioning.** The current generator takes only
   noise; only `BankNeighbourhoodCritic` takes a population label
   (`src/models/critic.py`, `score_population`). How the flag enters both
   networks is left to the plan. Constraints:
   - an unconditioned config (every existing config, v0 included) builds the
     same networks and loads the same checkpoints as today;
   - the gradient penalty interpolates real and fake rows that carry the same
     flag, and scores the interpolate under that flag;
   - the checkpoint and `run_config.yaml` record that the model is
     flag-conditioned, the way they record `output_normalize`
     (fffd7ba), so a sampler cannot read a v1 checkpoint as unconditioned;
   - a resume that changes the conditioning is refused, as a resume that
     flips `output_normalize` is (11c2316).
4. **Checkpoint selection.** Correction to the brainstorm: `select_on: gate`
   does not build search corpora. `src/train/selection.py` scores LID median
   and relative contrast of a fake sample against the real holdout, on `h`,
   pooled. For v1 it must not ignore the flag. Comparing a pooled fake
   sample with a pooled holdout would select for the body (97% of rows) and
   say nothing about the tail. The plan chooses how (for example, score each
   class against its own holdout class and combine), and must say why the
   tail half of the score is not noise at the holdout size it picks. v0
   measured both `best_generator.pt` and the final checkpoint at B6; v1 does
   the same, so a bad selector cannot hide a good final checkpoint.
5. **Oversampling.** The tail share of each batch is left to the plan. State
   the chosen share and why. Whatever share is used, sampling for the gate
   is by flag, not by the training share.
6. **Gate sampler.** `kv_gate measure-run` samples `h_keys`, `h_query`,
   `h_id` from one unconditioned generator (`src/eval/kv_gate.py`, around
   `measure_h_layer`). For a flag-conditioned checkpoint it draws `h_keys`
   with the body flag and `h_query`, `h_id` with the tail flag. For a v0
   checkpoint its output must not change: v0's committed `best.jsonl` and
   `final.jsonl` are pinned by `tests/test_kv_gate.py`.
7. **Configs.** `configs/scout_kv/v1/L{LL}.yaml`, one per global layer,
   differing only in `data.kv_layer` and `output_dir`, as v0's are (pinned by
   `tests/test_scout_kv_configs.py`). Their only differences from v0 are the
   flag settings and anything the flag forces (holdout, selector, sampling).

Tests follow the project's rules (`make check`, mutation-checked new tests).
The two that matter most: a test that fails if the gate sampler draws keys
and queries under the same flag, and a test that fails if a v0 checkpoint's
sampled corpus changes.

## How v1 is judged

The five statistics over the 96 (layer, KV head) corpora, measured exactly as
v0's were (`kv_gate measure-run`, then `check` against `gates/scout_kv.yaml`).

**Success, as the owner put it:** v1 passes all five bands, or at least moves
every statistic toward real. "Toward real" is made checkable here, and this
definition is my call (revised after the builder check, which showed the
first version could not be met):

- **Target interval.** Pooled: the band in `gates/scout_kv.yaml`. Per layer:
  the min-max range of the five reference draws
  (`docs/results/scout-kv/reference/seed0.jsonl` ... `seed4.jsonl`).
- **L19 exception (owner's decision, 2026-10-22).** At L19 the training
  books are harder to search than the reference books: real books 0-7
  measure `hnsw_reach_k10` 0.375, `hnsw_recall_k10_ef176` 0.831,
  `ivf_ratio_k128` 0.369, against reference ranges 0.500-0.875, 0.854-0.882
  and 0.133-0.172 (`docs/results/scout-kv/controls/train_books_real.jsonl`,
  recomputed 2026-10-22). A v1 that learns its training data could overshoot
  the reference there. So at L19 only, each statistic's target interval
  stretches from the reference range out to the books 0-7 value (for
  example the IVF ratio target is 0.133-0.369). The pooled gate verdict is
  unaffected: it uses the bands as set. Not needed at L43, where books 0-7
  sit inside the reference range or at its edge (recall 0.641 against
  0.574-0.637).
- **Distance** = how far a value lies outside the target interval, on
  either side; 0 inside. Overshooting past the far edge is distance, not
  progress.
- **Spread** = max minus min of that statistic over the five reference draws,
  at the same level (pooled or per layer).
- **A statistic moves toward real** if v1 is inside the target interval, or
  v1's distance is smaller than v0's by more than the spread. The first
  clause is needed because in some cells v0's whole distance is smaller
  than the spread (MEASURED 2026-10-22 from the committed records: pooled
  `hnsw_recall_k10_ef176` v0 0.9274, band top 0.88, spread 0.0544; L19
  `hnsw_reach_k10` v0 1.000, range 0.500-0.875, spread 0.375; L19
  `ivf_ratio_k128` v0 0.0955, range 0.133-0.172, spread 0.0387).
- `ivf_recall_k128_np24`, which v0 already passes, must stay inside its band.
- **Key-side recall** has zero reference spread (1.000 in every draw at L19
  and L43), so the spread rule degenerates. For it, the noise allowance is
  the spread of v0's own value across the two `measure-run` sampling seeds
  (see the pilot), and v1 counts as moving if it rises by more than that.
- **Checkpoints.** v1 is judged per checkpoint kind: all-best or all-final,
  never mixed across layers. v1 meets a criterion if either kind meets it
  in full. The report shows both.
- Caveat to carry into the report: there is no v0 retrain spread for this
  family, and same-config retrains elsewhere moved statistics by more than
  the reference spread (NYTimes v0, 30k LID 18.0 to 13.8). A move of less
  than twice the spread is labelled borderline in the report. A borderline
  move still counts for the go/no-go, and the label is there so the owner
  sees it. The owner chose (2026-10-22) not to add a v0 retrain to the
  pilot to measure this. The two sampling seeds and the borderline label
  are the only noise controls.

## Pilot, then the full run

Training runs on `tig-gpu-alt` (L40 46 GB, 23-core cgroup quota), which
holds the capture (books 0-19 plus weights) and the v0 checkpoints. v0 took
2,463-2,756 s per layer on the A100 (MEASURED,
`docs/results/scout-kv/training/README.md`). The L40 time per layer is not
known: ESTIMATE (unverified) slower than the A100. Training needs
`OMP_NUM_THREADS=8` (measured during v0: torch's default thread count on a
cgroup-capped box is about 5x slower).

1. **Pilot (owner's decision): L19 and L43 only.** These are the layers where
   the controls showed the largest real-vs-pooled differences (L43 IVF ratio
   pooled 1.12-1.15 against real 2.97-3.48; held-out queries differed most at
   L19). Two training runs, no v0 retrain (owner's decision).
2. **Measure the pilot** with `kv_gate measure-run --layers 19,43` on best
   and final checkpoints, at two sampling seeds (`--seed 0` and `--seed 1`).
   v0's best checkpoint at the same two layers is measured at seed 1 too, so
   v0 also has two sampling seeds. Sampling seeds are cheap: no training. Then
   run `kv_gate compare-layers` against v0 and the five reference draws
   restricted to those layers. Per-layer views are diagnostics (spec of
   2026-10-18), so this is a go/no-go read, not a gate verdict. Per-layer
   `hnsw_reach_k10` is a share of 8 corpora and moves in steps of 0.125.
3. **Go/no-go (my call), using the rule in "How v1 is judged":** go to all
   12 layers if, at both L19 and L43, `ivf_ratio_k128`, `hnsw_reach_k10` and
   `hnsw_recall_k10_ef176` each move toward real, at both sampling seeds, for
   one checkpoint kind, and key-side recall is not lower than v0's by more
   than v0's sampling-seed spread. v0 fails the key-side band (0.9757
   pooled), so "not lower" is the pilot bar; moving toward >= 0.99 is the
   full-run bar. If only one layer moves, or nothing moves, stop and report
   to the owner before the full run.
4. **Record the L40 wall time per layer** from the pilot, and use it to cost
   the full run before starting it.
5. **Full run:** the other 10 layers, then `measure-run` on all 12 and
   `check`.
6. **Report:** `docs/results/scout-kv/v1/report.html`, one self-contained
   HTML file with base64-inlined figures, committed. It compares v1, v0 and
   the reference on the five statistics, pooled and per layer, and includes
   the intrinsic diagnostics on `h` that v0's report has
   (`docs/results/scout-kv/v0/intrinsic.jsonl`), computed per flag. The hub
   diagnostic gets its own figure: real `h` has hubs at layers 15-47 (skew
   38-47) that v0 lacks (4-14), and whether the flag brings them back is
   worth knowing either way.

Box transfers follow the standing rules: hash on the box, tar-stream the
minimal set, verify hashes locally.

## Left to the builder

- How the flag enters the generator and the critic (input concatenation, an
  embedding, a projection term, or other), within the constraints in item 3.
- The tail oversampling share (item 5) and the holdout split (item 2).
- The flag-aware selection score (item 4).
- Test layout, and whether the pilot runs through `gpuq` or a plain
  `nohup` script as v0's B5 did. Whatever runs it, the job's ID and log path
  come back before anyone waits on it.
