# scout_kv v2: books with a shared latent

Design, 2026-10-28. Approved by the owner 2026-10-28. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built. Scope: the `v2`
rung of the `scout_kv` family. Read first:
`docs/ai/specs/2026-10-18-kv-cache-scout-design.md` (the family, data split,
synthetic corpus and gate) and
`docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md` (v1, the
body/tail flag). The evidence for every choice below is in
`docs/results/scout-kv/v2-probe/README.md` (branch `scout-kv-v2-probe`), which
has a claim table per run. Numbers quoted here are MEASURED there unless
labelled ESTIMATE.

## Why

The owner's aim for the family: a generator realistic enough to produce
synthetic KV-cache search problems, close to real both in search difficulty
and in the distribution of the vectors.

v1 draws every token independently given a body/tail flag. It fails four of
five gate bands (PR #67), and its tokens have no relation to each other:
the cosine between two v1 tokens is 0.00 at any distance. Real books are
different. A real token's nearest neighbours mostly come from its own book
(the "same-book share": 0.41 at L3 rising to 0.90 at L27, against 0.125 by
chance with 8 books, `docs/results/scout-kv/locality/locality.json`), and
nearby tokens are alike (L27 cosine 0.72 at distance 1, 0.50 at 256).

The owner proposed generating along the book so tokens depend on each other.
The v2 probe compared three ways to do that and then ablated the winner:

| finding | evidence |
|---|---|
| A smooth random signal per book (the "drifting latent"), fed to the generator next to the noise, reproduces the same-book share | L27 0.89–0.94 vs real 0.895; L19 0.53–0.77 vs 0.645 |
| A token-by-token recurrent generator is slower and unstable across seeds | probe main run |
| v1's per-token critic must stay as the main critic; an 8-token window critic on its own costs 5 points of body LID | ablation |
| Both critics side by side, window term at weight 0.1, keep v1's LID for a position-only generator | two-critic run, Cboth L27 25.1–25.3, v1 25.0 |
| The latent costs body LID: about 4–5 points at L27 and 2–3 at L19 against v1 | Bflag L27 20.3–21.0, Bflag25 L19 22.7–23.7; v1 25.0 / 26.0; real 30.0 / 29.2 |
| That cost sits in the latent's long length scales; capping them at 1,024 tokens recovers most of the LID but loses most of the same-book share | Bsmall L27 LID 22.6–23.1, share 0.58–0.64 |
| Sinusoidal position features are not needed; the tail bit alone does as well | Bflag vs Bboth |
| v1's 25% tail share in training moves tail LID toward real | L19 tail LID Bflag25 14.6–15.5, real 13.1, Cv1 19.2–19.5 |

The owner then chose (2026-10-28) between search difficulty and book
clustering, because no probe generator gets both: every probe generator
whose tokens are related makes L19 search too easy (HNSW recall 0.94–0.99
against real 0.854–0.882), and only independent-token generators (v1, and
the probe's v1 copy Cv1) are near real. **The owner chose book clustering
first.** v2 keeps the drifting latent and accepts too-easy L19 search as a
known open item; a separate probe looks for a kind of token relation that
does not make search easier.

## What v2 is

One generator per global layer (12), producing the 5,120-dim attention input
`h`, trained on books 0–11, keys and queries made afterwards with the model's
own `W_k`, `W_q`, as in v0 and v1. The generator is the probe's **Bflag25**:

- **Per-book latent.** Each synthetic book draws its own coefficients; the
  latent at position `t` is
  `u_c(t) = Σ_m (a_cm cos(ω_cm t) + b_cm sin(ω_cm t)) / √M`, with 64 channels
  `c`, M = 8 random Fourier features per channel, `a, b ~ N(0, 1)` per book,
  and `ω_cm ~ N(0, 1) / ℓ_c` with the length scales `ℓ_c` log-spaced from 2 to
  16,384 tokens. The frequencies are fixed at construction from the training
  seed and stored with the model. This is `DriftLatent` in
  `docs/results/scout-kv/v2-probe/probe.py`.
- **Generator.** v1's trunk: input = noise (128) + latent (64) + tail flag,
  hidden 1,024–2,048–4,096, output 5,120, LeakyReLU 0.2. Each token gets its
  own noise. Tokens of one book share only the latent. The probe feeds the
  flag as one bit; v1 feeds it as a 2-column one-hot (`src/models/conditional.py`).
  Either is acceptable, as long as generator and both critics use the same
  encoding.
- **Critics, trained side by side,** each with its own Adam optimizer and its
  own gradient penalty, on the same batches:
  - v1's per-token critic: `h` + tail flag → 2,048–1,024–512 → 1, gradient
    penalty per token.
  - The window critic: 8 tokens at one stride from {1, 4, 16, 64, 256, 1,024},
    per-token trunk 1,024–512, a token head and a window head, gradient
    penalty over the window (`WindowCritic` in `probe.py`).
  - Generator loss = −(token score) − 0.1 × (window score).
- **Batches.** 64 windows of 8 tokens (512 tokens, v1's batch size). 25% of
  windows lie wholly inside the tail (the last 1,024 positions), the rest end
  in the body. The stride is uniform over the strides whose window fits. Real
  windows come from one random training book at the same positions.
- **Training.** 30,000 generator steps, `n_critic` 3, Adam 1e-4 with betas
  (0, 0.9) for all three networks, `λ_gp` 10 for both critics, EMA 0.999 on
  the generator. Data centred with the mean over every position 1 to T − 1
  of books 0–11, as the probe does. (v1 fits its mean on its training split,
  after a 2% body holdout and a 1,024-row tail holdout; v2 has no holdout
  because it does not use v1's checkpoint selector.)
- **Sampling.** 8 synthetic books of T = 32,768 positions, each with its own
  latent coefficients, generated in chunks of positions; position 0 (the
  sink) is set to the data mean. Written in capture format as books 12–19,
  with the real weights linked, so the gate scores them exactly as it scores
  real books.

The rung's delta, for the ladder: **v1 + a per-book drifting latent, a
window critic at weight 0.1, and λ_gp 10.** (v1's flag and 25% tail share are
kept.)

### Decided

- **Owner:** book clustering takes priority over search difficulty for v2
  (2026-10-28).
- **Owner (earlier):** generation along the book, so tokens depend on each
  other (the drifting latent is the probe's winning form of this).
- **My call:** the full latent (64 channels, 2–16,384), not the capped one.
  The capped latent loses the clustering the owner prioritised.
- **My call:** tail bit only, no sinusoidal position features (no cost in
  the probe, fewer inputs).
- **My call:** 25% tail share, as v1, because it moves tail LID toward real.
- **My call:** `λ_gp` 10 rather than v1's 5. All probe evidence is at 10, and
  the probe's v1 copy at 10 reproduced v1 (L27 LID 25.35–25.37 vs 25.03; L19
  HNSW recall 0.890 / 0.906 vs 0.869–0.902).
- **My call:** judge the final EMA weights only. v1's checkpoint selector
  scores independent rows against a pooled holdout and does not see
  structure across tokens. At L27, v1's selected checkpoint was step 30,000
  at both seeds anyway.
- **My call:** two training seeds (42 and 1) per layer. The v1 diagnostics
  showed per-layer search statistics are dominated by retrain noise
  (L27 HNSW recall 0.808 vs 0.715 between seeds).

### Not in v2

- Fixing the too-easy L19 search. Known in advance; it is the follow-up
  probe's question (below).
- Hubness. Real body hubness is ~50 at L19/L27; probe generators and v1
  are 1.5–8.5, with one run at 16.8. Reported, not targeted.
- Changing the gate, its bands or the reference (`AGENTS.md`, "What requires
  a human").
- Values, and anything past 32,768 tokens.

## What has to change in the code

The probe (`docs/results/scout-kv/v2-probe/probe.py`, ~500 lines, throwaway)
is the reference implementation. v2 moves it into `src/` and `configs/`
under the project's rules. Requirements; how to meet them is left to the
plan unless stated.

1. **Book-structured data.** v2 trains on whole books (positions, book
   index), not on stacked rows. Books 0–11 of one layer at bf16 are about
   4 GB (ESTIMATE: 12 × 32,768 × 5,120 × 2 bytes) and fit on the L40.
   `load_kv_capture` and every v0/v1 config must load exactly as before.
2. **Model.** Latent, generator and both critics as above. The checkpoint
   and `run_config.yaml` record that the model is a book-latent generator,
   its latent settings and its frequencies, so no sampler can read it as a
   v0 or v1 row generator, and a resume that changes any of these is
   refused (the pattern of fffd7ba and 11c2316).
3. **Trainer.** Whether v2 extends `src/train/train_wgan_gp.py` or gets its
   own trainer is the builder's call. Constraint: v0 and v1 configs train
   exactly as today; existing tests pinning that stay green.
4. **Gate path.** For a book-latent checkpoint, sampling writes 8 synthetic
   books in capture format and the gate scores them with
   `kv_gate measure-real --books 12-19`, which builds corpora from books the
   same way as for the real reference. This is what the probe did. The probe
   argues the two paths agree for independent tokens: Cv1 scored this way
   lands on v1's `measure-run` numbers (above). `measure-run` on a v0/v1
   checkpoint must not change (pinned by `tests/test_kv_gate.py`).
5. **Configs.** `configs/scout_kv/v2/L{LL}.yaml`, one per global layer,
   differing only in `data.kv_layer` and `output_dir`, as v0's and v1's are
   (pinned by `tests/test_scout_kv_configs.py`). Seed 1 configs are
   generated from these by changing only `seed` and `output_dir`, as the v1
   noise configs were.
6. **Diagnostics.** `locality.py`, `hub_position.py` and the probe's
   `intrinsic` (body/tail LID, contrast, hubness) run on every v2 run's
   books. These live under `docs/results/`; the plan says whether they move
   or are called from where they are.

Tests follow the project's rules (`make check`, mutation-checked new tests).
The ones that matter most:
- the latent is shared within a book and differs between books (fails if
  every token draws its own latent, or every book the same);
- tail-share windows lie wholly inside the last 1,024 positions;
- the generator loss weights the window score by 0.1 (fails if the weight
  is dropped or applied to the token term);
- a v0 and a v1 checkpoint's sampled corpora do not change.

## How v2 is judged

v2 is not expected to pass the gate. It is judged on whether it keeps v1's
vectors close while adding real book structure. Measured per layer, for
both training seeds. Thresholds are my calls, flagged for the owner's review.

1. **Book clustering (the target).** The same-book share of `h`'s nearest
   neighbours (`locality.py`, `h_l2.same_book`) lies within **±0.15** of real
   at that layer, at both seeds. Real per layer:
   L3 0.409, L7 0.431, L11 0.458, L15 0.530, L19 0.645, L23 0.816,
   L27 0.895, L31 0.877, L35 0.865, L39 0.770, L43 0.786, L47 0.841.
   Why ±0.15: the probe's two seeds differed by up to 0.21 at L19 (Bboth),
   so a tighter tolerance would be decided by noise. Bflag25 sits at +0.10 /
   +0.12 at L19.
2. **Vector spread (the cost).** Body LID is at most **6 points** below v1's at
   that layer, at both seeds. The probe measured a cost of 2.3–4.7.
   The v1 value: v1's **final** checkpoint (step 30,000, EMA), because v2 is
   judged on its final weights and v1's best and final LIDs differ by up to
   3.3 at some layers (L7 18.51 best vs 15.19 final,
   `docs/results/scout-kv/v1/full/intrinsic.jsonl`). v2 seed 1 is compared
   with v1 seed 1 where it exists (L19, L27, L35, L43; checkpoints on
   tig-gpu-alt under `runs/scout_kv/v1_seed1/`), and with v1 seed 42
   elsewhere; v2 seed 42 with v1 seed 42. v1 final LID is committed at only
   6 layers and v1 seed-1 LID only at L27, so the plan includes measuring v1
   final at all 12 layers (seed 42) and the 4 seed-1 layers, with the same
   intrinsic tool as v2. This needs sampling only, no training.
3. **Gate, reported not judged.** The five statistics, pooled
   (`kv_gate check` on the 12 layers' records) and per layer, against v1 at
   both seeds and the reference. L19 HNSW recall is expected too easy.
4. **Reported:** key-side same-book share (`key_ip.same_book`), body and tail
   hubness, tail LID, the lag-cosine curves against real same-book and
   cross-book (`docs/results/scout-kv/v2-probe/control/cross_book_lag.json`).

**v2 meets its aim** if criterion 1 holds at 10 or more of the 12 layers
and criterion 2 at all 12, both at both seeds.

## Pilot, then the full run

Training runs on `tig-gpu-alt` (L40), one run at a time (never two trainings
on one card). A Bflag25 run took 2,176–2,207 s to train and 260–300 s more to
sample and score (MEASURED: `train_log.jsonl` last rows and the queue log's
`TRAINED`/`DONE` times, `/workspace/logs/v2probe_l19.log`).

1. **Port check and early-layer check: L3, L19 and L27, seeds 42 and 1**
   (6 runs, ~4.2 h ESTIMATE). All probe evidence is from L19 and L27. L3 is
   the farthest from it: real tokens there are almost unrelated to their
   neighbours (cosine 0.002 at distance 1, `control/cross_book_lag.json`) but
   still cluster by book (same-book share 0.409). At L19 and L27 the v2 code
   must reproduce the probe. At L19, against Bflag25: body LID within
   ±1.5 of 22.72–23.67 and same-book share within ±0.1 of 0.742–0.766. At L27
   there is no Bflag25 run; compare with Bflag (3% tail), expecting the same
   same-book share (±0.1 of 0.902–0.935) and body LID within ±1.5 of
   20.35–21.02. If the port misses, stop and diff against `probe.py` before
   going on. At L3, if criterion 1 or 2 misses at both seeds, stop and report
   to the owner before the full run.
2. **Full run:** the other 9 layers at both seeds (18 runs, ~12.5 h
   ESTIMATE). Cost it from the pilot's measured times before starting.
3. **Report:** `docs/results/scout-kv/v2/report.html`, one self-contained HTML
   file with base64-inlined figures, committed, with a claim table first. It
   shows the two criteria per layer, the gate statistics against v1 and
   real, and the diagnostics in item 4 above.

Box transfers follow the standing rules: hash on the box, tar-stream the
minimal set, verify hashes locally.

## Follow-up probe (separate, not blocking v2)

The question: what kind of relation between tokens keeps real search
difficulty? Known so far: it is not the window critic, the tail share, the
position features, or queries matching their own book's keys (generated
queries hit own-book keys at 0.11–0.13 of the time, real 0.095 at L19).
Real tokens are related to each other about as strongly as Ctok's
(L19 cosine at distance 256: real 0.370, cross-book 0.329, Ctok 0.43), yet
real search stays hard. This probe is designed separately once v2's port
check is running; it uses the GPU only between v2 runs.

## Branches

The probe, its results, `locality.py` and `control/cross_book_lag.json` exist
only on branch `scout-kv-v2-probe` (not on `main`, which is at 215611d).
My call: that branch goes to `main` through its own PR first, and v2 is
built on a new branch from `main` after the merge.

## If early layers overshoot

- If the latent overshoots the same-book share at early layers, v2 does not
  switch to per-layer latent settings (that would break the rule that the 12
  configs differ only in layer and output directory). v2 records the miss,
  and a per-layer latent becomes a candidate for a later rung. My call.

## Left to the builder

- Whether v2 extends the existing trainer or gets its own (item 3).
- How the diagnostics are packaged (item 6).
- Test layout, and whether runs go through `gpuq` or a plain `nohup` script
  as the probe did. Whatever runs them, the job's ID and log path come back
  before anyone waits on them.
