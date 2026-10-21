# scout_kv controls: why v0 looks too easy (B7/B8)

Measured on tig-gpu-alt (L40), 2026-10-19, with the code at `eafd1ac` (sha256 of every source file is at the top of
`b7.log`). Plan: `docs/ai/plans/2026-10-19-scout-kv-controls.md`. Every number here is MEASURED unless marked ESTIMATE.
The figures are in `report.html` (self-contained). `summary.json` is the machine-readable source of every table, built by
`summarise.py`; `make_report.py` builds the HTML from it.

## Verdict under the decision rules fixed before the run

- **R1 fires, at all three seeds: the gate's measurement path flatters synthetic data.** `pooled_real` is real `h` from
  the reference books 12-19, split into keys, queries and key-side queries the way `synthetic_corpus` does it (one
  pool, no position). It is outside the real range in v0's direction on `hnsw_recall_k10_ef176` at 2 or 3 of the 3 layers
  and on `ivf_ratio_k128` at all 3, at seeds 0, 1 and 2. **Hard stop: v1 is not designed. The gate question goes to the
  owner.** Seeds 1 and 2 were run because at seed 0 the verdict hung on single cells (plan, "one draw against five").
- **R2 does not fire.** The training books 0-7 are outside the real range in v0's direction on 0 of 3 layers for reach and
  1 of 3 for recall@10 and the IVF ratio. Where they are outside otherwise (layer 19 reach and recall@10), they are harder
  than the reference, not easier. The generator was not trained on easier data than it is judged against.
- **v0 re-measured on the L40 through the new code path matches the committed `v0/best.jsonl`** within the tolerance fixed
  before the run (statuses identical in all 15 cells; the tolerances themselves are ESTIMATE judgements).
- **R3 (descriptive).** The Gaussian fit is at least as far from real as v0, in v0's direction, on 8 of 9 rule cells (the
  exception: layer 19 IVF ratio, 0.097 against 0.095). v0 has learned something beyond the covariance.
- **R4 (descriptive).** Equalising norms does not move `pooled_real` toward v0. It makes HNSW much harder (layer 43
  recall@10: 0.296 against pooled 0.770 and real 0.574-0.637). The spread of norms makes real search easier.
- **R5 (descriptive).** No neighbour-position scalar is associated with per-corpus hardness: every |rho| < 0.5 over the 96
  corpora (largest 0.282). `hub_slot_share` is 1.0 in 61 of the 96 corpora, so its two rho values are computed mostly over
  ties. That saturation is why `n_distinct_keys` was added in the rerun below.

> **Correction, 2026-10-21.** The `hub_median_pos` values committed at `eedcd92` (in `neighbours.jsonl` and
> `summary.json`) were wrong. When the top-10 lists used fewer distinct keys than `n_hub` (317), the hub set was
> padded with never-used keys in index order, which are book 12's earliest positions. That put the median near positions
> 1,100-1,400. Hubs now come only from keys that some query used (`ccfa551`). `neighbours.jsonl` was regenerated on the
> box (`neighbours_rerun.log`, 177 s). The rerun pairs with `reference/seed0` exactly as before: 96/96 within 1e-3, 92
> exact. Apart from run times, every other field is identical to the first run. Only `hub_median_pos` changed (61 OOD and 89 key-side corpora),
> and `n_distinct_keys` is new. No decision-rule result changed: R1, R2, R5 and the v0 re-measure are as stated above.

## What this says about what makes real `h` hard

Most of the gap between real and v0 on the IVF ratio, and part of it on HNSW recall, comes from **where the queries come
from**, not from the distribution of `h`. The real reference asks for the neighbours of a book's last 1,024 tokens among
its earlier tokens. Asked the synthetic way (every row from one pool), the same real `h` gives an IVF ratio of
1.12-1.15 at layer 43 against 2.97-3.48 for the real split, and HNSW recall@10 of 0.685-0.770 against 0.574-0.637. An
unconditional generator cannot reproduce a key/query shift that is tied to position.

Two further measurements describe the real queries:

- **Same-book neighbours are enriched at lags of 512-4,096 positions** (median ratio >= 2 over the own-book baseline at 7,
  8 and 4 of the 12 layers for the bins 512-1024, 1024-2048 and 2048-4096). At layers 19 and 23, 36% of same-book
  neighbours lie within 1,024 positions of the query, against a 1.6% baseline.
- **Real queries share a small set of target keys, and those keys sit late in the books.** For head 0 at seed 0, all
  51,200 top-10 slots of the 5,120 real-style queries fall on 11 distinct keys at layer 3, 673 at layer 19 and 199 at layer
  43, out of 31,736 (`distinct_keys.py` / `distinct_keys.txt`). These keys are not simply the longest: the correlation
  between key norm and use is about 0. Over all 8 heads (`n_distinct_keys`, added in the rerun), the median is 31-949
  distinct keys per layer. At 8 of the 9 layers from 15 on, the median position of those keys is 26,609-30,087 (the exception is layer 39, at 16,038). Keys sit at positions
  1-31,743, and their median is about 15,900. At most of these layers the same-book share is near the 1/8 baseline, so the
  queries go to late keys of any book, not mainly of their own. Neither count was a pre-registered statistic; both are
  observations for follow-up. The cross-book control tests whether this position effect, rather than the own-book link,
  carries the hardness. Its rule-grade version, the held-out-book control below, reads **reachable**: tail queries from
  books outside the key set are not easier than the reference.

## R6/R7: cross-book control (2026-10-21)

Plan: `docs/ai/plans/2026-10-21-scout-kv-cross-book.md`, with the rules fixed before the run. Box run B9 on tig-gpu-alt used
the code at `6c914a5` (`b9.log` has the code hashes). Each of the 4 draws took 651-673 s. All values are MEASURED, from
`summary.json` `xbook`.

Both controls take keys from the bodies of four reference books and queries from book tails (the last 1,024 positions).
The key-side queries are the same tail tokens, as in the reference.
- `same_book_tail` (the positive control) takes its queries from the tails of the same four books.
- `cross_book_tail` takes them from the tails of the other four books.
- Within a draw, both controls use the same key rows.
- Draws: partition a (key books 12-15) and partition b (key books 16-19), each at seeds 0 and 1.

**Verdict: R6 fails, so R7 is not read.** The plan's rule is to report and stop.
- `same_book_tail` is outside the real range on at least 2 of the 3 rule statistics in 2 of the 4 draws: a_s0 (3
  statistics) and a_s1 (2). The rule allowed at most 1 draw.
- Both failing draws are partition a, and both are outside on the *hard* side. At layer 19 the IVF ratio is 0.657 and
  0.630 against real 0.133-0.172, and recall@10 is 0.778 and 0.744 against 0.854-0.882. Partition b is inside on every
  rule statistic.
- Reading: with four key books, the construction's hardness depends on which books form the key set, most of all at layer
  19. The positive control therefore does not reproduce the 8-book reference, and the plan does not let the cross-book
  draws be judged against that reference.
- For the record, `cross_book_tail` is on v0's side in 0 of the 4 draws (`n_cross_flatter` 0). That number is not a
  reading under the rules.

**Descriptive (not a rule result).** Within each draw, the same key rows can be paired with own-book and with other-book
tail queries. The difference cross minus same is:

| layer | recall@10 ef176, cross minus same (a_s0, a_s1, b_s0, b_s1) | same, range over draws | IVF ratio, cross minus same | same, range over draws |
|---|---|---|---|---|
| 11 | +0.009, +0.013, +0.001, +0.001 | 0.438-0.696 | -0.002, -0.005, +0.005, -0.014 | 1.202-1.397 |
| 19 | -0.061, -0.050, +0.041, +0.036 | 0.744-0.857 | +0.092, +0.084, +0.012, +0.025 | 0.142-0.657 |
| 43 | +0.020, -0.001, +0.026, +0.045 | 0.458-0.696 | -0.029, -0.054, +0.032, +0.080 | 3.127-3.710 |

- **Layers 11 and 43:** whether queries come from the key books or from other books changes recall and the IVF ratio by
  much less than the spread across draws.
- **Layer 43 tail split:** the IVF ratio is 3.10-3.71 for both controls, against 1.12-1.15 for `pooled_real` and 0.76
  for v0. Queries from book tails, even other books' tails, keep the reference's IVF hardness at layer 43.
- **Layer 19:** the gap is larger and changes sign with the partition. This is also the layer with the strongest
  same-book recency (36% of same-book neighbours within 1,024 positions).
- These are observations from 4 draws at 3 layers. Testing them under a rule would need a construction whose positive
  control holds, for example 8 key books with the queries taken from held-out books. That is the held-out-book control
  below.

## R6a/R6/R7: held-out-book control (2026-10-21)

Plan: `docs/ai/plans/2026-10-21-scout-kv-held-out.md`, with the rules and the four draws fixed before the run. Box run B11
on tig-gpu-alt used the code at `3bb39ee` (`heldout/b11.log` has the code hashes). The seed-0 pairing step took 357 s and
each draw 714-720 s. All values are MEASURED, from `summary.json` `heldout`.

The keys are exactly the reference's: all eight reference books 12-19, built per KV head by `mixed_real_corpus` at
the reference's per-head seed.
- `rebuild` is the reference corpus itself at a fresh seed (5-8, none of them a reference seed). It is the positive
  control.
- `held_out_tail` keeps the same keys and the same index seed. Its real-style and key-side queries come from the tails
  (the last 1,024 positions) of eight books from 0-11, 128 rows each, so no query shares a book with any key. Its
  own-book share is 0 in every corpus.
- Draws: s5 (query books 0-7), s6 (4-11), s7 (0-3, 8-11), s8 (0-2, 4-6, 8, 10).

**Verdict: R7 reads reachable.**
- **R6a passes.** The seed-0 rebuild pairs with the committed `reference/seed0.jsonl` on all 24 rule-layer corpora (23
  exact, max own-book share difference 3.9e-5), so the code path is the reference's.
- **R6 passes.** `rebuild` is outside the real range on at least 2 of the 3 rule statistics in 1 of the 4 draws (s5);
  the rule allows 1.
- **R7: `held_out_tail` flatters in 0 of the 4 draws.** In every draw, 0 of the 3 rule statistics are on v0's side at 2
  or more of the 3 layers. Under the plan this reads **reachable**: queries from books outside the key set keep the real
  difficulty, so the difficulty does not need a query to come from the same book as its keys. That supports a
  position-conditioned generator as the v1 target.

Cells outside the real range (3 layers × 3 rule statistics × 4 draws = 36):

| control | on v0's (easy) side | on the hard side |
|---|---|---|
| `rebuild` | 9 | 8 |
| `held_out_tail` | 6 | 17 |

**Paired difference, held-out minus rebuild** (same keys, same index seed; s5, s6, s7, s8):

| layer | reach | recall@10 ef176 | IVF ratio |
|---|---|---|---|
| 11 | 0, 0, 0, 0 | -0.002, +0.004, +0.005, -0.014 | +0.022, +0.015, -0.004, +0.017 |
| 19 | -0.375, -0.125, 0, -0.375 | -0.091, -0.048, -0.047, -0.089 | +0.106, +0.145, +0.095, +0.171 |
| 43 | 0, 0, 0, +0.250 | +0.032, -0.016, +0.009, +0.039 | +0.359, +0.562, +0.051, +0.291 |

What the result does not say (descriptive, not rule results):
- **Held-out queries are harder, not just as hard, at layer 19.** In every draw, recall@10 drops by 0.047-0.091 and the
  IVF ratio rises by 0.095-0.171. Books 0-7 measured alone (`train_books_real`) are also harder than the reference at
  layer 19 (recall@10 0.831, IVF ratio 0.369, against 0.854-0.882 and 0.133-0.172). So at least part of the layer-19
  gap is a difference between books 0-11 and 12-19, not an own-book effect. The rule cannot separate the two. A
  generator trained on books 0-11 may therefore overshoot the reference at layer 19 rather than undershoot it.
- **Layer 11 does not care which book a query comes from** (|Δ recall| ≤ 0.014, |Δ IVF ratio| ≤ 0.022).
- **At layer 43, held-out queries keep the reference's IVF hardness and add to it** (IVF ratio 3.32-4.05 against rebuild
  2.96-3.55 and real 2.97-3.48). HNSW recall moves by -0.016 to +0.039.
- **The five-draw real range is narrow.** The positive control, which is exchangeable with the reference draws, falls
  outside it in 17 of 36 cells. Layer 43 recall@10 is above it in 3 of 4 rebuild draws (0.629-0.707 against
  0.574-0.637). The plan's base rate assumed 1 in 3 per cell (ESTIMATE). R6 passes because a statistic must be outside
  at 2 of the 3 layers to count. A gate band set from the five draws alone would reject some real draws; the bands in
  `gates/scout_kv.yaml` are wider than the five draws, and pooled over 96 corpora rather than per layer.
- This covers 3 of the 12 layers and does not run the 96-corpus gate.

## Per-layer values

Arrows mark a value above (↑) or below (↓) the range of the five real reference draws. `pooled_real` s0, s1 and s2 are
seeds 0, 1 and 2; every other control is seed 0 only.

### Layer 11

| statistic | real, 5 draws | train books 0-7 | pooled_real s0 | s1 | s2 | norm_equalised | gaussian | v0 |
|---|---|---|---|---|---|---|---|---|
| `hnsw_reach_k10` | 0.500-0.750 | 0.625 | 0.250 ↓ | 1.000 ↑ | 0.500 | 0.875 ↑ | 1.000 ↑ | 1.000 ↑ |
| `hnsw_recall_k10_ef176` | 0.521-0.718 | 0.701 | 0.526 | 0.737 ↑ | 0.473 ↓ | 0.916 ↑ | 0.976 ↑ | 0.931 ↑ |
| `ivf_ratio_k128` | 1.223-1.304 | 0.996 ↓ | 1.149 ↓ | 1.217 ↓ | 1.171 ↓ | 0.443 ↓ | 0.630 ↓ | 0.759 ↓ |
| `ivf_recall_k128_np24` | 0.593-0.643 | 0.660 ↑ | 0.640 | 0.613 | 0.638 | 0.778 ↑ | 0.684 ↑ | 0.707 ↑ |
| `hnsw_recall_k10_ef176_id` | 0.996-1.000 | 0.998 | 0.997 | 0.997 | 0.993 ↓ | 0.988 ↓ | 0.997 | 0.995 ↓ |

### Layer 19

| statistic | real, 5 draws | train books 0-7 | pooled_real s0 | s1 | s2 | norm_equalised | gaussian | v0 |
|---|---|---|---|---|---|---|---|---|
| `hnsw_reach_k10` | 0.500-0.875 | 0.375 ↓ | 1.000 ↑ | 1.000 ↑ | 1.000 ↑ | 0.375 ↓ | 1.000 ↑ | 1.000 ↑ |
| `hnsw_recall_k10_ef176` | 0.854-0.882 | 0.831 ↓ | 0.967 ↑ | 0.967 ↑ | 0.963 ↑ | 0.669 ↓ | 0.993 ↑ | 0.972 ↑ |
| `ivf_ratio_k128` | 0.133-0.172 | 0.369 ↑ | 0.054 ↓ | 0.065 ↓ | 0.062 ↓ | 0.157 | 0.097 ↓ | 0.095 ↓ |
| `ivf_recall_k128_np24` | 0.981-0.987 | 0.971 ↓ | 0.962 ↓ | 0.960 ↓ | 0.961 ↓ | 0.969 ↓ | 0.962 ↓ | 0.963 ↓ |
| `hnsw_recall_k10_ef176_id` | 1.000-1.000 | 1.000 ↓ | 0.988 ↓ | 0.988 ↓ | 0.985 ↓ | 0.990 ↓ | 0.995 ↓ | 0.958 ↓ |

### Layer 43

| statistic | real, 5 draws | train books 0-7 | pooled_real s0 | s1 | s2 | norm_equalised | gaussian | v0 |
|---|---|---|---|---|---|---|---|---|
| `hnsw_reach_k10` | 0.000-0.125 | 0.125 | 0.125 | 0.125 | 0.000 | 0.000 | 1.000 ↑ | 0.375 ↑ |
| `hnsw_recall_k10_ef176` | 0.574-0.637 | 0.641 ↑ | 0.770 ↑ | 0.704 ↑ | 0.685 ↑ | 0.296 ↓ | 0.967 ↑ | 0.824 ↑ |
| `ivf_ratio_k128` | 2.974-3.480 | 3.360 | 1.150 ↓ | 1.117 ↓ | 1.125 ↓ | 1.142 ↓ | 0.197 ↓ | 0.761 ↓ |
| `ivf_recall_k128_np24` | 0.516-0.591 | 0.555 | 0.758 ↑ | 0.744 ↑ | 0.782 ↑ | 0.798 ↑ | 0.909 ↑ | 0.766 ↑ |
| `hnsw_recall_k10_ef176_id` | 1.000-1.000 | 1.000 | 0.989 ↓ | 0.985 ↓ | 0.988 ↓ | 0.853 ↓ | 0.993 ↓ | 0.967 ↓ |

### Neighbour positions, median over 8 heads (OOD queries)

| layer | own-book share | same-book lag <= 1024 | baseline | key pos < 16 | baseline | top-1% key slot share | distinct top-10 keys | hub median pos (all keys ~15,900) |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0004 | 1.000 | 61.5 | 4,492 |
| 7 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0005 | 1.000 | 31 | 7,094 |
| 11 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0005 | 1.000 | 64 | 17,213 |
| 15 | 0.109 | 0.029 | 0.016 | 0.000 | 0.0005 | 1.000 | 202.5 | 29,654 |
| 19 | 0.108 | 0.362 | 0.016 | 0.005 | 0.0005 | 1.000 | 172.5 | 30,002 |
| 23 | 0.136 | 0.367 | 0.016 | 0.000 | 0.0005 | 1.000 | 221 | 29,482 |
| 27 | 0.197 | 0.084 | 0.016 | 0.000 | 0.0005 | 0.984 | 662.5 | 26,609 |
| 31 | 0.215 | 0.204 | 0.016 | 0.001 | 0.0005 | 0.968 | 949 | 29,432 |
| 35 | 0.163 | 0.157 | 0.016 | 0.002 | 0.0005 | 0.998 | 401.5 | 28,757 |
| 39 | 0.131 | 0.014 | 0.016 | 0.002 | 0.0004 | 0.999 | 322 | 16,038 |
| 43 | 0.127 | 0.080 | 0.016 | 0.032 | 0.0005 | 1.000 | 156 | 28,709 |
| 47 | 0.142 | 0.193 | 0.016 | 0.002 | 0.0005 | 1.000 | 237 | 30,087 |

## Files

| file | what |
|---|---|
| `neighbours.jsonl` | neighbour diagnostic, 96 corpora (all 12 layers), books 12-19, seed 0 |
| `train_books_real.jsonl` | real reference path over books 0-7, layers 11/19/43 |
| `pooled_real.jsonl`, `pooled_real_seed1.jsonl`, `pooled_real_seed2.jsonl` | real `h` split the synthetic way, seeds 0/1/2 |
| `norm_equalised.jsonl`, `gaussian.jsonl` | the other two controls, seed 0 |
| `v0_best_subset.jsonl` | v0 `best_generator.pt` re-measured through `measure-run --layers` |
| `b7.log`, `b7_seeds.log` | box logs (code hashes, per-step seconds) |
| `neighbours_rerun.log` | box log of the 2026-10-21 neighbour rerun (commit, per-corpus lines, pairing check) |
| `SHA256SUMS` | box-side hashes; every file was verified against them after transfer |
| `b7.sh`, `check_pairing.py`, `distinct_keys.py` | box scripts |
| `xbook/{a,b}_s{0,1}/{same,cross}_book_tail.jsonl`, `xbook.sh`, `b9.log` | cross-book control (R6/R7): records, box script, box log |
| `heldout/pair_s0/rebuild.jsonl`, `heldout/s{5,6,7,8}/{rebuild,held_out_tail}.jsonl`, `heldout/pair_check.json`, `heldout/b11.log`, `heldout.py`, `heldout.sh` | held-out-book control (R6a/R6/R7): records, R6a pairing report, box log, script and box script |
| `summarise.py`, `summary.json`, `make_report.py`, `report.html` | write-up |

## Claim table

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| R1 per seed | fires at seeds 0, 1, 2 | MEASURED | `summary.json` `R1_per_seed` | 2026-10-19 |
| R2 | does not fire | MEASURED | `summary.json` | 2026-10-19 |
| neighbour corpora pair with `seed0` | 96/96 within 1e-3, 92 exact, max 3.9e-5 | MEASURED | `check_pairing.py` on the box | 2026-10-19 |
| v0 re-measure matches | yes, 15/15 cells | MEASURED (tolerances ESTIMATE) | `summary.json` `v0_remeasure` | 2026-10-19 |
| B7 wall time | 1,991 s, plus 700 s for the two extra seeds | MEASURED | `b7.log`, `b7_seeds.log` | 2026-10-19 |
| distinct top-10 keys, head 0 | 11 / 673 / 199 at L3 / L19 / L43 | MEASURED | `distinct_keys.txt` | 2026-10-19 |
| largest R5 abs(rho) | 0.282 | MEASURED | `summary.json` `r5` | 2026-10-21 (unchanged by the rerun) |
| R6 (cross-book positive control valid) | no: outside in 2 of 4 draws (a_s0, a_s1), hard side | MEASURED | `summary.json` `R6_R7_xbook` | 2026-10-21 |
| cross_book_tail on v0's side | 0 of 4 draws (R7 not read) | MEASURED | `summary.json` `xbook` | 2026-10-21 |
| L43 IVF ratio, tail controls | 3.097-3.710 (8 values) | MEASURED | `summary.json` `xbook` | 2026-10-21 |
| B9 wall time per draw | 651-673 s | MEASURED | `b9.log` | 2026-10-21 |
| neighbour rerun pairs with `seed0` | 96/96 within 1e-3, 92 exact | MEASURED | `neighbours_rerun.log` | 2026-10-21 |
| median distinct top-10 keys per layer, 8 heads | 31-949 | MEASURED | `summary.json` `neighbours_per_layer` | 2026-10-21 |
| R6a (held-out path check) | passes: 24/24 within 1e-3, 23 exact, max 3.9e-5 | MEASURED | `heldout/pair_check.json` | 2026-10-21 |
| R6 (held-out positive control valid) | yes: rebuild outside in 1 of 4 draws (s5) | MEASURED | `summary.json` `R6_R7_heldout` | 2026-10-21 |
| R7 (held-out reading) | reachable: held_out_tail flatters in 0 of 4 draws | MEASURED | `summary.json` `R6_R7_heldout` | 2026-10-21 |
| held-out minus rebuild, layer 19 | recall@10 -0.047 to -0.091; IVF ratio +0.095 to +0.171 | MEASURED | `summary.json` `heldout` | 2026-10-21 |
| rebuild cells outside the 5-draw real range | 17 of 36 | MEASURED | `summary.json` `heldout` | 2026-10-21 |
| B11 wall time | 357 s pairing step; 714-720 s per draw | MEASURED | `heldout/b11.log` | 2026-10-21 |
| median hub key position, layers 15-47 | 26,609-30,087 at 8 of 9 layers; 16,038 at layer 39 (key median ~15,900) | MEASURED | `summary.json` `neighbours_per_layer` | 2026-10-21 |
