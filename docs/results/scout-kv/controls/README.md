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
  corpora (largest 0.282).

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
- **Real queries share a tiny set of target keys.** For head 0 at seed 0, all 51,200 top-10 slots of the 5,120 real-style
  queries fall on 11 distinct keys at layer 3, 673 at layer 19 and 199 at layer 43, out of 31,736. These keys are not
  simply the longest (correlation between key norm and use is about 0). `distinct_keys.py` / `distinct_keys.txt`. This
  count was not a pre-registered statistic; it is an observation for follow-up.

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

| layer | own-book share | same-book lag <= 1024 | baseline | key pos < 16 | baseline | top-1% key slot share |
|---|---|---|---|---|---|---|
| 3 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0004 | 1.000 |
| 7 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0005 | 1.000 |
| 11 | 0.125 | 0.000 | 0.016 | 0.000 | 0.0005 | 1.000 |
| 15 | 0.109 | 0.029 | 0.016 | 0.000 | 0.0005 | 1.000 |
| 19 | 0.108 | 0.362 | 0.016 | 0.005 | 0.0005 | 1.000 |
| 23 | 0.136 | 0.367 | 0.016 | 0.000 | 0.0005 | 1.000 |
| 27 | 0.197 | 0.084 | 0.016 | 0.000 | 0.0005 | 0.984 |
| 31 | 0.215 | 0.204 | 0.016 | 0.001 | 0.0005 | 0.968 |
| 35 | 0.163 | 0.157 | 0.016 | 0.002 | 0.0005 | 0.998 |
| 39 | 0.131 | 0.014 | 0.016 | 0.002 | 0.0004 | 0.999 |
| 43 | 0.127 | 0.080 | 0.016 | 0.032 | 0.0005 | 1.000 |
| 47 | 0.142 | 0.193 | 0.016 | 0.002 | 0.0005 | 1.000 |

## Files

| file | what |
|---|---|
| `neighbours.jsonl` | neighbour diagnostic, 96 corpora (all 12 layers), books 12-19, seed 0 |
| `train_books_real.jsonl` | real reference path over books 0-7, layers 11/19/43 |
| `pooled_real.jsonl`, `pooled_real_seed1.jsonl`, `pooled_real_seed2.jsonl` | real `h` split the synthetic way, seeds 0/1/2 |
| `norm_equalised.jsonl`, `gaussian.jsonl` | the other two controls, seed 0 |
| `v0_best_subset.jsonl` | v0 `best_generator.pt` re-measured through `measure-run --layers` |
| `b7.log`, `b7_seeds.log` | box logs (code hashes, per-step seconds) |
| `SHA256SUMS` | box-side hashes; every file was verified against them after transfer |
| `b7.sh`, `check_pairing.py`, `distinct_keys.py` | box scripts |
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
| largest R5 abs(rho) | 0.282 | MEASURED | `summary.json` `r5` | 2026-10-19 |
