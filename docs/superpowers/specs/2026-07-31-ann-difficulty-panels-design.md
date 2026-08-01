> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# ANN difficulty panels for the SIFT EDA report

Date: 2026-07-31
Branch: `eda/sift-eda`
Status: approved, ready for implementation planning

## Problem

`src/eval/eda_report.py` compares real SIFT against synthetic sets on distributional
shape. It answers "is this faithful". It does not answer "would this work as a
benchmark", which is the question that matters for SIFT1M's actual purpose.

The most recent run (`runs/eda/compare_100k.summary.json`) shows why a new
instrument is needed. Every generator matches real SIFT on global geometry:

| statistic | real | baseline_30k | ema_only_100k | improved_100k |
|---|---|---|---|---|
| `value_std` | 0.0712 | 0.0710 | 0.0711 | 0.0712 |
| `median_pairwise_distance` | 1.0995 | 1.0913 | 1.0975 | 1.0981 |
| `median_5nn_distance` | 0.5153 | 0.5101 | 0.5144 | 0.5131 |
| `effective_rank` | 27.99 | 26.96 | 27.39 | 27.56 |

Matching on all four of those is compatible with behaving nothing like SIFT under
nearest-neighbour search. Nothing currently in the report or in
`evaluate_distribution.py` measures search behaviour: the existing
`ann_proxy_recall` (`src/eval/evaluate_distribution.py:160`) reduces to a single
scalar `exp(-|ratio - 1|)` over mean k-NN distance, which compares distance
*magnitudes* rather than search *difficulty*.

## Scope

Add three sections to the existing HTML report, backed by four ANN-difficulty
metrics computed from the vectors alone. No new heavyweight dependencies; no
index is built.

Explicitly out of scope:

- Building a real index (faiss/hnswlib) and plotting recall-vs-effort curves.
- Replacing `ann_proxy_recall` in `evaluate_distribution.py`. The new module is
  structured so that could follow later, but this work does not touch that file.
- Sparsity/support panels and SIFT 4x4x8 cell-structure panels. These are worth
  doing and are a plausible next spec, but they answer "is it faithful", not
  "is it usable", and are not part of this work.

## Protocol decisions

**Self-query.** Each set is queried by held-out rows drawn from itself, against
the remainder of that same set. This is the "would this be as hard a benchmark
as SIFT" question stated directly.

Scoring real queries against a synthetic base using shared ground truth is
rejected: the true neighbours of real queries are not present in a synthetic
set, so recall would measure absence rather than difficulty. Synthetic data is a
good stand-in when it is *equally hard*, not when it returns the same vectors.

**Equal N, always.** LID, relative contrast and hubness all drift with sample
count. Every set is truncated to the same row count before measurement. This is
the discipline the existing `nn_distances` docstring already argues for
(`src/eval/eda_report.py:186`).

**No literature comparison.** Published LID for SIFT1M (~9.3) will not reproduce
here: that figure is the full 1M set, raw, queried with the real query set,
whereas this measures a 20k L2-normalized self-queried subsample. The section
notes must say that the reference is the `real` series *in the same report* and
never the published value. This mirrors the caveat `effective_rank` already
carries at `src/eval/eda_report.py:447`.

## Architecture

New module `src/eval/ann_difficulty.py`, depending only on numpy and sklearn. It
knows nothing about plotly, argparse, or the report's `Series` type.

```python
@dataclass
class AnnMetrics:
    lid: np.ndarray               # per surviving query
    relative_contrast: np.ndarray # per surviving query, same mask as lid
    k_occurrence: np.ndarray      # per point
    cell_occupancy: np.ndarray    # per cluster, ascending
    num_rows: int                 # rows actually measured
    k: int
    discarded_queries: int

def compute(x, *, k=100, k_hub=10, nlist=256, max_rows=20000, seed=42) -> AnnMetrics
def summary(m: AnnMetrics) -> Dict[str, float]
```

A single brute-force k-NN pass inside `compute` produces both distances and
indices; all four metrics read from that one result. `summary` reduces to the
scalars that join the report's existing statistics table.

`summary` returns exactly these keys, which are merged into the report's existing
per-series statistics table:

- `lid_median`
- `relative_contrast_median`
- `hubness_skew`
- `ivf_gini`
- `lid_discarded_queries`

The separation exists so the metrics are importable and testable without
invoking argparse or plotly, and so `eda_report.py` stays a presentation layer
rather than growing past 1000 lines.

### Data flow in `eda_report.py`

`main()` calls `ann_difficulty.compute` once per `Series` immediately after
`load_series`. The resulting `AnnMetrics` is passed to two consumers:

1. `summary_stats`, which merges `summary(m)` into its per-series dict so the new
   numbers appear in the existing single table rather than a second one.
2. Three new figure builders, which read the raw per-query and per-point arrays.

No metric is computed twice.

## Metrics

### Local intrinsic dimensionality

Hill / Amsaleg maximum-likelihood estimator over each query's `k` nearest
neighbour distances `r_1 <= ... <= r_k`:

```
LID(q) = -[ (1/k) * sum_{i=1..k} log(r_i / r_k) ]^{-1}
```

The `i = k` term contributes zero and is retained so the divisor is `k`.

Queries with `r_1 <= 0` are discarded, not clamped, and the count is reported as
`lid_discarded_queries`. Real SIFT has a 0.062% duplicate row fraction so a small
number of queries will hit this. Discarding is mildly biased, since duplicates
are exactly the low-LID region, but at that rate the effect is negligible and
clamping would silently fabricate values instead.

### Relative contrast

Per query, the mean distance to a random target sample divided by `r_1`. The
target sample is 2000 rows drawn once per set from the truncated set using
`seed`, shared across all queries in that set so the numerator is measured
against a fixed reference. Computed on the same surviving query mask as LID so
the two panels are comparable row-for-row.

### Hubness

`N_k(x)` counts how many times `x` appears in other points' k-NN lists, computed
from the indices matrix at `k_hub = 10` (the standard in the hubness literature)
by reusing the first 10 columns of the `k = 100` cache. Reported as the full
distribution plus its skewness. Graph indexes such as HNSW degrade when a few
hubs dominate, and a GAN has no direct training pressure to reproduce this.

### IVF cell balance

`MiniBatchKMeans(nlist, random_state=seed)` fit on each set independently, since
an IVF index would be built on whichever set was shipped. Occupancy is sorted
ascending and reported as a Lorenz curve plus a Gini coefficient.

## Figures

1. **ANN difficulty profile** — `make_subplots(1, 2)`: LID histogram alongside
   relative contrast histogram, overlaid per series with a shared legend via
   `legendgroup`. Follows the existing `fig_pca_spectrum` pattern
   (`src/eval/eda_report.py:343`).
2. **Hubness** — overlaid `N_k` histogram with log y. Requires no new plotting
   code; `overlay_hist_fig(..., log_y=True)` already covers it.
3. **IVF cell balance** — Lorenz curves, one line per series, with the
   perfect-balance diagonal as a dashed reference.

One genuinely new figure function; the other two reuse existing helpers.

### Placement

The three sections go immediately after the statistics table and before "Pooled
value distribution" — verdict first, diagnostics after. A reader who sees LID 6.1
against real's 9.4 then scrolls into the support and marginal panels to find out
why. The report grows from 8 sections to 11.

## New CLI flags

| flag | default | effect |
|---|---|---|
| `--ann-k` | 100 | neighbours per query for LID and relative contrast |
| `--ann-hub-k` | 10 | neighbour depth for the k-occurrence count |
| `--ann-max-rows` | 20000 | equal-N truncation for all difficulty metrics |
| `--ivf-nlist` | 256 | cluster count for the cell-balance panel |

`--ann-max-rows` also replaces the hardcoded `max_rows=20000` default in
`nn_distances` (`src/eval/eda_report.py:186`), so every equal-N truncation in the
report is governed by one number.

## Failure modes

| condition | behaviour |
|---|---|
| `N <= k` | clamp `k = N - 1`, state the clamp in the section note |
| `nlist > N // 2` | clamp, state it in the section note |
| every query discarded | render the section with an explanatory note and write `null` scalars to `summary.json`; do not crash a long run at the last panel |
| duplicate rows | excluded from the LID/RC query mask, count surfaced as `lid_discarded_queries` |
| `MiniBatchKMeans` | explicit `n_init` and `random_state=seed` so reruns are byte-identical |

## Testing

The repository currently has no tests. This work adds `pytest` to
`requirements.txt` and creates `tests/`. Getting real coverage on these metrics
is the main reason the separate module was worth the extra file.

`tests/test_ann_difficulty.py`, cheapest first:

1. **k-occurrence conservation** — `sum(N_k) == n * k_hub` exactly. Deterministic
   and sharp; catches any off-by-one in index bookkeeping.
2. **Gini** — tested against the pure helper with hand-built occupancy arrays.
   Uniform occupancy gives ~0; one dominant cluster gives a high value. No
   k-means involved.
3. **Hubness skew** — a planted hub (one centroid point plus a surrounding shell)
   must score clearly above a uniform set. Relative assertion, not absolute.
4. **Duplicate handling** — a set with repeated rows produces no `inf` or `nan`
   and a non-zero `discarded_queries`.
5. **Equal-N truncation** — 50k rows in with `max_rows=20000` yields
   `num_rows == 20000`, and two runs at the same seed agree exactly.
6. **Relative contrast concentration** — RC at d=64 is below RC at d=2.
   Directional and robust.
7. **LID recovers dimension** — uniform in a d-ball at d=4, n=20000, k=100 lands
   within 20% of 4. This carries the loosest tolerance of the set; the Hill
   estimator's bias grows with d/n, so d is kept small deliberately rather than
   asserting something that flakes.

`tests/test_eda_report.py`:

8. **Wiring smoke test** — run `main()` against two small random arrays in
   `tmp_path` with `--no-png`, assert the HTML is written and `summary.json`
   parses with the new keys present. This is what catches a broken figure call;
   the unit tests above never would.

## Consequences

- `eda_report.py` grows by roughly 120 lines of figure code; `ann_difficulty.py`
  is roughly 250 lines.
- Runtime rises by one brute-force k-NN pass per set (~50 GFLOP at 20k rows,
  k=100) plus one MiniBatchKMeans fit. On a four-series comparison this is a few
  minutes, bounded by `--ann-max-rows`.
- `summary.json` gains four scalars and `lid_discarded_queries` per series.
