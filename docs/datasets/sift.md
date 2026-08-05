# SIFT

128-dimensional SIFT image descriptors, non-negative and quantized to uint8.
The vectors come from local image keypoints, and the one structural fact that
decides how this family is modelled is the heavy mass sitting at exactly
zero.

## Source

    python -m src.data.fetch sift

Fetches `sift-128-euclidean` into the shared cache and writes
`data/sift_250k.npy` and `data/sift_1m.npy`. The HDF5 is large and immutable;
the fetcher downloads it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `128` |
| Search metric | `l2` |
| Upstream | `sift-128-euclidean` |

## Structure

128-dimensional SIFT descriptors, non-negative and quantized to uint8, with
heavy mass at exactly zero. Points therefore sit on a lattice: exact ties and
true duplicates are common and dominate the top of any neighbour list. A
dense MLP generator cannot reproduce that support, which is why `gated`
exists — a softplus magnitude times a sampled binary gate, giving exact
zeros.

That structure is the one thing an aggregate cannot show you: the 128 values
are a 4x4 grid of spatial cells holding 8-bin gradient orientation
histograms, and a generator can match every marginal while producing 128
numbers that are not a plausible histogram. The descriptor glyph panel draws
individual descriptors instead — `Descriptor glyphs` in `eda_report`, or
`src/eval/plot_descriptor_grid.py` to draw straight from run checkpoints.
It is the only panel here that is SIFT-specific: the `(cell, orientation
bin)` mapping does not exist for the other families, so it is skipped for
them. See `PROJECT_DOCUMENTATION.md` for how to read it.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (best variant) |
|---|---|---|
| LID median | not yet measured | — |
| Relative contrast | not yet measured | — |
| Hubness skew | not yet measured | — |
| IVF cell-balance Gini | not yet measured | — |

Fill the real column with:

    python -m src.eval.eda_report \
        --real-path data/sift_250k.npy \
        --output-dir runs/sift/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

Read the four values out of runs/sift/profile/summary.json (written by the command above).

## Model family

`gated` — a softplus magnitude times a sampled binary gate reproduces the
exact zeros and quantized lattice that a dense MLP generator cannot.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift/v0.yaml` | `runs/long_baseline` | trained |
| `v1` | + generator EMA (`ema_decay: 0.999`) | `configs/sift/v1.yaml` | `runs/x100k_ema_only` | trained |
| `v1_5` | + distance reg (`alpha: 0.1`, 256 points) | `configs/sift/v1_5.yaml` | `runs/x100k_improved` | trained |
| `v2` | + gated generator | `configs/sift/v2.yaml` | `runs/x100k_sparse_clamp4` | trained |

Train `v0` (or any rung, by swapping the config):

    python -m src.train.train_wgan_gp --config configs/sift/v0.yaml

Run directory names predate the per-dataset scheme and are kept as-is, since
the artifacts under them are already named that way. Run length is an
independent axis, not a variant: `bench_*` are 3k generator steps, `long_*`
30k, `x100k_*` 100k.

## Gate

`gates/sift.yaml` is the gate. The bands live there rather than in this
prose so a program can read them, and this section does not repeat the
numbers: two copies of a threshold is one copy too many.

Pass bands are per statistic, not a combined score, because the four fail in
different directions. A set can look too easy on relative contrast while being
too clustered on Gini, and a single score would average that away instead of
naming it. The gate file also pins the measurement conditions the bands were
set under, since these statistics are not comparable across different N, k or
nlist.

Every band is currently null. Bands are set once this family has a trained
ladder to show what is achievable; until then the gate file records that they
are unset, and the checker says so instead of passing.

Check a run against it:

    python -m src.eval.check_gate --dataset sift --run-dir runs/sift/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
