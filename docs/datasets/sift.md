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

## Noise floor

How far each gated statistic moves when *nothing* changes but the seed. A band
tighter than this is unenforceable, and a ladder rung whose improvement is
smaller than this is indistinguishable from a reseed. Measured 2026-08-06 from
two 30k-step `v0` runs identical in every training hyperparameter except seed
(42 and 43), configs `configs/sift/noisefloor_{a,b}.yaml`. Both were sampled at
a *fixed* seed of 42, so only the training seed varies, and both were measured
in a single `eda_report` invocation under the canonical conditions above.

| Statistic | Seed-to-seed spread | As % of real | Distance from real, in units of that spread |
|---|---|---|---|
| LID median | 0.164 | 0.9% | 2.7x |
| Relative contrast median | 0.048 | 2.1% | **0.4x -- noise exceeds signal** |
| Hubness skew | 0.062 | 3.3% | 1.5x |
| IVF cell-balance Gini | 0.007 | 2.3% | **0.6x -- noise exceeds signal** |

The last column is the one that matters: it compares the seed-to-seed spread
against how far the generator sits from the real corpus. For relative contrast
and IVF Gini, reseeding moves the statistic *further than the generator's entire
deviation from SIFT*. The two runs do not even agree on the sign of the contrast
gap -- one lands above the real value, the other below. Neither statistic can
carry a meaningful band at this ladder's current distance from real, and neither
should be used to attribute a rung-to-rung improvement. LID median is the one
comfortably usable statistic; hubness skew is marginal.

**This is n=2.** A single paired difference has one degree of freedom: it
establishes the floor's order of magnitude and nothing more, and the true spread
could be wider. Three to five seeds are needed before any of these numbers
justifies writing a band into `gates/sift.yaml`. No band was set from this
measurement -- `gates/sift.yaml` is unchanged and every band there is still null.

Reproduce with:

    python -m src.train.train_wgan_gp --config configs/sift/noisefloor_a.yaml
    python -m src.train.train_wgan_gp --config configs/sift/noisefloor_b.yaml

then sample each `best_generator.pt` at a fixed seed, pass both to one
`eda_report --synthetic-path LABEL=PATH`, and difference the two labels'
entries in `summary.json`. A 30k-step run is ~34 min on one RTX 4060.
