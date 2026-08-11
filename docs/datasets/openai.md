# OpenAI

1536-dimensional text embeddings from an OpenAI embedding model, already
unit-norm. The vectors come from a DBpedia text corpus, and the one
structural fact that decides how this family is modelled is that ambient
dimension is very high while intrinsic dimension is low.

## Source

    python -m src.data.fetch openai

Writes `data/openai_250k.npy`. This is the one family ann-benchmarks does
not publish as an HDF5: it names `dbpedia-openai-*-angular` as datasets but
generates them on demand from a HuggingFace dataset, so the fetcher reads
that dataset's parquet shards instead. They are large and immutable; the
fetcher downloads them once and is safe to run concurrently.

Only the 250k subset is written, unlike the five HDF5-backed families, which
default to 250k and 1M. `configs/openai/v0.yaml` names `openai_250k.npy` and
this family's canonical N is 20,000, so a 1M subset would cost 6GB to go
unread.

| | |
|---|---|
| Dimension | `1536` |
| Search metric | `angular` |
| Upstream | `KShivendu/dbpedia-entities-openai-1M` (HuggingFace) |
| Benchmark name | `dbpedia-openai-1000k-angular` (generated, not hosted) |

## Structure

1536-dimensional text embeddings, already unit-norm. Ambient dimension is
very high while intrinsic dimension is low, so LID and relative contrast are
the statistics that carry information; per-dimension marginals say almost
nothing at this width.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (best variant) |
|---|---|---|
| LID median | `31.88` | — |
| Relative contrast | `1.375` | — |
| Hubness skew | `1.555` | — |
| IVF cell-balance Gini | `0.552` | — |

Measured 2026-08-10 on `openai_250k.npy`, the corpus `configs/openai/v0.yaml`
names. The synthetic column stays empty until this family has a trained
rung; there is nothing to put in it yet.

Two properties of the corpus were confirmed rather than assumed while
measuring: every row is unit-norm to float32 precision (norm spread under
`3e-8`), and no two rows are duplicates. The first is what makes `angular`
measurable as L2 here at all.

Fill the real column with:

    python -m src.eval.eda_report \
        --real-path data/openai_250k.npy \
        --output-dir runs/openai/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

Read the four values out of runs/openai/profile/summary.json (written by the command above).

At this width the report drops its per-dimension marginals and correlation
panels, which are quadratic in the dimension and say little here; pass
`--max-panel-dim 1536` to force them back on.

`ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are neither unit-norm nor exactly zero rather than normalizing
them itself -- an exact zero is what `maybe_l2_normalize` leaves behind, so
it is accepted rather than treated as a caller mistake.

## Structural profile

The four gate statistics say how hard this corpus is to search. They do not
say what a generator for it should look like. That question -- how many
directions the data actually uses, whether it fills the sphere or a cone,
and which statistics are stable enough to gate on -- is answered by:

    python -m src.eval.openai_structure \
        --real-path data/openai_250k.npy \
        --output-dir runs/openai/structure

It writes `openai_structure.html` and `structure.json`, and reports the
noise floor: the spread of each gate statistic across disjoint draws of the
same real corpus. A band narrower than that spread would reject real data,
so it bounds how tight any band can be. It sets no bands.

It also measures the L2-versus-cosine question directly, which is the
empirical form of the argument above: whether the two induce the same
neighbour sets, and by what factor the distance-ratio statistics differ.

### Noise floor

Ten disjoint 20,000-row draws of the same real corpus, measured 2026-08-10.
The spread is what redrawing real data moves a statistic by, so a band
narrower than it would reject the real corpus. It bounds how tight any band
for that statistic can be; it does not say where the band goes.

| Statistic | min | median | max | spread | % of median |
|---|---|---|---|---|---|
| LID median | `31.71` | `31.80` | `31.96` | `0.245` | `0.77%` |
| Relative contrast | `1.3754` | `1.3769` | `1.3813` | `0.0059` | `0.43%` |
| Hubness skew | `1.441` | `1.495` | `1.582` | `0.140` | `9.38%` |
| IVF Gini | `0.5561` | `0.5697` | `0.5895` | `0.0334` | `5.87%` |

All four are usable here, which is not true of every family: SIFT's contrast
and Gini and GloVe's hubness skew each came out noise-dominated at their own
canonical N. Relative contrast and LID are stable to under a percent and can
carry tight bands. Hubness skew is the loosest at 9.4% and IVF Gini next at
5.9%, so bands for those two have to be at least that wide before they admit
real data at all.

### Angular versus L2, measured

The argument that `angular` can be measured as L2 between unit-norm rows
holds on this corpus, and by a wide margin:

| | |
|---|---|
| Neighbour-set agreement at k=100 | `0.9999935` |
| Hubness skew, L2 vs cosine | `1.5702607` vs `1.5702458` |
| LID median, L2 vs cosine | `31.810` vs `15.905` |

The neighbour sets are the same to within a rounding tie, so hubness — which
depends only on which points are neighbours — is invariant to five
significant figures. LID under cosine is exactly half the L2 value, which is
what `cos = L2^2/2` implies for a log-ratio estimator, so it rescales rather
than reorders. Re-running the profile under `--metric angular` reproduced the
L2 numbers bit for bit.

## Model family

`mlp` today, `spherical` when phase (b) lands.

### What the geometry says about `v0`

Measured 2026-08-10. These are readings, not decisions: no config below has
been changed, and each would be a ladder rung someone chooses deliberately.

**This corpus is a narrow cone, not a sphere.** The mean vector has norm
`0.83` — on an isotropic unit sphere it would be near zero — and random pairs
sit at cosine `0.688`, where an isotropic 1536-dimensional sphere would give
about `0.000 ± 0.026`. The typical vector is at cosine `0.83` to the mean
direction.

That is the case for `spherical` mattering more here than elsewhere. An
`mlp` reaching the sphere by dividing at the end has to place essentially all
of its output inside a cone of half-angle ~34 degrees, so almost all of the
space it can express is off-manifold. The anisotropy is a property of the
embedding model, not of this subset.

**Intrinsic dimension is roughly 23 to 32**, against an ambient 1536: two-NN
gives `22.7` and LID median `31.9`. `latent_dim: 512` is an order of
magnitude above that. A larger latent than manifold is not wrong on its own —
the generator can learn a degenerate map — but nothing in this measurement
argues for 512, and a rung at 64 or 128 would be a cheap thing to compare
against.

**Centering is the largest single lever, and the one with a catch.** The
spectrum about the origin, which is what `preprocess.center: false` hands the
generator, is dominated by the shared mean direction:

| | about the origin | about the mean |
|---|---|---|
| Top component's share of variance | `69.0%` | `3.6%` |
| Participation ratio | `2.10` | `175.3` |
| Components for 90% of variance | `155` | `459` |

By participation ratio the uncentered corpus effectively uses two directions
and the centered one 175 — a factor of 83. Most of what a `center: false`
generator spends capacity reproducing is a constant offset shared by every
row.

The catch is that centering is not free at evaluation time. Centered rows are
no longer unit-norm, and `ann_difficulty.compute` refuses rows that are
neither unit-norm nor exactly zero, so a centered run has to be mapped back
through its transform before it can be gated — which is what
`run_metadata.json` exists for, and which `compare_variants` already requires
for exactly this case. A centering rung is worth trying and is not a
one-line config change.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/openai/v0.yaml` | — | not trained |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/openai/v0.yaml

## Gate

`gates/openai.yaml` is the gate. The bands live there rather than in this
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

    python -m src.eval.check_gate --dataset openai --run-dir runs/openai/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
