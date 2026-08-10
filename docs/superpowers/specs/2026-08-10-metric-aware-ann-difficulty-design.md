# Metric-aware ANN difficulty

Phase (c) of `docs/superpowers/specs/2026-08-04-multi-dataset-ann-emulation-design.md`,
narrowed to what measurement actually requires.

Closes issue #22 (`ann_difficulty.py` is the last consumer that could inherit
`--dataset`) and issue #16 (re-measure the angular families' real profiles).

## Problem

`data.metric` records `l2` or `angular` per family and is inert: nothing reads
it. `src/eval/ann_difficulty.py` measures every family under L2, including the
four `angular` ones (`deep`, `glove`, `nytimes`, `openai`).

The obvious reading — that angular families are being measured under the wrong
distance and their numbers are wrong — is not what is happening. Three facts,
each checked rather than assumed:

**Every set that reaches `ann_difficulty.compute` already has unit-norm rows.**
All six family configs set `data.preprocess.l2_normalize: true`, and
`compare_variants.build_report_args` hardcodes `preprocess="l2"`, which
`series.load_series` applies to the real set and every overlay alike.

**On unit-norm rows the metric choice is almost entirely cosmetic.** Euclidean
and cosine distance are related by `‖a−b‖ = √(2·cos_dist)`, which is strictly
monotone, so the two rank neighbours identically. Measured on 3000 unit-norm
96d rows at k=100:

| quantity | euclidean | cosine |
|---|---|---|
| LID median | 40.662956 | 20.331474 |
| k-NN sets | — | identical |
| rows differing in neighbour *order* | — | 14 of 3000 (float tie-breaks) |

The LID ratio is exactly 2.000000, because `r_i/r_k` under cosine is the square
of the ratio under Euclidean, so every log doubles. Hubness skew and IVF Gini
do not move at all. Switching to cosine would rescale one statistic by a
constant and change nothing about what it can discriminate.

Where the metric does bite is **unnormalized** rows: on rows with norms spread
over `[0.2, 5.0]`, only 12.5% of neighbour slots agree between the two. That
case does not arise today.

**Only one family has measured numbers.** `glove.md`, `nytimes.md` and
`openai.md` all record "not yet measured". `deep.md` is the only page with a
filled-in profile, and `docs/datasets/deep_ladder_summary.json` records
`preprocess: l2` — so DEEP was measured on unit-norm rows.

So the real defect is not the distance function. It is that **nothing ties the
measurement geometry to `data.metric`**. `--preprocess` is a free CLI flag that
merely happens to default to `l2`, and `compare_variants` hardcodes it. An
`angular` corpus measured at `--preprocess none` would be silently wrong today,
and nothing anywhere would flag it.

## What `angular` means here

**`angular` is L2 on the unit sphere.** Rows must already be unit-norm; the
measurement is then the chord distance between them, which orders neighbours
exactly as cosine does and keeps every estimator's math on a true metric.

Rejected alternatives:

- **Cosine distance (`1 − cos`).** Matches the issue's literal framing, but
  halves LID by a constant on unit rows, moves nothing else, and is not a
  metric — the triangle inequality fails, which the Hill LID estimator's
  volume-growth argument leans on.
- **`arccos(cos)`, the geodesic metric.** A true metric, but needs a custom
  sklearn callable and so a slower k-NN, in exchange for values that differ
  from the chord ones only in the third decimal at these angles.

## Code shape

### `src/eval/ann_difficulty.py`

`compute` gains `metric: str = "l2"`. The default preserves every existing
call site.

```python
METRICS = ("l2", "angular")
```

Declared locally rather than imported from `src.data.dataset`, which imports
torch. This module's docstring commits to staying usable and testable without
heavy dependencies, and `_subsample` is already duplicated here for the same
reason: the dependency direction is worth more than the sharing.

The angular path adds a precondition and nothing else:

```python
def require_unit_norm(x, metric, atol=1e-4) -> None:
    """Refuse to measure `angular` on rows that are not on the unit sphere."""
```

Called from `compute` *after* `_subsample`, because the subsample is what gets
measured. It raises `ValueError` naming `--preprocess` and reporting the
observed norm range.

**Refusing rather than normalizing is the design.** Normalizing inside
`compute` would let the report's `preprocess:` line say `none` while the
difficulty panels were measured on normalized rows. A measurement function
that silently transforms its input is how that divergence gets shipped.

**Zero rows are accepted.** `series.maybe_l2_normalize` deliberately leaves an
all-zero row at zero rather than dividing by ~0, so norms of exactly `0.0` are
a legitimate output of our own preprocessing. Norms are accepted when ~1 or
exactly 0, and rejected otherwise.

`knn`, `survivor_mask`, `lid_mle`, `relative_contrast`, `cell_occupancy`,
`k_occurrence`, `hubness_skew`, `gini` and `summary` are all unchanged.

### Threading it through

The chain is `compare_variants` → `argparse.Namespace` → `EdaConfig` →
`build_context` → `compute`. The field appears at every link or
`test_report_args_match_eda_report_fields` fails.

- `src/eval/eda/config.py` — `METRIC_DEFAULT = "l2"` beside the other shared
  ANN defaults; `EdaConfig.metric`; `from_args` reads `args.metric`.
- `src/eval/eda/cli.py` — `--metric`, `default="l2"`, `choices=["l2",
  "angular"]`.
- `src/eval/eda/pipeline.py` — pass `metric=cfg.metric` to `compute`, and
  record it in `summary.json` under `ann_settings`, so a gate result carries
  the geometry it was measured under.
- `src/eval/compare_variants.py` — resolve the family's metric, and add
  `metric` to `build_report_args`.

### Where `compare_variants` reads the metric

From each variant's **repo config** (`variant.config_path`), never from
`run_dir/run_config.yaml`.

Run configs predate the `data.metric` field, so a DEEP run config has no
metric and would fall back to `l2` — silently wrong for the exact family this
change exists for. This is issue #18's argument pointed the other way: a run
config is evidence of what ran, not a statement about what the corpus is.

Read from every manifest-listed entry rather than only the resolved ones, so
the measurement geometry cannot depend on which checkpoints happen to be on
the box. Configs disagreeing within one family is a `SystemExit` naming the
offenders; a missing config file likewise.

`build_report_args`'s hardcoded `preprocess="l2"` stays. It already supplies
exactly what `angular` requires, for every family.

`compare_variants` gains no `--metric` flag. The metric is a property of the
corpus, already recorded per family in config; a flag would be a second place
to state it, and so a place for it to go stale. `eda.cli` has one only because
it is invoked against bare `.npy` paths with no config in sight.

## Documentation

Seven statements across five files become false the moment this lands, and are
corrected in the same change:

- `docs/datasets/deep.md` — the "will need re-measuring once angular distance
  support lands (phase (c))" caveat, and the sentence in the gate-band section
  claiming the numbers move again when phase (c) re-measures the family.
- `docs/datasets/glove.md`, `nytimes.md`, `openai.md` — the same caveat.
- `PROJECT_DOCUMENTATION.md` — the `data.metric` section ("otherwise inert
  today: nothing consumes it yet"), and the phase (c) note in the gate
  section.

The correction states that these numbers were measured on unit-norm rows at
`preprocess: l2`, which *is* the angular geometry, so they stand.
`deep_ladder_summary.json` records that `preprocess: l2`, so the claim is
checkable from this repo alone.

`PROJECT_DOCUMENTATION.md` and the four family pages are authoritative and are
policed by `tests/test_docs_references.py`, so every path, anchor and symbol
they cite must still resolve.

## Testing

The premise gets pinned, not only the plumbing:

- **`angular` and `l2` produce identical summaries on unit-norm rows.** This
  is the entire justification for the chosen definition and for leaving
  `deep.md`'s numbers in place. If it stops holding, the documentation
  correction is false.
- `angular` raises on non-unit rows, and the message names `--preprocess`.
- Zero rows are accepted; a row at norm 0.5 is not.
- `l2` is unaffected by non-unit rows.
- `compute`'s default metric is `l2`, so untouched callers are untouched.
- `compare_variants` reads the metric from the manifest's configs; a family
  whose configs disagree exits.
- `EdaConfig.from_args` carries `metric`; the CLI defaults it to `l2`.

`test_report_args_match_eda_report_fields` covers Namespace parity for free.

## Scope

In: the four source files above, their tests, and the seven documentation
corrections across five files.

Out:

- **First-time measurement of `glove`, `nytimes` and `openai`.** They have no
  measured profile at all, under any metric. That is data work needing the
  corpora on the box, and GloVe already appears to be in flight elsewhere.
- **Re-running DEEP.** Its numbers stand under this definition; that is the
  finding, not a shortcut.
- **Spherical k-means for IVF cell balance.** FAISS clusters an
  inner-product index by renormalizing centroids each iteration, which plain
  `MiniBatchKMeans` on unit rows does not do. Normalize-then-measure applies
  the same reasoning here as everywhere else in this change; revisiting it is
  its own question with its own evidence.
- **Issues #17, #19, #20, #21.** Independent.
