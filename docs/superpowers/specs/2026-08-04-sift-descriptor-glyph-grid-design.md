# SIFT descriptor glyph grid

A qualitative sanity-check figure: do the descriptors our GAN variants generate
*look* like real SIFT descriptors?

## Motivation

Every existing eval panel is an aggregate — distance CDFs, per-dimension
marginals, PCA spectra, ANN difficulty. All of them can look healthy while the
generator produces descriptors that are structurally wrong, because a matched
marginal says nothing about whether the 128 numbers form a plausible *gradient
histogram*.

A SIFT descriptor is not an arbitrary point in R^128. It is a 4x4 grid of
spatial cells, each holding an 8-bin histogram of gradient orientations
(Lowe 2004). Real descriptors are sparse and spiky: most cells are dominated by
one or two directions, because most image patches contain edges. Rendering a
descriptor as a 4x4 grid of 8-ray stars makes that structure directly visible,
and makes its absence obvious.

## What this is not

The glyph is a diagram of the descriptor, not the image patch it came from.
SIFT is lossy and orientation-normalised; recovering the patch would need a
learned inversion model and a patch database, which is far outside this scope.

## Data layout

The descriptor index convention is:

    index = (row * 4 + col) * 8 + orientation_bin

so `(128,)` reshapes to `(4, 4, 8)` indexed `[row][col][bin]`.

The row/col scan order and the zero direction of the orientation bins differ
between SIFT implementations (VLFeat, OpenCV, Lowe's original binary), and we
do not know which extractor produced SIFT1M. The glyph may therefore be rotated
or transposed relative to the true patch geometry. This does not affect the
comparison: the convention is applied identically to real and generated
vectors, so "does this look like real SIFT" remains valid. It would only matter
for a claim about underlying image content, which we do not make.

## Two facts that shape the design

**1. Preprocessing is pure L2 normalisation.** All four variant configs set
`center: false`, `whiten: false`, `l2_normalize: true`. This is the only reason
dimension *k* still means "cell *i*, bin *j*". Centering is a constant offset
and whitening is a dense linear mix; under either, a glyph would become a
picture of mixed bins while still looking plausible. The tool therefore refuses
to render any variant whose `run_config.yaml` enables centering or whitening,
rather than drawing a silent lie.

**2. Three of four variants can emit impossible values.** `MLPGenerator.forward`
(`src/models/generator.py:27`) is a bare `Linear` with no output activation, so
its samples are neither non-negative nor unit-norm. `GatedGenerator.forward`
(line 114) is `softplus x gate` followed by L2 normalisation, so it is strictly
non-negative and unit-norm. v0, v1 and v1_5 resolve to `mlp` (no
`generator_type` key); only v2 is `gated`.

Real SIFT bins are gradient-magnitude histogram counts and are non-negative by
construction. Negative bins are not a subtle distributional mismatch — they are
physically impossible. Clamping them to zero at render time would flatter
v0/v1/v1_5 and hide the most obvious defect in the comparison, so the renderer
shows them instead.

## Architecture

Two modules, splitting pure geometry from I/O.

The split exists because the risky part — the index convention and the angle
mapping — is exactly the part that cannot be verified on the development
machine, which has neither `data/` vectors nor `runs/` checkpoints (both live
on tig-gpu). Isolating it behind pure functions lets tests pin the convention
locally, leaving only plumbing to run on the GPU box.

### `src/eval/descriptor_glyph.py`

Pure NumPy. No I/O, no Plotly import.

```python
CELL_ROWS = CELL_COLS = 4
ORIENTATION_BINS = 8
DESCRIPTOR_DIM = 128   # 4 * 4 * 8

def descriptor_to_cells(vec: np.ndarray) -> np.ndarray:
    """(128,) -> (4, 4, 8) indexed [row][col][orientation_bin]."""

def shared_scale(descriptors: np.ndarray, percentile: float = 99.0) -> float:
    """Value -> ray-length factor, from the non-zero |value| across every
    descriptor that will be plotted. Returns 0.0 for an all-zero input."""

def glyph_segments(
    cells: np.ndarray,      # (4, 4, 8)
    origin: tuple[float, float],   # centre of the whole 4x4 glyph
    pitch: float,           # centre-to-centre cell spacing
    scale: float,           # from shared_scale, identical for every glyph
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(pos_x, pos_y, neg_x, neg_y): NaN-separated ray endpoints, split
    into non-negative and negative bins. NaN is the gap marker rather than
    None so the arrays stay numeric and directly assertable in tests;
    Plotly breaks scatter lines on NaN identically."""
```

Three decisions live here:

- **Negatives get their own coordinate arrays.** Ray length is always
  `|value| * scale`; the sign only decides which pair the segment lands in, so
  the CLI can colour negative rays as a warning. Nothing is clamped.
- **One shared scale for the whole figure**, computed once by `shared_scale`
  from the 99th percentile of the *non-zero* `|value|` across every descriptor
  plotted — real and generated together — and passed unchanged to every
  `glyph_segments` call. Zeros are excluded because real SIFT is sparse: over
  the raw values the percentile can land inside the run of zeros, yielding a
  scale of 0.0 and an empty figure. Excluding them also makes the reference
  mean "a typical meaningful bin".
  Per-glyph normalisation would rescale each glyph to fill its box, making a
  near-flat generated descriptor look as structured as a real one. The
  percentile rather than the max stops one outlier bin shrinking everything
  else into invisibility. Because it is a percentile, some rays exceed it;
  those are clipped to a maximum length of `pitch / 2`, so a ray never crosses
  into a neighbouring cell.
- **Angle mapping** is `bin j -> j * 45 degrees`, counter-clockwise from `+x`,
  with row 0 at the top so the grid reads like an image.

### `src/eval/plot_descriptor_grid.py`

CLI. Flags follow `compare_variants` for consistency: `--real-path`,
`--real-format`, `--output-dir`, `--root`, `--num-samples` (per row, default 8),
`--seed` (default 42), `--no-png`, `--plotlyjs`.

## Figure

Rows, top to bottom: `real-a`, `real-b`, then one row per resolved variant in
`VARIANTS` order (v0, v1, v1_5, v2). Columns are `--num-samples` independent
draws.

Two real rows rather than one, because the reader cannot otherwise calibrate:
without a sense of how much two real descriptors differ from each other, a
variant row is just a vibe. The real-a/real-b gap is the baseline against which
every row below it is read.

Samples are drawn at random under a fixed seed, never selected. Nearest-real-
neighbour pairing was considered and rejected: it flatters the generator and
raises memorisation questions the figure would then have to answer.

## Data flow

1. Load real vectors with `load_descriptors`, then L2-normalise, putting real
   and generated in the same space the critic saw.
2. Draw `2 * num_samples` distinct indices from a seeded RNG; split into
   `real-a` and `real-b`.
3. `compare_variants.resolve_variants` determines which checkpoints exist. For
   each, rebuild the generator from its `run_config.yaml`, seed with
   `compare_variants.variant_seed(seed, name)`, and `sample_generator`.

   The seeding is load-bearing. `GatedGenerator._sample_gate` samples gate
   noise in `eval()` mode too (documented at `src/models/generator.py:32`), so
   v2 is only reproducible under an explicit seed. Reusing `variant_seed` also
   keeps a row identical whether or not other variants' checkpoints happen to
   be on the machine.
4. Lay all glyphs in one shared axes at `(col * pitch, row * pitch)`. One
   positive-ray trace per row, each with its own colour and legend entry, plus
   a single shared negative-ray trace in red. Row labels as left-margin
   annotations; a rule between the real block and the variant block. The axes
   are aspect-locked so glyphs stay square, with ticks and grid hidden.
5. Write `descriptor_grid.html`, plus PNG through the same kaleido path
   `eda_report` uses, unless `--no-png`.

## Error handling

Degrade and carry on, matching `compare_variants`' existing stance that a
partial comparison is still worth reading:

- A variant with no run directory, no `best_generator.pt`, or no
  `run_config.yaml` is skipped with a printed reason.
- If no variant resolves at all, the two real rows still render. This is the
  case on the development laptop, and it keeps the real-vs-real baseline
  workable without the GPU box.

Refuse loudly:

- Real path missing or unreadable.
- Descriptor dimension other than 128 — the 4x4x8 mapping is undefined.
- Fewer than `2 * num_samples` real vectors available.
- A variant whose `run_config.yaml` enables `center` or `whiten`, for the
  reason given above.

## Testing

All of it runs on the development machine with no data and no checkpoints.
Run with `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`.

### `tests/test_descriptor_glyph.py`

NumPy only, no torch.

- **Index convention pinned exactly**: `descriptor_to_cells(np.arange(128))`
  gives `cells[r, c, b] == (r*4 + c)*8 + b`. Stops the layout drifting silently.
- **One-hot gives one ray**: a single non-zero bin yields exactly one positive
  segment, from the expected cell centre at the expected angle. Parametrised
  over several `(row, col, bin)` triples including corners.
- **Angle and orientation**: bin 0 points along `+x`; bin 2 is 90 degrees
  counter-clockwise from it; cell `(0,0)` sits above cell `(3,0)` in *y*.
- **Negatives are separated, not clamped**: one negative bin produces a ray of
  length `|value|` in the negative arrays and nothing in the positive ones.
  This test protects the honesty property that motivates the whole design.
- **Shared scale is linear**: doubling `scale` doubles ray length, up to the
  clip.
- **Clipping**: a bin far above the percentile produces a ray of exactly
  `pitch / 2`, never longer, so it cannot bleed into a neighbouring cell.
- **`shared_scale` ignores outliers**: a batch of typical descriptors plus one
  with a huge spike gives nearly the same scale as the batch alone.
- **`shared_scale` ignores zeros**: a sparse batch, mostly zeros, still yields
  a scale that maps its non-zero values to visible rays.
- **Edge cases**: the zero vector produces no rays rather than dividing by
  zero; `shared_scale` of all-zero input returns `0.0` without warning;
  lengths 127 and 129 raise `ValueError`.

### `tests/test_plot_descriptor_grid.py`

Integration against `tmp_path`.

- Synthetic real `.npy`, no run directories: writes HTML with exactly the two
  real rows. The laptop-developable path.
- A real tiny checkpoint: the variant row renders.
- `run_config.yaml` with `whiten: true`: raises, reason in the message.
- Missing checkpoint: skipped with a message, poster still written.
- Real data with dim != 128: raises.

### Shared fixtures

`tests/test_compare_variants.py` already has `_write_tiny_gated_run` (line 86)
and `_make_run_dir` (line 28), both module-private and both needed here.
Promote them to `tests/conftest.py` as fixtures rather than duplicating
checkpoint-writing logic. `test_compare_variants.py` changes only to drop the
helpers and use the fixtures; its existing tests must pass unchanged.

## Out of scope

- Inverting a descriptor back to an image patch.
- Any numeric score ranking the variants — this figure is qualitative by
  intent. Ranking is what the existing `eda_report` panels are for.
- Adding a panel to `eda_report.py`. At 1017 lines it is already triple the
  size of any other module, and its panel contract is aggregates over ~50k
  vectors, not eight individual ones.
