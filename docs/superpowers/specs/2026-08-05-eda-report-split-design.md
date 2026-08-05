# Splitting `eda_report.py` into a panel package

Date: 2026-08-05
Status: design approved, not yet implemented

## Problem

`src/eval/eda_report.py` is 1328 lines, the largest module in the repo, and
`AGENTIC-REVIEW.md:196` already names it as "the file most likely to be edited
by an agent and the hardest for one to hold in context".

The size alone is not the whole problem. `run()` is a single ~290-line function
that interleaves five unrelated concerns: computing the ANN metrics, building
thirteen figures, writing the prose note attached to each one, assembling the
HTML, and emitting `summary.json`. Three panels are conditional, one panel's
figure function also returns a `summary.json` payload, and two panels' titles or
notes are computed from the parsed arguments. There is no seam at which a panel
can be read, changed or added without loading the whole file.

Concretely, an agent asked to "add a panel" or "fix the hubness note" today has
to hold 1328 lines and edit inside a 290-line function whose control flow mixes
presentation with computation.

## Non-goals

This is a pure refactor. No new panels, no behaviour changes, no bug fixes, no
new metrics. Anything discovered along the way is recorded in `FOLLOWUPS.md`
rather than fixed here, so the diff stays reviewable as a move.

## Module layout

A `src/eval/eda/` package. `src/eval/eda_report.py` shrinks to a CLI entrypoint
so that `python -m src.eval.eda_report` — the command baked into
`docs/datasets/sift.md`, `README.md` and `check_gate.py`'s error strings —
keeps working unchanged.

| Module | Holds | ~LOC |
|---|---|---|
| `config.py` | `EdaConfig` dataclass and `from_args()`; the `*_DEFAULT` constants | 90 |
| `cli.py` | `parse_args` — argparse and nothing else | 120 |
| `series.py` | `Series`, `REAL_NAME`, `REAL_COLOR`, `SYNTH_PALETTE`, `parse_synthetic_spec`, `subsample`, `maybe_l2_normalize`, `load_series` | 120 |
| `metrics.py` | `pairwise_distance_sample`, `nn_distances`, `wasserstein1`, `effective_rank`, `summary_stats`, `dimension_divergence` | 160 |
| `figures.py` | `shared_edges`, `overlay_hist_fig`, the nine `fig_*` builders | 330 |
| `glyphs.py` | `GLYPH_*` constants, `glyph_rows`, `fig_descriptor_glyphs` | 150 |
| `notes.py` | `ann_condition_note`, `ann_discarded_note`, `ANN_NOTE_SUFFIX` | 90 |
| `panels.py` | `Context`, `Panel`, `PANELS` — the thirteen panels with their prose | 260 |
| `html.py` | `REPORT_CSS`, `CDN_SRC`, `plotlyjs_head`, `format_stat`, `stats_table_html`, `build_report`, `export_pngs` | 130 |
| `run.py` | build the context, walk `PANELS`, write HTML, PNGs and `summary.json` | 60 |

`__init__.py` holds a package docstring and no re-exports. Every name has
exactly one import path; there is no second way to reach it.

Two functions change shape rather than just moving:

- **`parse_synthetic_spec` moves to `series.py`.** It parses a series
  specification, not a command line, and `compare_variants` uses the
  `LABEL=PATH` format without going through argparse.
- **`fig_dim_divergence` splits in two.** Today it computes the per-dimension
  Wasserstein distances, draws them, *and* returns the `worst_dimensions`
  payload that ends up in `summary.json`. After the split,
  `metrics.dimension_divergence(series, top_k)` returns a `DimDivergence`
  holding the distances, the shared ordering and the worst-dimension list, and
  `figures.fig_dim_divergence(divergence, series)` only draws. A summary
  payload stops travelling out of a figure function.

`glyphs.py` is separate from `figures.py` because `plot_descriptor_grid.py`
imports `fig_descriptor_glyphs` and the glyph constants directly, and that is
its whole dependency on this package.

## The panel registry

```python
@dataclass(frozen=True)
class Context:
    config: EdaConfig
    series: list[Series]
    ann_metrics: dict[str, ann_difficulty.AnnMetrics]
    divergence: DimDivergence | None    # None when there is no synthetic overlay


@dataclass(frozen=True)
class Panel:
    title: Callable[[Context], str] | str
    note: Callable[[Context], str] | str
    build: Callable[[Context], go.Figure | None]
```

**`build` returning `None` means "omit this panel".** That is the single skip
mechanism, and it covers all three of today's conditionals uniformly:

- *Descriptor glyphs* — needs every series to be 128-dimensional and large
  enough. The applicability test and the drawing share the work of choosing
  rows (`glyph_rows` returns `[]` to mean "cannot draw"), which is exactly why
  a separate `applies(ctx)` predicate would be wrong here.
- *Vector norms* — only when `preprocess == "none"`.
- *Per-dimension mismatch* — only when there is at least one synthetic set.

`title` and `note` accept either a literal string or a callable because two
panels are dynamic: the within-set k-NN title embeds `--knn`, and the three ANN
panels' notes embed the measured conditions via `notes.ann_condition_note`.

`Context` deliberately does not carry the summary statistics. No panel reads
them — they feed the header table and `summary.json` — so `run.py` computes them
alongside the context and keeps them out of the panel contract.

`PANELS` is a module-level list in the current report order. Adding a panel is
appending one `Panel(...)` and adding one `fig_*` to `figures.py` — two files in
context instead of 1328 lines.

## Orchestration

```python
def run(args: argparse.Namespace) -> Path:
    ctx = build_context(EdaConfig.from_args(args))
    sections = [
        (resolve(p.title, ctx), resolve(p.note, ctx), fig)
        for p in PANELS
        if (fig := p.build(ctx)) is not None
    ]
    # write HTML, best-effort PNGs, summary.json
```

`run` keeps its exact signature, `run(args: argparse.Namespace) -> Path`.
`compare_variants.build_report_args` hand-builds that Namespace and
`tests/test_compare_variants.py::test_report_args_match_eda_report_fields`
asserts field-for-field parity with `parse_args`; both stay untouched. The
Namespace is converted to `EdaConfig` on entry and nothing downstream sees
argparse.

`EdaConfig.from_args` preserves today's defensive
`getattr(args, "glyph_samples", GLYPH_SAMPLES_DEFAULT)`, since
`compare_variants` has historically built Namespaces missing that field.

## Consumers

- **`src/eval/eda_report.py`** — reduced to `parse_args`/`run` wiring and
  `main()`. Roughly twenty lines. No re-exports.
- **`src/eval/plot_descriptor_grid.py`** — imports move from
  `from src.eval import eda_report` to `from src.eval.eda import glyphs, html,
  series`. It uses `fig_descriptor_glyphs`, `GLYPH_REAL_COLORS`,
  `plotlyjs_head`, `export_pngs` and `maybe_l2_normalize`.
- **`src/eval/compare_variants.py`** — the five `*_DEFAULT` constants come from
  `src.eval.eda.config`; `run` from `src.eval.eda.run`.
- **`src/eval/check_gate.py`** — reads `summary.json` only. Unchanged, and its
  reference to the `python -m src.eval.eda_report` command stays correct.
- **`AGENTIC-REVIEW.md:196`** — the line recommending this split is updated to
  record that it happened.

## Verification

The refactor must be provably behaviour-preserving.

**Golden snapshot.** Before any edit, on the base commit, generate a report from
fixed-seed synthetic data — a real set plus two overlays, sized so every panel
including the glyphs is exercised — with `--plotlyjs cdn --no-png`, and save
`eda_report.html` and `summary.json` outside the repo. Regenerate with the same
seed and arguments after the refactor and require equality.

**The UUID wrinkle.** Plotly stamps a fresh random UUID into each figure's
`<div id=...>` on every render, so raw HTML bytes never match across two runs.
The comparison normalises those UUIDs to a fixed placeholder first, then
requires byte equality. That still covers every figure's embedded trace JSON, so
a changed number, colour, trace order or note fails the check. `summary.json` is
compared unmodified.

**Test suite.** Existing tests move to files mirroring the new modules —
`test_eda_series.py`, `test_eda_metrics.py`, `test_eda_notes.py`,
`test_eda_html.py`, `test_eda_panels.py`, `test_eda_run.py` — with their bodies
unchanged apart from import paths. `tests/test_plot_descriptor_grid.py`,
`tests/test_compare_variants.py` and `tests/test_check_gate.py` also name
`eda_report` and get the same import-path-only treatment. No test is rewritten
and no assertion is weakened; a test that no longer compiles against the new
layout is a signal the split is wrong, not a test to edit.

**Definition of done.** `make check` green (ruff lint, ruff format check, the
full pytest suite), the golden comparison clean, and `python -m
src.eval.eda_report --help` unchanged.

## Risks

- **Circular imports.** `panels.py` imports `figures`, `glyphs`, `metrics` and
  `notes`; `run.py` imports `panels` and `html`. The dependency graph is a DAG
  with `config.py` and `series.py` at the bottom and `run.py` at the top.
  `Context` lives in `panels.py` rather than a shared module, because nothing
  below `panels.py` needs it.
- **Silent prose drift.** The section notes are long strings being moved by
  hand. The golden HTML comparison is what catches a dropped sentence or a
  changed entity reference; it is not optional.
- **Import churn in a shared repo.** Other worktrees on this repo may hold
  in-flight edits to `eda_report.py`. This branch renames the module's contents
  wholesale, so it should land before, not alongside, other work touching that
  file.
