# eda_report Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/eval/eda_report.py` (1328 lines) into a `src/eval/eda/` package with a declarative panel registry, without changing a single byte of the report it produces.

**Architecture:** Ten focused modules replace one file. The thirteen report panels become `Panel` objects in `panels.py`, each holding a title, a prose note and a figure builder; `build` returning `None` means the panel is omitted. A `Context` dataclass carries everything a panel needs. `eda_report.py` survives as a ~20-line CLI entrypoint so `python -m src.eval.eda_report` keeps working.

**Tech Stack:** Python 3.12, numpy, plotly, scikit-learn, pytest, ruff.

## Global Constraints

- **This is a pure refactor.** No behaviour changes, no new panels, no bug fixes, no new metrics. Anything suspicious found along the way goes into `FOLLOWUPS.md` as a note, never into the code.
- **Every commit must be green.** `make check` (ruff lint, ruff format check, pytest) passes at every task boundary. Baseline is 433 passing tests.
- **Every commit must be golden-clean.** The report comparison from Task 0 must pass at every task boundary. This is the real safety net; the test suite alone does not cover the prose notes.
- Run everything from the worktree root, `/home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/eda-report-split`. The interpreter is `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python` — there is no `.venv` inside the worktree.
- **Moved code is moved verbatim.** When a task says "move", the function body, docstring and comments are copied character-for-character. Do not reword a comment, rename a local, or "tidy" a docstring in the same commit as a move. The only permitted edits are import statements and the signature changes this plan names explicitly.
- `src/eval/eda/__init__.py` contains a package docstring and nothing else. No re-exports — every name has exactly one import path.
- `run(args: argparse.Namespace) -> Path` keeps its exact signature throughout. `compare_variants.build_report_args` hand-builds that Namespace and a parity test asserts it matches `parse_args` field-for-field.

**Naming change from the spec:** the spec calls the orchestrator module `run.py`. This plan names it **`pipeline.py`** instead. `compare_variants` imports the module and calls `module.run(...)`, and `from src.eval.eda import run` followed by `run.run(...)` reads badly and collides with the local name `run` in tests that monkeypatch it. `pipeline.run(...)` is unambiguous. Nothing else about the spec changes.

---

## File structure

| File | Responsibility |
|---|---|
| `src/eval/eda/__init__.py` | Package docstring only |
| `src/eval/eda/config.py` | `EdaConfig` + `from_args`; the six `*_DEFAULT` constants |
| `src/eval/eda/cli.py` | `parse_args` — argparse and nothing else |
| `src/eval/eda/series.py` | `Series`, colours, `parse_synthetic_spec`, `subsample`, `maybe_l2_normalize`, `load_series` |
| `src/eval/eda/metrics.py` | `pairwise_distance_sample`, `nn_distances`, `wasserstein1`, `effective_rank`, `summary_stats`, `DimDivergence`, `dimension_divergence` |
| `src/eval/eda/figures.py` | `shared_edges`, `overlay_hist_fig`, eight `fig_*` builders |
| `src/eval/eda/glyphs.py` | `GLYPH_*` constants, `glyph_rows`, `fig_descriptor_glyphs` |
| `src/eval/eda/notes.py` | `ANN_NOTE_SUFFIX`, `ann_condition_note`, `ann_discarded_note` |
| `src/eval/eda/panels.py` | `Context`, `Panel`, `PANELS` — the thirteen panels and their prose |
| `src/eval/eda/html.py` | `REPORT_CSS`, `CDN_SRC`, `plotlyjs_head`, `format_stat`, `stats_table_html`, `build_report`, `export_pngs` |
| `src/eval/eda/pipeline.py` | `build_context`, `run` |
| `src/eval/eda_report.py` | CLI entrypoint: `main()` |

Dependency order (each imports only from those above it): `config` → `series` → `metrics` → `figures`/`glyphs`/`notes` → `panels` → `html` → `pipeline`.

Tasks run in that order so that every commit leaves a working tree.

---

### Task 0: Golden snapshot harness

Nothing else can start until there is a way to prove the report did not change.

**Files:**
- Create: `/tmp/claude-1000/-home-fibonadithya-TIG-wgan-synthetic/65ca9f9c-9966-4a1c-aba3-ba55a5717d50/scratchpad/golden/make_data.py`
- Create: `.../scratchpad/golden/snapshot.sh`
- Create: `.../scratchpad/golden/compare.py`

These live in the scratchpad, not the repo: they are a verification tool for this branch, not a project artefact.

- [ ] **Step 1: Write the fixture generator**

`make_data.py`:

```python
"""Deterministic stand-in corpora for the eda_report golden comparison.

Not realistic SIFT -- realism is irrelevant here. What matters is that the
numbers are fixed under a seed and that the shapes exercise every panel.
"""

import sys
from pathlib import Path

import numpy as np


def sift_like(n: int, dim: int, seed: int) -> np.ndarray:
    """Non-negative, zero-heavy, quantized -- enough shape to fill every panel."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=0.7, scale=30.0, size=(n, dim))
    x[rng.random((n, dim)) < 0.35] = 0.0
    return np.clip(np.rint(x), 0, 255).astype(np.float32)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "real128.npy", sift_like(3000, 128, 1))
    np.save(out / "synth_a.npy", sift_like(2000, 128, 2))
    np.save(out / "synth_b.npy", sift_like(2000, 128, 3))
    np.save(out / "real64.npy", sift_like(3000, 64, 4))
    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the snapshot driver**

`snapshot.sh` takes one argument — the directory to write into (`before` or `after`) — and produces three reports covering all thirteen panels plus the three skip paths.

```bash
#!/usr/bin/env bash
# Generate the three golden reports into $1.
#
#   l2      : the default path. 12 panels (no vector-norms panel).
#   none    : --preprocess none, which is the only way to get the
#             vector-norms panel. 13 panels.
#   real64  : no overlay and dim != 128, so both the per-dimension
#             mismatch panel and the glyph panel drop out. 10 panels.
set -euo pipefail

DEST="$1"
ROOT=/home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/eda-report-split
PY=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python
FIX="$(dirname "$0")/fixtures"

cd "$ROOT"
mkdir -p "$DEST"

common=(--synthetic-path "a=$FIX/synth_a.npy"
        --synthetic-path "b=$FIX/synth_b.npy"
        --max-vectors 1500 --num-pairs 5000 --bins 30
        --ann-max-rows 800 --knn-max-rows 800 --ann-k 20 --ivf-nlist 16
        --seed 7 --plotlyjs cdn --no-png)

"$PY" -m src.eval.eda_report --real-path "$FIX/real128.npy" \
    "${common[@]}" --preprocess l2 --output-dir "$DEST/l2"

"$PY" -m src.eval.eda_report --real-path "$FIX/real128.npy" \
    "${common[@]}" --preprocess none --output-dir "$DEST/none"

"$PY" -m src.eval.eda_report --real-path "$FIX/real64.npy" \
    --max-vectors 1500 --num-pairs 5000 --bins 30 \
    --ann-max-rows 800 --knn-max-rows 800 --ann-k 20 --ivf-nlist 16 \
    --seed 7 --plotlyjs cdn --no-png --preprocess l2 \
    --output-dir "$DEST/real64"

echo "snapshot written to $DEST"
```

- [ ] **Step 3: Write the comparator**

`compare.py`. The UUID normalisation is the whole point: plotly stamps a fresh random id into each figure's container div and into the matching `Plotly.newPlot(...)` call on every render, so raw bytes never match twice.

```python
"""Compare two eda_report snapshots, ignoring plotly's per-render UUIDs."""

import re
import sys
from pathlib import Path

UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def normalized(path: Path) -> str:
    return UUID.sub("UUID", path.read_text(encoding="utf-8"))


def main() -> None:
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    failures = []
    checked = 0
    for name in ("l2", "none", "real64"):
        for f in ("eda_report.html", "summary.json"):
            b, a = before / name / f, after / name / f
            if not b.exists() or not a.exists():
                failures.append(f"{name}/{f}: missing")
                continue
            checked += 1
            # summary.json holds absolute output paths, which differ by
            # design between the two snapshot directories. Normalize the
            # snapshot root out before comparing; everything else in the
            # file -- every statistic, every worst-dimension entry -- must
            # match exactly.
            lhs = normalized(b).replace(str(before), "SNAPSHOT")
            rhs = normalized(a).replace(str(after), "SNAPSHOT")
            if lhs != rhs:
                failures.append(f"{name}/{f}: DIFFERS")
    if failures:
        print("GOLDEN FAILED")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"GOLDEN OK ({checked} files identical)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate fixtures and capture the baseline**

```bash
cd /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/eda-report-split
G=/tmp/claude-1000/-home-fibonadithya-TIG-wgan-synthetic/65ca9f9c-9966-4a1c-aba3-ba55a5717d50/scratchpad/golden
PY=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python
$PY $G/make_data.py $G/fixtures
chmod +x $G/snapshot.sh
$G/snapshot.sh $G/before
$PY -m src.eval.eda_report --help > $G/help_before.txt
```

Expected: three output directories under `$G/before`, each with `eda_report.html` and `summary.json`, plus the captured `--help` text. Task 9 diffs against that `--help` capture: `parse_args` changes modules there, which changes what `description=__doc__` resolves to, and no golden report can detect it.

- [ ] **Step 5: Prove the comparator catches a real change**

A comparator that always says OK is worse than none. Verify it fails when the report genuinely changes, then verify it passes on an honest re-run.

```bash
# 1. Same code, second run -> must pass (proves UUID normalization works).
$G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
# Expected: GOLDEN OK (6 files identical)

# 2. Perturb one note, re-snapshot -> must fail.
sed -i 's/How often each point turns up/How often each point shows up/' \
    src/eval/eda_report.py
rm -rf $G/after && $G/snapshot.sh $G/after
$PY $G/compare.py $G/before $G/after
# Expected: GOLDEN FAILED, none/eda_report.html and l2/eda_report.html DIFFER

# 3. Revert the perturbation and confirm clean again.
git checkout src/eval/eda_report.py
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
# Expected: GOLDEN OK (6 files identical)
```

If step 2 reports OK, stop — the harness is not measuring anything and the rest of the plan is unsafe.

- [ ] **Step 6: Confirm the working tree is clean**

```bash
git status --short
```

Expected: no output. Nothing in this task touches the repo, so there is nothing to commit.

**From here on, "run the golden check" means:**

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```

---

### Task 1: `config.py` — defaults and `EdaConfig`

**Files:**
- Create: `src/eval/eda/__init__.py`, `src/eval/eda/config.py`
- Create: `tests/test_eda_config.py`
- Modify: `src/eval/eda_report.py` (import the constants instead of defining them)
- Modify: `src/eval/compare_variants.py:45,437-475`, `tests/test_compare_variants.py:12,166-174`

**Interfaces:**
- Produces: `ANN_K_DEFAULT = 100`, `ANN_HUB_K_DEFAULT = 10`, `ANN_MAX_ROWS_DEFAULT = 20000`, `IVF_NLIST_DEFAULT = 256`, `KNN_MAX_ROWS_DEFAULT = ANN_MAX_ROWS_DEFAULT`, `GLYPH_SAMPLES_DEFAULT = 8`; `EdaConfig` (frozen dataclass, 20 fields) with classmethod `from_args(args: argparse.Namespace) -> EdaConfig`.

- [ ] **Step 1: Create the package**

`src/eval/eda/__init__.py`:

```python
"""Exploratory data analysis report for descriptor corpora.

The CLI entrypoint is `src/eval/eda_report.py`, which is what
`python -m src.eval.eda_report` runs. This package holds the implementation,
split by responsibility:

    config    parsed settings as a typed value object
    cli       argparse
    series    loading and preprocessing the sets being compared
    metrics   the numbers
    figures   the plotly figures
    glyphs    the descriptor glyph panel, shared with plot_descriptor_grid
    notes     computed prose fragments for the ANN panels
    panels    the report's panel registry
    html      page assembly and static export
    pipeline  orchestration

This module deliberately re-exports nothing: every name has exactly one
import path, so a reader who sees `figures.fig_pca_spectrum` knows where it
lives without checking.
"""
```

- [ ] **Step 2: Write the failing test**

`tests/test_eda_config.py`:

```python
import argparse

from src.eval.eda import config


def _full_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        real_path="r.npy",
        real_format="auto",
        synthetic_path=["a=a.npy"],
        synthetic_format="auto",
        output_dir="out",
        preprocess="l2",
        max_vectors=50000,
        num_pairs=200000,
        knn=5,
        ann_k=100,
        ann_hub_k=10,
        ann_max_rows=20000,
        knn_max_rows=20000,
        ivf_nlist=256,
        bins=80,
        top_divergent=16,
        seed=42,
        no_png=False,
        glyph_samples=8,
        plotlyjs="inline",
    )


def test_from_args_carries_every_field_across():
    cfg = config.EdaConfig.from_args(_full_namespace())

    assert cfg.real_path == "r.npy"
    assert cfg.synthetic_path == ["a=a.npy"]
    assert cfg.preprocess == "l2"
    assert cfg.ann_max_rows == 20000
    assert cfg.plotlyjs == "inline"


def test_from_args_covers_every_field_the_parser_produces():
    """Parity guard: a new --flag must reach EdaConfig, not be dropped here."""
    ns = _full_namespace()
    cfg = config.EdaConfig.from_args(ns)

    assert set(vars(ns)) == {f.name for f in dataclasses.fields(cfg)}


def test_missing_glyph_samples_falls_back_to_the_default():
    """compare_variants has historically built Namespaces without this field.

    eda_report.run guarded it with getattr; that guard moves here.
    """
    ns = _full_namespace()
    del ns.glyph_samples

    assert config.EdaConfig.from_args(ns).glyph_samples == config.GLYPH_SAMPLES_DEFAULT


def test_synthetic_path_none_becomes_an_empty_list():
    """argparse leaves a repeatable append-action flag at None when unused."""
    ns = _full_namespace()
    ns.synthetic_path = None

    assert config.EdaConfig.from_args(ns).synthetic_path == []
```

Add `import dataclasses` at the top of the file alongside `import argparse`.

- [ ] **Step 3: Run it and watch it fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.eval.eda.config'`.

- [ ] **Step 4: Write `config.py`**

Move the six default constants out of `eda_report.py:61-79` **with their comments intact** — the comment explaining why `KNN_MAX_ROWS_DEFAULT` is separate from `ANN_MAX_ROWS_DEFAULT` is the reason the flag exists — and add `EdaConfig`.

```python
"""Settings for one EDA report run, as a typed value object.

`compare_variants` builds the argparse Namespace that `pipeline.run` consumes
by hand, so the Namespace stays the public contract. Everything downstream of
`run` takes an `EdaConfig` instead: a panel that needs `bins` should not be
able to reach `sys.argv`-shaped state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# Single source of truth for the ANN-difficulty flag defaults, shared with
# compare_variants.py so its hand-built Namespace cannot silently drift from
# what the CLI's own --ann-* / --ivf-nlist flags default to.
ANN_K_DEFAULT = 100
ANN_HUB_K_DEFAULT = 10
ANN_MAX_ROWS_DEFAULT = 20000
IVF_NLIST_DEFAULT = 256
# The within-set k-NN distance panel is not an ANN-difficulty panel, so it
# gets its own knob rather than riding on --ann-max-rows: tuning the cost of
# the difficulty metrics should not silently move a pre-existing panel. The
# default is the same number, so nothing changes unless a flag is passed.
KNN_MAX_ROWS_DEFAULT = ANN_MAX_ROWS_DEFAULT

GLYPH_SAMPLES_DEFAULT = 8


@dataclass(frozen=True)
class EdaConfig:
    """One run's settings. Field names match the CLI flags exactly."""

    real_path: str
    real_format: str
    synthetic_path: list[str]
    synthetic_format: str
    output_dir: str
    preprocess: str
    max_vectors: int
    num_pairs: int
    knn: int
    ann_k: int
    ann_hub_k: int
    ann_max_rows: int
    knn_max_rows: int
    ivf_nlist: int
    bins: int
    top_divergent: int
    seed: int
    no_png: bool
    glyph_samples: int
    plotlyjs: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> EdaConfig:
        """Build from the Namespace `pipeline.run` was handed.

        `glyph_samples` is read defensively because `compare_variants` has
        built Namespaces without it; every other field is required, so a
        Namespace missing one fails here rather than deep inside a panel.
        """
        return cls(
            real_path=args.real_path,
            real_format=args.real_format,
            synthetic_path=list(args.synthetic_path or []),
            synthetic_format=args.synthetic_format,
            output_dir=args.output_dir,
            preprocess=args.preprocess,
            max_vectors=args.max_vectors,
            num_pairs=args.num_pairs,
            knn=args.knn,
            ann_k=args.ann_k,
            ann_hub_k=args.ann_hub_k,
            ann_max_rows=args.ann_max_rows,
            knn_max_rows=args.knn_max_rows,
            ivf_nlist=args.ivf_nlist,
            bins=args.bins,
            top_divergent=args.top_divergent,
            seed=args.seed,
            no_png=args.no_png,
            glyph_samples=getattr(args, "glyph_samples", GLYPH_SAMPLES_DEFAULT),
            plotlyjs=args.plotlyjs,
        )
```

- [ ] **Step 5: Run the new test**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_config.py -v`
Expected: 4 passed.

- [ ] **Step 6: Repoint the three consumers of the constants**

In `src/eval/eda_report.py`, delete lines 61-79 (the constants and their comments) and `GLYPH_SAMPLES_DEFAULT` at line 79, then add near the other imports:

```python
from src.eval.eda.config import (
    ANN_HUB_K_DEFAULT,
    ANN_K_DEFAULT,
    ANN_MAX_ROWS_DEFAULT,
    GLYPH_SAMPLES_DEFAULT,
    IVF_NLIST_DEFAULT,
    KNN_MAX_ROWS_DEFAULT,
)
```

In `src/eval/compare_variants.py`, the six `eda_report.X_DEFAULT` references at lines 437, 443, 449, 459, 468 and 475 become `eda_config.X_DEFAULT`, with `from src.eval.eda import config as eda_config` added beside the existing `from src.eval import eda_report` on line 45.

In `tests/test_compare_variants.py`, the same six references at lines 166-174 become `eda_config.X_DEFAULT`, importing `from src.eval.eda import config as eda_config` at line 12.

- [ ] **Step 7: Verify**

```bash
make check
```
Expected: ruff clean, 437 passed (433 baseline + 4 new).

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

- [ ] **Step 8: Commit**

```bash
git add src/eval/eda/ src/eval/eda_report.py src/eval/compare_variants.py \
        tests/test_eda_config.py tests/test_compare_variants.py
git commit -m "refactor(eda): extract config defaults and EdaConfig

First step of the eda_report split. The six *_DEFAULT constants move to
src/eval/eda/config.py and gain a typed EdaConfig that everything
downstream of run() will take instead of an argparse Namespace. run()'s
signature is unchanged, so compare_variants' hand-built Namespace and its
parity test are untouched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `series.py` — the datasets being compared

**Files:**
- Create: `src/eval/eda/series.py`, `tests/test_eda_series.py`
- Modify: `src/eval/eda_report.py` (remove the moved code, import it back)
- Modify: `src/eval/plot_descriptor_grid.py:33,267-268`, `tests/test_plot_descriptor_grid.py`
- Modify: `tests/test_eda_report.py` (move the four relevant tests out)

**Interfaces:**
- Consumes: `config.EdaConfig` from Task 1.
- Produces: `REAL_NAME = "real"`, `REAL_COLOR`, `SYNTH_PALETTE`, `Series` (frozen-by-convention dataclass: `name: str`, `x: np.ndarray`, `color: str`, property `is_real`), `parse_synthetic_spec(spec: str) -> tuple[str, Path]`, `subsample(x, max_vectors, seed) -> np.ndarray`, `maybe_l2_normalize(x, mode, eps=1e-8) -> np.ndarray`, `load_series(cfg: EdaConfig) -> list[Series]`.

- [ ] **Step 1: Move the code**

Create `src/eval/eda/series.py` with this module docstring:

```python
"""The datasets a report compares, loaded and preprocessed.

A `Series` is one already-subsampled, already-normalized set with the colour
it is drawn in. Everything downstream works in terms of these, so no figure
or panel touches a file path or a preprocessing mode.
"""
```

Then move verbatim from `src/eval/eda_report.py`:

| From | Lines | What |
|---|---|---|
| `eda_report.py` | 48-59 | `REAL_NAME`, `REAL_COLOR`, `SYNTH_PALETTE` and the comment above the palette |
| `eda_report.py` | 90-100 | `Series` |
| `eda_report.py` | 225-237 | `parse_synthetic_spec` |
| `eda_report.py` | 245-257 | `subsample`, `maybe_l2_normalize` |
| `eda_report.py` | 922-947 | `load_series` |

`load_series` is the one signature change: `load_series(args: argparse.Namespace)` becomes `load_series(cfg: EdaConfig)`, and the five `args.` references inside become `cfg.`. The body is otherwise unchanged, including both `raise ValueError` messages.

Imports the module needs: `from __future__ import annotations`, `from dataclasses import dataclass`, `from pathlib import Path`, `import numpy as np`, `from src.data.dataset import load_descriptors`, `from src.eval.eda.config import EdaConfig`.

- [ ] **Step 2: Remove from `eda_report.py` and import back**

Delete those five blocks from `eda_report.py`. Add:

```python
from src.eval.eda.series import REAL_NAME, Series, load_series, maybe_l2_normalize
```

`run()` calls `load_series(args)` at line 1030. Change to:

```python
    cfg = EdaConfig.from_args(args)
    series = load_series(cfg)
```

importing `EdaConfig` from `src.eval.eda.config`. This is the first place `EdaConfig` is actually used, which is what makes Task 1's tests meaningful rather than decorative. Later tasks refer to this local as `cfg`; leave the remaining `args.` references alone for now, they migrate as their code moves.

`parse_synthetic_spec`, `subsample` and `SYNTH_PALETTE` are now only used inside `series.py`; do not import them back into `eda_report.py` or ruff will flag them unused.

- [ ] **Step 3: Move the tests**

Create `tests/test_eda_series.py` and move these four tests from `tests/test_eda_report.py` verbatim, changing only `eda_report.` to `series.`:

- `test_maybe_l2_normalize_gives_unit_rows` (line 198)
- `test_maybe_l2_normalize_leaves_a_zero_row_finite` (line 204)
- `test_maybe_l2_normalize_none_mode_passes_rows_through` (line 211)
- the `_stub_series` helper (line 216) — this one is **copied**, not moved: `tests/test_eda_notes.py` needs it in Task 6 and `tests/test_eda_report.py` still needs it until then.

Header of the new file:

```python
import numpy as np

from src.eval.eda import series
```

- [ ] **Step 4: Repoint `plot_descriptor_grid`**

`src/eval/plot_descriptor_grid.py:267-268` calls `eda_report.maybe_l2_normalize(row_a, "l2")`. Add `from src.eval.eda import series as eda_series` beside the existing `from src.eval import eda_report` on line 33 and change both calls to `eda_series.maybe_l2_normalize(...)`. Leave the surrounding comment at line 264 in place but change the words "`eda_report`'s normaliser" to "`eda.series`'s normaliser" — a comment naming the wrong module is worse than no comment.

- [ ] **Step 5: Verify**

```bash
make check
```
Expected: ruff clean, 437 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract series loading and preprocessing

Series, the palette, parse_synthetic_spec, subsample, maybe_l2_normalize
and load_series move to src/eval/eda/series.py. load_series now takes an
EdaConfig rather than an argparse Namespace -- the first real consumer of
Task 1's config object. Verbatim move otherwise.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `metrics.py` — the numbers

The one task with a genuine structural change: `fig_dim_divergence` currently computes, draws, and returns a `summary.json` payload. The computation moves here; Task 4 keeps only the drawing.

**Files:**
- Create: `src/eval/eda/metrics.py`, `tests/test_eda_metrics.py`
- Modify: `src/eval/eda_report.py`

**Interfaces:**
- Consumes: `series.Series`, `config.EdaConfig`.
- Produces: `pairwise_distance_sample(x, num_pairs, seed)`, `nn_distances(x, k, seed, max_rows)`, `wasserstein1(a, b, num_quantiles=512)`, `effective_rank(x)`, `summary_stats(s, knn, num_pairs, seed, knn_max_rows, metrics) -> dict`, `DimDivergence`, `dimension_divergence(series, top_k) -> DimDivergence`.

- [ ] **Step 1: Move the unchanged functions**

Create `src/eval/eda/metrics.py`:

```python
"""The numbers behind the report, with no plotly in sight.

Kept separate from `figures` so a statistic can be tested without rendering
anything, and so a reader can see what is measured without reading how it is
drawn.
"""
```

Move verbatim, docstrings and comments intact:

| From | Lines | What |
|---|---|---|
| `eda_report.py` | 260-289 | `pairwise_distance_sample`, `nn_distances`, `wasserstein1` |
| `eda_report.py` | 766-823 | `effective_rank`, `summary_stats` |

Imports: `from __future__ import annotations`, `from collections.abc import Sequence`, `import numpy as np`, `from sklearn.decomposition import PCA`, `from sklearn.neighbors import NearestNeighbors`, `from src.eval import ann_difficulty`, `from src.eval.eda.series import Series, subsample`.

Note `nn_distances` calls `subsample`, which now lives in `series.py` — that import is load-bearing.

- [ ] **Step 2: Write the failing test for the divergence split**

`tests/test_eda_metrics.py`:

```python
import numpy as np

from src.eval.eda import metrics, series


def _series(name: str, x: np.ndarray, color: str = "#000000") -> series.Series:
    return series.Series(name, x.astype(np.float32), color)


def test_dimension_divergence_orders_dimensions_by_worst_mismatch():
    """Dimension 1 is the one the synthetics get wrong, so it must rank first."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 3))
    bad = real.copy()
    bad[:, 1] += 5.0

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", bad)], top_k=2
    )

    assert div.order[0] == 1
    assert div.worst["a"][0]["dim"] == 1
    assert div.worst["a"][0]["wasserstein1"] > div.worst["a"][1]["wasserstein1"]


def test_dimension_divergence_orders_by_the_worst_across_all_synthetics():
    """One shared x-axis ordering, driven by the worst offender on each dim.

    Series 'a' is wrong on dim 0 and 'b' on dim 2, so both must outrank the
    dimension neither one misses.
    """
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 3))
    a, b = real.copy(), real.copy()
    a[:, 0] += 9.0
    b[:, 2] += 5.0

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", a), _series("b", b)], top_k=3
    )

    assert list(div.order[:2]) == [0, 2]


def test_dimension_divergence_reports_top_k_dimensions_per_series():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(120, 5))

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", real + 1.0)], top_k=2
    )

    assert set(div.worst) == {"a"}
    assert len(div.worst["a"]) == 2
    assert all(isinstance(e["dim"], int) for e in div.worst["a"])
    assert all(isinstance(e["wasserstein1"], float) for e in div.worst["a"])
```

- [ ] **Step 3: Run it and watch it fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_metrics.py -v`
Expected: FAIL, `AttributeError: module 'src.eval.eda.metrics' has no attribute 'dimension_divergence'`.

- [ ] **Step 4: Split the computation out of `fig_dim_divergence`**

Add to `metrics.py`. The body is lifted from `eda_report.py:617-622` and `640-643` unchanged; only the packaging is new.

```python
@dataclass(frozen=True)
class DimDivergence:
    """Per-dimension W1 against real, plus the shared plotting order.

    `distances` is one array per synthetic series; `order` is the single
    dimension ordering every series is drawn in, worst first, so bars line
    up across series; `worst` is the top-k slice that reaches summary.json.
    """

    distances: dict[str, np.ndarray]
    order: np.ndarray
    worst: dict[str, list[dict]]


def dimension_divergence(series: Sequence[Series], top_k: int) -> DimDivergence:
    """Rank dimensions by 1-D Wasserstein distance from real, per synthetic set.

    Dimensions are ordered by the worst mismatch across all synthetics, so the
    same x-axis ordering applies to every series and they stay comparable.
    """
    real = next(s for s in series if s.is_real)
    synths = [s for s in series if not s.is_real]
    dim = real.x.shape[1]

    distances = {
        s.name: np.array([wasserstein1(real.x[:, d], s.x[:, d]) for d in range(dim)])
        for s in synths
    }
    worst_overall = np.max(np.stack(list(distances.values())), axis=0)
    order = np.argsort(worst_overall)[::-1]
    worst = {
        name: [{"dim": int(d), "wasserstein1": float(v[d])} for d in order[:top_k]]
        for name, v in distances.items()
    }
    return DimDivergence(distances=distances, order=order, worst=worst)
```

Add `from dataclasses import dataclass` to the imports.

In `eda_report.py`, `fig_dim_divergence` (lines 605-644) loses its computation and its second return value:

```python
def fig_dim_divergence(
    divergence: metrics.DimDivergence, series: Sequence[Series]
) -> go.Figure:
    """Draw the per-dimension mismatch bars in the shared worst-first order."""
    fig = go.Figure()
    for s in series:
        if s.is_real:
            continue
        fig.add_bar(
            x=[f"dim {d}" for d in divergence.order],
            y=divergence.distances[s.name][divergence.order],
            name=s.name,
            marker_color=s.color,
        )
    fig.update_layout(
        title="Per-dimension marginal mismatch vs real (Wasserstein-1, worst first)",
        xaxis_title="dimension",
        yaxis_title="W1(real, synthetic)",
        barmode="group",
        template="plotly_white",
        height=440,
    )
    return fig
```

The trace order matters for the golden check: the original iterated `synths` (real filtered out first), and this iterates `series` skipping real, which yields the same order.

Update `run()` at `eda_report.py:1252-1254`:

```python
    worst_dims: dict[str, list[dict]] = {}
    if has_synth:
        divergence = metrics.dimension_divergence(series, args.top_divergent)
        worst_dims = divergence.worst
        div_fig = fig_dim_divergence(divergence, series)
```

- [ ] **Step 5: Remove the moved functions from `eda_report.py` and import back**

Delete lines 260-289 and 766-823. Add `from src.eval.eda import metrics` and use `metrics.pairwise_distance_sample`, `metrics.nn_distances`, `metrics.summary_stats` at their call sites in `run()` (lines 1044, 1185, 1208) and in the panel note sections. `wasserstein1` and `effective_rank` are only called from within `metrics.py` now.

- [ ] **Step 6: Verify**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_metrics.py -v
```
Expected: 3 passed.

```bash
make check
```
Expected: ruff clean, 440 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`. This is the task most likely to break it — the divergence ordering and the `summary.json` `worst_dimensions` payload both flow through the code just moved.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract metrics and split divergence from its figure

The statistics move to src/eval/eda/metrics.py. fig_dim_divergence no
longer computes W1 and no longer returns the summary.json payload: that is
now metrics.dimension_divergence returning a DimDivergence, and the figure
only draws. A summary payload stops travelling out of a figure function.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `figures.py` — the plotly builders

**Files:**
- Create: `src/eval/eda/figures.py`
- Modify: `src/eval/eda_report.py`

**Interfaces:**
- Consumes: `series.Series`, `metrics.DimDivergence`, `ann_difficulty.AnnMetrics`.
- Produces: `shared_edges(arrays, bins)`, `overlay_hist_fig(named_values, bins, title, xaxis_title, log_y=False)`, `fig_value_distribution`, `fig_per_dim_marginals`, `fig_dim_profiles`, `fig_pca_spectrum`, `fig_correlation`, `fig_ann_profile`, `fig_ivf_balance`, `fig_dim_divergence` — all taking the arguments they take today.

- [ ] **Step 1: Move the code**

Create `src/eval/eda/figures.py`:

```python
"""Plotly figures for the report's aggregate panels.

Each builder takes already-computed inputs and returns a `go.Figure`. None of
them read settings, write files, or decide whether their panel belongs in the
report -- that is `panels.py`'s job.

The descriptor glyph panel lives in `glyphs.py` instead, because
`plot_descriptor_grid` draws it from generator checkpoints and should not
have to import the aggregate figures to do so.
"""
```

Move verbatim from `eda_report.py`:

| Lines | What |
|---|---|
| 297-330 | `shared_edges`, `overlay_hist_fig` |
| 333-341 | `fig_value_distribution` |
| 344-397 | `fig_per_dim_marginals` |
| 400-431 | `fig_dim_profiles` |
| 434-472 | `fig_pca_spectrum` |
| 475-504 | `fig_correlation` |
| 507-568 | `fig_ann_profile` |
| 571-602 | `fig_ivf_balance` |
| — | `fig_dim_divergence` as rewritten in Task 3 |

Imports: `from __future__ import annotations`, `from collections.abc import Sequence`, `import numpy as np`, `import plotly.graph_objects as go`, `from plotly.subplots import make_subplots`, `from sklearn.decomposition import PCA`, `from src.eval import ann_difficulty`, `from src.eval.eda import metrics`, `from src.eval.eda.series import Series`.

- [ ] **Step 2: Remove from `eda_report.py` and import back**

Delete those blocks. Add `from src.eval.eda import figures` and prefix the nine call sites in `run()` with `figures.`.

- [ ] **Step 3: Verify**

```bash
make check
```
Expected: ruff clean, 440 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

No new tests here. These builders have no logic beyond assembling plotly calls, they are covered end-to-end by `tests/test_eda_report.py`, and the golden check compares their full trace JSON byte-for-byte — a unit test asserting "the figure has three traces" would be strictly weaker than what already runs.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract the plotly figure builders

Nine figure builders and their two helpers move to
src/eval/eda/figures.py. Verbatim move; no signature changes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `glyphs.py` — the descriptor glyph panel

Separate from `figures.py` because this is `plot_descriptor_grid`'s entire dependency on the package.

**Files:**
- Create: `src/eval/eda/glyphs.py`
- Modify: `src/eval/eda_report.py`, `src/eval/plot_descriptor_grid.py:33,39-40,90-94`
- Modify: `tests/test_eda_report.py` (glyph tests move), `tests/test_plot_descriptor_grid.py`
- Create: `tests/test_eda_glyphs.py`

**Interfaces:**
- Consumes: `series.Series`, `src.eval.descriptor_glyph`.
- Produces: `GLYPH_SECTION_TITLE`, `GLYPH_CELL_PITCH`, `GLYPH_PITCH`, `GLYPH_REAL_COLORS`, `GLYPH_NEGATIVE_COLOR`, `glyph_rows(series, num_samples, seed) -> list[tuple[str, np.ndarray, str]]`, `fig_descriptor_glyphs(rows) -> go.Figure`.

`GLYPH_SAMPLES_DEFAULT` stays in `config.py` — it is a flag default, not a drawing constant.

- [ ] **Step 1: Move the code**

Create `src/eval/eda/glyphs.py` and move verbatim:

| From | Lines | What |
|---|---|---|
| `eda_report.py` | 74-87 | the glyph constants and the comment block above them, minus `GLYPH_SAMPLES_DEFAULT` (moved in Task 1) |
| `eda_report.py` | 647-727 | `fig_descriptor_glyphs` |
| `eda_report.py` | 730-758 | `glyph_rows` |

Module docstring:

```python
"""The descriptor glyph panel.

Every other panel in the report is an aggregate over tens of thousands of
vectors; this one draws a handful of individual descriptors, because a
matched marginal says nothing about whether the 128 numbers form a plausible
gradient histogram.

Kept apart from `figures.py` because `plot_descriptor_grid` renders the same
panel straight from generator checkpoints, and this module is the whole of
what it needs.
"""
```

Imports: `from __future__ import annotations`, `from collections.abc import Sequence`, `import numpy as np`, `import plotly.graph_objects as go`, the existing `from src.eval.descriptor_glyph import (...)` block, `from src.eval.eda.series import Series`.

- [ ] **Step 2: Remove from `eda_report.py` and import back**

Delete those blocks; add `from src.eval.eda import glyphs`. In `run()`, the `glyph_rows(...)` call at line 1060 becomes:

```python
    rows = glyphs.glyph_rows(series, cfg.glyph_samples, cfg.seed)
```

using the `cfg` local Task 2 introduced. The `getattr(args, "glyph_samples", GLYPH_SAMPLES_DEFAULT)` guard disappears from this call site — `EdaConfig.from_args` owns it now, which is what Task 1's `test_missing_glyph_samples_falls_back_to_the_default` pins. `GLYPH_SECTION_TITLE` (line 1061) and `fig_descriptor_glyphs` (line 1083) become `glyphs.`-prefixed, and `eda_report.py` no longer imports `GLYPH_SAMPLES_DEFAULT`.

- [ ] **Step 3: Move the glyph tests**

Create `tests/test_eda_glyphs.py`. Move these five from `tests/test_eda_report.py` verbatim (they call `eda_report.run`, so they keep importing `eda_report` — only the `GLYPH_SECTION_TITLE` reference changes to `glyphs.GLYPH_SECTION_TITLE`):

- `test_glyph_section_is_included_for_128_dimensional_data` (line 81)
- `test_glyph_section_draws_two_real_rows_as_the_variation_baseline` (line 90)
- `test_glyph_section_is_skipped_for_other_dimensions` (line 101)
- `test_glyph_section_is_skipped_when_a_series_is_too_small` (line 114)
- `test_glyph_samples_zero_disables_the_section` (line 130)

They use the `make_args` helper at `tests/test_eda_report.py:30`. Move `make_args` into `tests/conftest.py` as a fixture-free module-level helper so both files share one copy rather than two that can drift. Import it in both files.

- [ ] **Step 4: Repoint `plot_descriptor_grid`**

- Line 33: add `from src.eval.eda import glyphs as eda_glyphs`.
- Line 40: `REAL_COLORS = eda_report.GLYPH_REAL_COLORS` becomes `eda_glyphs.GLYPH_REAL_COLORS`.
- Line 94: `eda_report.fig_descriptor_glyphs(rows)` becomes `eda_glyphs.fig_descriptor_glyphs(rows)`.
- Lines 3, 39 and 90 are comments naming `eda_report`; update them to name `eda.glyphs`.

- [ ] **Step 5: Verify**

```bash
make check
```
Expected: ruff clean, 440 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`. The `real64` snapshot is the one that proves the glyph skip path still fires.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract the descriptor glyph panel

glyph_rows, fig_descriptor_glyphs and the drawing constants move to
src/eval/eda/glyphs.py, which is now plot_descriptor_grid's whole
dependency on this package. The getattr guard on glyph_samples is gone --
EdaConfig.from_args owns it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `notes.py` — the computed prose fragments

**Files:**
- Create: `src/eval/eda/notes.py`, `tests/test_eda_notes.py`
- Modify: `src/eval/eda_report.py`, `tests/test_eda_report.py`

**Interfaces:**
- Consumes: `series.Series`, `ann_difficulty.AnnMetrics`.
- Produces: `ANN_NOTE_SUFFIX: str`, `ann_condition_note(series, ann_metrics, attrs) -> str`, `ann_discarded_note(series, ann_metrics) -> str`.

- [ ] **Step 1: Move the code**

Create `src/eval/eda/notes.py`:

```python
"""Prose fragments the ANN panels compute rather than hard-code.

A panel's fixed prose lives with the panel in `panels.py`. What lives here is
the part that depends on the run: which measurement conditions each series
was actually measured under, and whether any series contributed no queries at
all. Both exist so a reader cannot mistake one series' numbers for all of
them.
"""
```

Move verbatim from `eda_report.py:950-1023` (`ann_condition_note` and `ann_discarded_note`, both docstrings intact).

Move the suffix string from `eda_report.py:1087-1091` and name it:

```python
# Appended to all three ANN panel notes. Invariant 3 in AGENTS.md: these are
# self-queried subsample figures with no absolute meaning, and a reader who
# checks them against published SIFT1M numbers is drawing a false conclusion.
ANN_NOTE_SUFFIX = (
    " Compare against the <code>real</code> series in this report only. "
    "These numbers come from a self-queried subsample, so they are not "
    "comparable with published SIFT1M figures."
)
```

The string content is byte-identical to today's `ann_note_suffix` local; only the name and the added comment are new.

- [ ] **Step 2: Move the tests**

Create `tests/test_eda_notes.py`. Move these seven from `tests/test_eda_report.py` verbatim, changing `eda_report.` to `notes.`, and bring the `_stub_series` helper copy from Task 2:

- `test_ann_condition_note_states_one_condition_when_every_series_matches` (line 243)
- `test_ann_condition_note_spells_out_every_series_when_conditions_diverge` (line 255)
- `test_ann_condition_note_diverges_on_a_single_attribute_too` (line 271)
- `test_ann_discarded_note_is_empty_when_every_series_kept_some_queries` (line 282)
- `test_ann_discarded_note_names_a_series_whose_queries_were_all_discarded` (line 288)
- `test_ann_discarded_note_blames_k_equals_one_when_that_is_the_cause` (line 300)
- plus the `_stub_metrics` helper these depend on (around line 226)

Delete them from `tests/test_eda_report.py`, along with `_stub_series` and `_stub_metrics` if nothing there still uses them.

- [ ] **Step 3: Remove from `eda_report.py` and import back**

Delete lines 950-1023 and the `ann_note_suffix` local. Add `from src.eval.eda import notes` and prefix the four call sites in `run()` (lines 1102, 1105, 1118, 1142) plus the three `+ ann_note_suffix` references, which become `+ notes.ANN_NOTE_SUFFIX`.

- [ ] **Step 4: Verify**

```bash
make check
```
Expected: ruff clean, 440 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract the computed ANN note fragments

ann_condition_note, ann_discarded_note and the shared suffix move to
src/eval/eda/notes.py. The suffix gains a name and a comment pointing at
AGENTS.md invariant 3; its text is byte-identical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `html.py` — page assembly and export

**Files:**
- Create: `src/eval/eda/html.py`, `tests/test_eda_html.py`
- Modify: `src/eval/eda_report.py`, `src/eval/plot_descriptor_grid.py:102,113`
- Modify: `tests/test_eda_report.py`, `tests/test_plot_descriptor_grid.py:232`

**Interfaces:**
- Consumes: nothing from the package.
- Produces: `REPORT_CSS`, `CDN_SRC`, `plotlyjs_head(mode, out_dir) -> str`, `format_stat(value) -> str`, `stats_table_html(stats) -> str`, `build_report(sections, meta_html, head_script) -> str`, `export_pngs(sections, out_dir) -> list[str]`.

`sections` is `list[tuple[str, str, go.Figure]]` — (title, note, figure) — unchanged from today.

- [ ] **Step 1: Move the code**

Create `src/eval/eda/html.py`:

```python
"""Turning finished figures into a page on disk.

Everything here works on `(title, note, figure)` triples and knows nothing
about what a panel means. Static PNG export is best-effort: kaleido v1 shells
out to Chrome, which is not installed everywhere.
"""
```

Move verbatim from `eda_report.py`:

| Lines | What |
|---|---|
| 826-838 | `format_stat` |
| 841-854 | `stats_table_html` |
| 857-868 | `REPORT_CSS` |
| 870 | `CDN_SRC` |
| 873-884 | `plotlyjs_head` |
| 887-906 | `build_report` |
| 909-919 | `export_pngs` |

Imports: `from __future__ import annotations`, `from pathlib import Path`, `import numpy as np`, `import plotly.graph_objects as go`.

- [ ] **Step 2: Move the tests**

Create `tests/test_eda_html.py` and move verbatim, changing `eda_report.` to `html.`:

- `test_format_stat_renders_counts_as_integers_not_scientific_notation` (line 309)
- `test_stats_table_renders_a_large_discarded_count_as_a_tally` (line 316)
- any remaining tests in `tests/test_eda_report.py` below line 309 that call `format_stat` or `stats_table_html`

- [ ] **Step 3: Remove from `eda_report.py` and import back**

Delete those blocks; add `from src.eval.eda import html` and prefix the call sites in `run()` (lines 1275, 1278, 1286).

- [ ] **Step 4: Repoint `plot_descriptor_grid`**

- Line 33 area: add `from src.eval.eda import html as eda_html`.
- Line 102: `eda_report.plotlyjs_head(...)` becomes `eda_html.plotlyjs_head(...)`.
- Line 113: `eda_report.export_pngs(...)` becomes `eda_html.export_pngs(...)`.
- `tests/test_plot_descriptor_grid.py:232`: `monkeypatch.setattr(pdg.eda_report, "export_pngs", _boom)` becomes `monkeypatch.setattr(pdg.eda_html, "export_pngs", _boom)`. The docstring at line 227 mentions `eda_report.run`'s try/except — update it to `eda.pipeline.run`.

After this task `plot_descriptor_grid.py` no longer imports `eda_report` at all; delete line 33's `from src.eval import eda_report` and confirm with `grep -n "eda_report" src/eval/plot_descriptor_grid.py` returning nothing but prose in comments you have already updated.

- [ ] **Step 5: Verify**

```bash
make check
```
Expected: ruff clean, 440 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract HTML assembly and PNG export

REPORT_CSS, plotlyjs_head, format_stat, stats_table_html, build_report and
export_pngs move to src/eval/eda/html.py. plot_descriptor_grid no longer
imports eda_report at all.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `panels.py` — the registry

The point of the whole exercise. `run()`'s thirteen `sections.append(...)` blocks become thirteen `Panel` objects.

**Files:**
- Create: `src/eval/eda/panels.py`, `tests/test_eda_panels.py`
- Modify: `src/eval/eda_report.py`

**Interfaces:**
- Consumes: everything below it in the package.
- Produces: `Context` (frozen dataclass: `config`, `series`, `ann_metrics`, `divergence`), `Panel` (frozen dataclass: `title`, `note`, `build`) with methods `resolve_title(ctx) -> str` and `resolve_note(ctx) -> str`, and `PANELS: list[Panel]`.

- [ ] **Step 1: Write the failing test**

`tests/test_eda_panels.py`:

```python
import numpy as np
import plotly.graph_objects as go

from src.eval import ann_difficulty
from src.eval.eda import config, panels, series


def _context(dim: int = 128, num_synth: int = 2, preprocess: str = "l2"):
    rng = np.random.default_rng(0)
    sets = [series.Series("real", rng.random((300, dim), dtype=np.float32), "#000")]
    for i in range(num_synth):
        sets.append(
            series.Series(f"s{i}", rng.random((300, dim), dtype=np.float32), "#111")
        )
    cfg = config.EdaConfig.from_args(_namespace(preprocess=preprocess))
    metrics = {
        s.name: ann_difficulty.compute(
            s.x, k=10, k_hub=5, nlist=8, max_rows=200, seed=0
        )
        for s in sets
    }
    return panels.Context(
        config=cfg, series=sets, ann_metrics=metrics, divergence=None
    )


def test_every_panel_declares_a_title_and_a_note():
    """A panel with an empty title renders as an unlabelled <h2>."""
    ctx = _context()

    for panel in panels.PANELS:
        assert panel.resolve_title(ctx).strip()
        assert panel.resolve_note(ctx).strip()


def test_panel_titles_are_unique():
    """export_pngs slugs the title into a filename; duplicates collide."""
    ctx = _context()
    titles = [p.resolve_title(ctx) for p in panels.PANELS]

    assert len(titles) == len(set(titles))


def test_vector_norms_panel_is_omitted_under_l2_normalization():
    """Norms are all 1.0 after L2, so the panel would say nothing."""
    ctx = _context(preprocess="l2")
    panel = _by_title(panels.PANELS, "Vector norms", ctx)

    assert panel.build(ctx) is None


def test_vector_norms_panel_is_built_without_normalization():
    ctx = _context(preprocess="none")
    panel = _by_title(panels.PANELS, "Vector norms", ctx)

    assert isinstance(panel.build(ctx), go.Figure)


def test_mismatch_panel_is_omitted_without_a_synthetic_overlay():
    ctx = _context(num_synth=0)
    panel = _by_title(panels.PANELS, "Per-dimension mismatch", ctx)

    assert panel.build(ctx) is None


def test_glyph_panel_is_omitted_for_non_128_dimensional_data():
    ctx = _context(dim=64)
    panel = _by_title(panels.PANELS, "Descriptor glyphs", ctx)

    assert panel.build(ctx) is None
```

`_by_title(panels, title, ctx)` returns the single panel whose resolved title equals `title`, raising if there is not exactly one. `_namespace(preprocess=...)` is the shared helper from `tests/conftest.py` introduced in Task 5, extended to take a `preprocess` keyword. Write both helpers at the top of the test file / in `conftest.py` respectively.

- [ ] **Step 2: Run it and watch it fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_panels.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.eval.eda.panels'`.

- [ ] **Step 3: Write `Context` and `Panel`**

```python
"""The report's panels, as data.

Each panel is a title, a prose note explaining what a reader should conclude
from it, and a builder that returns its figure. `build` returning None means
"this panel does not apply to this run" -- the report simply omits it. That
one mechanism covers all three cases: the glyph panel needs 128-dimensional
data, the norms panel needs unnormalized vectors, and the mismatch panel
needs something to compare against.

To add a panel: append a Panel here and add its builder to figures.py. To
change what a panel claims, edit its note here. Nothing else in the package
needs to know the report has thirteen sections rather than twelve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from src.eval import ann_difficulty
from src.eval.eda import figures, glyphs, metrics, notes
from src.eval.eda.config import EdaConfig
from src.eval.eda.series import Series


@dataclass(frozen=True)
class Context:
    """Everything a panel is allowed to see.

    Deliberately does not carry the summary statistics: no panel reads them,
    they feed the header table and summary.json, and keeping them out of here
    keeps the panel contract to what a panel actually draws from.
    """

    config: EdaConfig
    series: list[Series]
    ann_metrics: dict[str, ann_difficulty.AnnMetrics]
    divergence: metrics.DimDivergence | None

    @property
    def has_synthetic(self) -> bool:
        return len(self.series) > 1


Text = str | Callable[[Context], str]


@dataclass(frozen=True)
class Panel:
    """One report section.

    `title` and `note` accept a callable because some are computed: the k-NN
    title embeds --knn, and the ANN notes embed the conditions each series was
    actually measured under.
    """

    title: Text
    note: Text
    build: Callable[[Context], go.Figure | None]

    def resolve_title(self, ctx: Context) -> str:
        return self.title(ctx) if callable(self.title) else self.title

    def resolve_note(self, ctx: Context) -> str:
        return self.note(ctx) if callable(self.note) else self.note
```

- [ ] **Step 4: Move the thirteen panels**

Each `sections.append((title, note, fig))` in `eda_report.py:1055-1263` becomes a `Panel` in `PANELS`, **in the same order**. Panel order is report order and the golden check will catch any reshuffle.

The note strings move byte-for-byte. Do not rewrap them, do not fix a typo, do not change an `--` to an em dash.

The three conditional panels get a builder that returns `None`:

```python
def _build_glyphs(ctx: Context) -> go.Figure | None:
    rows = glyphs.glyph_rows(ctx.series, ctx.config.glyph_samples, ctx.config.seed)
    # glyph_rows returns [] when the mapping does not apply -- a width other
    # than 128, or a series too small for its rows. The applicability test and
    # the row choice are the same work, which is why this is one call and not
    # a separate predicate.
    return glyphs.fig_descriptor_glyphs(rows) if rows else None


def _build_norms(ctx: Context) -> go.Figure | None:
    if ctx.config.preprocess != "none":
        return None
    return figures.overlay_hist_fig(
        [(s.name, np.linalg.norm(s.x, axis=1), s.color) for s in ctx.series],
        ctx.config.bins,
        "L2 norm",
        "norm",
    )


def _build_mismatch(ctx: Context) -> go.Figure | None:
    if ctx.divergence is None:
        return None
    return figures.fig_dim_divergence(ctx.divergence, ctx.series)
```

The two computed-text panels:

```python
def _knn_title(ctx: Context) -> str:
    return f"Within-set {ctx.config.knn}-NN distances"


def _lid_note(ctx: Context) -> str:
    return (
        LID_NOTE
        + notes.ann_condition_note(
            ctx.series, ctx.ann_metrics, (("num_rows", "rows"), ("k", "k"))
        )
        + notes.ann_discarded_note(ctx.series, ctx.ann_metrics)
        + notes.ANN_NOTE_SUFFIX
    )
```

where `LID_NOTE` is the fixed prose from `eda_report.py:1095-1101`, moved verbatim as a module constant. `_hubness_note` and `_ivf_note` follow the same shape with their own condition attrs — `(("num_rows", "rows"),)` and `(("nlist", "nlist"),)` respectively — and no discarded-note call, matching today exactly.

- [ ] **Step 5: Point `run()` at the registry**

In `eda_report.py`, replace the whole `sections` construction with:

```python
    ctx = panels.Context(
        config=cfg,
        series=series,
        ann_metrics=ann_metrics,
        divergence=(
            metrics.dimension_divergence(series, cfg.top_divergent)
            if len(series) > 1
            else None
        ),
    )
    sections = [
        (panel.resolve_title(ctx), panel.resolve_note(ctx), fig)
        for panel in panels.PANELS
        if (fig := panel.build(ctx)) is not None
    ]
    worst_dims = ctx.divergence.worst if ctx.divergence else {}
```

- [ ] **Step 6: Verify**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_eda_panels.py -v
```
Expected: 6 passed.

```bash
make check
```
Expected: ruff clean, 446 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`. **This is the task the golden check exists for.** Thirteen prose blocks moved by hand; a single dropped sentence shows up here and nowhere else. If it fails, diff the two HTML files directly rather than guessing.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor(eda): turn the report's sections into a panel registry

The thirteen sections run() appended by hand are now Panel objects in
src/eval/eda/panels.py, each a title, a note and a builder. A builder
returning None omits its panel, which replaces all three ad-hoc
conditionals. Adding a panel is now one Panel plus one figure builder.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `pipeline.py` and the entrypoint

**Files:**
- Create: `src/eval/eda/pipeline.py`
- Rewrite: `src/eval/eda_report.py`
- Create: `src/eval/eda/cli.py`
- Modify: `src/eval/compare_variants.py:45,564`, `tests/test_compare_variants.py:12,191,601`
- Rename: `tests/test_eda_report.py` → `tests/test_eda_run.py`

**Interfaces:**
- Produces: `cli.parse_args() -> argparse.Namespace`, `pipeline.build_context(cfg) -> Context`, `pipeline.run(args: argparse.Namespace) -> Path`.

- [ ] **Step 1: Move `parse_args` to `cli.py`**

Move `eda_report.py:103-222` verbatim into `src/eval/eda/cli.py`. The `description=__doc__` reference now resolves to `cli.py`'s docstring, which would change `--help` output — so `cli.py` takes the current `eda_report.py` module docstring (lines 1-22) as its own, and the entrypoint keeps a short one. Defaults come from `src.eval.eda.config`.

Verify with `python -m src.eval.eda_report --help` after Step 3; the text must be unchanged.

- [ ] **Step 2: Move `run` to `pipeline.py`**

Move what remains of `run()` verbatim, plus a `build_context` helper:

```python
def build_context(cfg: EdaConfig) -> panels.Context:
    """Load the data and compute everything every panel might need."""
    sets = load_series(cfg)
    ann_metrics = {
        s.name: ann_difficulty.compute(
            s.x,
            k=cfg.ann_k,
            k_hub=cfg.ann_hub_k,
            nlist=cfg.ivf_nlist,
            max_rows=cfg.ann_max_rows,
            seed=cfg.seed,
        )
        for s in sets
    }
    return panels.Context(
        config=cfg,
        series=sets,
        ann_metrics=ann_metrics,
        divergence=(
            metrics.dimension_divergence(sets, cfg.top_divergent)
            if len(sets) > 1
            else None
        ),
    )


def run(args: argparse.Namespace) -> Path:
    """Write the report. Takes a Namespace because compare_variants builds one."""
    cfg = EdaConfig.from_args(args)
    ctx = build_context(cfg)
    ...
```

The rest of `run` — the sections comprehension, `stats`, `meta_html`, writing the HTML, the best-effort PNG block, the `summary` dict and the four `print` calls — moves verbatim with `args.` becoming `cfg.`. The `summary` dict's `"synthetic_paths": args.synthetic_path or []` becomes `cfg.synthetic_path`, which `EdaConfig.from_args` already normalised to a list — the emitted JSON is identical.

- [ ] **Step 3: Rewrite `eda_report.py`**

```python
"""CLI entrypoint for the descriptor EDA report.

The implementation lives in `src/eval/eda/`; this module exists so that
`python -m src.eval.eda_report`, the command in `README.md`,
`docs/datasets/*.md` and `check_gate.py`'s error messages, keeps working.

It deliberately re-exports nothing. Import from `src.eval.eda.<module>`.
"""

from src.eval.eda.cli import parse_args
from src.eval.eda.pipeline import run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Repoint `compare_variants`**

- Line 45: `from src.eval import eda_report` becomes `from src.eval.eda import pipeline`.
- Line 564: `eda_report.run(report_args)` becomes `pipeline.run(report_args)`.
- Lines 3, 485-491: comments and the docstring naming `eda_report.run` / `eda_report.parse_args` become `eda.pipeline.run` / `eda.cli.parse_args`.
- `tests/test_compare_variants.py:601`: `monkeypatch.setattr(cv.eda_report, "run", fake_run)` becomes `monkeypatch.setattr(cv.pipeline, "run", fake_run)`.
- `tests/test_compare_variants.py:191`: `eda_report.parse_args()` becomes `cli.parse_args()`, importing `from src.eval.eda import cli`. The `sys.argv[0]` stub at line 184 stays `"eda_report.py"` — it is the invoked command's name, which has not changed.

- [ ] **Step 5: Rename the end-to-end test file**

```bash
git mv tests/test_eda_report.py tests/test_eda_run.py
```

What remains in it after Tasks 2, 5, 6 and 7 took their pieces is the end-to-end coverage: `test_run_returns_written_report_path`, `test_run_accepts_several_synthetic_sets`, `test_report_writes_html_and_summary_with_ann_sections`. These keep calling `eda_report.run` — no: change them to `pipeline.run`, since `eda_report` no longer has a `run` attribute. `test_report_writes_html_and_summary_with_ann_sections` calls `eda_report.main()` at line 183; that stays, as it is testing the entrypoint.

- [ ] **Step 6: Verify the CLI surface is unchanged**

`--help` text is a user-facing contract and the one thing the golden report check cannot see, because `parse_args` moving modules changes what `description=__doc__` resolves to. Task 0 Step 4 captured the baseline into `$G/help_before.txt`.

```bash
$PY -m src.eval.eda_report --help > $G/help_after.txt
diff $G/help_before.txt $G/help_after.txt
```
Expected: no output. A diff here means `cli.py`'s docstring is not the one `eda_report.py` had.

- [ ] **Step 7: Verify**

```bash
make check
```
Expected: ruff clean, 446 passed.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

```bash
wc -l src/eval/eda_report.py src/eval/eda/*.py
```
Expected: `eda_report.py` under 25 lines; no file in `src/eval/eda/` over ~340.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "refactor(eda): extract cli and pipeline, reduce eda_report to an entrypoint

parse_args moves to src/eval/eda/cli.py and run to
src/eval/eda/pipeline.py, which now builds a Context and walks PANELS.
src/eval/eda_report.py is twenty lines of entrypoint, so
'python -m src.eval.eda_report' is unchanged -- verified against a
captured --help. compare_variants calls pipeline.run directly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Documentation and final verification

**Files:**
- Modify: `AGENTIC-REVIEW.md:185-198`
- Modify: `AGENTS.md:114-115` (the "Where to look" table)
- Modify: `docs/superpowers/specs/2026-08-05-eda-report-split-design.md` (status line)

- [ ] **Step 1: Update `AGENTIC-REVIEW.md`**

The table row `| src/eval/eda_report.py | 1017 | 113 lines |` and the paragraph at lines 196-198 ("`eda_report.py` at 1017 lines is also the file most likely to be edited by an agent... Worth splitting when something next touches it; not worth a dedicated refactor.") both describe a state that no longer exists. Replace the paragraph with a note that the split happened, naming the package and the commit range, and keep the surrounding test-coverage findings — `src/sample/generate.py` being untested is still true and is not this branch's job.

- [ ] **Step 2: Update the `AGENTS.md` router**

The "Where to look" table's `Evaluation and metric definitions` row points at `src/eval/`, which is still right. Add a row:

```markdown
| The EDA report's panels and prose | `src/eval/eda/panels.py` |
```

This is the highest-value line in the change for a future agent: it routes "change what the report says" to one file.

- [ ] **Step 3: Mark the spec implemented**

Change the spec's `Status:` line to `implemented 2026-08-05, see src/eval/eda/`.

- [ ] **Step 4: Full verification**

```bash
make check
```
Expected: ruff clean, 446 passed, 0 failures.

```bash
rm -rf $G/after && $G/snapshot.sh $G/after && $PY $G/compare.py $G/before $G/after
```
Expected: `GOLDEN OK (6 files identical)`.

```bash
grep -rn "eda_report\." src/ tests/ | grep -v "eda_report.py:"
```
Expected: no output — nothing reaches into the entrypoint module for attributes any more.

```bash
git diff --stat main...HEAD
```
Sanity-check that the net line count is roughly flat. A pure refactor that adds 300 lines of production code has done something other than move things.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs: record the eda_report split

AGENTIC-REVIEW.md's recommendation to split this file is now history
rather than a suggestion. AGENTS.md gains a router line pointing 'change
what the report says' at src/eval/eda/panels.py.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review notes

Checked against the spec:

- Module layout — Tasks 1-9, with `run.py` renamed to `pipeline.py` and the reason recorded in Global Constraints.
- `parse_synthetic_spec` moving to `series.py` — Task 2.
- `fig_dim_divergence` splitting — Task 3.
- `Context`/`Panel`/`PANELS` with `build` returning `None` — Task 8.
- `run` keeping its Namespace signature — Task 9, with the parity test untouched.
- `EdaConfig.from_args` preserving the `glyph_samples` guard — Task 1, tested.
- Consumer updates: `plot_descriptor_grid` (Tasks 2, 5, 7), `compare_variants` (Tasks 1, 9), `check_gate` (unchanged, verified by Task 10's grep), `AGENTIC-REVIEW.md` (Task 10).
- Golden verification with UUID normalisation — Task 0, re-run at every task boundary, and its own failure mode tested in Task 0 Step 5.
- Test files mirroring modules — Tasks 1, 2, 3, 5, 6, 7, 8, 9.

Deviations from the spec, both deliberate:

1. `run.py` → `pipeline.py`, for the call-site readability reason in Global Constraints.
2. The spec's module table lists ten modules totalling ~1510 lines against today's 1328. That estimate included docstrings this plan adds. The Task 10 diffstat check is there to catch the estimate being wrong by more than a couple of hundred lines, which would mean something was rewritten rather than moved.
