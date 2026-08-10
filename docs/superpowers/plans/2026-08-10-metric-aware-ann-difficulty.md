# Metric-aware ANN difficulty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/eval/ann_difficulty.py` measure under the distance its corpus is actually searched with, by reading `data.metric` through `compare_variants`, and correct the documentation that this makes false.

**Architecture:** `angular` is defined as L2 on the unit sphere, so no estimator changes — `compute` gains a `metric` parameter and a precondition that refuses rows which are not on the sphere. The value is threaded `compare_variants` → `argparse.Namespace` → `EdaConfig` → `build_context` → `compute`, sourced from each variant's repo config.

**Tech Stack:** Python 3.12, numpy, scikit-learn, PyYAML, pytest, ruff.

Spec: `docs/superpowers/specs/2026-08-10-metric-aware-ann-difficulty-design.md`. Closes issues #22 and #16.

## Global Constraints

- **Metric vocabulary is exactly `l2` and `angular`.** No third value, no synonyms (`cosine`, `ip`).
- **Default is `l2` at every layer.** An existing caller that passes no metric must behave exactly as it does today.
- **`src/eval/ann_difficulty.py` must not import from `src.eval.eda` or `src.data.dataset`.** The first drags plotly and argparse, the second drags torch. Its module docstring commits to staying usable and testable without them.
- **No estimator changes.** `knn`, `survivor_mask`, `lid_mle`, `relative_contrast`, `k_occurrence`, `hubness_skew`, `cell_occupancy`, `gini` and `summary` keep their current bodies. Any diff that moves a measured number is out of scope.
- **Do not re-run or edit DEEP's numbers.** `docs/datasets/deep.md` and `docs/datasets/deep_ladder_summary.json` stand as-is.
- **`make check` is the gate** — ruff lint, ruff format check, pytest. Run from the repo root on Python 3.12.
- **This worktree has no `.venv`.** The tools live in the main checkout. `Makefile:7-8` declares `PYTHON ?= python` and `RUFF ?= ruff`, so override both:

  ```bash
  make check \
    PYTHON=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python \
    RUFF=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/ruff
  ```

- **Never run `make format`.** Its target is `ruff format src tests` — repo-wide, which AGENTS.md forbids. Format only the files you touched, by invoking `.venv/bin/ruff format <paths>` directly.
- **`PROJECT_DOCUMENTATION.md` and `docs/datasets/*.md` are authoritative** and policed by `tests/test_docs_references.py`: every path, anchor and symbol they cite must resolve, and no new line-number citations.

---

### Task 1: Metric-aware `compute`

Self-contained: no caller changes, no plumbing. The default keeps every existing call site on today's behaviour.

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Test: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `METRICS: tuple[str, ...] = ("l2", "angular")`
  - `require_unit_norm(x: np.ndarray, atol: float = 1.0e-4) -> None`
  - `compute(x, *, k=100, k_hub=10, nlist=256, max_rows=20000, seed=42, metric: str = "l2") -> AnnMetrics`

- [ ] **Step 1: Write the failing tests**

`tests/test_ann_difficulty.py` currently imports only `numpy`. Add `import pytest` at the top, and add `require_unit_norm` to the existing `from src.eval.ann_difficulty import (...)` block. Then append:

```python
# --- Metric awareness -----------------------------------------------------
#
# `angular` is L2 on the unit sphere. On unit-norm rows Euclidean distance is
# sqrt(2 * cosine distance), which is strictly increasing, so the two order
# neighbours identically and every statistic below comes out the same. That
# equivalence is the entire reason the angular path adds no estimator, and the
# reason `docs/datasets/deep.md`'s numbers stand unchanged -- so it is pinned
# here rather than left as a comment.


def _unit_rows(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_angular_and_l2_agree_exactly_on_unit_norm_rows():
    x = _unit_rows(600, 16, seed=0)
    kwargs = dict(k=20, k_hub=5, nlist=8, max_rows=0, seed=7)

    l2 = summary(compute(x, metric="l2", **kwargs))
    angular = summary(compute(x, metric="angular", **kwargs))

    assert l2 == angular


def test_angular_refuses_rows_that_are_not_on_the_unit_sphere():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(50, 8)) * 3.0).astype(np.float32)

    with pytest.raises(ValueError, match="--preprocess l2"):
        compute(x, k=5, k_hub=3, nlist=4, max_rows=0, metric="angular")


def test_angular_rejects_a_single_row_off_the_sphere():
    """One bad row is enough: it is a query and a neighbour for everyone else."""
    x = _unit_rows(60, 8, seed=2)
    x[17] *= 0.5

    with pytest.raises(ValueError, match="1 of 60"):
        compute(x, k=5, k_hub=3, nlist=4, max_rows=0, metric="angular")


def test_angular_accepts_the_zero_rows_maybe_l2_normalize_leaves_behind():
    """`maybe_l2_normalize` clamps the divisor, so a zero row stays exactly 0.

    That is a deliberate output of our own preprocessing, not a caller
    mistake, so it must not make an otherwise-normalized set unmeasurable.
    """
    x = _unit_rows(60, 8, seed=3)
    x[0] = 0.0

    m = compute(x, k=5, k_hub=3, nlist=4, max_rows=0, metric="angular")

    assert m.num_rows == 60


def test_l2_measures_rows_of_any_norm():
    rng = np.random.default_rng(4)
    x = (rng.normal(size=(60, 8)) * 3.0).astype(np.float32)

    m = compute(x, k=5, k_hub=3, nlist=4, max_rows=0, metric="l2")

    assert m.num_rows == 60


def test_compute_defaults_to_l2():
    """Back-compat: every existing call site passes no metric."""
    rng = np.random.default_rng(5)
    x = (rng.normal(size=(60, 8)) * 3.0).astype(np.float32)

    assert compute(x, k=5, k_hub=3, nlist=4, max_rows=0).num_rows == 60


def test_compute_rejects_an_unknown_metric():
    x = _unit_rows(20, 4, seed=6)

    with pytest.raises(ValueError, match="cosine"):
        compute(x, k=3, k_hub=2, nlist=2, max_rows=0, metric="cosine")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -v
```

Expected: `ImportError: cannot import name 'require_unit_norm'` — the whole module fails to collect. That is the correct first failure.

- [ ] **Step 3: Add `METRICS` and `require_unit_norm`**

In `src/eval/ann_difficulty.py`, after the imports and before `def gini(...)`:

```python
METRICS = ("l2", "angular")


def require_unit_norm(x: np.ndarray, atol: float = 1.0e-4) -> None:
    """Refuse rows that are not on the unit sphere.

    `angular` is measured as L2 between unit vectors: on the sphere Euclidean
    distance is sqrt(2 * cosine distance), strictly increasing, so it orders
    neighbours exactly as cosine does while keeping every estimator below on
    a true metric -- which cosine distance, failing the triangle inequality,
    is not.

    That equivalence holds only on the sphere, so rows that are not there are
    refused rather than normalized here. Normalizing would let the report's
    `preprocess:` line read `none` while the difficulty panels were measured
    on normalized rows, with nothing anywhere to surface the divergence.

    An exactly-zero row is accepted. `eda.series.maybe_l2_normalize` clamps
    its divisor rather than dividing by ~0, so a zero row is a deliberate
    output of our own preprocessing, not a caller mistake.
    """
    norms = np.linalg.norm(x, axis=1)
    offenders = ~(np.isclose(norms, 1.0, atol=atol) | (norms == 0.0))
    if not np.any(offenders):
        return
    bad = norms[offenders]
    raise ValueError(
        f"metric='angular' measures L2 on the unit sphere, so rows must "
        f"already be unit-norm; {int(offenders.sum())} of {norms.size} are "
        f"not (norms {bad.min():.6g} to {bad.max():.6g}). Pass --preprocess l2."
    )
```

- [ ] **Step 4: Give `compute` the parameter**

Change the signature and add the two checks. Keep every other line of the body as it is:

```python
def compute(
    x: np.ndarray,
    *,
    k: int = 100,
    k_hub: int = 10,
    nlist: int = 256,
    max_rows: int = 20000,
    seed: int = 42,
    metric: str = "l2",
) -> AnnMetrics:
    """Measure every difficulty metric for one set off a single k-NN pass.

    Callers must pass the same max_rows for every set they intend to compare:
    LID, relative contrast and hubness all drift with sample count, so
    unequal N makes the overlay meaningless.

    `metric` is the distance the corpus is searched under, from its config's
    `data.metric`. `angular` requires unit-norm rows and is then measured as
    L2 between them; see `require_unit_norm`. It defaults to `l2`, so a
    caller that does not know about metrics is unaffected.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {METRICS}.")
    x = np.ascontiguousarray(_subsample(x, max_rows, seed), dtype=np.float32)
    if metric == "angular":
        require_unit_norm(x)
    n = x.shape[0]
```

The precondition sits after `_subsample` because the subsample is what actually gets measured.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -v
```

Expected: PASS, including the pre-existing tests.

- [ ] **Step 6: Run the full suite**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q
```

Expected: PASS. Nothing else passes `metric` yet, so nothing else can have moved.

- [ ] **Step 7: Commit**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/ruff format \
  src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): measure ANN difficulty under the corpus's own metric

\`angular\` is L2 on the unit sphere: on unit-norm rows Euclidean is
sqrt(2 * cosine), so the two order neighbours identically and no estimator
changes. \`compute\` refuses rows off the sphere rather than normalizing
them, so the report's preprocess line cannot disagree with the geometry
measured.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Thread the metric from the configs to the panels

`EdaConfig` is the value object every panel sees; `eda.cli` is the standalone entry point; `compare_variants` is where the family's config is already resolved. All three change together in **one commit**.

They cannot be split. `test_report_args_match_eda_report_fields` asserts that `eda.cli.parse_args` and `build_report_args` produce identical field sets, so a commit adding `--metric` to only one leaves the suite red — and `AGENTS.md` is explicit that a red suite is a failure, not a warning. Splitting the other way (CLI flag last) leaves `EdaConfig.from_args` reading an `args.metric` the CLI no longer supplies, crashing the standalone entry point with no test to catch it.

**Files:**
- Modify: `src/eval/eda/config.py`
- Modify: `src/eval/eda/cli.py:29-36` (import block), and its argument list
- Modify: `src/eval/eda/pipeline.py:22-32` (the `compute` call), `pipeline.py:107-112` (`ann_settings`)
- Modify: `src/eval/compare_variants.py` — add `family_metric`, change `build_report_args:485`, wire `main:518-566`
- Modify: `tests/conftest.py:33-52` (`make_args`) and `tests/conftest.py:71-96` (`_write_run`)
- Modify: `tests/test_eda_config.py:8-29` (`_full_namespace`)
- Modify: `tests/test_compare_variants.py:179` (the `build_report_args` call)
- Test: `tests/test_eda_config.py`, `tests/test_eda_run.py`, `tests/test_compare_variants.py`

**Interfaces:**
- Consumes: `ann_difficulty.compute(..., metric=...)` and `METRICS` from Task 1.
- Produces:
  - `src.eval.eda.config.METRIC_DEFAULT = "l2"`
  - `EdaConfig.metric: str`
  - `--metric` on `eda.cli.parse_args`
  - `summary.json` key `ann_settings.metric`
  - `family_metric(variants: Sequence[Variant], root: Path) -> str`
  - `build_report_args(args: argparse.Namespace, specs: list[str], metric: str) -> argparse.Namespace` — note the **third positional parameter**, required, no default.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eda_config.py`:

```python
def test_from_args_carries_the_metric():
    assert config.EdaConfig.from_args(_full_namespace()).metric == "l2"
```

Append to `tests/test_eda_run.py`:

```python
def test_summary_records_the_metric_measured_under(tmp_path):
    """A gate result is unreadable without the geometry it was measured in."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)

    args = make_args(tmp_path, real, {})
    args.metric = "angular"
    pipeline.run(args)

    summary = json.loads((Path(args.output_dir) / "summary.json").read_text())

    assert summary["ann_settings"]["metric"] == "angular"


def test_angular_runs_end_to_end_because_preprocess_puts_rows_on_the_sphere(tmp_path):
    """`make_args` uses preprocess='l2', which is exactly angular's precondition."""
    rng = np.random.default_rng(1)
    real = (rng.normal(size=(200, 8)) * 5.0).astype(np.float32)

    args = make_args(tmp_path, real, {})
    args.metric = "angular"

    assert pipeline.run(args).exists()


def test_angular_with_preprocess_none_fails_loudly(tmp_path):
    """The defect this change exists to close: geometry untied from the corpus."""
    rng = np.random.default_rng(2)
    real = (rng.normal(size=(200, 8)) * 5.0).astype(np.float32)

    args = make_args(tmp_path, real, {})
    args.metric = "angular"
    args.preprocess = "none"

    with pytest.raises(ValueError, match="--preprocess l2"):
        pipeline.run(args)


def test_cli_defaults_the_metric_to_l2(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report.py",
            "--real-path",
            "real.npy",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert cli.parse_args().metric == "l2"
```

`tests/test_eda_run.py` already imports `json`, `sys`, `Path`, `numpy`, `make_args`, `cli` and `pipeline`. Add `import pytest` if it is not already there.

Then append to `tests/test_compare_variants.py`:

```python
# --- Family metric --------------------------------------------------------
#
# Read from the repo config, never from run_dir/run_config.yaml: run configs
# predate `data.metric`, so a run trained before the field existed falls back
# to l2 -- silently wrong for exactly the angular families this exists for.


def _write_family_config(root, rel_path, metric=None):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"descriptor_dim": 8}
    if metric is not None:
        data["metric"] = metric
    path.write_text(yaml.safe_dump({"data": data}))
    return path


def test_family_metric_reads_angular_from_the_variant_configs(tmp_path):
    _write_family_config(tmp_path, "configs/deep/v0.yaml", "angular")
    _write_family_config(tmp_path, "configs/deep/v1.yaml", "angular")
    variants = (
        cv.Variant("v0", "configs/deep/v0.yaml", "runs/deep/v0"),
        cv.Variant("v1", "configs/deep/v1.yaml", "runs/deep/v1"),
    )

    assert cv.family_metric(variants, tmp_path) == "angular"


def test_family_metric_defaults_to_l2_when_a_config_omits_it():
    """Older configs predate the field; l2 is what they were measured under."""
    variants = (cv.Variant("v0", "configs/sift/v0.yaml", "runs/v0"),)

    assert cv.family_metric(variants, cv.REPO_ROOT) == "l2"


def test_family_metric_refuses_configs_that_disagree(tmp_path):
    _write_family_config(tmp_path, "configs/x/v0.yaml", "l2")
    _write_family_config(tmp_path, "configs/x/v1.yaml", "angular")
    variants = (
        cv.Variant("v0", "configs/x/v0.yaml", "runs/v0"),
        cv.Variant("v1", "configs/x/v1.yaml", "runs/v1"),
    )

    with pytest.raises(SystemExit) as excinfo:
        cv.family_metric(variants, tmp_path)

    message = str(excinfo.value)
    assert "v0" in message and "v1" in message
    assert "angular" in message and "l2" in message


def test_family_metric_names_a_config_it_cannot_find(tmp_path):
    variants = (cv.Variant("v0", "configs/gone/v0.yaml", "runs/v0"),)

    with pytest.raises(SystemExit) as excinfo:
        cv.family_metric(variants, tmp_path)

    assert "configs/gone/v0.yaml" in str(excinfo.value)


def test_family_metric_reads_every_manifest_entry_not_only_resolved_ones(tmp_path):
    """Geometry must not depend on which checkpoints happen to be on this box."""
    _write_family_config(tmp_path, "configs/y/v0.yaml", "angular")
    _write_family_config(tmp_path, "configs/y/v1.yaml", "l2")
    variants = (
        cv.Variant("v0", "configs/y/v0.yaml", "runs/y/v0"),
        cv.Variant("v1", "configs/y/v1.yaml", "runs/y/v1"),
    )

    with pytest.raises(SystemExit):
        cv.family_metric(variants, tmp_path)


def test_main_hands_the_family_metric_to_the_report(
    monkeypatch, tmp_path, write_tiny_gated_run
):
    variant, _ = write_tiny_gated_run(tmp_path)
    _write_family_config(tmp_path, variant.config_path, "angular")
    manifest = write_manifest(
        tmp_path / "variants.yaml",
        [
            {
                "name": variant.name,
                "config": variant.config_path,
                "run_dir": variant.run_dir,
            }
        ],
    )

    seen = {}

    def fake_run(args):
        seen["metric"] = args.metric
        return Path(args.output_dir) / "report.html"

    monkeypatch.setattr(cv.pipeline, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_variants.py",
            "--real-path",
            str(tmp_path / "real.npy"),
            "--output-dir",
            str(tmp_path / "out"),
            "--root",
            str(tmp_path),
            "--variants-manifest",
            str(manifest),
            "--num-samples",
            "20",
            "--batch-size",
            "8",
        ],
    )

    cv.main()

    assert seen["metric"] == "angular"
```

Then fix the two existing call sites this task changes:

1. `tests/test_compare_variants.py:179` — `cv.build_report_args(args, specs=["v0=a.npy"])` becomes `cv.build_report_args(args, specs=["v0=a.npy"], metric="l2")`.
2. `test_main_reports_on_the_variants_a_custom_manifest_resolves` — its manifest has a second entry naming `configs/sift/v0.yaml`, which does not exist under `tmp_path`. Add `_write_family_config(tmp_path, "configs/sift/v0.yaml")` before the `cv.main()` call.


- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest \
  tests/test_eda_config.py tests/test_eda_run.py tests/test_compare_variants.py -v
```

Expected: `AttributeError: 'EdaConfig' object has no attribute 'metric'`, `AttributeError: 'Namespace' object has no attribute 'metric'`, and `AttributeError: module 'src.eval.compare_variants' has no attribute 'family_metric'`.

- [ ] **Step 3: Add the constant and the field**

In `src/eval/eda/config.py`, after `IVF_NLIST_DEFAULT = 256`:

```python
# The distance the corpus is searched under, from its config's `data.metric`.
# Shared with compare_variants for the same reason as the ANN defaults above.
# `l2` is the default because it is what every pre-existing report measured.
METRIC_DEFAULT = "l2"
```

In the `EdaConfig` dataclass, add `metric: str` immediately after `preprocess: str`, and in `from_args` add `metric=args.metric,` immediately after `preprocess=args.preprocess,`.

- [ ] **Step 4: Add the CLI flag**

In `src/eval/eda/cli.py`, add `METRIC_DEFAULT,` to the `from src.eval.eda.config import (...)` block (keep it alphabetical: it goes after `KNN_MAX_ROWS_DEFAULT`). Then add the argument immediately after `--ivf-nlist`:

```python
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC_DEFAULT,
        choices=list(METRICS),
        help=(
            "Distance the corpus is searched under, from the family's "
            "`data.metric`. 'angular' is measured as L2 on the unit sphere, "
            "so it requires --preprocess l2."
        ),
    )
```

Import `METRICS` from `src.eval.ann_difficulty` at the top of `cli.py`, so the accepted vocabulary is stated once:

```python
from src.eval.ann_difficulty import METRICS
```

- [ ] **Step 5: Pass it through the pipeline**

In `src/eval/eda/pipeline.py`, add `metric=cfg.metric,` to the `ann_difficulty.compute(...)` call in `build_context`, after `seed=cfg.seed,`.

In `run`, add `"metric": cfg.metric,` to the `ann_settings` dict, after `"nlist": cfg.ivf_nlist,`.

- [ ] **Step 6: Add `family_metric`**

In `src/eval/compare_variants.py`, after `resolve_variants` and before `_needs_inversion`:

```python
def family_metric(variants: Sequence[Variant], root: Path) -> str:
    """The distance this family's corpus is searched under.

    Read from each variant's repo config, never from
    `run_dir/run_config.yaml`. Run configs predate the `data.metric` field, so
    a run trained before it existed would fall back to `l2` -- silently wrong
    for exactly the angular families this exists for. A run config is evidence
    of what ran, not a statement about what the corpus is.

    Every manifest entry is read, not only the ones whose checkpoints resolved
    on this box, so the geometry a report is measured under cannot depend on
    which runs happen to be present.
    """
    by_metric: dict[str, list[str]] = {}
    for variant in variants:
        path = root / variant.config_path
        if not path.is_file():
            raise SystemExit(
                f"no config at {path} for variant {variant.name!r}. The "
                "manifest names it, and its data.metric decides the distance "
                "the report measures under."
            )
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        metric = str((doc.get("data") or {}).get("metric", eda_config.METRIC_DEFAULT))
        by_metric.setdefault(metric, []).append(variant.name)

    if len(by_metric) > 1:
        detail = "; ".join(
            f"{metric}: {', '.join(names)}" for metric, names in sorted(by_metric.items())
        )
        raise SystemExit(
            "variants disagree on data.metric, so there is no single distance "
            f"to measure this family under ({detail}). Variant numbers are "
            "per-family, so one ladder is one corpus and one metric; fix the "
            "configs, or compare only the variants that agree."
        )
    return next(iter(by_metric))
```

`Sequence` and `eda_config` are already imported (`compare_variants.py:37`, `:47`).

- [ ] **Step 7: Give `build_report_args` the parameter**

```python
def build_report_args(
    args: argparse.Namespace, specs: list[str], metric: str
) -> argparse.Namespace:
```

and add `metric=metric,` to the returned Namespace, after `preprocess="l2",`. Leave `preprocess="l2"` exactly as it is — it already supplies what `angular` requires, for every family.

Add to the docstring, after the existing paragraph:

```
    `metric` is passed rather than read off `args` because it is a property of
    the corpus, recorded per family in config. A `--metric` flag would be a
    second place to state it, and so a place for it to go stale.
```

- [ ] **Step 8: Wire it into `main`**

In `main`, insert between the `if not found:` block and `samples_dir = out_dir / "samples"`:

```python
    # After the resolve checks, so a fresh clone still hears about missing
    # runs first; before sampling, so a config problem does not cost the
    # caller several hundred thousand vectors.
    metric = family_metric(variants, root)
```

Then change the report call:

```python
    report_args = build_report_args(args, specs, metric)
```

- [ ] **Step 9: Update the shared test fixtures**

In `tests/conftest.py`, add `metric=eda_config.METRIC_DEFAULT,` to `make_args`'s Namespace, after `preprocess="l2",`.

In `tests/test_eda_config.py`, add `metric="l2",` to `_full_namespace()`, after `preprocess="l2",`. `test_from_args_covers_every_field_the_parser_produces` compares field sets, so it will fail until both sides carry it.


`--root` is documented as "Repo root that variant config and run paths resolve against" (`compare_variants.py:400`), and `family_metric` is the first code to actually use that half of the contract. The fixtures build Variants naming real repo configs while rooting at `tmp_path`, so they must now write them.

In `tests/conftest.py`, in `_write_run`, before the `return`:

```python
    # `family_metric` resolves config paths against --root, which these tests
    # point at tmp_path. Write the config the Variant names so the fixture is
    # a complete tree rather than one that only happens to work.
    config_full = tmp_path / config_path
    config_full.parent.mkdir(parents=True, exist_ok=True)
    config_full.write_text(
        yaml.safe_dump({"data": {"metric": "l2", "descriptor_dim": descriptor_dim}})
    )
```

`yaml` is already imported in `conftest.py:13`.


- [ ] **Step 10: Run the tests to verify they pass**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest \
  tests/test_eda_config.py tests/test_eda_run.py tests/test_compare_variants.py -v
```

Expected: PASS, including `test_report_args_match_eda_report_fields` and `test_from_args_covers_every_field_the_parser_produces` — the two parity guards this task has to satisfy on both sides at once.

- [ ] **Step 11: Run `make check`**

```bash
make check \
  PYTHON=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python \
  RUFF=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/ruff
```

Expected: PASS — ruff lint, ruff format check, and the full suite. The suite must be green before you commit; this task is deliberately one commit so it never lands red.

- [ ] **Step 12: Commit**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/ruff format \
  src/eval/eda/config.py src/eval/eda/cli.py src/eval/eda/pipeline.py \
  src/eval/compare_variants.py tests/conftest.py tests/test_eda_config.py \
  tests/test_eda_run.py tests/test_compare_variants.py
git add src/eval/eda/config.py src/eval/eda/cli.py src/eval/eda/pipeline.py \
  src/eval/compare_variants.py tests/conftest.py tests/test_eda_config.py \
  tests/test_eda_run.py tests/test_compare_variants.py
git commit -m "feat(eval): measure each family under the metric its configs record

compare_variants reads data.metric from each variant's repo config and
threads it through EdaConfig into the difficulty panels; summary.json records
it so a gate result carries the geometry it was measured under.

Repo config rather than run_config.yaml: run configs predate the field and
would fall back to l2, silently wrong for the angular families this exists
for. Resolved before sampling so a config problem is cheap, and after the run
checks so a fresh clone still hears about missing runs first.

One commit because two parity guards span the CLI and compare_variants, and
splitting the change leaves the suite red in between.

Closes #22

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Correct the documentation this makes false

Seven statements across five files. All are authoritative docs policed by `tests/test_docs_references.py`.

**Files:**
- Modify: `docs/datasets/deep.md` (two statements), `glove.md`, `nytimes.md`, `openai.md` (one each)
- Modify: `PROJECT_DOCUMENTATION.md` (two statements)
- Test: `tests/test_docs_references.py` (existing, no changes)

**Interfaces:**
- Consumes: the behaviour built in Tasks 1-2.
- Produces: no code.

- [ ] **Step 1: Find the exact statements**

```bash
grep -rn "phase (c)" docs/datasets/*.md PROJECT_DOCUMENTATION.md
grep -n "inert today" PROJECT_DOCUMENTATION.md
```

Expected: the caveat in each of `deep.md`, `glove.md`, `nytimes.md`, `openai.md`; the second `deep.md` statement in its gate-band section; and two in `PROJECT_DOCUMENTATION.md` (the `data.metric` section and the gate section).

- [ ] **Step 2: Correct the four family pages**

In each of `docs/datasets/glove.md`, `nytimes.md`, `openai.md`, replace the caveat:

> `ann_difficulty.py` currently measures everything under L2, including this
> family's `angular` corpus, so these numbers will need re-measuring once
> angular distance support lands (phase (c)).

with:

```markdown
`ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are not on the sphere rather than normalizing them itself.
```

In `docs/datasets/deep.md`, replace the same caveat with that text plus:

```markdown
The figures above were measured at `preprocess: l2`, as
`deep_ladder_summary.json` records, so they were already measured under this
geometry and stand unchanged.
```

- [ ] **Step 3: Correct the second `deep.md` statement**

In the gate-band section, the sentence claiming the numbers "move again when phase (c) re-measures this family under angular distance" is now false. Delete that clause, keeping the surrounding argument that bands still need a real seed sweep (issue #20) intact. The paragraph should end at "...a band set from either draw alone would be fitted to noise. Setting them needs a real seed sweep."

- [ ] **Step 4: Correct `PROJECT_DOCUMENTATION.md`**

In the `data.metric` section, replace:

> The value is validated at load time against the two accepted strings and is
> otherwise inert today: nothing consumes it yet. Reading it in
> `src/eval/ann_difficulty.py`, so difficulty is measured under the metric the
> corpus is actually searched with, is phase (c).

with:

```markdown
The value is validated at load time against the two accepted strings.
`src/eval/compare_variants.py` reads it from a family's variant configs and
threads it into `src/eval/ann_difficulty.py`, so difficulty is measured under
the metric the corpus is actually searched with. `angular` is measured as L2
between unit-norm rows, which orders neighbours exactly as cosine does;
`compute` refuses rows that are not unit-norm rather than normalizing them,
so a report's `preprocess` setting cannot disagree with the geometry it
measured under.
```

In the gate section, replace:

> `ann_difficulty.py` currently computes everything under L2, including for the
> four `angular` families. Reading `data.metric` and measuring under the
> corpus's own distance is phase (c) of the multi-dataset design; until it
> lands, angular-family numbers are internally consistent within a report but
> are not the distance the corpus is searched with.

with:

```markdown
`ann_difficulty.py` measures each family under its `data.metric`. The four
`angular` families are measured as L2 between unit-norm rows, which is the
distance their corpora are searched under; `l2` families are measured as
given. Reports must therefore run angular families at `--preprocess l2`,
which `compare_variants` does for every family.
```

- [ ] **Step 5: Verify the docs tests still pass**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v
```

Expected: PASS. Every path, anchor and symbol cited must still resolve, and no line-number citations may be introduced — `src/eval/ann_difficulty.py` and `compute` are referenced by name only, which is what these tests require.

- [ ] **Step 6: Verify no stale claim survives**

```bash
grep -rn "phase (c)" docs/datasets/*.md PROJECT_DOCUMENTATION.md
grep -rn "inert today\|nothing consumes it yet" PROJECT_DOCUMENTATION.md
```

Expected: no matches in either. (`docs/superpowers/` still mentions phase (c); that is correct — those are dated design notes, explicitly non-authoritative.)

- [ ] **Step 7: Run `make check`**

```bash
make check \
  PYTHON=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python \
  RUFF=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/ruff
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add docs/datasets/deep.md docs/datasets/glove.md docs/datasets/nytimes.md \
  docs/datasets/openai.md PROJECT_DOCUMENTATION.md
git commit -m "docs: the angular families are measured under their own metric

Four family pages and PROJECT_DOCUMENTATION.md said these numbers would need
re-measuring once phase (c) landed. They do not: angular is L2 on the unit
sphere, and DEEP's committed profile was measured at preprocess: l2, which is
that same geometry. deep_ladder_summary.json records it, so the claim is
checkable from this repo alone.

Closes #16

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done

`make check` green, and:

- `python -m src.eval.eda.cli --help` shows `--metric`.
- `python -m src.eval.compare_variants --dataset deep ...` measures under `angular` without being told.
- An angular family at `--preprocess none` fails with a message naming `--preprocess l2`.
- No authoritative doc claims the angular numbers are pending a re-measurement.

Open a PR titled **"Measure ANN difficulty under the corpus's own metric"**, body closing #22 and #16. Per `pr-is-not-done-until-checks-are-green`: watch the checks and fix until green before handing it over.
