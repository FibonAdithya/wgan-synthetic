# Multi-dataset Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase (a) of `docs/superpowers/specs/2026-08-04-multi-dataset-ann-emulation-design.md` — make the repo dataset-general in name, acquisition, configuration and documentation, without changing any model or any existing SIFT result.

**Architecture:** Four mechanical changes then two documentation rewrites. `src/data/sift1m_dataset.py` becomes `src/data/dataset.py` (rename only, no behaviour change). `src/deep/download.py` becomes `src/data/fetch.py` with a six-entry source registry, keeping its atomic single-flight caching verbatim. A `data.metric` config field records `l2` vs `angular` so phase (c)'s evaluation can read it. Configs gain a per-dataset directory. Then `README.md`, `PROJECT_DOCUMENTATION.md`, `data/README.md` are reframed and `docs/datasets/*.md` created.

**Tech Stack:** Python 3.12, numpy, torch, h5py, pyyaml, pytest. Run everything with the repo-root interpreter: `.venv/bin/python`.

## Global Constraints

- **No behaviour change to training, sampling or evaluation.** This phase is rename, relocation, registry and prose. Any diff that changes a number is out of scope.
- **The existing test suite must pass unchanged after every task.** `.venv/bin/python -m pytest` from the repo root. That the SIFT tests still pass untouched is what proves the rename was a rename.
- **No new model families.** `spherical` is phase (b). Every angular dataset's `v0` config uses `generator_type: mlp`, because a per-dataset ladder starts from plain WGAN-GP by definition.
- **Existing run directories under `runs/` do not move.** The spec keeps them; docs record the flat names as historical.
- **`.npy` is the on-disk interchange format.** `src/data/fetch.py` writes `.npy`, so neither the loader nor the trainer learns about HDF5.
- **Metric vocabulary is exactly `l2` and `angular`.** Used in configs, the source registry and the dataset docs. No third value, no synonyms (`cosine`, `ip`).
- **Dataset names are exactly `sift`, `gist`, `deep`, `glove`, `nytimes`, `openai`.** Used as registry keys, config directory names and dataset doc filenames.
- Docs under `docs/superpowers/` are AI working notes and open with the standard provenance blockquote. `README.md`, `PROJECT_DOCUMENTATION.md`, `data/README.md` and `docs/datasets/*.md` are human-maintained sources of truth and carry no such banner.

---

### Task 1: Rename the dataset module

Pure rename. The module is already dimension-agnostic; only its name says SIFT.

**Files:**
- Create: `src/data/dataset.py` (via `git mv`)
- Delete: `src/data/sift1m_dataset.py` (via `git mv`)
- Modify: `src/train/train_wgan_gp.py:18`, `src/eval/evaluate_distribution.py:14`, `src/eval/evaluate_file_to_file.py:9`, `src/eval/eda_report.py:38`, `src/eval/plot_distance_cdf.py:10`, `src/eval/plot_distance_cdf_pillow.py:11`, `src/eval/plot_embedding_clusters.py:12`
- Modify: `src/deep/download.py:3` (docstring reference; the module is deleted in Task 2 but must not carry a stale path until then)

**Interfaces:**
- Consumes: nothing.
- Produces: `src.data.dataset` exporting the same names as before — `PreprocessConfig`, `PreprocessState`, `apply_preprocess`, `load_descriptors`, `train_holdout_split`, `NumpyTensorDataset`, `build_training_data`. No signature changes. No compatibility shim: `src.data.sift1m_dataset` stops existing.

- [ ] **Step 1: Record the current test baseline**

Run: `.venv/bin/python -m pytest`
Expected: PASS. Note the count; it must be identical at the end of this task.

- [ ] **Step 2: Move the file with git**

```bash
git mv src/data/sift1m_dataset.py src/data/dataset.py
```

- [ ] **Step 3: Run the tests to verify they now fail**

Run: `.venv/bin/python -m pytest`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.sift1m_dataset'`

- [ ] **Step 4: Update every importer**

Seven call sites, all of the form `from src.data.sift1m_dataset import ...`. Rewrite each to `from src.data.dataset import ...`, leaving the imported names and everything else on the line untouched:

```bash
grep -rln "src\.data\.sift1m_dataset" src/ | xargs sed -i 's/src\.data\.sift1m_dataset/src.data.dataset/g'
```

Then confirm nothing under `src/` still refers to the old path:

```bash
grep -rn "sift1m_dataset" src/
```

Expected: one remaining hit, the docstring at `src/deep/download.py:3`.

- [ ] **Step 5: Fix the DEEP docstring reference**

In `src/deep/download.py`, line 3, change:

```
Writes .npy deliberately: the existing loader in src/data/sift1m_dataset.py
```

to:

```
Writes .npy deliberately: the existing loader in src/data/dataset.py
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest`
Expected: PASS, with exactly the count recorded in Step 1. No test file was edited — that is the point of this task.

- [ ] **Step 7: Commit**

```bash
git add -A src/
git commit -m "refactor(data): rename sift1m_dataset to dataset

The module is dimension-agnostic already; only its name tied it to one
dataset. Rename only -- no signature or behaviour change."
```

---

### Task 2: Source registry and generalized fetcher

`src/deep/download.py` becomes `src/data/fetch.py` covering all six families. The acquisition discipline — atomic write, single-flight `.part` lock, `.npy` output — is kept exactly; what generalizes is the source table and the CLI.

All six sets are taken from ann-benchmarks HDF5 rather than mixing in corpus-texmex `.fvecs`, so the fetcher handles one container format. Hand-placed `.fvecs` files still work, because `load_descriptors` reads them independently of this module.

**Files:**
- Create: `src/data/fetch.py`
- Create: `tests/test_fetch.py`
- Delete: `src/deep/download.py`, `src/deep/__init__.py`, `tests/test_deep_download.py`
- Modify: `requirements.txt:7` (the h5py comment names the deleted module)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Source` — frozen dataclass with fields `name: str`, `url: str`, `dim: int`, `metric: str`, `hdf5_key: str`, `default_rows: Tuple[int, ...]`.
  - `SOURCES: Dict[str, Source]` — keyed by the six dataset names.
  - `fetch(url: str, dest: Path, *, chunk_bytes: int = 1 << 20, poll_seconds: float = 5.0, timeout_seconds: float = 3600.0) -> Path` — unchanged from `src/deep/download.py`.
  - `subset(hdf5_path: Path, out_path: Path, *, num_rows: int, seed: int = 42, key: str = "train") -> Path` — as before plus the `key` argument.
  - `subset_name(dataset: str, rows: int) -> str` — returns e.g. `"sift_250k"`, `"deep_1m"`.

- [ ] **Step 1: Write the failing test for the source registry**

Create `tests/test_fetch.py` with:

```python
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.fetch import SOURCES, Source, fetch, subset, subset_name


def test_registry_covers_exactly_the_six_families():
    assert set(SOURCES) == {"sift", "gist", "deep", "glove", "nytimes", "openai"}


def test_every_source_declares_a_known_metric_and_a_positive_dim():
    for name, src in SOURCES.items():
        assert isinstance(src, Source), name
        assert src.metric in {"l2", "angular"}, name
        assert src.dim > 0, name
        assert src.url.endswith(".hdf5"), name
        assert src.name == name


def test_registry_dims_match_the_names_upstream_publishes():
    assert SOURCES["sift"].dim == 128
    assert SOURCES["gist"].dim == 960
    assert SOURCES["deep"].dim == 96
    assert SOURCES["glove"].dim == 100
    assert SOURCES["nytimes"].dim == 256
    assert SOURCES["openai"].dim == 1536


def test_sift_and_gist_are_l2_and_the_rest_are_angular():
    assert SOURCES["sift"].metric == "l2"
    assert SOURCES["gist"].metric == "l2"
    for name in ("deep", "glove", "nytimes", "openai"):
        assert SOURCES[name].metric == "angular"


def test_subset_name_labels_thousands_and_millions():
    assert subset_name("sift", 250_000) == "sift_250k"
    assert subset_name("deep", 1_000_000) == "deep_1m"
    assert subset_name("glove", 2_000_000) == "glove_2m"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.fetch'`

- [ ] **Step 3: Create the module with the registry**

Create `src/data/fetch.py`. Start with everything below; the `fetch` and `subset` bodies are copied verbatim from `src/deep/download.py` except where noted.

```python
"""Fetch ANN benchmark descriptor sets and cut reproducible subsets from them.

Writes .npy deliberately: the loader in src/data/dataset.py reads .npy and
.fvecs, so emitting .npy means neither the loader nor the trainer needs to
learn about HDF5.

All six families are taken from the ann-benchmarks HDF5 mirrors so this module
handles one container format. Sets obtained by other routes -- corpus-texmex
.fvecs, say -- are read directly by load_descriptors and do not come through
here.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import h5py
import numpy as np

BASE_URL = "http://ann-benchmarks.com"


@dataclass(frozen=True)
class Source:
    """One benchmark family: where it comes from and what shape it has.

    `metric` is the distance the real corpus is searched under, and is the
    value that lands in a dataset config's `data.metric`. It is not a
    preprocessing instruction -- l2_normalize is set independently.
    """

    name: str
    url: str
    dim: int
    metric: str
    hdf5_key: str = "train"
    default_rows: Tuple[int, ...] = (250_000, 1_000_000)


def _ann_benchmarks(name: str, slug: str, dim: int, metric: str) -> Source:
    return Source(name=name, url=f"{BASE_URL}/{slug}.hdf5", dim=dim, metric=metric)


SOURCES: Dict[str, Source] = {
    "sift": _ann_benchmarks("sift", "sift-128-euclidean", 128, "l2"),
    "gist": _ann_benchmarks("gist", "gist-960-euclidean", 960, "l2"),
    "deep": _ann_benchmarks("deep", "deep-image-96-angular", 96, "angular"),
    "glove": _ann_benchmarks("glove", "glove-100-angular", 100, "angular"),
    "nytimes": _ann_benchmarks("nytimes", "nytimes-256-angular", 256, "angular"),
    "openai": _ann_benchmarks(
        "openai", "dbpedia-openai-1000k-angular", 1536, "angular"
    ),
}


def subset_name(dataset: str, rows: int) -> str:
    """Stable on-disk stem, e.g. sift_250k or deep_1m."""
    if rows >= 1_000_000:
        label = f"{rows // 1_000_000}m"
    else:
        label = f"{rows // 1000}k"
    return f"{dataset}_{label}"
```

- [ ] **Step 4: Run the registry tests**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: FAIL — the five registry tests pass, but the import of `fetch` and `subset` fails since they are not defined yet.

- [ ] **Step 5: Port `fetch` and `subset`**

Append to `src/data/fetch.py` the `fetch` function copied verbatim from `src/deep/download.py` (docstring included — the atomicity and single-flight explanation is the valuable part), then `subset` with one change: a `key: str = "train"` keyword argument replacing the hardcoded `f["train"]`.

```python
def subset(
    hdf5_path: Path,
    out_path: Path,
    *,
    num_rows: int,
    seed: int = 42,
    key: str = "train",
) -> Path:
    """Write a random `num_rows`-row sample of the `key` split to `out_path`.

    Rows are drawn without replacement and returned in sorted index order,
    which h5py requires for fancy indexing and which also makes the read
    sequential rather than random over a multi-gigabyte file.
    """
    hdf5_path = Path(hdf5_path)
    out_path = Path(out_path)
    with h5py.File(hdf5_path, "r") as f:
        split = f[key]
        total = split.shape[0]
        take = min(num_rows, total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=take, replace=False))
        rows = split[idx, :]
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rows)
    return out_path
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: PASS, all five.

- [ ] **Step 7: Port the acquisition tests**

Copy the whole body of `tests/test_deep_download.py` into `tests/test_fetch.py` below the tests already there, with three edits: import from `src.data.fetch` instead of `src.deep.download`, drop `DEEP_URL` from the import list, and rewrite the three monkeypatch target strings from `"src.deep.download.X"` to `"src.data.fetch.X"`. The fixture and the assertions are unchanged — the `fake_hdf5` fixture already builds a 500x96 `train` dataset, which exercises the default `key`.

Add one test for the new argument:

```python
def test_subset_reads_the_requested_split(fake_hdf5: Path, tmp_path: Path):
    out = subset(fake_hdf5, tmp_path / "t.npy", num_rows=5, key="test")
    assert np.load(out).shape == (5, 96)
```

- [ ] **Step 8: Run the ported tests**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: PASS. If a monkeypatch target was missed, the failure is a real network call attempt or a hang — check every `monkeypatch.setattr` string points at `src.data.fetch`.

- [ ] **Step 9: Add the CLI**

Append to `src/data/fetch.py`:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=sorted(SOURCES),
        help="Which benchmark family to fetch.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/data-cache",
        help="Where the shared read-only HDF5 files live. Each is downloaded once.",
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help="Subset sizes to write. Defaults to the source's default_rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = SOURCES[args.dataset]
    cache = fetch(source.url, Path(args.cache_dir) / Path(source.url).name)
    print(f"hdf5: {cache}")
    for rows in args.rows or source.default_rows:
        out = subset(
            cache,
            Path(args.out_dir) / f"{subset_name(source.name, rows)}.npy",
            num_rows=rows,
            seed=args.seed,
            key=source.hdf5_key,
        )
        shape = np.load(out, mmap_mode="r").shape
        print(f"subset: {out} {shape}")
        if shape[1] != source.dim:
            raise ValueError(
                f"{source.name}: expected dim {source.dim}, file has {shape[1]}. "
                "The registry entry and the upstream file disagree."
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Verify the CLI wiring without hitting the network**

Run: `.venv/bin/python -m src.data.fetch --help`
Expected: usage text listing the six dataset choices.

- [ ] **Step 11: Delete the DEEP module and its tests**

```bash
git rm -r src/deep tests/test_deep_download.py
```

- [ ] **Step 12: Update the requirements comment**

In `requirements.txt`, replace the h5py comment:

```
# HDF5 reader for the DEEP dataset, used only by src/deep/download.py.
```

with:

```
# HDF5 reader for the ann-benchmarks mirrors, used only by src/data/fetch.py.
```

- [ ] **Step 13: Run the full suite**

Run: `.venv/bin/python -m pytest`
Expected: PASS. The DEEP tests are gone and the fetch tests replace them.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "feat(data): source registry and generalized fetcher

Generalizes src/deep/download.py to all six benchmark families, keeping
its atomic single-flight caching and .npy output verbatim. Adds a Source
registry recording each family's url, dim and search metric."
```

---

### Task 3: `data.metric` config field

The metric a corpus is searched under is a property of the dataset, not of a
call site. Recording it in config now means phase (c)'s evaluation reads it
instead of assuming L2, and means every config written in Task 4 already
carries it.

**Files:**
- Modify: `src/data/dataset.py` (add `metric` to `PreprocessConfig` and `PreprocessState`)
- Modify: `src/train/train_wgan_gp.py` (pass it through from the `data` config block)
- Test: `tests/test_dataset_metric.py` (create)

**Interfaces:**
- Consumes: `src.data.dataset` from Task 1.
- Produces: `PreprocessConfig.metric: str = "l2"`, validated to be `l2` or `angular`, serialized through `PreprocessState.to_serializable()` and restored by its loader, so `run_config.yaml` and checkpoint metadata carry it.

- [ ] **Step 1: Read the current preprocess plumbing**

Read `src/data/dataset.py` lines 41-135 — `PreprocessConfig`, `PreprocessState`, `to_serializable`, its from-payload counterpart around line 72, and `_fit_preprocess_state`. Note exactly how `config` round-trips; the new field follows the same path and must be restored in the same place.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dataset_metric.py`:

```python
import numpy as np
import pytest

from src.data.dataset import PreprocessConfig, PreprocessState


def test_metric_defaults_to_l2():
    assert PreprocessConfig().metric == "l2"


def test_metric_accepts_angular():
    assert PreprocessConfig(metric="angular").metric == "angular"


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        PreprocessConfig(metric="cosine")


def test_metric_survives_serialization_round_trip():
    state = PreprocessState(
        descriptor_dim=8, config=PreprocessConfig(metric="angular")
    )
    payload = state.to_serializable()
    assert payload["config"]["metric"] == "angular"
    assert PreprocessState.from_serializable(payload).config.metric == "angular"
```

If the class method that rebuilds a `PreprocessState` from a payload is named
something other than `from_serializable`, use the real name from Step 1 in the
last test.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dataset_metric.py -v`
Expected: FAIL — `TypeError: PreprocessConfig.__init__() got an unexpected keyword argument 'metric'` for three of them.

- [ ] **Step 4: Add the field and its validation**

In `src/data/dataset.py`, extend the dataclass:

```python
METRICS = ("l2", "angular")


@dataclass
class PreprocessConfig:
    center: bool = False
    whiten: bool = False
    l2_normalize: bool = True
    eps: float = 1.0e-8
    metric: str = "l2"

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Unknown metric {self.metric!r}; expected one of {METRICS}. "
                "This is the distance the real corpus is searched under, not a "
                "preprocessing step."
            )
```

`asdict` already carries the new field into `to_serializable`, so only the
restore path needs checking: confirm it reconstructs `PreprocessConfig(**payload["config"])` rather than naming fields one by one. If it names them, add `metric=payload["config"]["metric"]`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_dataset_metric.py -v`
Expected: PASS, all four.

- [ ] **Step 6: Plumb it through training**

In `src/train/train_wgan_gp.py`, find where `PreprocessConfig` is built from the `data.preprocess` config block. The metric belongs one level up, in `data`, since it describes the corpus rather than the transform. Read the `data` block's key with a default and pass it in:

```python
preprocess_cfg = PreprocessConfig(
    **data_cfg.get("preprocess", {}),
    metric=data_cfg.get("metric", "l2"),
)
```

Match the surrounding style — if the existing code enumerates preprocess keys explicitly, add `metric=data_cfg.get("metric", "l2")` to that call in the same form.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest`
Expected: PASS. Every existing config omits `data.metric`, defaults to `l2`, and behaves as before — that is the check.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(data): record the search metric in dataset config

data.metric is l2 or angular and rides through PreprocessState into
run artifacts, so evaluation can stop assuming L2. Defaults to l2, so
existing configs are unaffected."
```

---

### Task 4: Per-dataset config directories

**Files:**
- Move: `configs/sift_gan_v0.yaml` → `configs/sift/v0.yaml`, and likewise `v1`, `v1_5`, `v2`
- Create: `configs/gist/v0.yaml`, `configs/deep/v0.yaml`, `configs/glove/v0.yaml`, `configs/nytimes/v0.yaml`, `configs/openai/v0.yaml`
- Modify: `src/eval/compare_variants.py:49-52`
- Modify: `tests/test_compare_variants.py:40,41,54,64,74,75,122`

Configs not in the `sift_gan_*` ladder — `wgan_gp_sift1m*.yaml`, `x100k_gated.yaml`, `configs/sweeps/` — stay at the top level. They are historical sweep and smoke configs, not ladder rungs, and moving them would break command lines recorded in run artifacts for no gain.

**Interfaces:**
- Consumes: `data.metric` from Task 3; `SOURCES` dims from Task 2.
- Produces: `configs/<dataset>/v0.yaml` for all six families, each a plain-WGAN-GP baseline. Later ladder rungs are added when a dataset is actually trained.

- [ ] **Step 1: Move the SIFT ladder**

```bash
mkdir -p configs/sift
git mv configs/sift_gan_v0.yaml configs/sift/v0.yaml
git mv configs/sift_gan_v1.yaml configs/sift/v1.yaml
git mv configs/sift_gan_v1_5.yaml configs/sift/v1_5.yaml
git mv configs/sift_gan_v2.yaml configs/sift/v2.yaml
```

- [ ] **Step 2: Run the tests to see what breaks**

Run: `.venv/bin/python -m pytest tests/test_compare_variants.py -v`
Expected: PASS — the tests use those paths as opaque strings and never open them. The stale paths are still wrong and get fixed in Step 3; the point of running now is to know the test suite will not catch it for you.

- [ ] **Step 3: Update the variant table and its tests**

In `src/eval/compare_variants.py`, lines 49-52:

```python
VARIANTS: Tuple[Variant, ...] = (
    Variant("v0", "configs/sift/v0.yaml", "runs/long_baseline"),
    Variant("v1", "configs/sift/v1.yaml", "runs/x100k_ema_only"),
    Variant("v1_5", "configs/sift/v1_5.yaml", "runs/x100k_improved"),
    Variant("v2", "configs/sift/v2.yaml", "runs/x100k_sparse_clamp4"),
)
```

Then rewrite the seven literals in `tests/test_compare_variants.py`:

```bash
sed -i 's|configs/sift_gan_v|configs/sift/v|g' tests/test_compare_variants.py
grep -rn "sift_gan_v" src/ tests/ configs/
```

Expected from the grep: no output.

- [ ] **Step 4: Add `data.metric` to the four SIFT configs**

In each of `configs/sift/v0.yaml`, `v1.yaml`, `v1_5.yaml`, `v2.yaml`, add one line to the `data` block directly under `format: npy`:

```yaml
  metric: l2
```

- [ ] **Step 5: Write the five new baseline configs**

Each is `configs/<dataset>/v0.yaml`. They share this shape, with the per-dataset values from the table below. Hidden dimensions scale with the descriptor dimension because a 1536d output from a 1024-unit layer is a bottleneck, not a design.

```yaml
# <DATASET> v0 -- plain WGAN-GP baseline. Every dataset's ladder starts here
# so each later rung is one attributable delta. Untrained: this config
# defines the starting point, it does not record a result.
seed: 42
device: auto
output_dir: runs/<dataset>/v0

data:
  real_path: data/<subset_stem>.npy
  format: npy
  metric: <metric>
  descriptor_dim: <dim>
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: <latent>
  generator_hidden_dims: <g_hidden>
  critic_hidden_dims: <c_hidden>
  negative_slope: 0.2
  generator_type: mlp

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

| dataset | subset_stem | metric | dim | latent | g_hidden | c_hidden |
|---|---|---|---|---|---|---|
| deep | `deep_250k` | angular | 96 | 96 | `[512, 1024, 1024]` | `[1024, 512, 256]` |
| glove | `glove_250k` | angular | 100 | 128 | `[512, 1024, 1024]` | `[1024, 512, 256]` |
| nytimes | `nytimes_250k` | angular | 256 | 256 | `[512, 1024, 1024]` | `[1024, 512, 256]` |
| gist | `gist_250k` | l2 | 960 | 512 | `[1024, 2048, 2048]` | `[2048, 1024, 512]` |
| openai | `openai_250k` | angular | 1536 | 512 | `[1024, 2048, 2048]` | `[2048, 1024, 512]` |

`l2_normalize: true` on the angular families is deliberate: it makes the
training contract match the metric. On GIST it matches what SIFT's ladder
does, and whether it should is the kind of question that dataset's ladder
answers.

- [ ] **Step 6: Verify every config parses and declares a coherent metric**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
for p in sorted(Path("configs").glob("*/v*.yaml")):
    cfg = yaml.safe_load(p.read_text())
    d = cfg["data"]
    assert d["metric"] in ("l2", "angular"), p
    assert d["descriptor_dim"] > 0, p
    print(f"{p}: dim={d['descriptor_dim']} metric={d['metric']} "
          f"gen={cfg['model']['generator_type']}")
PY
```

Expected: nine lines — four SIFT, five new — with dims matching the table and every `generator_type` being `mlp` except SIFT's `v2`, which is `gated`.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(configs): give each dataset its own ladder directory

Moves the SIFT ladder under configs/sift/ and adds a plain-WGAN-GP v0 for
the five other families. Sweep and smoke configs stay at the top level;
they are not ladder rungs."
```

---

### Task 5: Per-dataset documentation

Six new human-maintained pages. Each is the place a reader learns what a
family is, what it measures, and where its ladder stands.

**Files:**
- Create: `docs/datasets/sift.md`, `gist.md`, `deep.md`, `glove.md`, `nytimes.md`, `openai.md`

**Interfaces:**
- Consumes: config paths from Task 4, the fetch CLI from Task 2.
- Produces: pages that `README.md` and `PROJECT_DOCUMENTATION.md` link to in Task 6.

- [ ] **Step 1: Write the template**

Every page has these sections in this order. The per-dataset values come from
the tables in Steps 2 and 3.

```markdown
# <Title>

<Two or three sentences: what the vectors are, where they come from, and the
one structural fact that decides how it is modelled.>

## Source

    python -m src.data.fetch <name>

Fetches `<url>` into the shared cache and writes `data/<name>_250k.npy` and
`data/<name>_1m.npy`. The HDF5 is large and immutable; the fetcher downloads
it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `<dim>` |
| Search metric | `<metric>` |
| Upstream | `<slug>` |

## Structure

<What the values look like: sign, quantization, zero mass, norm distribution,
and what that implies for the generator. This is the section that justifies
the model family.>

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
        --real-path data/<name>_250k.npy \
        --output-dir runs/<name>/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

Read the four values out of `runs/<name>/profile/summary.json`.

## Model family

`<family>` — <one sentence of why>.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/<name>/v0.yaml` | — | not trained |

## Gate

Pass bands are per statistic, not a combined score, because the four fail in
different directions. Bands are set once this family has a trained ladder to
show what is achievable; until then this section records that they are unset.
```

The canonical N and k above (`20000`, `100`, `10`) are the current defaults in
`src/eval/ann_difficulty.py` and hold for all six families. Locking them per
page rather than referencing the defaults is the point: a default can change,
and a gate result must not silently change meaning with it.

- [ ] **Step 2: Fill the Source and Model family values**

| file | name | dim | metric | slug | family |
|---|---|---|---|---|---|
| `sift.md` | sift | 128 | l2 | `sift-128-euclidean` | `gated` |
| `gist.md` | gist | 960 | l2 | `gist-960-euclidean` | `mlp` |
| `deep.md` | deep | 96 | angular | `deep-image-96-angular` | `mlp` today, `spherical` when phase (b) lands |
| `glove.md` | glove | 100 | angular | `glove-100-angular` | `mlp` today, `spherical` when phase (b) lands |
| `nytimes.md` | nytimes | 256 | angular | `nytimes-256-angular` | `mlp` today, `spherical` when phase (b) lands |
| `openai.md` | openai | 1536 | angular | `dbpedia-openai-1000k-angular` | `mlp` today, `spherical` when phase (b) lands |

- [ ] **Step 3: Write each Structure section**

These are the paragraphs that justify the model family. Use them close to verbatim.

- **sift.md** — 128-dimensional SIFT descriptors, non-negative and quantized to uint8, with heavy mass at exactly zero. Points therefore sit on a lattice: exact ties and true duplicates are common and dominate the top of any neighbour list. A dense MLP generator cannot reproduce that support, which is why `gated` exists — a softplus magnitude times a sampled binary gate, giving exact zeros.
- **gist.md** — 960-dimensional GIST descriptors, non-negative dense float with little exact-zero mass. Shares SIFT's non-negativity but not its sparsity, so the thing `gated` was built to fix may not be present; the ladder starts on `mlp` and tries `gated` as a rung rather than assuming it. The high ambient dimension is the dominant cost.
- **deep.md** — 96-dimensional image descriptors from a deep network, dense, signed and unit-norm. The smallest angular family, which makes it the right first target for the `spherical` generator.
- **glove.md** — 100-dimensional word embeddings, dense and signed. Word frequency produces a pronounced density gradient across the space, which is the mechanism that generates hubs. Hubness skew is therefore the statistic this family is most likely to fail, and the most informative one when it does.
- **nytimes.md** — 256-dimensional document embeddings, dense and signed, with strong cluster structure by topic. Cluster structure is what IVF cell balance measures, so this family stresses that panel hardest.
- **openai.md** — 1536-dimensional text embeddings, already unit-norm. Ambient dimension is very high while intrinsic dimension is low, so LID and relative contrast are the statistics that carry information; per-dimension marginals say almost nothing at this width.

- [ ] **Step 4: Fill SIFT's ladder table**

SIFT is the only family with trained runs. Its ladder table replaces the
template's single row, taken from `PROJECT_DOCUMENTATION.md`'s current variant
table:

```markdown
| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift/v0.yaml` | `runs/long_baseline` | trained |
| `v1` | + generator EMA (`ema_decay: 0.999`) | `configs/sift/v1.yaml` | `runs/x100k_ema_only` | trained |
| `v1_5` | + distance reg (`alpha: 0.1`, 256 points) | `configs/sift/v1_5.yaml` | `runs/x100k_improved` | trained |
| `v2` | + gated generator | `configs/sift/v2.yaml` | `runs/x100k_sparse_clamp4` | trained |

Run directory names predate the per-dataset scheme and are kept as-is, since
the artifacts under them are already named that way. Run length is an
independent axis, not a variant: `bench_*` are 3k generator steps, `long_*`
30k, `x100k_*` 100k.
```

- [ ] **Step 5: Check every referenced path exists**

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path
missing = []
for page in sorted(Path("docs/datasets").glob("*.md")):
    for path in re.findall(r'`((?:configs|runs|data|src)/[^`]+)`', page.read_text()):
        if not Path(path).exists() and not path.startswith("data/"):
            missing.append(f"{page}: {path}")
print("\n".join(missing) or "all referenced paths exist")
PY
```

Expected: `all referenced paths exist`. `data/` paths are skipped — those files
are produced by the fetcher and are not in git.

- [ ] **Step 6: Commit**

```bash
git add docs/datasets
git commit -m "docs: a page per benchmark family

Source, structure, locked canonical N and k, model family, ladder and
gate status for each of the six families. Measured profiles are marked
unmeasured with the command that fills them."
```

---

### Task 6: Reframe the top-level documentation

The last task, because it links to everything the earlier five produced.

**Files:**
- Modify: `README.md` (rewrite)
- Modify: `PROJECT_DOCUMENTATION.md` (rewrite the framing sections; the metric definitions, EDA and workflow sections are kept)
- Modify: `data/README.md` (generalize the contract)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no code.

- [ ] **Step 1: Rewrite `README.md`**

Replace the SIFT1M framing. The file keeps its documentation-map and
quick-start structure and gains a dataset table. Specifically:

- Title and opening: the project trains GANs that reproduce the
  nearest-neighbour search difficulty of six benchmark families, so ANN
  algorithms can be developed against synthetic corpora. Not "match SIFT1M's
  distribution".
- Documentation map: add `docs/datasets/` as human-maintained source of truth,
  alongside the existing `README.md` / `PROJECT_DOCUMENTATION.md` /
  `data/README.md` entries and the `docs/superpowers/` non-authoritative note.
  Keep the "where these disagree, `PROJECT_DOCUMENTATION.md` wins" rule.
- Replace the "Model variants" section with a dataset table: family, dim,
  metric, ladder status, link to its page. SIFT is the only `trained` row.
- Quick start: gains `python -m src.data.fetch <dataset>` as step 2, and the
  train step points at `configs/sift/v0.yaml` rather than
  `configs/wgan_gp_sift1m.yaml`.
- Keep the existing "What this project provides" and "Notes" sections, with
  "SIFT-like descriptor loader" reworded to "descriptor loader and
  preprocessing pipeline".

- [ ] **Step 2: Rewrite the framing sections of `PROJECT_DOCUMENTATION.md`**

Kept unchanged: "Metric definitions", "Visualization tools", "Run artifact
structure", "Workflow for a new user", "Reproducibility notes", "Tuning
guidance". Those are accurate and dataset-independent.

Changed:

- **Goal** — state the ANN-difficulty objective and name the four gate
  statistics (LID, relative contrast, hubness skew, IVF cell-balance Gini).
  Add the short version of why distributional fidelity does not imply ANN
  difficulty: the distributional metrics are dominated by the bulk of the
  distance distribution while ANN difficulty is set by its far-left tail, and
  no symmetric two-sample statistic constrains hubness at all. Point at
  `docs/superpowers/specs/2026-08-04-multi-dataset-ann-emulation-design.md`
  for the full argument.
- **Datasets** — new section, directly after Goal: the six-family table
  (family, dim, metric, structure, model family) and a pointer to
  `docs/datasets/<name>.md` for each. Describe `src/data/fetch.py`: one source
  registry, ann-benchmarks HDF5 in, `.npy` subsets out, atomic and
  single-flight.
- **Data contract and preprocessing** — `src/data/sift1m_dataset.py` becomes
  `src/data/dataset.py`; document `data.metric` as `l2` or `angular`,
  defaulting to `l2`, describing the corpus rather than the transform.
- **Model architecture** — note that `descriptor_dim` and the hidden dims come
  from config and the models are dimension-agnostic; the 128d numbers shown
  are SIFT's config, not a repo-wide constant.
- **Model variants** — retitle to make the ladder per-dataset. Keep the full
  SIFT table and the "Why v2 exists" and "`generator_type`" subsections
  verbatim; add that each family has its own ladder, numbered independently,
  and that variant numbers are comparable only within one dataset. Add
  `spherical` to the `generator_type` subsection as planned-not-yet-built,
  naming phase (b).
- **Evaluation stack** — promote the ANN-difficulty paragraph out of the
  visualization bullet list into its own subsection ahead of "Metrics", since
  it is the gate. Say that the pass criterion is a per-statistic band recorded
  in each dataset page, and that canonical N and k are locked per dataset.
  Note that `ann_difficulty.py` currently assumes L2 and that reading
  `data.metric` is phase (c).
- Every `configs/sift_gan_v*.yaml` path becomes `configs/sift/v*.yaml`.

- [ ] **Step 3: Rewrite `data/README.md`**

- Title and opening: the data contract, not "SIFT1M-like training".
- Accepted formats stay `.npy` `[N, D]` and `.fvecs`, with `D` per dataset
  rather than 128.
- Add: subsets written by `python -m src.data.fetch <dataset>` land at
  `data/<dataset>_<rows>.npy` — e.g. `data/deep_250k.npy`, `data/sift_1m.npy`
  — and are not tracked in git.
- `src/data/sift1m_dataset.py` becomes `src/data/dataset.py`.
- The "What the variant comparison expects" section keeps its content; update
  its config paths to `configs/sift/` and note the variant table now lives in
  `docs/datasets/sift.md`.

- [ ] **Step 4: Check every path the docs reference**

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path
missing = []
for page in ["README.md", "PROJECT_DOCUMENTATION.md", "data/README.md"]:
    for path in re.findall(r'`((?:configs|src|docs|tests)/[^`\s]+)`', Path(page).read_text()):
        if not Path(path).exists():
            missing.append(f"{page}: {path}")
print("\n".join(missing) or "all referenced paths exist")
PY
```

Expected: `all referenced paths exist`. A hit on `src/data/sift1m_dataset.py`
or `configs/sift_gan_v0.yaml` means a stale path survived the rewrite.

- [ ] **Step 5: Check no stale name survives anywhere outside the AI notes**

```bash
grep -rn "sift1m_dataset\|sift_gan_v\|src/deep\|src\.deep" \
  README.md PROJECT_DOCUMENTATION.md data/README.md docs/datasets/ src/ tests/ configs/
```

Expected: no output. `docs/superpowers/` is excluded on purpose — those are
dated working notes and are not rewritten when the code moves.

- [ ] **Step 6: Run the full suite one final time**

Run: `.venv/bin/python -m pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add README.md PROJECT_DOCUMENTATION.md data/README.md
git commit -m "docs: reframe around multi-dataset ANN-difficulty emulation

The deliverable is synthetic corpora that reproduce the nearest-neighbour
difficulty of six benchmark families, gated on LID, relative contrast,
hubness and IVF balance. Distributional metrics become diagnostics."
```

---

## Not in this plan

From the spec, deliberately left for later phases:

- **Phase (b)** — the `spherical` generator, its factory wiring and tests, and
  a first ladder on DEEP.
- **Phase (c)** — angular distance through `src/eval/ann_difficulty.py`, the
  `--dataset` argument on `compare_variants.py`, and gate bands recorded per
  dataset.
- Downstream index benchmarking, retuning the SIFT ladder, and any seventh
  dataset family — out of scope for the whole design.
