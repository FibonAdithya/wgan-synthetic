# scout_kv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `scout_kv` family: load captured Llama 4 Scout attention inputs `h`, train one WGAN-GP generator per global layer, and gate the generators on IVF/HNSW search difficulty against mixed-book real corpora.

**Architecture:** Code tasks 1-8 run locally, CPU only, under `make check`. They add an output-normalisation switch to the trainer, a dependency-free reader for the capture files, a `src/kv/` package (projections, corpora, search profiling), a gate module with its band file, a ported capture script, configs and docs. Box tasks B1-B6 are run by the orchestrator only, never by a subagent, because they need `ssh tig-gpu` with the sandbox disabled. They ship the code to the box, capture 16 more books, measure the real reference and its noise floor, train `v0`, and gate it.

**Tech Stack:** Python 3.12, numpy 2.5.1, torch 2.13.0 (box: 2.14.1+cu130 in `/workspace/kvenv`), faiss-cpu 1.15.1, pyyaml, pytest, ruff.

**Spec:** `docs/ai/specs/2026-10-18-kv-cache-scout-design.md`. Read it before any task. This plan argues from it.

## Global Constraints

- Family name `scout_kv`. Global layers `(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)`, 8 KV heads, 40 query heads, `head_dim` 128, query head `j` reads KV head `j // 5`.
- Corpus rule: position 0 (the sink) and the last 1,024 positions of a 32,768-token book are never indexed.
- Train books 0-11, reference books 12-19, by capture order. Nothing gated on is trained on.
- Gate statistics, exact names: `hnsw_reach_k10`, `hnsw_recall_k10_ef176`, `ivf_ratio_k128`, `ivf_recall_k128_np24`, `hnsw_recall_k10_ef176_id`.
- Indexes: IVFFlat inner product, `nlist` 256, `nprobe` grid `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]`. HNSW on the Xbox reduction, `M` 48, `efConstruction` 500, `efSearch` grid `[10, 16, 24, 32, 48, 64, 96, 128, 176, 256, 384, 512, 768, 1024, 1536, 2048]`. Recall target 0.9.
- Reference corpus counts with 8 books: keys 31,736, real-style queries 5,120 (1,024 `h` x 5), key-side queries 1,024.
- **Bands ship `null`.** No task sets a band value. Setting bands is a human decision.
- `faiss-cpu==1.15.1` is added to `requirements.txt` only in Task 0, and only after the human approves this plan.
- Preprocess for `scout_kv`: `center: true`, `whiten: false`, `l2_normalize: false`; model `output_normalize: false`.
- Default behaviour of every existing family must not change. `output_normalize` defaults to `true`.
- Never `git add -A` / `.`; stage explicit paths. Run `make check` (with `.venv/bin` on `PATH`) before every commit. The venv is the main repo's `.venv`.
- Every number written into a report or commit is labelled MEASURED or ESTIMATE.

## Review Focus

1. **Generator output silently normalised.** `generate.py` and `evaluate_distribution.py` each normalise in their own loops. A `scout_kv` user running them would get unit-norm `h` with no warning. Task 1 makes all five sites honour `output_normalize` and tests each one.
2. **A capture file that isn't bf16 `h`.** For example the weights files, which hold several tensors, or an F32 file. The reader must decode BF16, F16 and F32 correctly and fail loudly on any other dtype or a missing tensor name. Task 2 tests all of these against a fixture written by the real `safetensors` library.
3. **Books of unequal length, or a book count that doesn't divide the tail.** Mixed corpora assume equal `T` and `n_tail % n_books == 0`. Both must raise rather than silently produce unbalanced corpora. Task 4 tests both.
4. **Mismatched gate conditions.** Records measured with another `nlist`, another key count, or fewer than 96 corpora must fail the check rather than be compared. Task 6 tests a key-count mismatch, a missing corpus and a duplicate corpus.
5. **Capture re-run overwriting existing books or metadata.** Re-running capture with overlapping book indices must refuse, and it must never rewrite the 4-book `meta.json`. Task 7 tests the refusal logic as pure functions.

---

### Task 0: Pin faiss-cpu (human-approved)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the pin after `scipy`**

```
# Approximate nearest-neighbour indexes for the scout_kv gate (src/kv/search_profile.py):
# IVFFlat and HNSW, with the indexIVF_stats / hnsw_stats work counters the gate reads.
# The same version measured the profiling runs in docs/results/kv-cache-profile/.
faiss-cpu==1.15.1
```

- [ ] **Step 2: Install into the shared venv and smoke it**

Run: `.venv/bin/pip install faiss-cpu==1.15.1 && .venv/bin/python -c "import faiss; print(faiss.__version__)"`
Expected: `1.15.1`

- [ ] **Step 3: `make check`, commit**

```bash
PATH=$PWD/.venv/bin:$PATH make check
git add requirements.txt
git commit -m "deps: pin faiss-cpu 1.15.1 for the scout_kv search gate"
```

---

### Task 1: `output_normalize` switch in the trainer and samplers

**Files:**
- Modify: `src/train/train_wgan_gp.py` (`sample_generator` ~line 416; critic step ~675; generator step ~704; eval call ~796)
- Modify: `src/eval/compare_variants.py:422` (`generate_samples`)
- Modify: `src/eval/plot_descriptor_grid.py:207`
- Modify: `src/sample/generate.py:64-71`
- Modify: `src/eval/evaluate_distribution.py:55-70`
- Modify: `PROJECT_DOCUMENTATION.md` (`## Model architecture`: one paragraph on `output_normalize`)
- Test: `tests/test_output_normalize.py`

**Interfaces:**
- Produces: `sample_generator(generator, num_samples, latent_dim, batch_size, device, normalize: bool = True) -> np.ndarray`; `output_normalize(model_cfg: Mapping) -> bool` in `src/train/train_wgan_gp.py`, returning `bool(model_cfg.get("output_normalize", True))`. Task 6 calls both.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_output_normalize.py
"""output_normalize: false must reach every place generator output is drawn.

Each test catches one call site that still normalises unconditionally."""

import numpy as np
import torch
import yaml

from src.train.train_wgan_gp import output_normalize, sample_generator, train
from tests.test_train_smoke import make_config


class Scaled(torch.nn.Module):
    """Emits rows of norm 3, so any normalisation is visible."""

    def forward(self, z):
        return 3.0 * z / torch.linalg.vector_norm(z, dim=1, keepdim=True)


def test_sample_generator_default_still_normalises():
    """Catches the default flipping to unnormalised (would change every family)."""
    x = sample_generator(Scaled(), 8, 4, 4, torch.device("cpu"))
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-5)


def test_sample_generator_normalize_false_keeps_norms():
    """Catches sample_generator ignoring normalize=False."""
    x = sample_generator(Scaled(), 8, 4, 4, torch.device("cpu"), normalize=False)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 3.0, rtol=1e-5)


def test_output_normalize_reads_model_cfg_with_true_default():
    """Catches a default of False, or reading the key from the wrong section."""
    assert output_normalize({}) is True
    assert output_normalize({"output_normalize": False}) is False


def test_training_with_output_normalize_false_learns_scale(tmp_path):
    """Catches the two training-loop sites still normalising: real rows have
    norm ~4 (unnormalised Gaussian in 16 dims), so a normalised generator's
    eval samples would sit at exactly 1.0."""
    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"]["l2_normalize"] = False
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 2
    ckpt, _ = train(cfg)
    run_cfg = yaml.safe_load((ckpt.parent / "run_config.yaml").read_text())
    assert run_cfg["model"]["output_normalize"] is False
    from src.eval.evaluate_distribution import load_generator

    gen = load_generator(run_cfg, ckpt, torch.device("cpu"))
    x = sample_generator(gen, 64, 8, 64, torch.device("cpu"), normalize=False)
    assert not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-3)
```

```python
def _tiny_unnormalised_run(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"]["l2_normalize"] = False
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 2
    ckpt, _ = train(cfg)
    return ckpt, ckpt.parent / "run_config.yaml"


def test_generate_cli_honours_output_normalize(tmp_path, monkeypatch):
    """Catches generate.py's own hard-coded division surviving."""
    import sys

    from src.sample import generate

    ckpt, run_cfg = _tiny_unnormalised_run(tmp_path)
    out = tmp_path / "s.npy"
    monkeypatch.setattr(sys, "argv", ["generate", "--checkpoint", str(ckpt), "--config", str(run_cfg),
                                      "--num-samples", "32", "--output-path", str(out)])
    generate.main()
    assert not np.allclose(np.linalg.norm(np.load(out), axis=1), 1.0, atol=1e-3)


def test_evaluate_distribution_honours_output_normalize(tmp_path):
    """Catches sample_fake's hard-coded division surviving."""
    from src.eval.evaluate_distribution import load_generator, sample_fake

    ckpt, run_cfg = _tiny_unnormalised_run(tmp_path)
    gen = load_generator(yaml.safe_load(run_cfg.read_text()), ckpt, torch.device("cpu"))
    x = sample_fake(gen, 8, 32, 32, torch.device("cpu"), normalize=False)
    assert not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-3)
```

`sample_fake` gains `normalize: bool = True`. `evaluate_distribution.main` and `generate.main` pass `output_normalize(config["model"])`. If `generate.main` writes extra files (timing JSON) next to `--output-path`, the test's `tmp_path` absorbs them.

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_output_normalize.py -v`
Expected: FAIL. `ImportError: cannot import name 'output_normalize'`.

- [ ] **Step 3: Implement**

In `src/train/train_wgan_gp.py`:

```python
def output_normalize(model_cfg: Mapping[str, Any]) -> bool:
    """Whether generator output is L2-normalised before the critic and in samples.

    True for every family built so far, which all search on the unit sphere.
    scout_kv sets it False: inner-product search reads vector norms, so a
    unit-norm h would erase part of what the gate measures.
    """
    return bool(model_cfg.get("output_normalize", True))
```

Inside `sample_generator`, add `normalize: bool = True` and write `x = generator(z); x = normalize_l2(x) if normalize else x`. Inside `train`, set `norm_out = output_normalize(model_cfg)` once after `model_cfg = config["model"]`. Replace both `normalize_l2(generator(z)...)` sites with `fake = generator(z).detach()` (critic) or `fake = generator(z)` (generator), each followed by `if norm_out: fake = normalize_l2(fake)`. Pass `normalize=norm_out` at the eval call (~796).

In `compare_variants.generate_samples` and `plot_descriptor_grid`, pass `normalize=output_normalize(config["model"])`. In `generate.py`, wrap the existing division in `if output_normalize(model_cfg):`. In `evaluate_distribution.sample_fake`, add `normalize: bool = True` and guard the division with it; its `main` passes `output_normalize(config["model"])`. Import `output_normalize` from `src.train.train_wgan_gp`. `evaluate_distribution` already imports from that module, so check for an import cycle by running the tests.

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_output_normalize.py -v && PATH=$PWD/.venv/bin:$PATH make check`
Expected: all pass. The existing count rises by exactly the new tests.

- [ ] **Step 5: Mutation check** (break, confirm the named test fails, restore):

| mutation | test that must fail |
|---|---|
| default in `output_normalize` -> `False` | `test_output_normalize_reads_model_cfg_with_true_default` |
| `sample_generator` normalises regardless of `normalize` | `test_sample_generator_normalize_false_keeps_norms` |
| eval call (~796) drops `normalize=norm_out` | `test_training_with_output_normalize_false_learns_scale` is not enough on its own here (it samples after training). Also assert, in that test, that `meta["metrics"][-1]` has `fake_norm_mean` (or the existing norm statistic `collapse_stats` logs) not equal to 1.0. Read `collapse_stats` for the exact key before writing the assertion. |
| `generate.py` division unguarded | `test_generate_cli_honours_output_normalize` |
| `sample_fake` division unguarded | `test_evaluate_distribution_honours_output_normalize` |

The two training-step sites (~675, ~704) aren't separately observable through samples. Their change is checked by review: both must read `if norm_out:`.

- [ ] **Step 6: Document and commit**

Add to `PROJECT_DOCUMENTATION.md#model-architecture`: "`model.output_normalize` (default `true`): generator output is L2-normalised before the critic, in in-training samples and in every sampler. `scout_kv` sets it `false`, because inner-product search reads norms."

```bash
git add src/train/train_wgan_gp.py src/eval/compare_variants.py src/eval/plot_descriptor_grid.py src/sample/generate.py src/eval/evaluate_distribution.py PROJECT_DOCUMENTATION.md tests/test_output_normalize.py
git commit -m "train: output_normalize switch, default on, honoured by every sampler"
```

---

### Task 2: Capture-file reader and `format: kv_capture`

**Files:**
- Create: `src/data/kv_capture.py`
- Create: `tests/fixtures/kv/tiny_bf16.safetensors`, `tests/fixtures/kv/tiny_multi.safetensors`. Written on the box by the orchestrator in Step 0; under 1 KB each.
- Modify: `src/data/dataset.py` (`build_training_data` signature and the load branch)
- Modify: `src/train/train_wgan_gp.py:468-477` (pass the three `kv_*` keys)
- Modify: `data/README.md` (one section on `kv_capture`)
- Test: `tests/test_kv_capture.py`

**Interfaces:**
- Produces:
  - `read_safetensor(path: Path, name: str) -> np.ndarray` (float32; BF16/F16/F32 only)
  - `GLOBAL_LAYERS: tuple[int, ...]`
  - `h_path(capture_dir: Path, book: int, layer: int) -> Path`, i.e. `capture_dir / f"doc{book}" / f"h_L{layer:02d}.safetensors"`
  - `load_kv_capture(capture_dir: Path, layer: int, books: Sequence[int], n_sink: int = 1) -> np.ndarray`, shape `(sum(T_b - n_sink), 5120)`
  - `build_training_data(..., kv_layer: int | None = None, kv_books: Sequence[int] | None = None, kv_n_sink: int = 1)`
- Used by: Tasks 3, 6, B4.

- [ ] **Step 0 (orchestrator, on the box): write the fixtures with the real library**

```bash
ssh tig-gpu '/workspace/kvenv/bin/python - <<EOF
import torch
from safetensors.torch import save_file
h = torch.tensor([[1.5, -2.0, 3.140625], [0.0, 0.0009765625, -256.0]], dtype=torch.bfloat16)
save_file({"h": h}, "/workspace/tiny_bf16.safetensors")
save_file({"k_proj": torch.arange(6, dtype=torch.float32).reshape(2, 3),
           "q_proj": torch.tensor([[0.5, -1.0]], dtype=torch.float16)},
          "/workspace/tiny_multi.safetensors")
EOF
sha256sum /workspace/tiny_*.safetensors; ls -l /workspace/tiny_*.safetensors'
ssh tig-gpu 'tar czf - -C /workspace tiny_bf16.safetensors tiny_multi.safetensors' | tar xzf - -C tests/fixtures/kv
sha256sum tests/fixtures/kv/*.safetensors   # must match the box
```

Every value is exactly representable in its dtype, so the expected arrays are exact.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kv_capture.py
from pathlib import Path

import numpy as np
import pytest

from src.data.kv_capture import h_path, load_kv_capture, read_safetensor

FIX = Path(__file__).parent / "fixtures" / "kv"


def test_reads_bf16_written_by_the_real_library():
    """Catches a wrong bf16 decode (byte order, shift, or treating it as f16)."""
    x = read_safetensor(FIX / "tiny_bf16.safetensors", "h")
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, [[1.5, -2.0, 3.140625], [0.0, 0.0009765625, -256.0]])


def test_reads_f32_and_f16_by_name_from_a_multi_tensor_file():
    """Catches using the first tensor's offsets for every name, or no f16 path."""
    np.testing.assert_array_equal(read_safetensor(FIX / "tiny_multi.safetensors", "k_proj"),
                                  np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(read_safetensor(FIX / "tiny_multi.safetensors", "q_proj"),
                                  [[0.5, -1.0]])


def test_missing_tensor_name_raises():
    """Catches silently returning another tensor."""
    with pytest.raises(KeyError, match="v_proj"):
        read_safetensor(FIX / "tiny_multi.safetensors", "v_proj")


def test_unsupported_dtype_raises(tmp_path):
    """Catches decoding I64 (or anything else) as if it were float."""
    import json, struct

    header = json.dumps({"h": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}).encode()
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(ValueError, match="I64"):
        read_safetensor(p, "h")


def _write_book(root, book, layer, arr):
    """Writes a BF16 file the way the box does, via the reader's own inverse.
    Only used for layout tests; decoding is pinned by the fixtures above."""
    from src.data.kv_capture import write_bf16_safetensor

    p = h_path(root, book, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_bf16_safetensor(p, "h", arr)


def test_load_kv_capture_drops_the_sink_per_book_and_keeps_book_order(tmp_path):
    """Catches dropping the sink only once, or concatenating books out of order."""
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = 100 + np.arange(12, dtype=np.float32).reshape(4, 3)
    _write_book(tmp_path, 0, 3, a)
    _write_book(tmp_path, 5, 3, b)
    x = load_kv_capture(tmp_path, 3, [5, 0], n_sink=1)
    np.testing.assert_array_equal(x, np.vstack([b[1:], a[1:]]))


def test_build_training_data_kv_capture_branch(tmp_path):
    """Catches the trainer path ignoring kv_layer / kv_books."""
    from src.data.dataset import PreprocessConfig, build_training_data

    rng = np.random.default_rng(0)
    for bk in (0, 1):
        _write_book(tmp_path, bk, 7, rng.normal(size=(50, 4)).astype(np.float32))
    xt, xh, _ = build_training_data(str(tmp_path), "kv_capture", 4, 0.1,
                                    PreprocessConfig(l2_normalize=False), 0,
                                    kv_layer=7, kv_books=[0, 1], kv_n_sink=1)
    assert xt.shape[0] + xh.shape[0] == 98
```

```python
def test_write_then_read_round_trips_bf16_exact_values(tmp_path):
    """The reader is pinned by the real-library fixture, so a round trip here
    checks the writer: catches truncation instead of round-to-nearest-even."""
    from src.data.kv_capture import write_bf16_safetensor

    exact = np.array([[1.5, -2.0, 3.140625]], np.float32)
    write_bf16_safetensor(tmp_path / "a.safetensors", "h", exact)
    np.testing.assert_array_equal(read_safetensor(tmp_path / "a.safetensors", "h"), exact)
    # 1 + 2**-8 is exactly halfway between bf16 neighbours 1.0 and 1 + 2**-7: ties to even -> 1.0;
    # 1 + 3 * 2**-8 is halfway between 1 + 2**-7 and 1 + 2**-6: ties to even -> 1 + 2**-6.
    tie = np.array([[1 + 2**-8, 1 + 3 * 2**-8]], np.float32)
    write_bf16_safetensor(tmp_path / "b.safetensors", "h", tie)
    np.testing.assert_array_equal(read_safetensor(tmp_path / "b.safetensors", "h"), [[1.0, 1 + 2**-6]])
```

`write_bf16_safetensor(path, name, arr)` rounds float32 to bf16 with round-to-nearest-even, then writes the header and bytes. It is used by the layout tests and by Tasks 4 and 6.

- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError: src.data.kv_capture`)

- [ ] **Step 3: Implement `src/data/kv_capture.py`**

```python
"""Reader for the scout_kv capture files.

The capture writes safetensors: an 8-byte little-endian header length, a JSON
header mapping each tensor name to dtype, shape and byte offsets, then the raw
bytes. That layout is fixed by the format, so it is read here directly rather
than through the safetensors package, which would be a new pinned dependency
for 30 lines of parsing. Only the dtypes the capture produces are accepted.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

GLOBAL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}


def h_path(capture_dir: Path, book: int, layer: int) -> Path:
    return Path(capture_dir) / f"doc{book}" / f"h_L{layer:02d}.safetensors"


def _header(f) -> tuple[dict, int]:
    (n,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(n)), 8 + n


def read_safetensor(path: Path, name: str) -> np.ndarray:
    with open(path, "rb") as f:
        header, base = _header(f)
        if name not in header:
            raise KeyError(f"{name} not in {path}: has {sorted(k for k in header if k != '__metadata__')}")
        meta = header[name]
        dtype, shape, (start, end) = meta["dtype"], meta["shape"], meta["data_offsets"]
        if dtype not in _DTYPES:
            raise ValueError(f"{path}:{name} has dtype {dtype}; only {sorted(_DTYPES)} are supported")
        f.seek(base + start)
        raw = f.read(end - start)
    if dtype == "F32":
        out = np.frombuffer(raw, "<f4")
    elif dtype == "F16":
        out = np.frombuffer(raw, "<f2").astype(np.float32)
    else:
        out = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    return out.reshape(shape).astype(np.float32, copy=False)


def write_bf16_safetensor(path: Path, name: str, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = ((a + 0x7FFF + ((a >> 16) & 1)) >> 16).astype("<u2")
    header = json.dumps({name: {"dtype": "BF16", "shape": list(arr.shape),
                                "data_offsets": [0, rounded.nbytes]}}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + rounded.tobytes())


def load_kv_capture(capture_dir: Path, layer: int, books: Sequence[int], n_sink: int = 1) -> np.ndarray:
    if layer not in GLOBAL_LAYERS:
        raise ValueError(f"layer {layer} is not a global layer {GLOBAL_LAYERS}")
    return np.vstack([read_safetensor(h_path(capture_dir, b, layer), "h")[n_sink:] for b in books])
```

In `build_training_data`, add the three keyword parameters. Before `load_descriptors`, branch: `if file_format == "kv_capture": x = load_kv_capture(Path(descriptor_path), kv_layer, kv_books, kv_n_sink)`, raising `ValueError` if `kv_layer` or `kv_books` is None. In the trainer, pass `kv_layer=data_cfg.get("kv_layer")`, `kv_books=data_cfg.get("kv_books")`, `kv_n_sink=int(data_cfg.get("kv_n_sink", 1))`.

- [ ] **Step 4: Run tests, `make check`**

- [ ] **Step 5: Mutation check.** Change `<< 16` to `<< 8`; the fixture test must fail. Change `[n_sink:]` to `[:]`; the sink test must fail. Restore both.

- [ ] **Step 6: Document in `data/README.md`** (a `## kv_capture` section: directory layout, the three `data.kv_*` keys, bf16 to float32, the sink dropped per book). **Commit** all files listed above.

---

### Task 3: Projections — `src/kv/transforms.py`

**Files:**
- Create: `src/kv/__init__.py` (empty), `src/kv/transforms.py`
- Test: `tests/test_kv_transforms.py`

**Interfaces:**
- Consumes: `read_safetensor` (Task 2).
- Produces:
  - `HEAD_DIM = 128`
  - `load_layer_weights(capture_dir: Path, layer: int) -> dict[str, torch.Tensor]`, with keys `k_proj (1024, 5120)`, `q_proj (5120, 5120)` and, when present, `v_proj`
  - `n_rep(wq, wk) -> int`
  - `head_keys(h: Tensor, wk: Tensor, g: int) -> Tensor (n, 128)`
  - `head_queries(h: Tensor, wq: Tensor, g: int, n_rep: int) -> Tensor (n * n_rep, 128)`, ordered so rows `i*n_rep ... i*n_rep + n_rep - 1` come from `h[i]`
  - `pos_scale(pos: Tensor, attn_scale: float, floor_scale: float) -> Tensor`
  - `xbox(base: np.ndarray) -> np.ndarray` (adds 1 column)
  - `pad0(q: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Failing tests**

```python
# tests/test_kv_transforms.py
import numpy as np
import torch

from src.kv.transforms import head_keys, head_queries, n_rep, pad0, pos_scale, xbox


def _w(seed, rows, cols):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=(rows, cols)), dtype=torch.float32)


def test_head_keys_takes_rows_g_times_128():
    """Catches slicing the wrong 128-row block of W_k."""
    h, wk = _w(0, 3, 16), _w(1, 2 * 128, 16)
    np.testing.assert_allclose(head_keys(h, wk, 1), h @ wk[128:256].T, rtol=1e-6)


def test_head_queries_gqa_mapping_and_row_order():
    """Catches query head j reading KV head other than j // n_rep, or the 5
    queries of one h not being consecutive rows."""
    h, wk, wq = _w(0, 2, 16), _w(1, 2 * 128, 16), _w(2, 2 * 5 * 128, 16)
    r = n_rep(wq, wk)
    assert r == 5
    q = head_queries(h, wq, 1, r)
    assert q.shape == (2 * 5, 128)
    for i in range(2):
        for j in range(5):
            jj = 5 * 1 + j
            np.testing.assert_allclose(q[i * 5 + j], h[i] @ wq[jj * 128:(jj + 1) * 128].T, rtol=1e-5)


def test_pos_scale_steps_at_floor_scale():
    """Catches an off-by-one in floor((pos+1)/8192): pos 8190 is unscaled, 8191 is the first scaled."""
    s = pos_scale(torch.tensor([0, 8190, 8191, 16383]), 0.1, 8192.0)
    np.testing.assert_allclose(s, [1.0, 1.0, 1 + 0.1 * np.log(2), 1 + 0.1 * np.log(3)], rtol=1e-6)


def test_xbox_preserves_mips_ranking():
    """Catches a wrong augmentation (e.g. norm instead of sqrt(M^2 - |x|^2))."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 8)).astype(np.float32) * rng.uniform(0.2, 3, (200, 1)).astype(np.float32)
    q = rng.normal(size=(20, 8)).astype(np.float32)
    mips = np.argsort(-(q @ base.T), 1)[:, :10]
    xb, qx = xbox(base), pad0(q)
    l2 = np.argsort(((qx[:, None, :] - xb[None]) ** 2).sum(-1), 1)[:, :10]
    np.testing.assert_array_equal(mips, l2)
```

- [ ] **Step 2: Run, expect `ModuleNotFoundError`.**

- [ ] **Step 3: Implement**

```python
"""h -> per-head keys and queries for Llama 4 Scout's global (no-RoPE) layers.

k = W_k h and q = s(pos) W_q h reproduce the model to bf16 precision
(docs/results/kv-cache-profile/). s(pos) does not change any query's ranking
in flat, IVF or Xbox-HNSW search, so search code may ignore it; attention
error does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.kv_capture import read_safetensor

HEAD_DIM = 128


def load_layer_weights(capture_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    path = Path(capture_dir) / "weights" / f"layer{layer:02d}.safetensors"
    out = {n: torch.from_numpy(read_safetensor(path, n)) for n in ("k_proj", "q_proj")}
    try:
        out["v_proj"] = torch.from_numpy(read_safetensor(path, "v_proj"))
    except KeyError:
        pass
    return out


def n_rep(wq: torch.Tensor, wk: torch.Tensor) -> int:
    return wq.shape[0] // wk.shape[0]


def head_keys(h, wk, g):
    return h @ wk[g * HEAD_DIM:(g + 1) * HEAD_DIM].T


def head_queries(h, wq, g, n_rep):
    w = wq[g * n_rep * HEAD_DIM:(g + 1) * n_rep * HEAD_DIM]
    return (h @ w.T).reshape(-1, HEAD_DIM)


def pos_scale(pos, attn_scale, floor_scale):
    p = pos.to(torch.float64)
    return (1.0 + attn_scale * torch.log1p(torch.floor((p + 1.0) / floor_scale))).to(torch.float32)


def xbox(base: np.ndarray) -> np.ndarray:
    n2 = (base.astype(np.float64) ** 2).sum(1)
    extra = np.sqrt(np.maximum(n2.max() - n2, 0.0)).astype(np.float32)
    return np.hstack([base, extra[:, None]])


def pad0(q: np.ndarray) -> np.ndarray:
    return np.hstack([q, np.zeros((len(q), 1), np.float32)])
```

- [ ] **Step 4: Tests pass, `make check`.**
- [ ] **Step 5: Mutation check.** Change `(pos + 1.0)` to `pos`; the pos-scale test must fail. In `head_queries` use `g * HEAD_DIM` as the start; the GQA test must fail. Restore both.
- [ ] **Step 6: Commit** `src/kv/__init__.py src/kv/transforms.py tests/test_kv_transforms.py`.

---

### Task 4: Corpora — `src/kv/corpus.py`

**Files:**
- Create: `src/kv/corpus.py`
- Test: `tests/test_kv_corpus.py`

**Interfaces:**
- Consumes: `head_keys`, `head_queries`, `n_rep` (Task 3).
- Produces:
  - `@dataclass Corpus: keys: np.ndarray; q_ood: np.ndarray; q_id: np.ndarray; key_book: np.ndarray; q_book: np.ndarray` (float32 arrays; int book labels, -1 for synthetic)
  - `real_sizes(T: int, n_books: int, n_sink: int, n_tail: int) -> tuple[int, int, int]`, returning `(n_keys, n_query_h, n_id)`
  - `mixed_real_corpus(h_books: Sequence[torch.Tensor], wk, wq, g, *, n_sink: int, n_tail: int, seed: int) -> Corpus`
  - `synthetic_corpus(h_keys: Tensor, h_query: Tensor, h_id: Tensor, wk, wq, g) -> Corpus`

- [ ] **Step 1: Failing tests**

```python
# tests/test_kv_corpus.py
import numpy as np
import pytest
import torch

from src.kv.corpus import mixed_real_corpus, real_sizes, synthetic_corpus

D, T = 16, 64


def _setup(n_books, seed=0):
    rng = np.random.default_rng(seed)
    # Encode (book, position) in h so selected rows are identifiable after projection:
    # with W_k = identity on the first 2 dims, key[:, :2] == (book, position).
    hs = []
    for b in range(n_books):
        h = np.zeros((T, D), np.float32)
        h[:, 0], h[:, 1] = b, np.arange(T)
        h[:, 2:] = rng.normal(size=(T, D - 2))
        hs.append(torch.from_numpy(h))
    wk = torch.zeros(2 * 128, D); wk[0, 0] = wk[1, 1] = 1.0
    wq = torch.as_tensor(rng.normal(size=(2 * 5 * 128, D)), dtype=torch.float32)
    return hs, wk, wq


def test_real_sizes_eight_books_matches_the_spec():
    """Catches the 4-book count (31,740) leaking into the 8-book reference."""
    assert real_sizes(32768, 8, 1, 1024) == (31736, 1024, 1024)


def test_mixed_corpus_never_indexes_sink_or_tail_and_balances_books():
    """Catches indexing position 0 or the tail, or unequal per-book shares."""
    hs, wk, wq = _setup(4)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=3)
    book, pos = c.keys[:, 0], c.keys[:, 1]
    assert pos.min() >= 1 and pos.max() < T - 16
    assert np.bincount(book.astype(int)).tolist() == [(T - 16 - 1) // 4] * 4
    assert (c.key_book == book).all()


def test_single_book_indexes_exactly_positions_1_to_before_tail():
    """With one book every indexable position is drawn, so the set is exact.
    Catches indexing position 0 or any tail position, deterministically."""
    hs, wk, wq = _setup(1)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)
    assert sorted(c.keys[:, 1].astype(int).tolist()) == list(range(1, T - 16))


def test_mixed_corpus_queries_come_from_tail_five_per_h():
    """Catches queries from indexed positions, or a wrong query count."""
    hs, wk, wq = _setup(4)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=3)
    assert c.q_ood.shape == (16 * 5, 128) and c.q_id.shape == (16, 128)
    assert (c.q_id[:, 1] >= T - 16).all()
    assert np.bincount(c.q_book).tolist() == [4 * 5] * 4


def test_mixed_corpus_is_seed_reproducible_and_seed_sensitive():
    """Catches unseeded sampling (flaky gate) or a seed that is ignored."""
    hs, wk, wq = _setup(4)
    a = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=1)
    b = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=1)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=2)
    np.testing.assert_array_equal(a.keys, b.keys)
    assert not np.array_equal(a.keys, c.keys)


def test_unequal_book_lengths_raise():
    hs, wk, wq = _setup(2)
    hs[1] = hs[1][:-1]
    with pytest.raises(ValueError, match="length"):
        mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)


def test_tail_not_divisible_by_books_raises():
    hs, wk, wq = _setup(3)
    with pytest.raises(ValueError, match="divisible"):
        mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)


def test_synthetic_corpus_shapes():
    """Catches keys and key-side queries drawn from the same h (they must be independent)."""
    hs, wk, wq = _setup(1)
    h = hs[0]
    c = synthetic_corpus(h[:40], h[40:48], h[48:56], wk, wq, 0)
    assert c.keys.shape == (40, 128) and c.q_ood.shape == (40, 128) and c.q_id.shape == (8, 128)
    np.testing.assert_array_equal(c.q_id[:, 1], np.arange(48, 56))
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement**

```python
"""Real (mixed-book) and synthetic KV corpora for one (layer, KV head)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.kv.transforms import head_keys, head_queries, n_rep


@dataclass
class Corpus:
    keys: np.ndarray
    q_ood: np.ndarray
    q_id: np.ndarray
    key_book: np.ndarray
    q_book: np.ndarray


def real_sizes(T: int, n_books: int, n_sink: int, n_tail: int) -> tuple[int, int, int]:
    return n_books * ((T - n_tail - n_sink) // n_books), n_tail, n_tail


def _np(t):
    return t.detach().float().cpu().numpy()


def mixed_real_corpus(h_books: Sequence[torch.Tensor], wk, wq, g, *, n_sink, n_tail, seed) -> Corpus:
    B = len(h_books)
    T = h_books[0].shape[0]
    if any(h.shape[0] != T for h in h_books):
        raise ValueError(f"books differ in length: {[h.shape[0] for h in h_books]}")
    if n_tail % B:
        raise ValueError(f"n_tail {n_tail} is not divisible by {B} books")
    Tk = T - n_tail
    per, tail_per = (Tk - n_sink) // B, n_tail // B
    r = n_rep(wq, wk)
    rng = np.random.default_rng(seed)
    ks, qs, ids, kb, qb = [], [], [], [], []
    for b, h in enumerate(h_books):
        kpos = torch.as_tensor(np.sort(rng.choice(np.arange(n_sink, Tk), per, replace=False)))
        tpos = torch.as_tensor(np.sort(rng.choice(np.arange(Tk, T), tail_per, replace=False)))
        hk, ht = h[kpos.to(h.device)], h[tpos.to(h.device)]
        ks.append(_np(head_keys(hk, wk, g)))
        qs.append(_np(head_queries(ht, wq, g, r)))
        ids.append(_np(head_keys(ht, wk, g)))
        kb.append(np.full(per, b))
        qb.append(np.full(tail_per * r, b))
    return Corpus(np.vstack(ks), np.vstack(qs), np.vstack(ids), np.concatenate(kb), np.concatenate(qb))


def synthetic_corpus(h_keys, h_query, h_id, wk, wq, g) -> Corpus:
    r = n_rep(wq, wk)
    return Corpus(_np(head_keys(h_keys, wk, g)), _np(head_queries(h_query, wq, g, r)),
                  _np(head_keys(h_id, wk, g)), np.full(len(h_keys), -1), np.full(len(h_query) * r, -1))
```

The real-corpus queries are unscaled here, because `s(pos)` doesn't change rankings. The attention-error diagnostic, which needs `s(pos)`, is not part of this plan's gate path. It stays in the spike script until a later task.

- [ ] **Step 4: Tests pass, `make check`.**
- [ ] **Step 5: Mutation check.** Change `np.arange(n_sink, Tk)` to `np.arange(0, Tk)`; `test_single_book_indexes_exactly_positions_1_to_before_tail` must fail. Change `seed` to a constant `0`; the seed-sensitivity test must fail. Restore both.
- [ ] **Step 6: Commit.**

---

### Task 5: Search profiling — `src/kv/search_profile.py`

**Files:**
- Create: `src/kv/search_profile.py`
- Test: `tests/test_kv_search_profile.py`

**Interfaces:**
- Consumes: `xbox`, `pad0` (Task 3).
- Produces:
  - `NPROBES`, `EFS`, `KS = (10, 128)`, `OP_NPROBE = 24`, `OP_EF = 176`
  - `recall(found, gt) -> float`
  - `work_at_target(curve: list[tuple[int, float, float]], target: float) -> tuple[float | None, bool]`
  - `profile_corpus(keys, q_ood, q_id, *, nlist=256, hnsw_m=48, ef_construction=500, seed=0, target=0.9) -> dict`, returning keys `plain_{ood,id}_k{10,128}`, each `{"ivf_scan_frac", "ivf_reached", "hnsw_ndis", "hnsw_reached", "ivf_recall_op", "hnsw_recall_op", "ivf_curve", "hnsw_curve"}`. Same field meanings as `docs/results/kv-cache-profile/spike/profile_v2.py`.
  - `conditions(nlist, hnsw_m, ef_construction, target) -> dict`, the locked index conditions recorded with each result

- [ ] **Step 1: Failing tests**

```python
# tests/test_kv_search_profile.py
import numpy as np
import pytest

import faiss  # pinned in requirements.txt (Task 0); a missing install must fail CI, not skip

from src.kv.search_profile import profile_corpus, recall, work_at_target


def test_recall_counts_overlap_and_ignores_minus_one():
    """Catches counting -1 padding as a hit, or per-row instead of pooled recall."""
    gt = np.array([[0, 1], [2, 3]])
    found = np.array([[1, -1], [2, 3]])
    assert recall(found, gt) == 0.75


def test_work_at_target_interpolates_between_bracketing_points():
    """Catches returning the upper point's work instead of interpolating."""
    curve = [(1, 0.5, 10.0), (2, 0.8, 20.0), (4, 1.0, 40.0)]
    w, ok = work_at_target(curve, 0.9)
    assert ok and w == pytest.approx(30.0)


def test_work_at_target_unreached():
    assert work_at_target([(1, 0.5, 1.0), (2, 0.8, 2.0)], 0.9) == (None, False)


def test_work_at_target_first_point_already_reaches():
    """Catches interpolating from a phantom (0, 0) point."""
    assert work_at_target([(1, 0.95, 7.0)], 0.9) == (7.0, True)


def _profile(keys, q_ood, q_id):
    faiss.omp_set_num_threads(1)  # deterministic HNSW build
    return profile_corpus(keys, q_ood, q_id, nlist=16, hnsw_m=16, ef_construction=64, seed=0)


def test_key_side_queries_are_easy_and_fields_are_complete():
    """Catches a profile missing a field or an ID baseline that is not easy."""
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(3000, 16)).astype(np.float32)
    q_id = rng.normal(size=(64, 16)).astype(np.float32)
    p = _profile(keys, q_id.repeat(5, 0), q_id)
    for name in ("plain_ood_k10", "plain_id_k10", "plain_ood_k128", "plain_id_k128"):
        assert set(p[name]) >= {"ivf_scan_frac", "ivf_reached", "hnsw_ndis", "hnsw_reached",
                                "ivf_recall_op", "hnsw_recall_op", "ivf_curve", "hnsw_curve"}
    assert p["plain_id_k10"]["hnsw_reached"]
    assert p["plain_id_k10"]["hnsw_curve"][-1][1] == pytest.approx(1.0, abs=1e-3)


def test_ood_queries_cost_more_hnsw_work_than_id():
    """The property the gate rests on: queries off the key distribution (here a
    shifted, scaled cloud whose MIPS answers are large-norm outliers) need more
    HNSW distance computations than key-like queries. Catches swapping the
    ood/id workloads or reading ndis from the wrong counter."""
    rng = np.random.default_rng(1)
    keys = rng.normal(size=(4000, 16)).astype(np.float32)
    q_id = rng.normal(size=(64, 16)).astype(np.float32)
    q_ood = (rng.normal(size=(320, 16)) * 0.3 + 2.5).astype(np.float32)
    p = _profile(keys, q_ood, q_id)
    assert p["plain_ood_k10"]["hnsw_recall_op"] <= p["plain_id_k10"]["hnsw_recall_op"]
    ho, hi = p["plain_ood_k10"]["hnsw_ndis"], p["plain_id_k10"]["hnsw_ndis"]
    assert ho is None or hi is None or ho > hi
```

Before Step 3, the implementer runs the last test against the spike's own functions (import `docs/results/kv-cache-profile/spike/profile_v2.py` through `importlib`) to confirm the property holds at these sizes. If it doesn't, they change the planted distribution, not the assertion, and record the change in the test docstring.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** by porting `recall`, `work_at_target`, `sweep_ivf`, `sweep_hnsw` and `build` verbatim from `docs/results/kv-cache-profile/spike/profile_v2.py` (lines 26-125), with the module constants copied exactly from Global Constraints. Then add:

```python
def conditions(nlist, hnsw_m, ef_construction, target):
    return {"nlist": nlist, "hnsw_m": hnsw_m, "ef_construction": ef_construction,
            "target": target, "nprobes": list(NPROBES), "efs": list(EFS)}


def profile_corpus(keys, q_ood, q_id, *, nlist=256, hnsw_m=48, ef_construction=500, seed=0, target=0.9):
    keys = np.ascontiguousarray(keys, np.float32)
    ivf, hnsw = build(keys, nlist, hnsw_m, ef_construction, seed)
    out = {}
    for qname, q in (("ood", q_ood), ("id", q_id)):
        q = np.ascontiguousarray(q, np.float32)
        sims = q @ keys.T
        qx = pad0(q)
        for k in KS:
            gt = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            ic, _ = sweep_ivf(ivf, q, gt, len(keys), k)
            hc, _ = sweep_hnsw(hnsw, qx, gt, k)
            iw, ir = work_at_target(ic, target)
            hw, hr = work_at_target(hc, target)
            out[f"plain_{qname}_k{k}"] = {
                "ivf_scan_frac": iw, "ivf_reached": ir, "hnsw_ndis": hw, "hnsw_reached": hr,
                "ivf_recall_op": dict((p, r) for p, r, _ in ic)[OP_NPROBE],
                "hnsw_recall_op": dict((p, r) for p, r, _ in hc)[OP_EF],
                "ivf_curve": ic, "hnsw_curve": hc}
    return out
```

`recall` compares sets, so the unsorted `argpartition` ground truth is fine. With 5,120 x 31,736 float32 similarities (650 MB) per corpus, compute `sims` in query blocks of 1,024 if memory is a concern. The test sizes don't need that.

- [ ] **Step 4: Tests pass, `make check`.**
- [ ] **Step 5: Mutation check.** In `work_at_target`, return `w` without interpolating; the interpolation test must fail. In `profile_corpus`, swap `q_ood` and `q_id` in the loop; the work test must fail. Restore both.
- [ ] **Step 6: Commit.**

---

### Task 6: Gate — `src/eval/kv_gate.py`, `gates/scout_kv.yaml`

**Files:**
- Modify: `src/eval/check_gate.py`. Extract the per-band validation loop body (lines ~113-143) into `validate_band(name: str, band: Any, path: Path) -> None`, behaviour unchanged.
- Create: `src/eval/kv_gate.py`, `gates/scout_kv.yaml`
- Test: `tests/test_kv_gate.py`

**Interfaces:**
- Consumes: `load_kv_capture`, `h_path`, `GLOBAL_LAYERS` (Task 2); `load_layer_weights` (Task 3); `mixed_real_corpus`, `synthetic_corpus`, `real_sizes` (Task 4); `profile_corpus`, `conditions` (Task 5); `sample_generator`, `output_normalize` (Task 1); `load_generator` (`src/eval/evaluate_distribution.py`); `load_preprocess_state`, `invert_samples` (`src/eval/compare_variants.py`); `validate_band`, `check_statistic`, `band_bounds`, `EXIT_*` (`check_gate`).
- Produces:
  - `STATISTICS` (the five names); `GATES_FILE = check_gate.GATES_DIR / "scout_kv.yaml"`; `GLOBAL_LAYERS` (re-exported from `src.data.kv_capture`)
  - `pooled_statistics(records: list[dict]) -> dict[str, float | None]`
  - `load_kv_gate(path: Path) -> dict`
  - `evaluate(records, gate) -> dict` with `verdict` in `{"pass", "fail", "unset"}`, `statistics` and `reasons`
  - CLI `python -m src.eval.kv_gate {measure-real,measure-run,check}`
  - Record schema, one JSON line per corpus: `{"layer", "kv_head", "source", "seed", "books", "n_keys", "n_q_ood", "n_q_id", "n_sink", "n_tail", "conditions": {...}, "own_book_share_top10" (real only), "plain_*": {...}, "seconds"}`

Statistic definitions, over records `R` with the 96 distinct `(layer, kv_head)`:

| name | value |
|---|---|
| `hnsw_reach_k10` | `mean(r["plain_ood_k10"]["hnsw_reached"] for r in R)` |
| `hnsw_recall_k10_ef176` | `median(r["plain_ood_k10"]["hnsw_recall_op"])` |
| `ivf_ratio_k128` | `median(r["plain_ood_k128"]["ivf_scan_frac"] / r["plain_id_k128"]["ivf_scan_frac"])` over records where both scan fractions are not `None` (use `is not None`, never truthiness, because 0.0 is a legitimate scan fraction); `None` if no record qualifies |
| `ivf_recall_k128_np24` | `median(r["plain_ood_k128"]["ivf_recall_op"])` |
| `hnsw_recall_k10_ef176_id` | `median(r["plain_id_k10"]["hnsw_recall_op"])` |

`gates/scout_kv.yaml`:

```yaml
# Gate bands for the scout_kv family. Checked by:
#
#     python -m src.eval.kv_gate check --records <records.jsonl>
#
# Unlike the other gates/*.yaml, these statistics are search difficulty, not
# the four intrinsic statistics: see docs/datasets/scout_kv.md and
# docs/ai/specs/2026-10-18-kv-cache-scout-design.md. Each is pooled over all
# 96 (global layer, KV head) corpora.
#
# ALL BANDS ARE UNSET. Setting them is a human decision, made from the
# reference noise floor (5 draws over books 12-19). Until then the checker
# exits 2 with verdict "unset".
dataset: scout_kv
canonical:
  n_corpora: 96
  n_keys: 31736
  n_q_ood: 5120
  n_q_id: 1024
  n_sink: 1
  n_tail: 1024
  nlist: 256
  hnsw_m: 48
  ef_construction: 500
  target: 0.9
statistics:
  hnsw_reach_k10: null
  hnsw_recall_k10_ef176: null
  ivf_ratio_k128: null
  ivf_recall_k128_np24: null
  hnsw_recall_k10_ef176_id: null
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_kv_gate.py
import json
import subprocess
import sys

import numpy as np
import pytest

from src.eval import kv_gate
from src.eval.check_gate import EXIT_FAIL, EXIT_PASS, EXIT_UNSET

GATE = kv_gate.GATES_FILE  # gates/scout_kv.yaml


def _rec(L, g, reached=True, rec10=0.8, ivf_o=0.2, ivf_i=0.1, n_keys=31736):
    def part(r, reach, iv, ivr):
        return {"hnsw_reached": reach, "hnsw_recall_op": r, "ivf_scan_frac": iv,
                "ivf_reached": iv is not None, "ivf_recall_op": ivr, "hnsw_ndis": 100.0}
    return {"layer": L, "kv_head": g, "n_keys": n_keys, "n_q_ood": 5120, "n_q_id": 1024,
            "n_sink": 1, "n_tail": 1024,
            "conditions": {"nlist": 256, "hnsw_m": 48, "ef_construction": 500, "target": 0.9},
            "plain_ood_k10": part(rec10, reached, ivf_o, 0.8), "plain_id_k10": part(1.0, True, ivf_i, 0.9),
            "plain_ood_k128": part(rec10, reached, ivf_o, 0.8), "plain_id_k128": part(1.0, True, ivf_i, 0.9)}


def _full(**kw):
    return [_rec(L, g, **kw) for L in kv_gate.GLOBAL_LAYERS for g in range(8)]


def test_pooled_statistics_values():
    """Catches mean/median mix-ups and the ratio direction (ood / id, not id / ood)."""
    R = _full()
    for r in R[:24]:
        r["plain_ood_k10"]["hnsw_reached"] = False
    s = kv_gate.pooled_statistics(R)
    assert s["hnsw_reach_k10"] == pytest.approx(72 / 96)
    assert s["ivf_ratio_k128"] == pytest.approx(2.0)
    assert s["hnsw_recall_k10_ef176_id"] == 1.0


def test_ratio_skips_unreached_and_is_none_when_all_unreached():
    """Catches dividing None, or a falsy-zero guard dropping a legitimate 0.0 scan."""
    R = _full()
    for r in R:
        r["plain_ood_k128"]["ivf_scan_frac"] = None
    assert kv_gate.pooled_statistics(R)["ivf_ratio_k128"] is None
    R = _full(ivf_o=0.0)
    assert kv_gate.pooled_statistics(R)["ivf_ratio_k128"] == 0.0


def test_shipped_gate_is_unset():
    """Catches a band being set in the shipped file (a human decision)."""
    gate = kv_gate.load_kv_gate(GATE)
    assert all(v is None for v in gate["statistics"].values())
    assert kv_gate.evaluate(_full(), gate)["verdict"] == "unset"


def _gate_with(**bands):
    gate = kv_gate.load_kv_gate(GATE)
    for k, v in bands.items():
        gate["statistics"][k] = v
    return gate


def test_set_band_passes_inside_and_fails_outside():
    """Catches an inverted comparison."""
    gate = _gate_with(**{n: {"min": -1e9, "max": 1e9} for n in kv_gate.STATISTICS})
    gate["statistics"]["hnsw_reach_k10"] = {"min": 0.5, "max": 0.6}
    R = _full()
    for r in R[:40]:
        r["plain_ood_k10"]["hnsw_reached"] = False  # reach = 56/96 = 0.583
    assert kv_gate.evaluate(R, gate)["verdict"] == "pass"
    for r in R[:60]:
        r["plain_ood_k10"]["hnsw_reached"] = False  # 36/96 = 0.375
    assert kv_gate.evaluate(R, gate)["verdict"] == "fail"


@pytest.mark.parametrize("mutate", ["key_count", "missing", "duplicate", "nlist"])
def test_condition_mismatch_fails(mutate):
    """Catches comparing a measurement taken under other conditions."""
    R = _full()
    if mutate == "key_count":
        R[0]["n_keys"] = 31740
    elif mutate == "missing":
        R.pop()
    elif mutate == "duplicate":
        R[-1] = dict(R[0])
    else:
        R[3]["conditions"]["nlist"] = 1024
    rep = kv_gate.evaluate(R, kv_gate.load_kv_gate(GATE))
    assert rep["verdict"] == "fail" and rep["reasons"]


def test_cli_check_exit_codes(tmp_path):
    """Catches the CLI exiting 0 on an unset gate."""
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _full()))
    out = subprocess.run([sys.executable, "-m", "src.eval.kv_gate", "check", "--records", str(p)],
                         capture_output=True, text=True)
    assert out.returncode == EXIT_UNSET, out.stderr
    assert json.loads(out.stdout)["verdict"] == "unset"


def _write_multi(path, tensors):
    """F32 multi-tensor safetensors, the layout of capture/weights/layerXX."""
    import struct

    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = np.ascontiguousarray(arr, "<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def test_measure_real_end_to_end_on_a_tiny_capture(tmp_path):
    """Catches measure-real not iterating KV heads, mis-sizing corpora, or not
    recording conditions. Tiny sizes: 2 books, 1 layer, T=64, D=16."""
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    for b in (0, 1):
        p = h_path(tmp_path, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(64, 16)).astype(np.float32))
    (tmp_path / "weights").mkdir()
    _write_multi(tmp_path / "weights" / "layer03.safetensors",
                 {"k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
                  "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32)})
    out = tmp_path / "r.jsonl"
    kv_gate.measure_real(tmp_path, books=[0, 1], layers=[3], seed=0, out=out,
                         n_sink=1, n_tail=16, nlist=4, hnsw_m=8, ef_construction=16, threads=1)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted((r["layer"], r["kv_head"]) for r in recs) == [(3, 0), (3, 1)]
    for r in recs:
        assert r["n_keys"] == 2 * ((64 - 16 - 1) // 2) and r["n_q_ood"] == 16 * 5 and r["n_q_id"] == 16
        assert r["conditions"]["nlist"] == 4 and r["books"] == [0, 1] and r["source"] == "real"
        assert 0.0 <= r["own_book_share_top10"] <= 1.0
```

`measure_real(capture_dir, books, layers, seed, out, *, n_sink=1, n_tail=1024, nlist=256, hnsw_m=48, ef_construction=500, threads=15)` is the exact signature; the CLI's `--books` accepts `12-19` or `12,13,...`. If an `evaluate` test can't catch its mutation, the implementer reports that rather than weakening the test.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `kv_gate.py`.**
  - **`load_kv_gate`** checks that `canonical` has exactly the keys in the YAML above and that `statistics` has exactly `STATISTICS`, and validates each band with `check_gate.validate_band`.
  - **`evaluate`:**
    - Check conditions first. There must be 96 records with distinct `(layer, kv_head)` covering `GLOBAL_LAYERS x range(8)`. Every record's `n_keys`, `n_q_ood`, `n_q_id`, `n_sink`, `n_tail` and `conditions[nlist|hnsw_m|ef_construction|target]` must equal canonical. Any mismatch gives `fail` with a reason naming the record and key.
    - Then call `check_statistic(name, stats, band)` for each statistic.
    - The verdict follows `check_gate`'s rule: any fail gives `fail`; otherwise any unset gives `unset`; otherwise `pass`.
  - **`measure_real`:** for each layer, load the books' `h` (torch, CUDA if available), call `load_layer_weights`, and for each `g` build `mixed_real_corpus(..., seed=1009 * layer + g + 7919 * seed)`. Then `profile_corpus`, compute the own-book share of the exact top-10 (as `mix_check.py` does), and append the record.
  - **`measure_run`** takes `--run-dir L=path` pairs (all 12 required). For each layer:
    - Load the generator from `best_generator.pt` (or `--checkpoint-name`), and draw `n_keys + 2 * n_tail` rows with `torch.manual_seed(7919 * seed + layer)` and `sample_generator(..., normalize=output_normalize(cfg["model"]))`.
    - Call `invert_samples(x, load_preprocess_state(run_dir))`.
    - Split the rows into keys, query `h` and key-side `h`; call `synthetic_corpus` for each `g`, then profile.
  - **`check`:** print `evaluate`'s report as JSON and exit with its code.

- [ ] **Step 4: Tests pass, `make check`** (including all existing `test_check_gate.py` tests, unchanged).
- [ ] **Step 5: Mutation check.** Flip the ratio to `id / ood`; `test_pooled_statistics_values` must fail. Remove the duplicate check; the `duplicate` case must fail. Restore both.
- [ ] **Step 6: Commit** `src/eval/check_gate.py src/eval/kv_gate.py gates/scout_kv.yaml tests/test_kv_gate.py`.

---

### Task 7: Capture script — `scripts/kv/capture_scout.py`

**Files:**
- Create: `scripts/kv/capture_scout.py`, ported from `docs/results/kv-cache-profile/spike/capture_scout.py`
- Create: `scripts/kv/__init__.py` only if needed for test import; otherwise the test loads the file by path with `importlib`
- Test: `tests/test_kv_capture_script.py`

**Interfaces:**
- Produces pure helpers:
  - `select_books(token_counts: Sequence[int], ctx: int, first_book: int, n_books: int) -> list[int]`: PG-19 row indices of books `first_book ... first_book + n_books - 1`, counting only rows with `>= ctx` tokens
  - `check_meta_compatible(existing: dict, new: dict) -> None`: raises `ValueError` unless `global_layers`, `attn_scale`, `floor_scale` and `gqa` are equal
  - `planned_dirs(out_dir: Path, books: Sequence[int]) -> list[Path]`: raises `FileExistsError` if any `doc{b}` exists

Changes from the spike, and only these:
1. `--first-book` (default 0) and `--n-books` (default 4) replace `--n-docs`. Books are numbered globally, so the output is `doc{first_book + i}`.
2. Before the model loads, call `planned_dirs`, so an overlap fails in seconds rather than after a 20-minute load.
3. Each book writes `doc{b}/book.json` (row, title, token counts, verify results). Top-level `meta.json` is written only if absent. If present, it is read and `check_meta_compatible` is called, and it is never rewritten.
4. `weights/layerXX.safetensors` gains `v_proj` (bf16, from the model) if it lacks it. Rewriting the file with all three tensors is allowed, because it is regenerable from the model.
5. The tokenizer still reads row text, but the row loop uses `select_books` on the token counts.

Everything else (the experts patch, the SDPA capture, the pre-hook, the loader check that aborts on missing keys, `verify`, the smoke passes and the 0.5% `REL_ERR_LIMIT`) is kept verbatim.

- [ ] **Step 1: Failing tests for the three helpers**

```python
# tests/test_kv_capture_script.py
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "capture_scout", Path(__file__).parent.parent / "scripts" / "kv" / "capture_scout.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)  # must not import torch/transformers at module level


def test_select_books_skips_short_rows_and_counts_globally():
    """Catches counting short rows as books (shifts every later book index)."""
    counts = [40000, 100, 50000, 33000, 10, 32768, 70000]
    assert cs.select_books(counts, 32768, 0, 2) == [0, 2]
    assert cs.select_books(counts, 32768, 2, 3) == [3, 5, 6]


def test_select_books_too_few_raises():
    with pytest.raises(ValueError, match="only"):
        cs.select_books([40000, 10], 32768, 0, 2)


def test_existing_doc_dir_refused(tmp_path):
    """Catches a re-run silently overwriting book 4."""
    (tmp_path / "doc4").mkdir()
    with pytest.raises(FileExistsError, match="doc4"):
        cs.planned_dirs(tmp_path, [4, 5])


def test_meta_compat():
    """Catches capturing new books under a different scale or layer set."""
    m = {"global_layers": [3, 7], "attn_scale": 0.1, "floor_scale": 8192, "gqa": "x", "docs": []}
    cs.check_meta_compatible(m, dict(m, docs=[1]))
    with pytest.raises(ValueError, match="attn_scale"):
        cs.check_meta_compatible(m, dict(m, attn_scale=0.2))
```

Move the heavy imports (torch, transformers, pyarrow, safetensors) in the ported script inside `main()`, so this test runs on CI without them.

- [ ] **Step 2: Run, expect failure.** **Step 3: Port.** **Step 4: Tests pass, `make check`.**
- [ ] **Step 5: Mutation check.** Make `select_books` count every row; the first test must fail. Restore.
- [ ] **Step 6: Commit.**

---

### Task 8: Configs, family page, invariant note

**Files:**
- Create: `configs/scout_kv/v0/L03.yaml` ... `L47.yaml` (12 files)
- Create: `docs/datasets/scout_kv.md`
- Modify: `AGENTS.md` (invariant 1), `PROJECT_DOCUMENTATION.md` (`## ANN difficulty — the gate`: one paragraph; `## Datasets`: one row or line), `README.md` documentation map if it lists every family page
- Test: `tests/test_scout_kv_configs.py`

Template (`L43.yaml`; the others differ only in `kv_layer` and `output_dir`):

```yaml
# scout_kv v0, global layer 43. One of 12 per-layer configs that differ only in
# data.kv_layer and output_dir (pinned by tests/test_scout_kv_configs.py).
# Design: docs/ai/specs/2026-10-18-kv-cache-scout-design.md.
# real_path is box-specific: the capture lives only on tig-gpu.
seed: 42
device: cuda:0
output_dir: runs/scout_kv/v0/L43

data:
  real_path: /workspace/kv/capture
  format: kv_capture
  kv_layer: 43
  kv_books: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  kv_n_sink: 1
  descriptor_dim: 5120
  holdout_fraction: 0.02
  synthetic_if_missing: false
  synthetic_num_vectors: 0
  metric: l2
  preprocess:
    center: true
    whiten: false
    l2_normalize: false

model:
  latent_dim: 128
  generator_hidden_dims: [1024, 2048, 4096]
  critic_hidden_dims: [2048, 1024, 512]
  negative_slope: 0.2
  generator_type: mlp
  output_normalize: false

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  select_on: gate
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 5000
```

`holdout_fraction: 0.02` (7,864 rows) keeps the in-training gate selector's LID computation on 5,120-dim rows affordable. It only ranks checkpoints of one run.

- [ ] **Step 1: Failing tests**

```python
# tests/test_scout_kv_configs.py
from pathlib import Path

import yaml

from src.data.kv_capture import GLOBAL_LAYERS

ROOT = Path(__file__).resolve().parent.parent / "configs" / "scout_kv" / "v0"


def _load(L):
    return yaml.safe_load((ROOT / f"L{L:02d}.yaml").read_text())


def test_one_config_per_global_layer_and_nothing_else():
    assert sorted(p.name for p in ROOT.glob("*.yaml")) == [f"L{L:02d}.yaml" for L in GLOBAL_LAYERS]


def test_configs_differ_only_in_layer_and_output_dir():
    """Catches one layer's config drifting (a silent per-layer hyperparameter change)."""
    base = _load(43)
    for L in GLOBAL_LAYERS:
        c = _load(L)
        assert c["data"]["kv_layer"] == L and c["output_dir"] == f"runs/scout_kv/v0/L{L:02d}"
        for d in (c, base):
            d["data"].pop("kv_layer"); d.pop("output_dir")
        assert c == base
        base = _load(43)


def test_train_books_exclude_reference_books():
    """Catches training on books 12-19, which the gate's reference uses."""
    assert set(_load(43)["data"]["kv_books"]) == set(range(12))


def test_norms_are_preserved():
    """Catches any normalisation sneaking back in."""
    c = _load(43)
    assert c["data"]["preprocess"]["l2_normalize"] is False
    assert c["model"]["output_normalize"] is False
```

- [ ] **Step 2: Fail. Step 3: write the 12 configs** with a short throwaway generator script, not committed. **Step 4: pass.**
- [ ] **Step 5: Docs.**
  - **`docs/datasets/scout_kv.md`:** what a sample is; the book split; locked conditions; the gate statistics; "bands unset"; the mixed-book result with its MEASURED numbers from `docs/results/kv-cache-profile/mixed-book/comparison.txt`. Cite code by symbol and docs by anchor (enforced by `tests/test_docs_references.py`).
  - **`AGENTS.md` invariant 1:** append "Exception: `scout_kv` gates on IVF/HNSW search difficulty (`src/eval/kv_gate.py`); for it the four statistics are diagnostics. See `docs/datasets/scout_kv.md`."
  - **`PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate`:** add a paragraph saying the same.
- [ ] **Step 6: `make check`, commit.**

---

### Box tasks (orchestrator only)

Each step prints its MEASURED timing. Long jobs run with `nohup`, write their PID to `/workspace/logs/<name>.pid`, and are polled with `kill -0` (never `pgrep -f`). Every result that leaves the box goes by `tar` stream with sha256 checked on both sides.

### B1: Ship the code to the box

- [ ] `git archive --format=tar HEAD | ssh tig-gpu 'rm -rf /workspace/repo && mkdir -p /workspace/repo && tar xf - -C /workspace/repo && echo <HEAD sha> > /workspace/repo/COMMIT'`
- [ ] Install the repo's pins except torch into `kvenv`: `grep -v '^torch==' requirements.txt | ssh tig-gpu '/workspace/kvenv/bin/pip install -r /dev/stdin'`. Record `pip freeze` diff.
- [ ] On the box: `cd /workspace/repo && /workspace/kvenv/bin/python -m pytest -q tests/test_kv_*.py tests/test_output_normalize.py tests/test_scout_kv_configs.py`. Expected: all pass on torch 2.14.1.

### B2: Capture books 4-19

- [ ] `python scripts/kv/capture_scout.py --model-dir /workspace/models/scout --pg19-parquet <parquet> --out-dir /workspace/kv/capture --first-book 4 --n-books 16 --gpu-weights 24GiB`, in the background. ESTIMATE about 45 min.
- [ ] Afterwards: 16 new `doc*/book.json`, each with `worst_rel_err < 0.005`; `df -h` above 8 GB free; `meta.json` sha256 unchanged (`f78f360b...`).
- [ ] Copy every `book.json` off the box (tar plus sha256) and commit them under `docs/results/scout-kv/capture/`.

### B3: Real reference and noise floor

- [ ] `python -m src.eval.kv_gate measure-real --capture-dir /workspace/kv/capture --books 12-19 --seed {0..4} --out /workspace/kv/ref/seed{s}.jsonl`, sequentially. ESTIMATE about 19 min each.
- [ ] One diagnostic draw over books 12-15, seed 0, `--out ref/four_books.jsonl`. Its key count (31,740) differs from canonical by design, so `check` will report a condition mismatch.
- [ ] `check` on each seed file. Expected: verdict `unset` and no condition failures.
- [ ] Copy off, then commit the five pooled-statistics summaries plus a table of mean, min, max and spread per statistic under `docs/results/scout-kv/reference/`. The raw jsonl goes in only if under 1 MB each.

### B4: Train v0 on L03, timed

- [ ] `python -m src.train.train_wgan_gp --config configs/scout_kv/v0/L03.yaml`, in the background. Measure wall time, peak GPU memory and steps/s.
- [ ] **Stop and report to the human** with the measured time per layer and the projected total for 11 more, before queuing them.

### B5: Train the other 11 layers (after the human's go-ahead)

- [ ] Sequential by default. Two at once only if B4 measured GPU memory at under 35 GB and GPU utilisation at under 50%.

### B6: Gate v0

- [ ] `measure-run` with all 12 `best_generator.pt`, seed 0, then again with the final checkpoints.
- [ ] `check` both. The verdict will be `unset`; the report holds the pooled values.
- [ ] Write a self-contained HTML report under `docs/results/scout-kv/v0/` comparing v0 (both checkpoints) with the 5 reference draws, per statistic and per layer, with a claim table. Verify every figure renders by screenshot.
- [ ] Diagnostics in the same report: the four intrinsic statistics (`src/eval/ann_difficulty.py::compute` at the canonical k 100, k_hub 10, nlist 256, metric l2) on 20,000 real `h` rows from books 12-19 and on 20,000 v0 samples, per layer; plus the per-layer and per-head values of every gate statistic.
- [ ] Hand the band decision to the human with the noise-floor table.

## Not in this plan

- **Attention-error diagnostic (SoftmaxLift and top-k, flat/IVF/HNSW).** The spec lists it as a diagnostic. It needs `W_v` (exported by Task 7) and `s(pos)`, and it stays in `docs/results/kv-cache-profile/spike/profile_v2.py` until a follow-up plan ports it. Nothing in the gate depends on it.
- **Later rungs** (RMSNorm-form output, a projection-aware critic). Designed only after v0 is gated.
