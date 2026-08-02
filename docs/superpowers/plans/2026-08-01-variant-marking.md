> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# Variant Marking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the EMA and sparse-generator branches into `eda/sift-eda`, name the four trained model variants consistently across configs, code and the EDA report, and separate human-maintained docs from AI working notes.

**Architecture:** Four v-numbered variants (`v0`/`v1`/`v1_5`/`v2`), each one config delta from the previous, get a config file apiece reconciled against the runs already trained. The generator architecture axis is hard-renamed `sparse` → `gated`. A new `src/eval/compare_variants.py` drives the existing multi-overlay EDA report across all four by importing `eda_report` directly, which requires extracting a `run(args)` entry point from its `main()`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, plotly, scikit-learn, pytest, PyYAML.

## Global Constraints

- Repo root for all commands: `/home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/eda-sift`.
- Run Python as `python3` and pytest as `python3 -m pytest` from the repo root; the venv lives at the main worktree's `.venv`.
- Variant names are exactly `v0`, `v1`, `v1_5`, `v2` — used identically in config filenames, the docs table, and EDA report legend labels.
- `generator_type` accepts exactly `mlp` and `gated` after Task 3. No `sparse` alias, no deprecation shim.
- Never rename the checkpoint key `generator_weights` or its `"live"`/`"ema"` values — unrelated to this work.
- Do not retrain anything. Configs are reconciled against existing runs.
- `runs/` is gitignored; edits there are local-only and never committed.
- Every commit message ends with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

**Created:**
- `configs/sift_gan_v0.yaml` — plain WGAN-GP config.
- `configs/sift_gan_v1.yaml` — v0 + generator EMA.
- `configs/sift_gan_v1_5.yaml` — v1 + distance regularizer.
- `configs/sift_gan_v2.yaml` — v1_5 + gated generator.
- `src/eval/compare_variants.py` — resolves variant checkpoints, samples each, drives `eda_report`.
- `tests/test_compare_variants.py` — tests for the driver.
- `docs/superpowers/README.md` — marks the tree as AI working notes.

**Modified:**
- `src/models/generator.py` — `SparseGenerator` → `GatedGenerator`, `kind == "sparse"` → `kind == "gated"`.
- `src/eval/eda_report.py` — extract `run(args)` from `main()`.
- `configs/bench_sparse.yaml` — `generator_type: gated`.
- `tests/test_generator.py`, `tests/test_generator_factory.py`, `tests/test_evaluate_distribution.py`, `tests/test_ema.py`, `tests/test_train_smoke.py` — rename fallout.
- `README.md` — doc map + variant quick-reference.
- `PROJECT_DOCUMENTATION.md` — Model variants section, updated EDA section.
- `data/README.md` — what the driver expects on disk.
- `docs/superpowers/plans/2026-07-31-*.md`, `docs/superpowers/specs/2026-07-31-*.md` — AI banner.

**Local-only (never committed):**
- `runs/x100k_sparse_clamp4/run_config.yaml` — hand-edited `sparse` → `gated`.

---

### Task 1: Merge the EMA + sparse branch

**Files:**
- Modify (via merge): `requirements.txt`, `src/eval/evaluate_distribution.py`, `src/train/train_wgan_gp.py`, `src/models/generator.py`, `src/sample/generate.py`, `pytest.ini`, `configs/`, `tests/`

**Interfaces:**
- Consumes: nothing.
- Produces: a tree where `build_generator(model_cfg: Mapping[str, Any], output_dim: int) -> nn.Module` exists in `src/models/generator.py`, `load_generator(config: Dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module` exists in `src/eval/evaluate_distribution.py`, and `sample_generator(generator: nn.Module, num_samples: int, latent_dim: int, batch_size: int, device: torch.device) -> np.ndarray` exists in `src/train/train_wgan_gp.py`. Later tasks import all three.

- [ ] **Step 1: Confirm a clean starting tree**

```bash
git status --short
git log --oneline -1
```

Expected: no output from `status` (clean), and the `log` line is the design-spec commit `docs: design for variant marking...`. If the tree is dirty, stop and report — do not stash.

- [ ] **Step 2: Merge**

```bash
git merge feat-wgan-gp-v2-ema
```

Expected: conflicts. The four likely files are `requirements.txt`, `src/eval/evaluate_distribution.py`, `src/train/train_wgan_gp.py`, and `pytest.ini`.

- [ ] **Step 3: Resolve `requirements.txt` by keeping both sides**

Both branches appended to the same file. The union, in this order:

```
numpy
torch
pyyaml
tqdm
scikit-learn
scipy
plotly
# Static PNG export in src/eval/eda_report.py. Kaleido v1 drives a headless
# Chrome; run `plotly_get_chrome` once if no browser is installed. The HTML
# report is produced without it -- pass --no-png to skip export entirely.
kaleido
# Test suite. Run from the repo root:
#   python3 -m pytest
pytest
```

Verify against both sides before writing — if either branch pins a version or lists a package not shown above, keep it:

```bash
git show :2:requirements.txt
git show :3:requirements.txt
```

- [ ] **Step 4: Resolve the remaining conflicts by keeping both sides**

For `src/eval/evaluate_distribution.py`, `src/train/train_wgan_gp.py`, and `pytest.ini`, inspect each conflict hunk and keep both additions. These branches added different features to the same files rather than editing the same lines semantically — no hunk should require choosing one side over the other. If a hunk genuinely conflicts (both sides rewrote the same function body), stop and report it rather than guessing.

```bash
git diff --diff-filter=U --name-only
```

Expected after resolving: no output.

- [ ] **Step 5: Verify no conflict markers survived**

```bash
grep -rn '<<<<<<<\|>>>>>>>\|=======' --include='*.py' --include='*.yaml' --include='*.txt' --include='*.ini' . | grep -v '\.git/'
```

Expected: no output. (`=======` may legitimately appear in markdown; the filters above exclude markdown for that reason.)

- [ ] **Step 6: Run the full suite**

```bash
python3 -m pytest
```

Expected: PASS. The combined suite is `test_ann_difficulty`, `test_ema`, `test_generator`, `test_generator_factory`, `test_tensor_stats`, `test_train_smoke`, `test_evaluate_distribution`. If any test fails, fix the merge resolution — do not edit tests to make them pass; that is Task 3's job and only for the rename.

- [ ] **Step 7: Commit the merge**

```bash
git add -A
git commit -m "$(cat <<'EOF'
merge: bring EMA and sparse generator into the EDA branch

Consolidates feat-wgan-gp-v2-ema so the EDA report can compare every
trained variant from one tree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Retire the parallel sparse worktree

**Files:**
- No files in this repo change. This task deletes a branch and a worktree after proving nothing is lost.

**Interfaces:**
- Consumes: the merged tree from Task 1.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: List what the parallel branch has that HEAD does not**

```bash
git diff HEAD...worktree-sparse-generator --stat
```

Expected: an empty or near-empty diffstat. The branch's contributions (`build_generator` factory, `tensor_stats` support metrics, `configs/bench_sparse.yaml`, `tests/test_generator.py`, `tests/test_generator_factory.py`, `tests/test_tensor_stats.py`, `tests/test_train_smoke.py`) are expected to have reached `feat-wgan-gp-v2-ema` already through merge `b9d9ef7`.

- [ ] **Step 2: Inspect anything the diff reports**

For every file the previous step listed, read the actual difference:

```bash
git diff HEAD...worktree-sparse-generator -- <path>
```

If any hunk is a real improvement not present in HEAD, cherry-pick or hand-apply it now, run `python3 -m pytest`, and commit it before continuing. **If anything is carried over, stop and report what and why before deleting.**

- [ ] **Step 3: Confirm the worktree has no uncommitted work**

```bash
git -C /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator status --short
```

Expected: no output. If there is uncommitted work, stop and report — do not delete.

- [ ] **Step 4: Remove the worktree and branch**

```bash
git worktree remove /home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sparse-generator
git branch -D worktree-sparse-generator
```

Leave `preserve-wgan-improvements-wip` and `experiment/wgan-improvements` alone.

- [ ] **Step 5: Verify**

```bash
git worktree list
git branch
```

Expected: `sparse-generator` absent from both. Nothing to commit — this task changes no tracked files.

---

### Task 3: Hard rename `sparse` → `gated`

**Files:**
- Modify: `src/models/generator.py` (class `SparseGenerator` at line 31, `build_generator` at line 122)
- Modify: `configs/bench_sparse.yaml`
- Test: `tests/test_generator_factory.py`, `tests/test_generator.py`, `tests/test_evaluate_distribution.py`, `tests/test_ema.py`, `tests/test_train_smoke.py`
- Local-only: `runs/x100k_sparse_clamp4/run_config.yaml`

**Interfaces:**
- Consumes: `build_generator(model_cfg, output_dim)` from Task 1.
- Produces: `GatedGenerator` (same constructor signature as `SparseGenerator`: `latent_dim`, `output_dim`, `hidden_dims`, `negative_slope`, `gate_temperature=0.5`, `logit_clamp=10.0`), and `build_generator` accepting `generator_type` values `"mlp"` and `"gated"` only. Tasks 4 and 6 rely on `"gated"` being the accepted value.

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_generator_factory.py` — replace every occurrence of `generator_type="sparse"` with `generator_type="gated"`, rename `SparseGenerator` to `GatedGenerator` in the import and assertions, and rename the test functions. Add a test that the old value is now rejected:

```python
from src.models.generator import Generator, GatedGenerator, build_generator


def test_gated():
    cfg = dict(BASE_CFG, generator_type="gated")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, GatedGenerator)


def test_sparse_is_no_longer_accepted():
    with pytest.raises(ValueError, match="Unknown generator_type"):
        build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=128)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_generator_factory.py -v
```

Expected: FAIL — `ImportError: cannot import name 'GatedGenerator'`.

- [ ] **Step 3: Rename in `src/models/generator.py`**

Two edits. The class declaration at line 31:

```python
class GatedGenerator(nn.Module):
```

and the factory branch in `build_generator`:

```python
    if kind == "gated":
        return GatedGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
        )
    raise ValueError(f"Unknown generator_type: {kind}")
```

Update the class docstring and any internal comment that says "sparse" to say "gated" where it names the type, but leave prose describing *sparsity* as a property (the gate does produce sparse output) — that is still accurate. Do not touch `_sample_gate` or `forward` logic.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_generator_factory.py -v
```

Expected: PASS.

- [ ] **Step 5: Fix the remaining test files**

```bash
grep -rn "SparseGenerator\|\"sparse\"\|'sparse'" tests/ src/
```

Update every hit in `tests/test_generator.py`, `tests/test_evaluate_distribution.py`, `tests/test_ema.py`, `tests/test_train_smoke.py`: import and assert `GatedGenerator`, and pass `generator_type="gated"`. Rename test function names containing `sparse` to `gated` so failures read correctly.

- [ ] **Step 6: Update the benchmark config**

In `configs/bench_sparse.yaml`, change `generator_type: sparse` to `generator_type: gated`. Leave the filename and `output_dir` as they are — it is a historical benchmark artifact.

- [ ] **Step 7: Run the full suite**

```bash
python3 -m pytest
```

Expected: PASS, with no remaining references:

```bash
grep -rn "SparseGenerator" src/ tests/ configs/
```

Expected: no output.

- [ ] **Step 8: Repair the local run config the rename breaks**

`runs/x100k_sparse_clamp4/run_config.yaml` still says `generator_type: sparse` and will no longer load. It is gitignored, so edit it in place:

```bash
sed -i 's/generator_type: sparse/generator_type: gated/' \
  /home/fibonadithya/TIG/wgan-synthetic/runs/x100k_sparse_clamp4/run_config.yaml
grep -n generator_type /home/fibonadithya/TIG/wgan-synthetic/runs/x100k_sparse_clamp4/run_config.yaml
```

Expected: `generator_type: gated`.

- [ ] **Step 9: Verify the existing checkpoint still loads**

Checkpoints store `generator_weights` (`"live"`/`"ema"`) but not `generator_type`, and state-dict keys derive from submodule attribute names (`trunk`, `magnitude_head`, `gate_head`) which this rename does not change — so loading should be unaffected. Prove it rather than assume it. Only run this if the checkpoint exists locally:

```bash
ls /home/fibonadithya/TIG/wgan-synthetic/runs/x100k_sparse_clamp4/best_generator.pt
```

If present:

```bash
python3 - <<'EOF'
from pathlib import Path
import torch, yaml
from src.eval.evaluate_distribution import load_generator

root = Path("/home/fibonadithya/TIG/wgan-synthetic/runs/x100k_sparse_clamp4")
config = yaml.safe_load((root / "run_config.yaml").read_text())
g = load_generator(config, root / "best_generator.pt", torch.device("cpu"))
print("loaded:", type(g).__name__)
EOF
```

Expected: `loaded: GatedGenerator`. If the file is absent (the checkpoints live on the training box), note that in the task report and move on — the config edit is still correct and required.

- [ ] **Step 10: Commit**

```bash
git add src/models/generator.py tests/ configs/bench_sparse.yaml
git commit -m "$(cat <<'EOF'
refactor(models): rename generator_type sparse to gated

"sparse" named the output property rather than the mechanism, and read as
a peer of the variant numbering rather than the architecture axis beneath
it. generator_type is now mlp | gated and SparseGenerator is
GatedGenerator.

Deliberately no compatibility alias. Checkpoints do not persist
generator_type -- the architecture is rebuilt from the run config at load
time and state-dict keys are unchanged -- so the only artifact affected is
runs/x100k_sparse_clamp4/run_config.yaml, which is gitignored and was
edited in place.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Variant configs

**Files:**
- Create: `configs/sift_gan_v0.yaml`, `configs/sift_gan_v1.yaml`, `configs/sift_gan_v1_5.yaml`, `configs/sift_gan_v2.yaml`

**Interfaces:**
- Consumes: `generator_type: gated` from Task 3.
- Produces: the four config paths above. Task 6 hardcodes them as defaults; Task 7 documents them.

Each config is exactly one delta from the previous. The shared block is copied verbatim from `runs/long_baseline/run_config.yaml` so these reproduce work already done.

- [ ] **Step 1: Write `configs/sift_gan_v0.yaml`**

```yaml
# Variant v0 -- plain WGAN-GP. The baseline every other variant is one
# delta from. Matches runs/long_baseline and runs/bench_baseline.
seed: 42
device: auto
output_dir: runs/sift_gan_v0

data:
  real_path: data/sift_base.npy
  format: npy
  descriptor_dim: 128
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  distance_reg_alpha: 0.0
  distance_reg_max_points: 128
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 2: Write `configs/sift_gan_v1.yaml`**

```yaml
# Variant v1 -- v0 plus generator EMA. Sole delta from v0:
# training.ema_decay. Matches runs/long_ema_only and runs/x100k_ema_only.
seed: 42
device: auto
output_dir: runs/sift_gan_v1

data:
  real_path: data/sift_base.npy
  format: npy
  descriptor_dim: 128
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.0
  distance_reg_max_points: 128
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 3: Write `configs/sift_gan_v1_5.yaml`**

```yaml
# Variant v1_5 -- v1 plus the pairwise-distance regularizer. Sole delta
# from v1: training.distance_reg_alpha (and the point budget it needs).
# Matches runs/long_improved, runs/x100k_improved, runs/bench_improved.
seed: 42
device: auto
output_dir: runs/sift_gan_v1_5

data:
  real_path: data/sift_base.npy
  format: npy
  descriptor_dim: 128
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.1
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 4: Write `configs/sift_gan_v2.yaml`**

```yaml
# Variant v2 -- v1_5 plus the gated non-negative generator, which
# reproduces SIFT's exact-zero support that a dense MLP cannot. Sole delta
# from v1_5: model.generator_type and its two gate hyperparameters.
# Matches runs/x100k_sparse_clamp4.
seed: 42
device: auto
output_dir: runs/sift_gan_v2

data:
  real_path: data/sift_base.npy
  format: npy
  descriptor_dim: 128
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: gated
  gate_temperature: 0.5
  logit_clamp: 10.0

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.1
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 5: Verify each config parses and builds its generator**

```bash
python3 - <<'EOF'
import yaml
from pathlib import Path
from src.models.generator import build_generator

for name in ["v0", "v1", "v1_5", "v2"]:
    cfg = yaml.safe_load(Path(f"configs/sift_gan_{name}.yaml").read_text())
    g = build_generator(cfg["model"], output_dim=cfg["data"]["descriptor_dim"])
    print(name, type(g).__name__,
          "ema=", cfg["training"].get("ema_decay", 0.0),
          "distreg=", cfg["training"]["distance_reg_alpha"])
EOF
```

Expected exactly:

```
v0 Generator ema= 0.0 distreg= 0.0
v1 Generator ema= 0.999 distreg= 0.0
v1_5 Generator ema= 0.999 distreg= 0.1
v2 GatedGenerator ema= 0.999 distreg= 0.1
```

Each line differs from the one above it in exactly one place. If not, a config is wrong.

- [ ] **Step 6: Commit**

```bash
git add configs/sift_gan_v0.yaml configs/sift_gan_v1.yaml configs/sift_gan_v1_5.yaml configs/sift_gan_v2.yaml
git commit -m "$(cat <<'EOF'
feat(configs): name the four trained variants v0 through v2

The variants that were actually trained were identifiable only by run
directory name, and no config reproduced them one-to-one. Each config is
now exactly one delta from the previous -- EMA, then the distance
regularizer, then the gated generator -- so an EDA overlay attributes each
change to a single cause.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Extract a callable entry point from `eda_report`

**Files:**
- Modify: `src/eval/eda_report.py` (the `main()` at line 590)
- Test: `tests/test_eda_report.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `run(args: argparse.Namespace) -> Path` in `src/eval/eda_report.py`, returning the path of the written HTML report. `main()` becomes `run(parse_args())`. Task 6 calls `run` directly.

The driver in Task 6 must not shell out, and `main()` currently reads `parse_args()` internally with no injection point. Extracting `run(args)` is the minimal change that opens one.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eda_report.py`:

```python
import argparse
from pathlib import Path

import numpy as np

from src.eval import eda_report


def make_args(tmp_path, real, synthetic):
    real_path = tmp_path / "real.npy"
    np.save(real_path, real)
    specs = []
    for label, arr in synthetic.items():
        p = tmp_path / f"{label}.npy"
        np.save(p, arr)
        specs.append(f"{label}={p}")
    return argparse.Namespace(
        real_path=str(real_path),
        real_format="npy",
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=str(tmp_path / "out"),
        preprocess="l2",
        max_vectors=200,
        num_pairs=500,
        knn=3,
        bins=16,
        top_divergent=4,
        seed=42,
        no_png=True,
        plotlyjs="cdn",
    )


def test_run_returns_written_report_path(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {"v0": rng.normal(size=(200, 8)).astype(np.float32)}

    out = eda_report.run(make_args(tmp_path, real, synth))

    assert isinstance(out, Path)
    assert out.exists()
    assert out.suffix == ".html"
    assert "v0" in out.read_text()


def test_run_accepts_several_synthetic_sets(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {
        "v0": rng.normal(size=(200, 8)).astype(np.float32),
        "v1": rng.normal(size=(200, 8)).astype(np.float32),
        "v2": rng.normal(size=(200, 8)).astype(np.float32),
    }

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    for label in ("v0", "v1", "v2"):
        assert label in html
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_eda_report.py -v
```

Expected: FAIL — `AttributeError: module 'src.eval.eda_report' has no attribute 'run'`.

- [ ] **Step 3: Extract `run(args)`**

In `src/eval/eda_report.py`, rename the existing `def main() -> None:` to `def run(args: argparse.Namespace) -> Path:` and delete its first line (`args = parse_args()`). The body is otherwise unchanged. Find where it writes the HTML file near the end of the function and return that path:

```python
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    ...
    return report_path
```

Use whatever variable already holds the output path rather than introducing a second one — only add the `return`. Then add a new `main()` immediately after:

```python
def main() -> None:
    run(parse_args())
```

Leave the `if __name__ == "__main__":` block at the bottom pointing at `main()`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_eda_report.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify the CLI still works end to end**

```bash
python3 - <<'EOF'
import numpy as np
rng = np.random.default_rng(0)
np.save("/tmp/eda_real.npy", rng.normal(size=(300, 16)).astype(np.float32))
np.save("/tmp/eda_a.npy", rng.normal(size=(300, 16)).astype(np.float32))
EOF
python3 -m src.eval.eda_report \
  --real-path /tmp/eda_real.npy \
  --synthetic-path v0=/tmp/eda_a.npy \
  --output-dir /tmp/eda_cli_check \
  --max-vectors 300 --num-pairs 500 --no-png --plotlyjs cdn
ls /tmp/eda_cli_check
```

Expected: an HTML report and `summary.json` in `/tmp/eda_cli_check`.

- [ ] **Step 6: Commit**

```bash
git add src/eval/eda_report.py tests/test_eda_report.py
git commit -m "$(cat <<'EOF'
refactor(eval): extract run(args) from eda_report main

main() read parse_args() internally, so the report could only be driven
from a command line. run(args) takes a Namespace and returns the written
report path, which lets src.eval.compare_variants call it in-process
instead of shelling out. Behaviour is unchanged; main() is now
run(parse_args()).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Variant comparison driver

**Files:**
- Create: `src/eval/compare_variants.py`
- Test: `tests/test_compare_variants.py`

**Interfaces:**
- Consumes: `eda_report.run(args) -> Path` (Task 5), `build_generator(model_cfg, output_dim)` (Task 3), `load_generator(config, checkpoint_path, device)` from `src.eval.evaluate_distribution`, `sample_generator(generator, num_samples, latent_dim, batch_size, device)` from `src.train.train_wgan_gp`, and the four config paths from Task 4.
- Produces: `VARIANTS: Tuple[Variant, ...]`, `Variant(name, config_path, run_dir)`, `resolve_variants(variants, root) -> Tuple[List[Variant], List[Tuple[Variant, str]]]`, and `main() -> None`.

A variant whose checkpoint is missing is skipped with a message rather than aborting — the checkpoints live on the training box and are frequently absent locally, so a partial comparison must still produce a report.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare_variants.py`:

```python
from pathlib import Path

from src.eval import compare_variants as cv


def test_variants_are_the_four_named_ones():
    assert [v.name for v in cv.VARIANTS] == ["v0", "v1", "v1_5", "v2"]


def test_every_variant_config_exists():
    for v in cv.VARIANTS:
        assert Path(v.config_path).exists(), f"missing config for {v.name}"


def _make_run_dir(root, name, with_checkpoint=True, with_config=True):
    d = root / name
    d.mkdir(parents=True)
    if with_config:
        (d / "run_config.yaml").write_text("model: {}\n")
    if with_checkpoint:
        (d / "best_generator.pt").write_bytes(b"")
    return d


def test_resolve_skips_variants_with_no_checkpoint(tmp_path):
    variants = (
        cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),
        cv.Variant("v1", "configs/sift_gan_v1.yaml", "runs/b"),
    )
    _make_run_dir(tmp_path / "runs", "a")
    _make_run_dir(tmp_path / "runs", "b", with_checkpoint=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0"]
    assert [v.name for v, _ in skipped] == ["v1"]
    assert "best_generator.pt" in skipped[0][1]


def test_resolve_skips_variants_with_no_run_config(tmp_path):
    variants = (cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),)
    _make_run_dir(tmp_path / "runs", "a", with_config=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert "run_config.yaml" in skipped[0][1]


def test_resolve_reports_a_missing_run_dir(tmp_path):
    variants = (cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/nope"),)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert [v.name for v, _ in skipped] == ["v0"]


def test_resolve_finds_everything_when_present(tmp_path):
    variants = (
        cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),
        cv.Variant("v2", "configs/sift_gan_v2.yaml", "runs/b"),
    )
    _make_run_dir(tmp_path / "runs", "a")
    _make_run_dir(tmp_path / "runs", "b")

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0", "v2"]
    assert skipped == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_compare_variants.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.compare_variants'`.

- [ ] **Step 3: Write the driver**

Create `src/eval/compare_variants.py`:

```python
"""Overlay every trained variant on the real SIFT data in one EDA report.

src.eval.eda_report can already overlay any number of synthetic sets; this
drives it across the four named variants so the comparison does not have to
be retyped. Each variant is one config delta from the one before it -- EMA,
then the distance regularizer, then the gated generator -- so a difference
visible in the report attributes to a single cause.

A variant whose checkpoint is not on this machine is skipped with a message
rather than aborting: checkpoints usually live on the training box, and a
partial comparison is still worth reading.

Example:
    python -m src.eval.compare_variants \
        --real-path data/sift_base.npy \
        --output-dir runs/eda_variants
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import yaml

from src.eval import eda_report
from src.eval.evaluate_distribution import get_device, load_generator
from src.train.train_wgan_gp import sample_generator


@dataclass(frozen=True)
class Variant:
    """One named model variant and where its trained artifacts live."""

    name: str
    config_path: str
    run_dir: str


# Ordered so the report legend reads as a progression. Each entry is one
# config delta from the previous. Run directories point at the longest run of
# each variant that exists; v0 was never taken to 100k steps.
VARIANTS: Tuple[Variant, ...] = (
    Variant("v0", "configs/sift_gan_v0.yaml", "runs/long_baseline"),
    Variant("v1", "configs/sift_gan_v1.yaml", "runs/x100k_ema_only"),
    Variant("v1_5", "configs/sift_gan_v1_5.yaml", "runs/x100k_improved"),
    Variant("v2", "configs/sift_gan_v2.yaml", "runs/x100k_sparse_clamp4"),
)

CHECKPOINT_NAME = "best_generator.pt"
RUN_CONFIG_NAME = "run_config.yaml"


def resolve_variants(
    variants: Sequence[Variant], root: Path
) -> Tuple[List[Variant], List[Tuple[Variant, str]]]:
    """Split variants into those whose artifacts are on disk and those not.

    The run config is required alongside the checkpoint because the generator
    architecture is rebuilt from it -- the checkpoint records which weights it
    holds ("live"/"ema") but not which generator produced them.
    """
    found: List[Variant] = []
    skipped: List[Tuple[Variant, str]] = []
    for variant in variants:
        run_dir = root / variant.run_dir
        checkpoint = run_dir / CHECKPOINT_NAME
        run_config = run_dir / RUN_CONFIG_NAME
        if not run_dir.is_dir():
            skipped.append((variant, f"no run directory at {run_dir}"))
        elif not checkpoint.exists():
            skipped.append((variant, f"no {CHECKPOINT_NAME} in {run_dir}"))
        elif not run_config.exists():
            skipped.append((variant, f"no {RUN_CONFIG_NAME} in {run_dir}"))
        else:
            found.append(variant)
    return found, skipped


def generate_samples(
    variant: Variant, root: Path, num_samples: int, batch_size: int, out_dir: Path
) -> Path:
    """Sample a variant's best checkpoint to an .npy file, and return its path."""
    run_dir = root / variant.run_dir
    config = yaml.safe_load((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))
    device = get_device(config["device"])
    generator = load_generator(config, run_dir / CHECKPOINT_NAME, device)
    x = sample_generator(
        generator,
        num_samples=num_samples,
        latent_dim=int(config["model"]["latent_dim"]),
        batch_size=batch_size,
        device=device,
    )
    out_path = out_dir / f"{variant.name}.npy"
    np.save(out_path, x)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root that variant config and run paths resolve against.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Vectors to draw from each variant.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-vectors", type=int, default=50000)
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top-divergent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    found, skipped = resolve_variants(VARIANTS, root)
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        raise SystemExit(
            "No variant has both a checkpoint and a run config on this machine. "
            "Copy them from the training box, or pass --root at the tree holding them."
        )

    torch.manual_seed(args.seed)
    specs = []
    for variant in found:
        print(f"sampling {variant.name} from {variant.run_dir}")
        path = generate_samples(
            variant, root, args.num_samples, args.batch_size, samples_dir
        )
        specs.append(f"{variant.name}={path}")

    report_args = argparse.Namespace(
        real_path=args.real_path,
        real_format=args.real_format,
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=args.output_dir,
        preprocess="l2",
        max_vectors=args.max_vectors,
        num_pairs=args.num_pairs,
        knn=args.knn,
        bins=args.bins,
        top_divergent=args.top_divergent,
        seed=args.seed,
        no_png=args.no_png,
        plotlyjs=args.plotlyjs,
    )
    report_path = eda_report.run(report_args)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
```

`preprocess` is fixed at `"l2"` rather than exposed: generator output is unit-norm, so the real data must be L2-normalized for the overlay to mean anything. `synthetic_format` is fixed at `"npy"` because this module writes those files itself.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_compare_variants.py -v
```

Expected: PASS.

- [ ] **Step 5: Smoke-test against a real checkpoint**

The repo ships one usable checkpoint at `runs/wgan_sift1m_smoke_improved/`. Point the driver at a temporary root that maps `v0` onto it:

```bash
python3 - <<'EOF'
import numpy as np, shutil
from pathlib import Path

src = Path("/home/fibonadithya/TIG/wgan-synthetic/runs/wgan_sift1m_smoke_improved")
root = Path("/tmp/cv_smoke")
dst = root / "runs/long_baseline"
dst.mkdir(parents=True, exist_ok=True)
for f in ("best_generator.pt", "run_config.yaml"):
    shutil.copy(src / f, dst / f)

rng = np.random.default_rng(0)
x = rng.normal(size=(2000, 128)).astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
np.save(root / "real.npy", x)
EOF

python3 -m src.eval.compare_variants \
  --root /tmp/cv_smoke \
  --real-path /tmp/cv_smoke/real.npy \
  --output-dir /tmp/cv_smoke/out \
  --num-samples 2000 --max-vectors 2000 --num-pairs 2000 \
  --no-png --plotlyjs cdn
```

Expected: `skipping v1: …`, `skipping v1_5: …`, `skipping v2: …` (those run dirs do not exist under the temp root), then `sampling v0 …`, then a `report: …` line. Confirm the report exists and names `v0`:

```bash
ls /tmp/cv_smoke/out
grep -c v0 /tmp/cv_smoke/out/*.html
```

Expected: a non-zero count. If the smoke checkpoint is absent, report that the smoke test could not run rather than claiming it passed.

- [ ] **Step 6: Run the full suite**

```bash
python3 -m pytest
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/eval/compare_variants.py tests/test_compare_variants.py
git commit -m "$(cat <<'EOF'
feat(eval): drive the EDA report across all four variants

eda_report could already overlay any number of synthetic sets, but the
invocation was retyped by hand each time and the legend was labelled by
file path. compare_variants resolves each variant's checkpoint, samples it,
and labels the overlay v0/v1/v1_5/v2 to match the configs and docs.

Variants whose checkpoints are not on this machine are skipped with a
message rather than aborting -- they usually live on the training box, and
a partial comparison is still worth reading.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Documentation split and variant table

**Files:**
- Create: `docs/superpowers/README.md`
- Modify: `README.md`, `PROJECT_DOCUMENTATION.md`, `data/README.md`
- Modify: `docs/superpowers/plans/2026-07-31-ann-difficulty-panels.md`, `docs/superpowers/plans/2026-07-31-sparse-generator.md`, `docs/superpowers/specs/2026-07-31-ann-difficulty-panels-design.md`, `docs/superpowers/specs/2026-07-31-sparse-generator-design.md`

**Interfaces:**
- Consumes: config paths from Task 4, `python -m src.eval.compare_variants` from Task 6.
- Produces: nothing consumed by later tasks.

Note the plan and spec files from *both* branches are present after the Task 1 merge — four `2026-07-31-*` files plus the two `2026-08-01-*` files created by this work.

- [ ] **Step 1: Add the doc map to `README.md`**

Insert immediately after the opening description paragraph, before `## What this project provides`:

```markdown
## Documentation map

Human-maintained, and the source of truth:

- `README.md` — this file. Setup and the commands you run day to day.
- `PROJECT_DOCUMENTATION.md` — technical reference: architecture, training
  objective, data contract, evaluation, and the model variant table.
- `data/README.md` — the data contract and what the evaluation tools expect
  on disk.

AI working notes, kept for provenance and **not** authoritative:

- `docs/superpowers/` — design specs and implementation plans written by
  Claude during development. See `docs/superpowers/README.md`. Where these
  disagree with `PROJECT_DOCUMENTATION.md`, the latter wins.

## Model variants

Four variants were trained, each one config change from the previous:

| Variant | Delta | Config |
|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift_gan_v0.yaml` |
| `v1` | + generator EMA | `configs/sift_gan_v1.yaml` |
| `v1_5` | + pairwise-distance regularizer | `configs/sift_gan_v1_5.yaml` |
| `v2` | + gated non-negative generator | `configs/sift_gan_v2.yaml` |

Full detail, including which run directory holds each, is in
`PROJECT_DOCUMENTATION.md`. To see all four overlaid on real SIFT in one
report:

    python -m src.eval.compare_variants \
        --real-path data/sift_base.npy \
        --output-dir runs/eda_variants
```

- [ ] **Step 2: Add the Model variants section to `PROJECT_DOCUMENTATION.md`**

Insert a new section immediately after `## Model architecture` and before the data/evaluation sections:

```markdown
---

## Model variants

Four variants were trained. Each is exactly one config change from the one
above it, so a difference visible in an EDA overlay attributes to a single
cause.

| Variant | Delta from previous | Config | Runs |
|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift_gan_v0.yaml` | `long_baseline`, `bench_baseline` |
| `v1` | + generator EMA (`ema_decay: 0.999`) | `configs/sift_gan_v1.yaml` | `long_ema_only`, `x100k_ema_only` |
| `v1_5` | + distance reg (`distance_reg_alpha: 0.1`, 256 points) | `configs/sift_gan_v1_5.yaml` | `long_improved`, `x100k_improved`, `bench_improved` |
| `v2` | + gated generator (`generator_type: gated`) | `configs/sift_gan_v2.yaml` | `x100k_sparse_clamp4` |

Run length is an independent axis and is not a variant: `bench_*` are 3k
generator steps, `long_*` are 30k, `x100k_*` are 100k. The run directory
names predate this scheme and are kept as-is because the artifacts under
them are already named that way.

### Why v2 exists

Raw SIFT descriptors carry heavy mass at exactly zero. A dense MLP generator
cannot reproduce that support — it emits smooth values everywhere — and the
critic does not reliably penalize it, so Wasserstein estimates look
flattering while the marginals are plainly wrong. v2's generator multiplies a
softplus magnitude by a sampled binary gate, producing exact zeros. See
`src/models/generator.py` (`GatedGenerator`).

### `generator_type`

The architecture axis in the `model` config block, accepting `mlp` (default)
and `gated`. It sits underneath the variant numbering: v0, v1 and v1_5 all
use `mlp` and differ only in training settings.

Checkpoints do not record `generator_type` — the architecture is rebuilt from
the run config at load time. A checkpoint is therefore only loadable
alongside the `run_config.yaml` written next to it. Checkpoints do record
`generator_weights` (`"live"` or `"ema"`), which says which weights the file
holds, not which architecture produced them.
```

- [ ] **Step 3: Update the EDA section of `PROJECT_DOCUMENTATION.md`**

In the existing `- Distributional EDA report:` block, append after the bullet describing `--synthetic-path`:

```markdown
  - `src/eval/compare_variants.py` drives this across all four variants at
    once, labelling the overlays `v0`/`v1`/`v1_5`/`v2` to match the variant
    table. It resolves each variant's `best_generator.pt` and
    `run_config.yaml`, samples the generator, and calls the report in
    process. Variants whose checkpoints are not on the local machine are
    skipped with a message, so a partial comparison still produces a report.
```

and add a second command block after the existing `src.eval.eda_report` example:

````markdown
```bash
.venv/bin/python -m src.eval.compare_variants \
  --real-path data/sift_base.npy \
  --output-dir runs/eda_variants \
  --num-samples 100000
```
````

- [ ] **Step 4: Document the driver's expectations in `data/README.md`**

Append:

```markdown
## What the variant comparison expects

`python -m src.eval.compare_variants` reads, for each variant, a run
directory containing both:

- `best_generator.pt` — the checkpoint.
- `run_config.yaml` — written by the training script. Required, because the
  generator architecture is rebuilt from it; the checkpoint alone is not
  enough to reconstruct the model.

The directories it looks in are listed in the variant table in
`PROJECT_DOCUMENTATION.md`. Pass `--root` to point at a different tree.
Variants missing either file are skipped with a message rather than failing
the run, since checkpoints commonly live only on the training box.

Samples are written to `<output-dir>/samples/<variant>.npy` and reused as the
report's input, so they can be inspected independently.
```

- [ ] **Step 5: Create `docs/superpowers/README.md`**

```markdown
# AI working notes

Everything in this directory was generated by Claude during development:
design specs in `specs/`, implementation plans in `plans/`.

**These are not the source of truth.** They record what was intended at the
time of writing and are not updated as the code changes. Where any file here
disagrees with `PROJECT_DOCUMENTATION.md`, `README.md`, or the code itself,
those win.

They are kept because the reasoning behind a decision is often more useful
than the decision, and that reasoning is not recoverable from a diff.

## Contents

- `specs/` — design documents, written before implementation. Each states a
  problem, the options considered, and the option chosen.
- `plans/` — task-by-task implementation plans derived from a spec.

## Related, but not tracked

`.superpowers/sdd/` at the repo root holds per-task briefs, reports, and
review diffs produced while executing a plan. It is gitignored: it is
tooling state scoped to a single execution run, superseded by the commits it
produced, and of no use to anyone reading the repo later.
```

- [ ] **Step 6: Banner the four existing spec and plan files**

Prepend to each of `docs/superpowers/plans/2026-07-31-ann-difficulty-panels.md`, `docs/superpowers/plans/2026-07-31-sparse-generator.md`, `docs/superpowers/specs/2026-07-31-ann-difficulty-panels-design.md`, and `docs/superpowers/specs/2026-07-31-sparse-generator-design.md`, above the existing `# ` heading:

```markdown
> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

```

The two `2026-08-01-*` files already carry this banner — do not double it. Verify all six:

```bash
head -1 docs/superpowers/specs/*.md docs/superpowers/plans/*.md
```

Expected: every file's first line is `> **AI-generated working note.** ...`.

- [ ] **Step 7: Check the docs against reality**

Every path, config name, command and value quoted in the docs must exist:

```bash
ls configs/sift_gan_v0.yaml configs/sift_gan_v1.yaml configs/sift_gan_v1_5.yaml configs/sift_gan_v2.yaml
python3 -m src.eval.compare_variants --help
grep -rn "sparse" README.md PROJECT_DOCUMENTATION.md data/README.md
```

The `grep` should return only prose about *sparsity as a property* (SIFT's zeros, the gate producing sparse output), never `generator_type: sparse` or `SparseGenerator`. Fix any stale hit.

- [ ] **Step 8: Commit**

```bash
git add README.md PROJECT_DOCUMENTATION.md data/README.md docs/superpowers/
git commit -m "$(cat <<'EOF'
docs: split human docs from AI notes, document the four variants

README and PROJECT_DOCUMENTATION are the source of truth and now carry the
variant table; docs/superpowers is marked as AI working notes that lose to
them on any disagreement. Documents why v2's gated generator exists, that
generator_type is mlp | gated, and that a checkpoint needs its run_config to
be loadable at all.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Full suite green**

```bash
python3 -m pytest
```

- [ ] **No stale identifiers anywhere**

```bash
grep -rn "SparseGenerator" src/ tests/ configs/ README.md PROJECT_DOCUMENTATION.md data/README.md
grep -rn "generator_type: sparse" src/ tests/ configs/
```

Expected: no output from either.

- [ ] **The parallel worktree is gone**

```bash
git worktree list
git branch
```

Expected: no `sparse-generator` worktree, no `worktree-sparse-generator` branch.

- [ ] **The branch is clean and ready to PR**

```bash
git status --short
git log --oneline main..HEAD | head -20
```

Expected: clean tree; the log shows the merge plus five commits from Tasks 3–7 on top of the EDA work.
