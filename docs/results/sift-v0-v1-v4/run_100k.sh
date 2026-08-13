#!/usr/bin/env bash
# The v0/v1/v4 comparison, run-length matched at 100k generator steps.
#
# v1 needs no sampling: /workspace/keep/sift-ladder/samples_v1.npy is already a
# seed-42 draw from x100k_ema_only, which is the 100k rung. v0 and v4 are drawn
# here from their 100k checkpoints.
set -euo pipefail

OUT=/workspace/keep/sift-v0-v1-v4
LADDER=/workspace/keep/sift-ladder
export PYTHONPATH="$PWD"

# --- Cell 1: cpu copies of the two run configs, then sample --------------------
# Sampling on CPU keeps this job out of the GPU's way; it takes under a second.
python - <<'PY'
import yaml
from pathlib import Path

for src, dst in [
    ("/workspace/keep/sift-v0-v1-v4/v0_x100k/run_config.yaml",
     "/workspace/keep/sift-v0-v1-v4/v0_x100k_cpu.yaml"),
    ("/workspace/keep/v34-sift1m/v4_x100k/run_config.yaml",
     "/workspace/keep/sift-v0-v1-v4/v4_x100k_cpu.yaml"),
]:
    cfg = yaml.safe_load(Path(src).read_text())
    assert cfg["training"]["num_gen_steps"] == 100000, (src, cfg["training"]["num_gen_steps"])
    cfg["device"] = "cpu"
    Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {dst}")
PY

python -m src.sample.generate \
    --checkpoint "$OUT/v0_x100k/best_generator.pt" \
    --config "$OUT/v0_x100k_cpu.yaml" \
    --num-samples 20000 --seed 42 \
    --output-path "$OUT/samples_v0_100k.npy"

python -m src.sample.generate \
    --checkpoint /workspace/keep/v34-sift1m/v4_x100k/best_generator.pt \
    --config "$OUT/v4_x100k_cpu.yaml" \
    --num-samples 20000 --seed 42 \
    --output-path "$OUT/samples_v4_100k.npy"

# --- Cell 2: one eda_report over real and the three 100k rungs -----------------
python -m src.eval.eda_report \
    --real-path /workspace/data-cache/sift_1m.npy \
    --synthetic-path "v0=$OUT/samples_v0_100k.npy" \
    --synthetic-path "v1=$LADDER/samples_v1.npy" \
    --synthetic-path "v4=$OUT/samples_v4_100k.npy" \
    --output-dir "$OUT/eda100k" \
    --preprocess l2 \
    --max-vectors 50000 \
    --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --ivf-nlist 256 --seed 42

# --- Cell 3: distribution diagnostics against the real-vs-real floor -----------
python /workspace/keep/sift-v0-v1-v4/dist_diag.py

echo "=== done ==="
ls -la "$OUT/eda100k" "$OUT/eda100k/png"
