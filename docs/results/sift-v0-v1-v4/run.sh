#!/usr/bin/env bash
# Two cells, one job: sample the 30k EMA control, then one 4-way eda_report.
#
# Batched into a single job on purpose -- the box has a ~10-core cgroup quota
# and the queue lane is one slot, so two small cells sequentially beats two
# queued jobs. Every data path is absolute and outside the repo, because the
# runner gives each job a fresh worktree that dies with the job.
set -euo pipefail

OUT=/workspace/keep/sift-v0-v1-v4
LADDER=/workspace/keep/sift-ladder
EMA30K=/workspace/keep/wgan-synthetic/long_ema_only
CORPUS=/workspace/data-cache/sift_1m.npy

mkdir -p "$OUT"

# --- Cell 1: sample the run-length-matched v1 control -----------------------
# long_ema_only/run_config.yaml says `device: auto`, which resolve_device now
# rejects. Rewrite that one key rather than editing the recorded run config:
# the config beside a checkpoint is the record of what was trained.
python - <<'PY'
import yaml
from pathlib import Path

src = Path("/workspace/keep/wgan-synthetic/long_ema_only/run_config.yaml")
dst = Path("/workspace/keep/sift-v0-v1-v4/v1_30k_cpu.yaml")
cfg = yaml.safe_load(src.read_text())
assert cfg["training"]["num_gen_steps"] == 30000, cfg["training"]["num_gen_steps"]
assert cfg["training"]["ema_decay"] == 0.999, cfg["training"].get("ema_decay")
cfg["device"] = "cpu"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {dst}")
PY

python -m src.sample.generate \
    --checkpoint "$EMA30K/best_generator.pt" \
    --config "$OUT/v1_30k_cpu.yaml" \
    --num-samples 20000 \
    --seed 42 \
    --output-path "$OUT/samples_v1_30k.npy"

# --- Cell 2: one eda_report over real + four rungs --------------------------
# A single invocation is load-bearing: it gives every rung one shared
# real-side subsample, so rung-to-rung differences carry no sampling noise.
# The --ann-*/--ivf flags are already the canonical SIFT defaults; passed
# explicitly so this script records the measurement conditions itself.
python -m src.eval.eda_report \
    --real-path "$CORPUS" \
    --synthetic-path "v0=$LADDER/samples_v0.npy" \
    --synthetic-path "v1=$LADDER/samples_v1.npy" \
    --synthetic-path "v1_30k=$OUT/samples_v1_30k.npy" \
    --synthetic-path "v4=$LADDER/samples_v4.npy" \
    --output-dir "$OUT/eda" \
    --preprocess l2 \
    --max-vectors 50000 \
    --ann-max-rows 20000 \
    --ann-k 100 \
    --ann-hub-k 10 \
    --ivf-nlist 256 \
    --seed 42

echo "=== done ==="
ls -la "$OUT" "$OUT/eda"
