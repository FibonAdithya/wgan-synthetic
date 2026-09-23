#!/usr/bin/env bash
# One gpuq job: retrain SIFT v0 at 100k steps on sift_1m, then measure it and
# the existing v4 100k checkpoint in ONE report, and emit the binned
# distribution data the write-up's figure needs.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch writeup-latex \
#     --lane gpu --timeout-s 28800 -- bash scripts/sift_v0_x100k_job.sh
#
# Why a retrain: the 2026-08-13 v0 100k run wrote its checkpoints to
# /workspace/keep, which the 2026-09-16 rebuild wiped, so no distribution
# figure can be drawn from them. See configs/sift/v0_sift1m_x100k.yaml for the
# config's provenance.
#
# Why v4 is re-sampled here rather than reusing its committed summary: measuring
# v0 and v4 in one eda_report makes the comparison paired. The committed
# numbers come from two separate reports and can only be compared through the
# real mean. v4 is NOT retrained -- its checkpoint survived and is reused.
#
# Do NOT pass --vram-mb: train_wgan_gp takes a per-card lock (src/train/gpu_lock.py),
# so a second job admitted beside this one would die with GpuBusyError.
#
# Timeout: the original v0 100k took 112 min on an RTX 4060. ESTIMATE
# (unverified) 2-3 h on the RTX 3060 Ti; 28,800 s leaves ample room.
#
# runs/sift is symlinked to /workspace/sift-v4, so checkpoints are written to
# persistent disk as they happen and survive a crash; nothing is declared as a
# --artifact because runs/ is gitignored. The two small JSONs are read back over
# ssh afterwards.
set -euo pipefail
P=${WGAN_PYTHON:-python}
V0=runs/sift/v0_sift1m_x100k
V4=runs/sift/v4_sift1m_x100k
REAL=/workspace/data-cache/sift_1m.npy
EXPECT=4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10"

echo "$EXPECT  $REAL" | sha256sum -c -
mkdir -p runs /workspace/sift-v4 && ln -sfn /workspace/sift-v4 runs/sift
# Refuse to write over an earlier v0 retrain's results.
test ! -e "$V0"
# The v4 checkpoint this job reuses must already be there.
test -f "$V4/best_generator.pt"
test -f "$V4/run_config.yaml"

"$P" -m src.train.train_wgan_gp --config configs/sift/v0_sift1m_x100k.yaml

"$P" -m src.sample.generate --checkpoint "$V0/best_generator.pt" --config "$V0/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$V0/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$V0/checkpoint_step_100000.pt" --config "$V0/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$V0/synthetic_50k_step100000.npy"
# v4 re-sampled from its surviving checkpoint at the same seed and count.
"$P" -m src.sample.generate --checkpoint "$V4/best_generator.pt" --config "$V4/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$V0/synthetic_50k_v4_best.npy"

# One report over all three, so v0 and v4 are a paired comparison.
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$REAL" \
  --synthetic-path "v0_best=$V0/synthetic_50k_best.npy" \
  --synthetic-path "v0_step100000=$V0/synthetic_50k_step100000.npy" \
  --synthetic-path "v4_best=$V0/synthetic_50k_v4_best.npy" \
  --output-dir "$V0/eda" $CANON --no-png --plotlyjs cdn

# Binned distribution data for the write-up figure. Small JSON, so it can be
# read back over ssh instead of moving the 25 MB sample files.
"$P" scripts/sift_dist_panels.py --real-path "$REAL" \
  --synthetic-path "v0=$V0/synthetic_50k_best.npy" \
  --synthetic-path "v4=$V0/synthetic_50k_v4_best.npy" \
  --out "$V0/panels.json"

sha256sum "$V0"/best_generator.pt "$V0"/checkpoint_step_100000.pt \
  "$V0"/eda/summary.json "$V0"/panels.json
ls -l "$V0"/eda/summary.json "$V0"/panels.json
