#!/usr/bin/env bash
# One gpuq job: retrain SIFT v4 at 100k steps on sift_1m, sample its selected
# and final checkpoints, and measure both against real SIFT.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch sift-v4-gate \
#     --lane gpu --timeout-s 28800 -- bash scripts/sift_v4_x100k_job.sh
#
# Why a retrain: the 2026-08-10 run's checkpoints went with the box rebuilds,
# and a gate cannot admit a generator that no longer exists. The config is
# unchanged, so the committed numbers in
# docs/results/v4-logratio/eda_ladder_100k_summary.json are what this should
# land near (n=1 seed, and CUDA nondeterminism alone moves it).
#
# Do NOT pass --vram-mb: train_wgan_gp takes a per-card lock, so a second job
# admitted beside this one would die with GpuBusyError.
#
# Timeout: no wall time was recorded for the old run (RTX 4060). ESTIMATE
# (unverified) 3-4 h on the RTX 3060 Ti; 28,800 s leaves twice that.
#
# runs/sift is symlinked to /workspace/sift-v4, so checkpoints are written to
# persistent disk as they happen and survive a crash; nothing is declared as a
# --artifact because runs/ is gitignored.
set -euo pipefail
P=${WGAN_PYTHON:-python}
RUN=runs/sift/v4_sift1m_x100k
REAL=/workspace/data-cache/sift_1m.npy
EXPECT=4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d
KEEP=/workspace/sift-v4/v4_sift1m_x100k
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10"

echo "$EXPECT  $REAL" | sha256sum -c -
# Refuse to write over an earlier retrain's results.
test ! -e "$KEEP"
mkdir -p runs /workspace/sift-v4 && ln -sfn /workspace/sift-v4 runs/sift
"$P" -m src.train.train_wgan_gp --config configs/sift/v4_sift1m_x100k.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_100000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step100000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$REAL" \
  --synthetic-path "v4_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v4_step100000=$RUN/synthetic_50k_step100000.npy" \
  --output-dir "$RUN/eda" $CANON --no-png --plotlyjs cdn
sha256sum "$RUN"/best_generator.pt "$RUN"/checkpoint_step_100000.pt "$RUN"/eda/summary.json
