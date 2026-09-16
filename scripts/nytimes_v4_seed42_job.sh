#!/usr/bin/env bash
# One gpuq job: train NYTimes v4 (seed 42), sample it, and measure it against
# the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v4 \
#     --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v4_seed42_job.sh
#
# Do NOT pass --vram-mb: declaring VRAM lets the scheduler admit a second job
# onto the card, which train_wgan_gp's per-card lock then blocks until it dies
# with GpuBusyError after 1800 s.
#
# Timeout from the v3 job, which took 68 minutes for the same 30,000 steps.
# The regulariser adds an O(n^2) pairwise distance over lid_reg_max_points=256
# rows per generator step, so allow margin; 10,800 s is the v3 job's.
#
# runs/ is gitignored, so nothing is declared as --artifact; the run directory
# is copied to /workspace/nytimes-v4 at the end instead.
set -euo pipefail
P=${WGAN_PYTHON:-/venv/main/bin/python}
RUN=runs/nytimes/v4_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v4/v4_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$P"
test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v4_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v4_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v4_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
