#!/usr/bin/env bash
# One gpuq job: continue NYTimes v3 (seed 42) from its step-30,000 checkpoint
# to 100,000 steps, sample the final and selected checkpoints, and measure
# them against the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v3-100k \
#     --lane gpu --timeout-s 18000 -- bash scripts/nytimes_v3_seed42_100k_job.sh
#
# Do NOT pass --vram-mb: declaring VRAM lets the scheduler admit a second job
# onto the card, which train_wgan_gp's per-card lock then blocks until it dies
# with GpuBusyError after 1800 s.
#
# The timeout is sized from the 30k run: 68 minutes for 30,000 steps plus
# sampling and the report, so 70,000 more steps is about 160 minutes; 18,000 s
# is the margin the v2c continuation had.
#
# runs/ is gitignored, so nothing here is declared as a --artifact. Instead
# runs/nytimes is a symlink to /workspace/nytimes-v3, so every checkpoint is
# written straight to the box's persistent disk and a crash partway through
# leaves them behind (the v3 job did this with a submit-time wrapper). The
# run lands at /workspace/nytimes-v3/v3_seed42_100k, beside the 30k run it
# resumes. Box-specific by construction, like the config it runs.
#
# The interpreter comes from the runner's project venv on PATH (gpuq puts
# it there for every job); WGAN_PYTHON overrides it when needed.
set -euo pipefail
P=${WGAN_PYTHON:-python}
RUN=runs/nytimes/v3_seed42_100k
RESUME=/workspace/nytimes-v3/v3_seed42/checkpoint_step_30000.pt
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v3/v3_seed42_100k
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN" && test -f "$RESUME"
# Refuse to write over an earlier continuation's results.
test ! -e "$KEEP"
mkdir -p runs && ln -sfn /workspace/nytimes-v3 runs/nytimes
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v3_seed42_100k.yaml --resume "$RESUME"
# A resume restores best_score from the 30k run (0.108758 at step 9,000, under
# select_on: gate). If no later eval beats it, no new best_generator.pt is
# written and the selection is the 30k run's own file, so carry that one over
# rather than fail.
if [ ! -f "$RUN/best_generator.pt" ]; then
  cp /workspace/nytimes-v3/v3_seed42/best_generator.pt "$RUN/best_generator.pt"
  echo "best_generator.pt not beaten after resume; carried over from the 30k run"
fi
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_100000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step100000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v3_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v3_step100000=$RUN/synthetic_50k_step100000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
ls -la "$KEEP"
