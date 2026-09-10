#!/usr/bin/env bash
# One gpuq job: continue NYTimes v0 (seed 42) from its step-30,000 checkpoint
# to 100,000 steps, sample the final and selected checkpoints, and measure
# them against the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-eda \
#     --lane gpu --timeout-s 14400 -- bash scripts/nytimes_v0_seed42_100k_job.sh
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v0 at the end instead, the way the
# GloVe seed sweep did. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v0_seed42_100k
RESUME=/workspace/nytimes-v0/v0_seed42/checkpoint_step_30000.pt
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v0/v0_seed42_100k
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN" && test -f "$RESUME"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v0_seed42_100k.yaml --resume "$RESUME"
# A resume restores best_cov from the 30k run (0.094 at step 1,000). If no
# later eval beats it, no new best_generator.pt is written and the selection
# is the 30k run's own file, so carry that one over rather than fail.
if [ ! -f "$RUN/best_generator.pt" ]; then
  cp /workspace/nytimes-v0/v0_seed42/best_generator.pt "$RUN/best_generator.pt"
  echo "best_generator.pt not beaten after resume; carried over from the 30k run"
fi
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_100000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step100000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v0_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v0_step100000=$RUN/synthetic_50k_step100000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
