#!/usr/bin/env bash
# One gpuq job: train NYTimes v0 (seed 42), sample it, and measure it against
# the real corpus both as shipped and with zero/duplicate rows removed.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-eda \
#     --lane gpu -- bash scripts/nytimes_v0_seed42_job.sh
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v0 at the end instead, the way the
# GloVe seed sweep did. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v0_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v0/v0_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v0_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$REAL" --synthetic-path "v0=$RUN/synthetic_50k.npy" \
  --output-dir "$RUN/eda_as_shipped" $CANON --no-png --plotlyjs cdn
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" --synthetic-path "v0=$RUN/synthetic_50k.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
"$P" -m src.eval.check_gate --dataset nytimes --run-dir "$RUN/eda_as_shipped" --stats-name v0 --allow-unset || true
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
