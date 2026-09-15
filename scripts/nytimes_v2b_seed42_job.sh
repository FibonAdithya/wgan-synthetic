#!/usr/bin/env bash
# One gpuq job: train NYTimes v2b (seed 42), sample it, and measure it against
# the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v2b \
#     --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v2b_seed42_job.sh
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v2b at the end instead, the way
# the v2 job did. The sha256 lines at the end are what the laptop-side fetch
# verifies against. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v2b_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v2b/v2b_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v2b_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v2b_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v2b_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
(cd "$KEEP" && sha256sum run_config.yaml run_metadata.json eda_clean/summary.json)
