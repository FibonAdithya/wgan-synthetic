#!/usr/bin/env bash
# One gpuq job: rescale the trunk term of the NYTimes v2c checkpoints and
# measure the gate statistics of each rescaled sample. See
# tools/probes/trunk_scale_probe.py for what the numbers mean.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch probe/nytimes-trunk-scale \
#     --lane cpu --timeout-s 5400 -- bash scripts/nytimes_trunk_scale_probe_job.sh
#
# CPU lane: decoding 50,000 latents through a 3-layer MLP is trivial, and the
# cost is the brute-force 100-NN over 20,000 rows that ann_difficulty runs per
# set (3 checkpoints x 13 sets + real = 40 sets). The result is copied to
# /workspace because runs/ is gitignored and cannot be an --artifact.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
V2C=/workspace/nytimes-v2/v2c_seed42
V2C100K=/workspace/nytimes-v2/v2c_seed42_100k
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
OUT=runs/nytimes/probes/trunk_scale_v2c_eq.json
KEEP=/workspace/nytimes-v2/probes

test -f "$CLEAN" && test -f "$V2C/checkpoint_step_30000.pt" && test -f "$V2C100K/checkpoint_step_100000.pt"
"$P" tools/probes/trunk_scale_probe.py \
  --config "$V2C/run_config.yaml" --real-path "$CLEAN" \
  --checkpoint "best_step4000=$V2C/best_generator.pt" \
  --checkpoint "step30000=$V2C/checkpoint_step_30000.pt" \
  --checkpoint "step100000=$V2C100K/checkpoint_step_100000.pt" \
  --output "$OUT" --equalise
mkdir -p "$KEEP" && cp "$OUT" "$KEEP"/ && sha256sum "$KEEP"/trunk_scale_v2c_eq.json
