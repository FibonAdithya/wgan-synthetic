#!/usr/bin/env bash
# One gpuq job: decode the NYTimes v3 checkpoints at overridden shared radii
# and measure the gate statistics of each. See tools/probes/radius_probe.py
# for what the numbers mean.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch probe/nytimes-radius \
#     --lane cpu --timeout-s 7200 -- bash scripts/nytimes_radius_probe_job.sh
#
# CPU lane, and deliberately so: the probe takes no card lock, so it runs
# alongside whatever is training. Do NOT pass --vram-mb -- declaring VRAM lets
# the scheduler admit a second job onto the card, and train_wgan_gp's per-card
# lock then blocks it until it dies with GpuBusyError.
#
# Cost: decoding 50,000 latents through the trunk and tangent heads is trivial
# and happens once per checkpoint; the cost is the brute-force 100-NN over
# 20,000 rows that ann_difficulty runs per set (3 checkpoints x 9 sets + real
# = 28 sets). The precedent trunk-scale probe did 40 sets inside 5400s on the
# old box; 7200 here leaves room for the 10-core cgroup quota being shared
# with a GPU sweep's dataloader workers.
#
# The result is copied to /workspace because runs/ is gitignored and therefore
# cannot be declared as an --artifact.
set -euo pipefail
# The box rebuild of 2026-09-16 removed /opt/venvs/wgan-synthetic; the project
# venv is /venv/main. Overridable so a rebuild does not require a new commit.
P=${WGAN_PYTHON:-/venv/main/bin/python}
V3=/workspace/nytimes-v3/v3_seed42
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
OUT=runs/nytimes/probes/radius_v3.json
KEEP=/workspace/nytimes-v3/probes

test -f "$P"
test -f "$CLEAN"
test -f "$V3/run_config.yaml"
test -f "$V3/checkpoint_step_9000.pt"
test -f "$V3/checkpoint_step_20000.pt"
test -f "$V3/checkpoint_step_30000.pt"

# step9000 is the run's selected checkpoint (learned r 1.196), step30000 the
# drifted end (learned r 1.448), step20000 a mid-drift point (r 1.373) that
# turns the two-point contrast into a trend. 1.448 appears in --radii as well
# as being step30000's own learned r, so that checkpoint measures it twice;
# the two rows must agree exactly, which is a free consistency check.
"$P" tools/probes/radius_probe.py \
  --config "$V3/run_config.yaml" --real-path "$CLEAN" \
  --checkpoint "step9000=$V3/checkpoint_step_9000.pt" \
  --checkpoint "step20000=$V3/checkpoint_step_20000.pt" \
  --checkpoint "step30000=$V3/checkpoint_step_30000.pt" \
  --radii 1.10,1.20,1.25,1.30,1.40,1.448 \
  --device cpu \
  --output "$OUT"

mkdir -p "$KEEP" && cp "$OUT" "$KEEP"/ && sha256sum "$KEEP"/radius_v3.json
