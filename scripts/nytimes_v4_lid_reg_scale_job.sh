#!/usr/bin/env bash
# One gpuq job: size lid_reg_alpha for the NYTimes family, before v4 exists.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-v4 \
#     --lane cpu --timeout-s 3600 -- bash scripts/nytimes_v4_lid_reg_scale_job.sh
#
# cpu lane and --device cpu: this takes no card lock and must not allocate on
# the GPU, so it can run beside whatever is training. Do NOT pass --vram-mb.
#
# The sizing config is built here rather than committed, because alpha is what
# this job measures: committing a config with a guessed alpha first, then
# amending it, would leave a provisional number in history that someone could
# train from. The probe reads only batch_size, lid_reg_k, lid_reg_max_points,
# the model block, descriptor_dim, real_path and device, so v3_seed42 plus the
# two k keys is a faithful stand-in for v4.
set -euo pipefail
P=${WGAN_PYTHON:-/venv/main/bin/python}
V3=/workspace/nytimes-v3/v3_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CFG=/tmp/nytimes_v4_sizing.yaml
OUT=runs/nytimes/probes/lid_reg_scale.json
KEEP=/workspace/nytimes-v4/probes

test -f "$P"
test -f "$REAL"
test -f "$V3/checkpoint_step_9000.pt"
test -f configs/nytimes/v3_seed42.yaml

# real_path stays the RAW corpus, the same file the trainer loads: the penalty
# has to be sized against the distribution it will actually see. The trainer
# drops 197 exact-zero rows at load and keeps duplicates; this probe drops
# neither, but log_ratio_penalty's own survivor mask discards the degenerate
# rows (r_1 == 0, r_1 == r_k) from the estimate either way. Do not silently
# switch this to the cleaned corpus -- that file also has 31,949 duplicates
# removed, which changes a local-neighbourhood statistic.
"$P" - "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/nytimes/v3_seed42.yaml"))
cfg["training"]["lid_reg_k"] = 20
cfg["training"]["lid_reg_max_points"] = 256
yaml.safe_dump(cfg, open(sys.argv[1], "w"), sort_keys=False)
print(f"wrote {sys.argv[1]}")
PY

# --adv-loss 0.87 is v3's run-mean |adv_loss| (MEASURED, from the committed
# docs/results/nytimes-v3/run_metadata.json). Sizing at launch instead would
# use the first-20-step mean of 0.33 and oversize alpha roughly threefold; the
# penalty has to do its work in the collapse window, not at initialisation.
# The trained reference is v3's step-9,000 checkpoint for the same reason.
#
# Note: the probe caps its real pool at min(200000, rows), so this measures
# against 200,000 of the corpus's 250,000 rows, not all of them.
"$P" tools/probes/lid_reg_scale_probe.py \
  --config "$CFG" \
  --device cpu \
  --v2-checkpoint "$V3/checkpoint_step_9000.pt" \
  --v2-config configs/nytimes/v3_seed42.yaml \
  --adv-loss 0.87 \
  --target-fraction 0.05 \
  --output "$OUT"

mkdir -p "$KEEP" && cp "$OUT" "$KEEP"/ && sha256sum "$KEEP"/lid_reg_scale.json
