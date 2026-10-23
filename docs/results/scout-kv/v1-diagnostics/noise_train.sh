#!/bin/bash
# Retrain-noise control for scout_kv v1: train v1 again at training seed SEED
# (default 1; the committed v1 runs used 42) for the given layers, one after
# another. The configs are configs/scout_kv/v1/L{LL}.yaml with only `seed` and
# `output_dir` changed, written to /workspace/kv/v1_noise/configs/ on the box,
# so no new rung appears under configs/. Intermediate checkpoints are pruned as
# in pilot_train.sh.
# Usage: noise_train.sh LAYER [LAYER ...]
set -u
cd /workspace/repo_v1
export OMP_NUM_THREADS=8
PY=/workspace/kvenv/bin/python
SEED=${SEED:-1}
CFG=/workspace/kv/v1_noise/configs
mkdir -p $CFG
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD) seed=$SEED layers=$*"
for L in "$@"; do
  L=$(printf %02d $((10#$L)))
  d=runs/scout_kv/v1_seed$SEED/L$L
  if [ -f $d/best_generator.pt ] && [ -f $d/checkpoint_step_30000.pt ] && [ ! -f $d/checkpoint_step_25000.pt ]; then
    echo "SKIP L$L already trained and pruned"; continue
  fi
  $PY - "$L" "$SEED" "$CFG" <<'EOF' || { echo "FAILED config L$L"; exit 1; }
import sys, yaml
L, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = yaml.safe_load(open(f"configs/scout_kv/v1/L{L}.yaml"))
c["seed"] = seed
c["output_dir"] = f"runs/scout_kv/v1_seed{seed}/L{L}"
yaml.safe_dump(c, open(f"{out}/L{L}_seed{seed}.yaml", "w"), sort_keys=False)
EOF
  avail=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' ')
  if [ "$avail" -lt 6 ]; then echo "DISK_SHORT before L$L avail=${avail}G"; exit 3; fi
  t0=$(date +%s)
  $PY -m src.train.train_wgan_gp --config $CFG/L${L}_seed$SEED.yaml > /workspace/logs/v1_noise_seed${SEED}_L$L.log 2>&1 || { echo "FAILED L$L rc=$?"; exit 1; }
  t1=$(date +%s)
  for f in best_generator.pt checkpoint_step_30000.pt run_config.yaml run_metadata.json; do
    [ -f $d/$f ] || { echo "MISSING L$L $f"; exit 2; }
  done
  rm -f $d/checkpoint_step_5000.pt $d/checkpoint_step_10000.pt $d/checkpoint_step_15000.pt $d/checkpoint_step_20000.pt $d/checkpoint_step_25000.pt
  echo "DONE L$L seconds=$((t1-t0))"
done
echo "NOISE_TRAIN_COMPLETE $(date -u +%FT%TZ)"
