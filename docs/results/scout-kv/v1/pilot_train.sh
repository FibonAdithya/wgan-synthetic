#!/bin/bash
# B2: train the scout_kv v1 pilot layers (19, 43) one after another, pruning
# intermediate checkpoints as v0's B5 did (docs/results/scout-kv/training/b5.sh).
# Usage: pilot_train.sh [LAYER ...]   (default: 19 43; B5 passes the other ten)
set -u
cd /workspace/repo_v1
export OMP_NUM_THREADS=8
PY=/workspace/kvenv/bin/python
TAG=${TAG:-v1_pilot}
LAYERS=${*:-19 43}
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD) layers=$LAYERS"
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> /workspace/logs/${TAG}_gpu.csv; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER' EXIT
for L in $LAYERS; do
  L=$(printf %02d $((10#$L)))
  d=runs/scout_kv/v1/L$L
  if [ -f $d/best_generator.pt ] && [ -f $d/checkpoint_step_30000.pt ] && [ ! -f $d/checkpoint_step_25000.pt ]; then
    echo "SKIP L$L already trained and pruned"; continue
  fi
  avail=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' ')
  if [ "$avail" -lt 6 ]; then echo "DISK_SHORT before L$L avail=${avail}G"; exit 3; fi
  t0=$(date +%s)
  $PY -m src.train.train_wgan_gp --config configs/scout_kv/v1/L$L.yaml > /workspace/logs/${TAG}_L$L.log 2>&1 || { echo "FAILED L$L rc=$?"; exit 1; }
  t1=$(date +%s)
  for f in best_generator.pt checkpoint_step_30000.pt run_config.yaml run_metadata.json; do
    [ -f $d/$f ] || { echo "MISSING L$L $f"; exit 2; }
  done
  rm -f $d/checkpoint_step_5000.pt $d/checkpoint_step_10000.pt $d/checkpoint_step_15000.pt $d/checkpoint_step_20000.pt $d/checkpoint_step_25000.pt
  echo "DONE L$L seconds=$((t1-t0)) avail_after=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' ')G"
done
echo "${TAG}_COMPLETE $(date -u +%FT%TZ)"
