#!/bin/bash
# B5: train the other 11 scout_kv v0 layers one after another, pruning each as in B4.
set -u
cd /workspace/repo
export OMP_NUM_THREADS=8
PY=/workspace/kvenv/bin/python
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> /workspace/logs/b5_gpu.csv; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER' EXIT
for L in 07 11 15 19 23 27 31 35 39 43 47; do
  d=runs/scout_kv/v0/L$L
  if [ -f $d/best_generator.pt ] && [ -f $d/checkpoint_step_30000.pt ] && [ ! -f $d/checkpoint_step_25000.pt ]; then
    echo "SKIP L$L already trained and pruned"; continue
  fi
  avail=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' ')
  if [ "$avail" -lt 6 ]; then echo "DISK_SHORT before L$L avail=${avail}G"; exit 3; fi
  t0=$(date +%s)
  $PY -m src.train.train_wgan_gp --config configs/scout_kv/v0/L$L.yaml > /workspace/logs/b5_L$L.log 2>&1 || { echo "FAILED L$L rc=$?"; exit 1; }
  t1=$(date +%s)
  for f in best_generator.pt checkpoint_step_30000.pt run_config.yaml run_metadata.json; do
    [ -f $d/$f ] || { echo "MISSING L$L $f"; exit 2; }
  done
  rm -f $d/checkpoint_step_5000.pt $d/checkpoint_step_10000.pt $d/checkpoint_step_15000.pt $d/checkpoint_step_20000.pt $d/checkpoint_step_25000.pt
  echo "DONE L$L seconds=$((t1-t0)) avail_after=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' ')G"
done
echo B5_COMPLETE
