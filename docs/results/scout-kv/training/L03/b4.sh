#!/bin/bash
cd /workspace/repo
export OMP_NUM_THREADS=8
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> /workspace/logs/b4_gpu.csv; sleep 10; done ) &
SAMPLER=$!
start=$(date +%s)
/workspace/kvenv/bin/python -m src.train.train_wgan_gp --config configs/scout_kv/v0/L03.yaml > /workspace/logs/b4_train.log 2>&1
rc=$?
end=$(date +%s)
kill $SAMPLER
echo "B4_EXIT rc=$rc wall_seconds=$((end-start)) start=$start end=$end" >> /workspace/logs/b4_train.log
