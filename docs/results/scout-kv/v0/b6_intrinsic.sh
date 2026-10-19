#!/bin/bash
while kill -0 39477 2>/dev/null; do sleep 30; done
cd /workspace/repo
export OMP_NUM_THREADS=15
t0=$(date +%s)
PYTHONPATH=. /workspace/kvenv/bin/python docs/results/scout-kv/v0/intrinsic.py --capture-dir /workspace/kv/capture --runs runs/scout_kv/v0 --out /workspace/kv/v0/intrinsic.jsonl
echo "INTRINSIC_EXIT rc=$? seconds=$(( $(date +%s) - t0 ))"
