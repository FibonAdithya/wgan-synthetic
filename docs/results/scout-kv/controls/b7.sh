#!/bin/bash
# B7: neighbour positions (all 12 layers), then the controls on layers 11, 19 and 43.
set -u
cd /workspace/repo
export OMP_NUM_THREADS=8
export PYTHONPATH=/workspace/repo
PY=/workspace/kvenv/bin/python
CAP=/workspace/kv/capture
OUT=/workspace/kv/controls
mkdir -p $OUT

step() {
  name=$1; shift
  t0=$(date +%s)
  "$@" || { echo "FAILED $name rc=$?"; exit 1; }
  echo "DONE $name seconds=$(( $(date +%s) - t0 ))"
}

step neighbours $PY docs/results/scout-kv/controls/neighbours.py \
  --capture-dir $CAP --seed 0 --out $OUT/neighbours.jsonl
step train_books_real $PY -m src.eval.kv_gate measure-real --capture-dir $CAP \
  --books 0-7 --layers 11,19,43 --seed 0 --threads 15 --out $OUT/train_books_real.jsonl
step controls $PY docs/results/scout-kv/controls/controls.py \
  --capture-dir $CAP --out-dir $OUT --seed 0 --threads 15
step v0_best_subset $PY -m src.eval.kv_gate measure-run --capture-dir $CAP --layers 11,19,43 \
  --run-dir 11=runs/scout_kv/v0/L11 --run-dir 19=runs/scout_kv/v0/L19 --run-dir 43=runs/scout_kv/v0/L43 \
  --ema --seed 0 --threads 15 --out $OUT/v0_best_subset.jsonl
echo B7_COMPLETE
