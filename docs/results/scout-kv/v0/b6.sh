#!/bin/bash
# B6: gate v0 under both checkpoints, EMA weights in both cases.
set -u
cd /workspace/repo
export OMP_NUM_THREADS=15
PY=/workspace/kvenv/bin/python
OUT=/workspace/kv/v0
mkdir -p $OUT
RUNS=""
for L in 03 07 11 15 19 23 27 31 35 39 43 47; do
  RUNS="$RUNS --run-dir $((10#$L))=runs/scout_kv/v0/L$L"
done
for ck in best:best_generator.pt final:checkpoint_step_30000.pt; do
  name=${ck%%:*}; file=${ck#*:}
  t0=$(date +%s)
  $PY -m src.eval.kv_gate measure-run --capture-dir /workspace/kv/capture $RUNS \
    --checkpoint-name $file --ema --seed 0 --out $OUT/$name.jsonl || { echo "FAILED measure $name rc=$?"; exit 1; }
  echo "DONE measure $name seconds=$(( $(date +%s) - t0 ))"
  $PY -m src.eval.kv_gate check --records $OUT/$name.jsonl > $OUT/check_$name.json
  echo "CHECK $name exit=$?"
done
echo B6_MEASURE_COMPLETE
