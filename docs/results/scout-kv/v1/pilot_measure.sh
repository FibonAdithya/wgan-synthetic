#!/bin/bash
# B3: gate the v1 pilot (L19, L43) at two sampling seeds, both checkpoint kinds,
# EMA weights, and re-measure v0 best at the same layers and seeds on this box
# (the committed v0/best.jsonl was measured on the A100; see the plan's B3).
set -u
cd /workspace/repo_v1
export OMP_NUM_THREADS=15
PY=/workspace/kvenv/bin/python
OUT=/workspace/kv/v1/pilot
CAP=/workspace/kv/capture
mkdir -p $OUT
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD)"
measure() {  # name run_root checkpoint seed
  local name=$1 root=$2 file=$3 seed=$4
  if [ -f $OUT/$name.jsonl ]; then echo "SKIP $name exists"; return; fi
  local t0=$(date +%s)
  $PY -m src.eval.kv_gate measure-run --capture-dir $CAP \
    --run-dir 19=$root/L19 --run-dir 43=$root/L43 --layers 19,43 \
    --checkpoint-name $file --ema --seed $seed --out $OUT/$name.jsonl \
    || { echo "FAILED $name rc=$?"; exit 1; }
  echo "DONE $name seconds=$(( $(date +%s) - t0 ))"
}
for seed in 0 1; do
  measure v1_best_seed$seed  /workspace/repo_v1/runs/scout_kv/v1 best_generator.pt $seed
  measure v1_final_seed$seed /workspace/repo_v1/runs/scout_kv/v1 checkpoint_step_30000.pt $seed
  measure v0_best_seed$seed  /workspace/repo/runs/scout_kv/v0    best_generator.pt $seed
done
cd $OUT && sha256sum *.jsonl > SHA256SUMS && cat SHA256SUMS
echo "PILOT_MEASURE_COMPLETE $(date -u +%FT%TZ)"
