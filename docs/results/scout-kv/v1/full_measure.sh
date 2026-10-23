#!/bin/bash
# B6: gate v1 on all 12 layers (best and final, seed 0, EMA), re-measure v0 best
# at seeds 0 and 1 on this box, and compute per-flag intrinsic diagnostics.
# Runs alongside B5: each layer is measured as soon as its training is done and
# pruned. Per-layer parts are joined in GLOBAL_LAYERS order at the end.
# Threads: training uses 8, this uses 15, inside the box's 23-core quota.
set -u
cd /workspace/repo_v1
export OMP_NUM_THREADS=15
PY=/workspace/kvenv/bin/python
CAP=/workspace/kv/capture
V1=/workspace/repo_v1/runs/scout_kv/v1
V0=/workspace/repo/runs/scout_kv/v0
OUT=/workspace/kv/v1/full
mkdir -p $OUT/parts
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD)"
ready() {  # trained and pruned, as pilot_train.sh leaves a layer
  local d=$V1/L$1
  [ -f $d/best_generator.pt ] && [ -f $d/checkpoint_step_30000.pt ] && [ -f $d/run_metadata.json ] && [ ! -f $d/checkpoint_step_25000.pt ]
}
measure() {  # name root layer checkpoint seed
  local name=$1 root=$2 L=$3 file=$4 seed=$5 part=$OUT/parts/$1_L$3.jsonl
  [ -f $part ] && { echo "SKIP $name L$L"; return; }
  rm -f $part.tmp
  local t0=$(date +%s)
  $PY -m src.eval.kv_gate measure-run --capture-dir $CAP --run-dir $((10#$L))=$root/L$L \
    --layers $((10#$L)) --checkpoint-name $file --ema --seed $seed --out $part.tmp \
    && mv $part.tmp $part || { echo "FAILED $name L$L rc=$?"; exit 1; }
  echo "DONE $name L$L seconds=$(( $(date +%s) - t0 ))"
}
for L in 19 43 03 07 11 15 23 27 31 35 39 47; do
  while ! ready $L; do
    pgrep -f "pilot_train.sh" >/dev/null || { ready $L || { echo "TRAINING_GONE before L$L"; exit 4; }; }
    sleep 60
  done
  measure v1_best_seed0  $V1 $L best_generator.pt 0
  measure v1_final_seed0 $V1 $L checkpoint_step_30000.pt 0
  measure v0_best_seed0  $V0 $L best_generator.pt 0
  measure v0_best_seed1  $V0 $L best_generator.pt 1
  ip=$OUT/parts/intrinsic_L$L.jsonl
  if [ ! -f $ip ]; then
    rm -f $ip.tmp
    t0=$(date +%s)
    PYTHONPATH=. $PY docs/results/scout-kv/v1/intrinsic.py --capture-dir $CAP --runs $V1 \
      --layers $((10#$L)) --out $ip.tmp && mv $ip.tmp $ip || { echo "FAILED intrinsic L$L"; exit 1; }
    echo "DONE intrinsic L$L seconds=$(( $(date +%s) - t0 ))"
  fi
done
for name in v1_best_seed0 v1_final_seed0 v0_best_seed0 v0_best_seed1 intrinsic; do
  : > $OUT/$name.jsonl
  for L in 03 07 11 15 19 23 27 31 35 39 43 47; do cat $OUT/parts/${name}_L$L.jsonl >> $OUT/$name.jsonl; done
done
for name in v1_best_seed0 v1_final_seed0 v0_best_seed0; do
  $PY -m src.eval.kv_gate check --records $OUT/$name.jsonl > $OUT/check_$name.json
  echo "CHECK $name exit=$?"
done
cd $OUT && sha256sum *.jsonl check_*.json > SHA256SUMS && cat SHA256SUMS
echo "FULL_MEASURE_COMPLETE $(date -u +%FT%TZ)"
