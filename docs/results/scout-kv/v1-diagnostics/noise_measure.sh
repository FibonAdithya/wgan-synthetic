#!/bin/bash
# Gate the seed-1 v1 retrains (noise_train.sh) at sampling seed 0, EMA, best and
# final, one layer per measure-run, matching docs/results/scout-kv/v1/full_measure.sh.
# Compare with docs/results/scout-kv/v1/full/v1_{best,final}_seed0.jsonl (training seed 42).
# Usage: noise_measure.sh LAYER [LAYER ...]
set -u
cd /workspace/repo_v1
export OMP_NUM_THREADS=15
PY=/workspace/kvenv/bin/python
SEED=${SEED:-1}
RUNS=/workspace/repo_v1/runs/scout_kv/v1_seed$SEED
OUT=/workspace/kv/v1_noise/measure
mkdir -p $OUT
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD)"
for L in "$@"; do
  L=$(printf %02d $((10#$L)))
  for ck in best:best_generator.pt final:checkpoint_step_30000.pt; do
    name=${ck%%:*}; file=${ck#*:}; part=$OUT/v1s${SEED}_${name}_L$L.jsonl
    [ -f $part ] && { echo "SKIP $name L$L"; continue; }
    rm -f $part.tmp; t0=$(date +%s)
    $PY -m src.eval.kv_gate measure-run --capture-dir /workspace/kv/capture \
      --run-dir $((10#$L))=$RUNS/L$L --layers $((10#$L)) --checkpoint-name $file \
      --ema --seed 0 --out $part.tmp && mv $part.tmp $part || { echo "FAILED $name L$L"; exit 1; }
    echo "DONE $name L$L seconds=$(( $(date +%s) - t0 ))"
  done
done
cd $OUT && sha256sum *.jsonl > SHA256SUMS
echo "NOISE_MEASURE_COMPLETE $(date -u +%FT%TZ)"
