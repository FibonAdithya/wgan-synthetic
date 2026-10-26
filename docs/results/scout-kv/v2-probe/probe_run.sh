#!/bin/bash
# Train, sample and score v2-probe runs one after another (one GPU card; never in parallel).
# Each run writes 8 synthetic books, scored by the unchanged gate (measure-real),
# locality.py, hub_position.py and probe.py intrinsic.
# Usage: probe_run.sh STEPS LAYER:ARM:SEED [LAYER:ARM:SEED ...]
set -u
cd /workspace/repo_v1
export PYTHONPATH=$PWD OMP_NUM_THREADS=15
PY=/workspace/kvenv/bin/python
CAP=/workspace/kv/capture
P=docs/results/scout-kv/v2-probe/probe.py
STEPS=$1; shift
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD) steps=$STEPS runs=$*"
for spec in "$@"; do
  IFS=: read -r L A S <<< "$spec"
  R=/workspace/kv/v2probe/L$L/${A}_s$S
  mkdir -p "$(dirname $R)"
  t0=$(date +%s)
  if [ ! -f $R/final.pt ]; then
    $PY $P train --arm $A --layer $L --seed $S --steps $STEPS --lambda-gp 10 --capture-dir $CAP --out $R \
      > $R.train.log 2>&1 || { echo "FAILED train $spec"; exit 1; }
  fi
  echo "TRAINED $spec seconds=$(( $(date +%s) - t0 ))"
  [ -d $R/capture ] || $PY $P sample --run $R --capture-dir $CAP || { echo "FAILED sample $spec"; exit 1; }
  if [ ! -f $R/gate.jsonl ]; then
    rm -f $R/gate.jsonl.tmp
    $PY -m src.eval.kv_gate measure-real --capture-dir $R/capture --books 12-19 --layers $L \
      --seed 0 --out $R/gate.jsonl.tmp && mv $R/gate.jsonl.tmp $R/gate.jsonl || { echo "FAILED gate $spec"; exit 1; }
  fi
  [ -f $R/locality.json ] || $PY docs/results/scout-kv/locality/locality.py --capture-dir $R/capture \
    --layers $L --out $R/locality.json || { echo "FAILED locality $spec"; exit 1; }
  [ -f $R/hub_position.json ] || $PY docs/results/scout-kv/v1-diagnostics/hub_position.py \
    --capture-dir $R/capture --layers $L --out $R/hub_position.json || { echo "FAILED hubs $spec"; exit 1; }
  [ -f $R/intrinsic.jsonl ] || $PY $P intrinsic --run $R || { echo "FAILED intrinsic $spec"; exit 1; }
  echo "DONE $spec seconds=$(( $(date +%s) - t0 ))"
done
echo "PROBE_COMPLETE $(date -u +%FT%TZ)"
