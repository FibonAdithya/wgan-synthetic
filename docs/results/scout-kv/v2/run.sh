#!/bin/bash
# Train, sample and score scout_kv v2 runs one after another (one card; never in parallel).
# Usage: run.sh LAYER:SEED [LAYER:SEED ...]     e.g. run.sh 19:42 3:1 07:42
# Launch detached: nohup run.sh ... > /workspace/logs/NAME.log 2>&1 &
# Each step is skipped when its output exists, so a rerun after a failure resumes.
# Env: RUNS (/workspace/kv/v2), PY, CAP (/workspace/kv/capture), REPO, DISK_MIN_GB (10).
set -euo pipefail
REPO=${REPO:-/workspace/repo_v2}
PY=${PY:-/workspace/kvenv/bin/python}
CAP=${CAP:-/workspace/kv/capture}
RUNS=${RUNS:-/workspace/kv/v2}
DISK_MIN_GB=${DISK_MIN_GB:-10}
cd "$REPO"
export PYTHONPATH=$PWD
D=docs/results/scout-kv
mkdir -p "$RUNS/configs"
echo "START $(date -u +%FT%TZ) commit=$(git rev-parse HEAD) runs=$*"
for spec in "$@"; do
  IFS=: read -r L S <<< "$spec"
  LN=$((10#$L))
  LL=$(printf %02d "$LN")
  R=$RUNS/L${LL}_s${S}
  CFG=$RUNS/configs/L${LL}_s${S}.yaml
  t0=$(date +%s)
  [ -f "$CFG" ] || "$PY" "$D/v2/seed_config.py" "configs/scout_kv/v2/L${LL}.yaml" "$S" "$R" "$CFG" \
    || { echo "FAILED config $R"; exit 1; }
  if [ ! -f "$R/final.pt" ]; then
    avail=$(df --output=avail -B1G "$RUNS" | tail -1 | tr -d ' ')
    if [ "$avail" -lt "$DISK_MIN_GB" ]; then echo "DISK_SHORT before $R avail=${avail}G"; exit 3; fi
    OMP_NUM_THREADS=15 "$PY" -m src.train.train_book_latent --config "$CFG" > "$R.train.log" 2>&1 \
      || { echo "FAILED train $R"; exit 1; }
  fi
  echo "TRAINED $R seconds=$(( $(date +%s) - t0 ))"
  if [ ! -d "$R/capture" ]; then
    rm -rf "$R/capture.tmp"
    "$PY" -m src.sample.sample_books --run-dir "$R" --capture-dir "$CAP" \
      --out-dir "$R/capture.tmp" --seed 0 || { echo "FAILED sample $R"; exit 1; }
    mv "$R/capture.tmp" "$R/capture"
  fi
  if [ ! -f "$R/gate.jsonl" ]; then
    rm -f "$R/gate.jsonl.tmp"
    "$PY" -m src.eval.kv_gate measure-real --capture-dir "$R/capture" --books 12-19 --layers "$LN" \
      --seed 0 --source v2 --out "$R/gate.jsonl.tmp" || { echo "FAILED gate $R"; exit 1; }
    mv "$R/gate.jsonl.tmp" "$R/gate.jsonl"
  fi
  if [ ! -f "$R/locality.json" ]; then
    "$PY" "$D/locality/locality.py" --capture-dir "$R/capture" --layers "$LN" \
      --out "$R/locality.json" || { echo "FAILED locality $R"; exit 1; }
  fi
  if [ ! -f "$R/hub_position.json" ]; then
    "$PY" "$D/v1-diagnostics/hub_position.py" --capture-dir "$R/capture" --layers "$LN" \
      --out "$R/hub_position.json" || { echo "FAILED hub_position $R"; exit 1; }
  fi
  if [ ! -f "$R/intrinsic.jsonl" ]; then
    rm -f "$R/intrinsic.jsonl.tmp"
    "$PY" "$D/v2/intrinsic.py" --capture-dir "$R/capture" --layer "$LN" --source "v2_s${S}" \
      --out "$R/intrinsic.jsonl.tmp" || { echo "FAILED intrinsic $R"; exit 1; }
    mv "$R/intrinsic.jsonl.tmp" "$R/intrinsic.jsonl"
  fi
  echo "DONE $R seconds=$(( $(date +%s) - t0 ))"
done
echo "V2_COMPLETE $(date -u +%FT%TZ)"
