#!/usr/bin/env bash
# Box run for the scout_kv v3 real-data test, on tig-gpu-alt, from the repo root.
#   setsid nohup bash docs/results/scout-kv/v3-surgery/run.sh > /workspace/kv/v3surgery.log 2>&1 &
# Both steps skip finished work, so rerunning after a crash resumes.
set -euo pipefail
PY=/workspace/kvenv/bin/python
CAP=/workspace/kv/capture
OUT=/workspace/kv/v3surgery
export PYTHONPATH=$PWD
mkdir -p "$OUT"
echo "commit $(git rev-parse HEAD)"
sha256sum src/kv/surgery.py src/eval/kv_gate.py src/kv/neighbours.py src/kv/corpus.py \
    src/kv/search_profile.py src/eval/ann_difficulty.py \
    docs/results/scout-kv/v3-surgery/v2_neighbours.py
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
date -u +%FT%TZ

t0=$SECONDS
$PY docs/results/scout-kv/v3-surgery/v2_neighbours.py --out-dir "$OUT/v2_neighbours"
echo "DONE v2_neighbours seconds=$((SECONDS - t0))"

t0=$SECONDS
$PY -m src.kv.surgery --capture-dir "$CAP" --layers 11,15,43 --seeds 0,1 --out-dir "$OUT"
echo "DONE surgery seconds=$((SECONDS - t0))"
date -u +%FT%TZ
echo V3_SURGERY_COMPLETE
