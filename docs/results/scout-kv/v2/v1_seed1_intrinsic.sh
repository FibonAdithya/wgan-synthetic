#!/bin/bash
# v1 seed-1 body/tail LID at L19, L35, L43 (sampling only; uses the card briefly).
# Run from the repo root on the box, before run.sh, not during it.
set -euo pipefail
PY=${PY:-/workspace/kvenv/bin/python}
CAP=${CAP:-/workspace/kv/capture}
OUT=${OUT:-/workspace/kv/v2/v1_seed1_intrinsic.jsonl}
export PYTHONPATH=$PWD
mkdir -p "$(dirname "$OUT")"
"$PY" docs/results/scout-kv/v1/intrinsic.py --capture-dir "$CAP" \
  --runs /workspace/repo_v1/runs/scout_kv/v1_seed1 --layers 19,35,43 \
  --out "$OUT"
