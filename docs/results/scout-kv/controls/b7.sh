#!/bin/bash
# B7: neighbour positions (all 12 layers), then the controls on layers 11, 19 and 43.
# Rerunnable: a finished step leaves $OUT/.done.NAME and is skipped; a failed step's
# outputs are moved to $OUT/failed-NAME-<epoch>/ so the rerun starts that step clean.
set -u
REPO_DIR=${REPO_DIR:-/workspace/repo}
PY=${PY:-/workspace/kvenv/bin/python}
CAP=${CAP:-/workspace/kv/capture}
OUT=${OUT:-/workspace/kv/controls}
cd "$REPO_DIR" || exit 1
export OMP_NUM_THREADS=8
export PYTHONPATH="$REPO_DIR"
mkdir -p "$OUT"

# Record which code ran.
sha256sum src/eval/kv_gate.py src/kv/controls.py src/kv/corpus.py src/kv/neighbours.py \
  src/kv/transforms.py docs/results/scout-kv/controls/controls.py \
  docs/results/scout-kv/controls/neighbours.py docs/results/scout-kv/controls/check_pairing.py \
  docs/results/scout-kv/controls/b7.sh

# step NAME OUTPUT... -- CMD...
step() {
  name=$1; shift
  outs=()
  while [ "$1" != "--" ]; do outs+=("$1"); shift; done
  shift
  if [ -e "$OUT/.done.$name" ]; then echo "SKIP $name"; return 0; fi
  t0=$(date +%s)
  "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    fdir="$OUT/failed-$name-$(date +%s)"
    mkdir -p "$fdir"
    for f in "${outs[@]}"; do [ -e "$OUT/$f" ] && mv "$OUT/$f" "$fdir/"; done
    echo "FAILED $name rc=$rc"
    exit 1
  fi
  touch "$OUT/.done.$name"
  echo "DONE $name seconds=$(( $(date +%s) - t0 ))"
}

step neighbours neighbours.jsonl -- $PY docs/results/scout-kv/controls/neighbours.py \
  --capture-dir $CAP --seed 0 --out $OUT/neighbours.jsonl
step train_books_real train_books_real.jsonl -- $PY -m src.eval.kv_gate measure-real --capture-dir $CAP \
  --books 0-7 --layers 11,19,43 --seed 0 --threads 15 --out $OUT/train_books_real.jsonl
step controls pooled_real.jsonl norm_equalised.jsonl gaussian.jsonl -- \
  $PY docs/results/scout-kv/controls/controls.py \
  --capture-dir $CAP --out-dir $OUT --seed 0 --threads 15
step v0_best_subset v0_best_subset.jsonl -- $PY -m src.eval.kv_gate measure-run --capture-dir $CAP --layers 11,19,43 \
  --run-dir 11=runs/scout_kv/v0/L11 --run-dir 19=runs/scout_kv/v0/L19 --run-dir 43=runs/scout_kv/v0/L43 \
  --ema --seed 0 --threads 15 --out $OUT/v0_best_subset.jsonl
echo B7_COMPLETE
