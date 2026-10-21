#!/bin/bash
# B9 (cross-book plan): same_book_tail and cross_book_tail on layers 11, 19, 43, for
# partitions a (key books 12-15) and b (key books 16-19) at seeds 0 and 1.
# Rerunnable like b7.sh: a finished step leaves $OUT/.done.NAME and is skipped.
set -u
REPO_DIR=${REPO_DIR:-/workspace/repo}
PY=${PY:-/workspace/kvenv/bin/python}
CAP=${CAP:-/workspace/kv/capture}
OUT=${OUT:-/workspace/kv/controls/xbook}
cd "$REPO_DIR" || exit 1
export OMP_NUM_THREADS=8
export PYTHONPATH="$REPO_DIR"
mkdir -p "$OUT"

# Record which code ran.
sha256sum src/eval/kv_gate.py src/kv/controls.py src/kv/corpus.py src/kv/transforms.py \
  docs/results/scout-kv/controls/controls.py docs/results/scout-kv/controls/xbook.sh

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

for part in a b; do
  if [ $part = a ]; then kb=12-15; else kb=16-19; fi
  for seed in 0 1; do
    d=${part}_s${seed}
    step $d $d/same_book_tail.jsonl $d/cross_book_tail.jsonl -- \
      $PY docs/results/scout-kv/controls/controls.py --capture-dir $CAP \
      --controls same_book_tail,cross_book_tail --key-books $kb --seed $seed \
      --out-dir $OUT/$d --threads 15
  done
done
echo XBOOK_COMPLETE
