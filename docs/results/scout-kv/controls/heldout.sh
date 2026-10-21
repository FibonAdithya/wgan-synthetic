#!/bin/bash
# B11 (held-out plan, docs/ai/plans/2026-10-21-scout-kv-held-out.md): rebuild and
# held_out_tail on layers 11, 19, 43. pair_s0 + pair_check is R6a (the seed-0
# rebuild must pair with reference/seed0); s5..s8 are the four draws.
# Rerunnable like b7.sh: a finished step leaves $OUT/.done.NAME and is skipped.
set -u
REPO_DIR=${REPO_DIR:-/workspace/repo}
PY=${PY:-/workspace/kvenv/bin/python}
CAP=${CAP:-/workspace/kv/capture}
OUT=${OUT:-/workspace/kv/controls/heldout}
cd "$REPO_DIR" || exit 1
export OMP_NUM_THREADS=8
export PYTHONPATH="$REPO_DIR"
mkdir -p "$OUT"
H=docs/results/scout-kv/controls

# Record which code ran.
sha256sum src/eval/kv_gate.py src/kv/controls.py src/kv/corpus.py src/kv/transforms.py \
  src/kv/search_profile.py $H/heldout.py $H/check_pairing.py $H/heldout.sh

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

step pair_s0 pair_s0/rebuild.jsonl -- \
  $PY $H/heldout.py --capture-dir $CAP --controls rebuild --seed 0 --query-books 0-7 \
  --out-dir $OUT/pair_s0 --threads 15
step pair_check pair_check.json -- \
  bash -c "$PY $H/check_pairing.py --records $OUT/pair_s0/rebuild.jsonl --layers 11,19,43 > $OUT/pair_check.json"

for d in s5:0-7 s6:4-11 s7:0,1,2,3,8,9,10,11 s8:0,1,2,4,5,6,8,10; do
  name=${d%%:*}; qb=${d#*:}; seed=${name#s}
  step $name $name/rebuild.jsonl $name/held_out_tail.jsonl -- \
    $PY $H/heldout.py --capture-dir $CAP --seed $seed --query-books $qb \
    --out-dir $OUT/$name --threads 15
done
echo HELDOUT_COMPLETE
