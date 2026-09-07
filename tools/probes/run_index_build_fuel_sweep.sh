#!/usr/bin/env bash
# Fuel of an IVF-Flat index build over the real corpora, one process per point
# (one CUDA context per process). Meant to run as a single gpuq job that holds
# the card; writes one JSON per point plus a manifest of exit codes, because a
# point that dies must show up as a row, not as a silently missing file.
#
#   gpuq submit --project wgan-synthetic --commit "$(git rev-parse HEAD)" \
#     --branch "$(git rev-parse --abbrev-ref HEAD)" --lane gpu --timeout-s 7200 \
#     --artifact docs/results/index-build-fuel \
#     -- bash tools/probes/run_index_build_fuel_sweep.sh
set -u
OUT=${OUT:-docs/results/index-build-fuel}
PY=${PY:-/opt/venvs/wgan-synthetic/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'corpus\tn\tn_lists\trc\twall_s\tfile\n' > "$MANIFEST"

run() { # corpus path n nlist
  local corpus=$1 path=$2 n=$3 nlist=$4
  local stem="$OUT/${corpus}_n${n}_nlist${nlist}"
  local t0
  t0=$(date +%s)
  "$PY" tools/probes/index_build_fuel_probe.py --real-path "$path" --corpus "$corpus" \
      --n "$n" --n-lists "$nlist" --output "$stem.json" > "$stem.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$corpus" "$n" "$nlist" "$rc" "$(( $(date +%s) - t0 ))" "$stem.json" >> "$MANIFEST"
  echo "$corpus n=$n nlist=$nlist rc=$rc $(( $(date +%s) - t0 ))s"
}

SIFT=/workspace/data-cache/sift_1m.npy
GLOVE=/workspace/data-cache/hdf5/glove-100-angular.hdf5      # train: 1,183,514 x 100
DEEP=/workspace/data-cache/hdf5/deep-image-96-angular.hdf5   # train: 9,990,000 x 96

for nlist in 1024 4096; do
  for n in 250000 500000 1000000; do run sift "$SIFT" "$n" "$nlist"; done
  for n in 250000 500000 1183514; do run glove "$GLOVE" "$n" "$nlist"; done
  for n in 250000 1000000 4000000 9990000; do run deep "$DEEP" "$n" "$nlist"; done
done
# n_lists scaling, for the 1B extrapolation where sqrt(n) rules put n_lists near 32k
for n in 1000000 9990000; do run deep "$DEEP" "$n" 16384; done
echo "sweep done"
