#!/usr/bin/env bash
# One gpuq cpu-lane job: restage sift_1m.npy on the rebuilt box and measure the
# real-side noise floor the SIFT gate bands will be centred on.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch sift-v4-gate \
#     --lane cpu --timeout-s 7200 -- bash scripts/sift_data_job.sh
#
# The fetch is deterministic (seed-42 subset of the ann-benchmarks HDF5), and
# the 2026-08-19 refetch hashed to the value checked below. A mismatch means
# the corpus is not the one every earlier SIFT result was measured on, so the
# job stops rather than measure it.
#
# Results land in /workspace/sift-v4/ because runs/ and data/ are gitignored
# and the job's worktree is discarded.
set -euo pipefail
P=${WGAN_PYTHON:-python}
CACHE=/workspace/data-cache
REAL=$CACHE/sift_1m.npy
EXPECT=4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d
KEEP=/workspace/sift-v4

if [ ! -f "$REAL" ]; then
  "$P" -m src.data.fetch sift --rows 1000000 --cache-dir "$CACHE" --out-dir "$CACHE" --seed 42
fi
echo "$EXPECT  $REAL" | sha256sum -c -
mkdir -p "$KEEP"
PYTHONPATH=. "$P" scripts/sift_real_noise_floor.py --real-path "$REAL" --out "$KEEP/sift_noise_floor.json"
sha256sum "$KEEP/sift_noise_floor.json"
