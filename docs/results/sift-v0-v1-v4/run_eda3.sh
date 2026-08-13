#!/usr/bin/env bash
# eda_report over real SIFT and three rungs: v0, v1, v4.
# Supersedes the four-overlay run; the v1_30k control was dropped.
set -euo pipefail

OUT=/workspace/keep/sift-v0-v1-v4
LADDER=/workspace/keep/sift-ladder

python -m src.eval.eda_report \
    --real-path /workspace/data-cache/sift_1m.npy \
    --synthetic-path "v0=$LADDER/samples_v0.npy" \
    --synthetic-path "v1=$LADDER/samples_v1.npy" \
    --synthetic-path "v4=$LADDER/samples_v4.npy" \
    --output-dir "$OUT/eda3" \
    --preprocess l2 \
    --max-vectors 50000 \
    --ann-max-rows 20000 \
    --ann-k 100 \
    --ann-hub-k 10 \
    --ivf-nlist 256 \
    --seed 42

echo "=== done ==="
ls -la "$OUT/eda3" "$OUT/eda3/png"
