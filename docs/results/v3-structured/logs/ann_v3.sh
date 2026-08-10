#!/bin/bash
cd /workspace/wgan-v3
export PYTHONPATH=/workspace/wgan-v3
export CUDA_VISIBLE_DEVICES=""
/venv/main/bin/python -u -m src.eval.eda_report \
  --real-path data/sift_base.npy \
  --synthetic-path samples_v3_30k.npy \
  --output-dir eda_v3_30k \
  --ann-k 100 --ann-hub-k 10 --ann-max-rows 20000 --ivf-nlist 256
echo "=== ANN DONE rc=$? ==="
