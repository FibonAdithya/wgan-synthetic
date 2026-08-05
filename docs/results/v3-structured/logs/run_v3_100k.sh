#!/bin/bash
cd /workspace/wgan-v3
export PYTHONPATH=/workspace/wgan-v3
export CUDA_VISIBLE_DEVICES=0
/venv/main/bin/python -u -m src.train.train_wgan_gp --config configs/x100k_structured.yaml
echo "=== V3_100K EXIT rc=$? ==="
