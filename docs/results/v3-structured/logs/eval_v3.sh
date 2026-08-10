#!/bin/bash
cd /workspace/wgan-v3
export PYTHONPATH=/workspace/wgan-v3
set -x
/venv/main/bin/python -u -m src.sample.generate \
  --checkpoint runs/sift_gan_v3/best_generator.pt \
  --config runs/sift_gan_v3/run_config.yaml \
  --num-samples 20000 --seed 7 \
  --output-path samples_v3_30k.npy
/venv/main/bin/python -u -c "
import numpy as np
x=np.load(\"samples_v3_30k.npy\"); r=np.load(\"data/sift_base.npy\", mmap_mode=\"r\")
rs=np.asarray(r[np.sort(np.random.default_rng(0).choice(r.shape[0],20000,replace=False))],dtype=np.float32)
for n,a in ((\"v3\",x),(\"real\",rs)):
    nz=(a>0).sum(1)
    print(f\"{n}: exact_zero_fraction {1-nz.mean()/a.shape[1]:.4f}  nnz mean {nz.mean():.2f} std {nz.std():.2f}\")
"
echo "=== EVAL DONE rc=$? ==="
