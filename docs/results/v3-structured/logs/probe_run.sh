#!/bin/bash
cd /workspace/wgan-v3
export PYTHONPATH=/workspace/wgan-v3
echo "=== existing run adv_loss scale ==="
/venv/main/bin/python -c "
import json,glob,statistics
for p in sorted(glob.glob(\"/workspace/wgan-synthetic/runs/*/run_metadata.json\")):
    m=[x for x in json.load(open(p)).get(\"metrics\",[]) if \"adv_loss\" in x]
    if m:
        t=m[-50:]
        print(p.split(\"/\")[-2],\"laststep\",m[-1].get(\"step\"),\"|adv_loss|~\",round(statistics.mean(abs(x[\"adv_loss\"]) for x in t),4),\"distance_reg~\",round(statistics.mean(abs(x.get(\"distance_reg\",0)) for x in t),5))
"
echo "=== lid_reg scale probe ==="
/venv/main/bin/python tools/probes/lid_reg_scale_probe.py --config configs/sift_gan_v4.yaml --trials 80
echo "=== PROBE DONE rc=$? ==="
