#!/usr/bin/env bash
# Train one deep variant under supervisor on tig-gpu.
#
# Run as a supervisor program, not a bare background process: a loose
# `python ... &` dies with the shell and its logs never reach the portal.
# Invoked as: deep_train_service.sh <variant>
set -euo pipefail

VARIANT="${1:?usage: deep_train_service.sh <v0|v1|v2>}"
WORK_DIR="/workspace/deep-gan"

cd "${WORK_DIR}"

# Courtesy preflight: other agents share this GPU. Report what is already
# resident so a heavy neighbour is visible in the log; the cap below is what
# actually protects them.
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Return freed blocks instead of holding a fragmented reserve.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /venv/main/bin/python - "${VARIANT}" <<'PYTHON'
import sys

import torch
import yaml

from src.train.train_wgan_gp import train

variant = sys.argv[1]

# Hard ceiling at 25% of the card, ~20x the measured 71 MiB peak. A runaway
# allocation then fails this job rather than another agent's.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)

with open(f"configs/deep_gan_{variant}.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

checkpoint, meta = train(config)
print(f"done: {checkpoint}")
print(f"final metrics: {meta['metrics'][-1] if meta['metrics'] else 'none'}")
PYTHON
