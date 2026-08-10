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

# This is a GPU training service: `device: auto` in the config silently
# resolves to CPU when CUDA is unavailable, and 30,000 steps on CPU runs for
# days instead of ~35 minutes, unattended, with nobody watching. A silent
# downgrade to CPU is never the desired behaviour here, so treat absent CUDA
# as a hard, loud failure instead of just skipping the VRAM cap below.
if not torch.cuda.is_available():
    raise SystemExit(
        "deep_train_service.sh: torch.cuda.is_available() is False -- no "
        "GPU visible to this process. Refusing to silently fall back to "
        "CPU training (30,000 steps would take days instead of ~35 "
        "minutes). Check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and the torch "
        "install on this box before retrying."
    )

# Hard ceiling at 25% of the card, ~20x the measured 71 MiB peak. A runaway
# allocation then fails this job rather than another agent's.
torch.cuda.set_per_process_memory_fraction(0.25)

with open(f"configs/deep/{variant}.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

checkpoint, meta = train(config)
print(f"done: {checkpoint}")
print(f"final metrics: {meta['metrics'][-1] if meta['metrics'] else 'none'}")
PYTHON
