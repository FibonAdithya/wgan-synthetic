#!/usr/bin/env bash
# One gpuq job: retrain NYTimes v0 (the base WGAN) for 100k steps, then measure
# its step-30,000 and step-100,000 checkpoints and the surviving v3 selection in
# ONE report against the cleaned corpus, and emit the binned distribution data
# the write-up's figures need.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch writeup-latex \
#     --lane gpu --timeout-s 21600 -- bash scripts/nytimes_v0_x100k_panels_job.sh
#
# Why a retrain: the v0 30k and 100k runs wrote their checkpoints and samples
# under /workspace/nytimes-v0, which the 2026-09-16 rebuild wiped, so no
# distribution figure can be drawn from them. The committed summaries survive,
# so their numbers are still readable; the samples are not. The config is
# configs/nytimes/v0_seed42_100k.yaml, unchanged. The original 100k run was the
# 30k run resumed; this one trains the same 100k schedule from step 0, so steps
# 30,000 and 100,000 come from one run. CUDA nondeterminism means the retrain
# will not reproduce the committed numbers exactly (same-seed runs diverge by
# step 250), which is why v0 and v3 are measured together here.
#
# v0 trains on the corpus as shipped, zero rows included, exactly as the
# original did (the zero-row drop at load arrived with v2). v3 is NOT retrained
# or re-sampled: its 50k sample file survived and is reused, pinned by hash.
#
# Do NOT pass --vram-mb: train_wgan_gp takes a per-card lock (src/train/gpu_lock.py),
# so a second job admitted beside this one would die with GpuBusyError.
#
# Timeout: the original run took 37 min for 30k steps plus 87 min for the next
# 70k on an RTX 3060. ESTIMATE (unverified) 2-3 h on the RTX 3060 Ti; 21,600 s
# leaves ample room.
#
# runs/nytimes is symlinked to /workspace/nytimes-v0-retrain, so checkpoints
# land on persistent disk as they are written and survive a crash; nothing is
# declared as a --artifact because runs/ is gitignored and commit_artifacts is
# off. The two small JSONs are read back over ssh afterwards.
set -euo pipefail
P=${WGAN_PYTHON:-python}
RUN=runs/nytimes/v0_seed42_100k
KEEP=/workspace/nytimes-v0-retrain
RAW=/workspace/data-cache/nytimes_250k.npy
RAW_SHA=ba8e8d512cc465e983661cbbabba64ead8251c3f313de86c3b1c3abe56c03428
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
CLEAN_SHA=edcff7ab374bc345b2340ee5cd3f1186a3c48d6502b69dde59e336c34e354809
V3=/workspace/nytimes-v3/live/nytimes/v3_seed42/synthetic_50k_best.npy
V3_SHA=0c06596a5f51d236319ea39434ca62c962699cac2119cacba6b1e1f031b14ee3
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

echo "$RAW_SHA  $RAW" | sha256sum -c -
echo "$CLEAN_SHA  $CLEAN" | sha256sum -c -
echo "$V3_SHA  $V3" | sha256sum -c -
# Refuse to write over an earlier retrain's results.
test ! -e "$KEEP/v0_seed42_100k"
mkdir -p runs "$KEEP" && ln -sfn "$KEEP" runs/nytimes

"$P" -m src.train.train_wgan_gp --config configs/nytimes/v0_seed42_100k.yaml

"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_100000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step100000.npy"

# One report over all three, so the base and the spherical generator are a
# paired comparison against the same real draw.
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v0_step30000=$RUN/synthetic_50k_step30000.npy" \
  --synthetic-path "v0_step100000=$RUN/synthetic_50k_step100000.npy" \
  --synthetic-path "v3_best=$V3" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn

# Binned distribution data for the write-up figures. Small JSON, so it can be
# read back over ssh instead of moving the 50 MB sample files.
"$P" scripts/nytimes_dist_panels.py --real-path "$CLEAN" \
  --synthetic-path "v0_step30000=$RUN/synthetic_50k_step30000.npy" \
  --synthetic-path "v0_step100000=$RUN/synthetic_50k_step100000.npy" \
  --synthetic-path "v3_best=$V3" \
  --out "$RUN/panels.json"

sha256sum "$RUN"/checkpoint_step_30000.pt "$RUN"/checkpoint_step_100000.pt \
  "$RUN"/eda_clean/summary.json "$RUN"/panels.json
ls -l "$RUN"/eda_clean/summary.json "$RUN"/panels.json
