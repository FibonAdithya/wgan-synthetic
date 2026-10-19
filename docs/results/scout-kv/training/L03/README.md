# scout_kv v0, layer 3: timed training run (B4)

`python -m src.train.train_wgan_gp --config configs/scout_kv/v0/L03.yaml`, run on tig-gpu
(A100-SXM4-80GB) on 2026-10-18 at commit 6c012e9 through `b4.sh`. The files here were copied
off the box by tar stream, and all six sha256 hashes matched on both sides. The checkpoints
stay on the box at `/workspace/repo/runs/scout_kv/v0/L03/`.

## Measurements

All MEASURED on the box.

| quantity | value | source |
|---|---|---|
| wall time, launch to exit | 2756 s (45.9 min) | `B4_EXIT` line in `b4_train.log` |
| data load before the first step | about 110 s | launch 20:28:10 UTC; GPU memory first rises at 20:30:00 in `b4_gpu.csv` |
| steps/s over the training phase | about 11.3, including the six evals | 30,000 steps in about 2646 s |
| peak GPU memory | 1683 MiB | largest `memory.used` in `b4_gpu.csv` |
| mean GPU utilisation | 64% | mean over the 264 samples taken while the model was loaded |
| best checkpoint | step 30,000, EMA weights, `select_on: gate`, best_score 0.7082 | `best_generator.pt` |
| training rows / holdout rows | 385,340 / 7,864 | `run_metadata.json` |
| exit code | 0 | `b4_train.log` |

The memory and utilisation figures come from `nvidia-smi` samples taken every 10 s, so a
peak shorter than 10 s would be missed. `memory.used` covers the whole card, including the
CUDA context.

## Thread setting

`b4.sh` exports `OMP_NUM_THREADS=8`. The first launch did not, and torch on the box then
starts 128 OpenMP threads against a cgroup quota of 15.36 cores. In that run, MEASURED:
all 128 threads were busy at about 12.6% CPU each, the main thread got 23%, the cgroup
was throttled in 100 of 100 scheduling periods over 10 s, GPU utilisation was about 17%,
and step 1,250 was the last line logged 628 s after training began. The log writes a line
every 250 steps, so that is 2.0-2.4 steps/s. The run was stopped after about 25 min. Its log
is `b4_train_omp128.log`. With 8 threads the same config ran at about 11.3 steps/s, 4.7-5.7
times faster.

## Disk after pruning

The run writes a 663 MB checkpoint every 5,000 steps (4.6 GB for the whole run). After the
intermediate checkpoints were deleted, `best_generator.pt`, `checkpoint_step_30000.pt`,
`run_config.yaml` and `run_metadata.json` remain: 1.3 GB.

`df -h /workspace` after pruning: 300G size, 286G used, 15G available (96%).

## Projection for the other 11 layers

ESTIMATE (unverified): 11 x 2756 s = about 8.4 h if run one after another, assuming the
other layers train at the same rate as layer 3.

ESTIMATE (unverified): disk does not fit as planned. Eleven more layers leave 11 x 1.3 GB =
14.6 GB of kept checkpoints, and each layer needs up to 4.6 GB while it runs, against 15 GB
free. The largest other use of the disk is `/workspace/models` (203 GB, the Scout weights).
Only the capture step needs it.
