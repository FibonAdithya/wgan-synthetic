# scout_kv v0 training runs (B4, B5)

One WGAN-GP generator per global layer, from `configs/scout_kv/v0/L{LL}.yaml`, trained on
tig-gpu (A100-SXM4-80GB) at commit 6c012e9 with `OMP_NUM_THREADS=8`. L03 was the timed run
(B4, see `L03/README.md`). The other 11 ran one after another through `b5.sh` on
2026-10-18/19. Each `L{LL}/` holds `run_config.yaml`, `run_metadata.json` and the training
log (`train.log`; L03's is `b4_train.log`). The files were copied off the box by tar stream.
All 36 B5 sha256 hashes matched on both sides. The checkpoints stay on the box at
`/workspace/repo/runs/scout_kv/v0/`. Each layer keeps `best_generator.pt` and
`checkpoint_step_30000.pt`.

## Per layer

All MEASURED. Wall time is from `b5.log` (L03 from `L03/b4_train.log`). The best step and
score were read from each `best_generator.pt` on the box.

| layer | wall s | best checkpoint step | best gate score |
|---|---|---|---|
| 3 | 2756 | 30000 | 0.7082 |
| 7 | 2592 | 5000 | 0.1838 |
| 11 | 2567 | 5000 | 0.1147 |
| 15 | 2563 | 30000 | 0.3031 |
| 19 | 2568 | 30000 | 0.2977 |
| 23 | 2585 | 30000 | 0.4567 |
| 27 | 2593 | 30000 | 0.3856 |
| 31 | 2579 | 30000 | 0.4031 |
| 35 | 2605 | 30000 | 0.4631 |
| 39 | 2463 | 30000 | 0.4426 |
| 43 | 2609 | 30000 | 0.4490 |
| 47 | 2543 | 30000 | 0.3946 |

The 11 B5 layers took 28,267 s (7.85 h) in total. Every run reached step 30,000, with
385,340 training rows and 7,864 holdout rows. Over B5, `b5_gpu.csv` shows a peak of
1683 MiB and a mean utilisation of 67% while a model was loaded. Those figures come from
`nvidia-smi` samples taken every 10 s.

`best_generator.pt` holds EMA weights. `checkpoint_step_30000.pt` holds the live weights and
the EMA weights both. The best score is the trainer's selection score under `select_on: gate`,
and lower is better. It is not one of the five scout_kv gate statistics.

For 10 of the 12 layers the best checkpoint is step 30,000, so the B6 measurements of the
best and final checkpoints, both taken with `--ema`, use the same weights at those layers.
They differ only at layers 7 and 11.
