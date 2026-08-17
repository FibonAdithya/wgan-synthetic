# GPU ANN benchmark (target recall@10 = 0.90)

All corpora are L2-normalized; see the design note. These figures are
not comparable with published SIFT1M results. Build time is train and
add phases, timed separately. The flat/exact index has no swept knob;
its row reports the single measured QPS as the exact-search ceiling,
not an interpolated value at the target recall. A QPS figure marked
'floor' was not evaluated at the target: every measured point on
that curve already cleared it, so the fastest (lowest-recall) point
is reported at the recall it was actually measured at, not at the
target -- see `metrics.RecallPoint`.

| Corpus | Index | Train (s) | Add (s) | Index (MB, est.) | QPS @ recall 0.90 | Peak recall |
|---|---|---|---|---|---|---|
| real | flat | 0.00 | 0.08 | 512.0 | 7,984.5 (exact ceiling) | 1.000 |
| v0 | flat | 0.00 | 0.09 | 512.0 | 7,886.5 (exact ceiling) | 1.000 |
| v1 | flat | 0.00 | 0.09 | 512.0 | 7,894.0 (exact ceiling) | 1.000 |
| v1_5 | flat | 0.00 | 0.09 | 512.0 | 7,917.0 (exact ceiling) | 1.000 |
| v2 | flat | 0.00 | 0.09 | 512.0 | 7,916.0 (exact ceiling) | 1.000 |
| v3 | flat | 0.00 | 0.09 | 512.0 | 7,943.4 (exact ceiling) | 1.000 |
| v4 | flat | 0.00 | 0.09 | 512.0 | 7,947.1 (exact ceiling) | 1.000 |
| real | torch_flat | 0.00 | 0.16 | 516.0 | 5,559.1 (exact ceiling) | 1.000 |
| v0 | torch_flat | 0.00 | 0.10 | 516.0 | 5,558.5 (exact ceiling) | 1.000 |
| v1 | torch_flat | 0.00 | 0.10 | 516.0 | 5,556.9 (exact ceiling) | 1.000 |
| v1_5 | torch_flat | 0.00 | 0.10 | 516.0 | 5,552.9 (exact ceiling) | 1.000 |
| v2 | torch_flat | 0.00 | 0.10 | 516.0 | 5,559.1 (exact ceiling) | 1.000 |
| v3 | torch_flat | 0.00 | 0.10 | 516.0 | 5,558.6 (exact ceiling) | 1.000 |
| v4 | torch_flat | 0.00 | 0.11 | 516.0 | 5,555.3 (exact ceiling) | 1.000 |
| real | torch_flat_fp16 | 0.00 | 0.12 | 260.0 | 5,436.5 (exact ceiling) | 0.984 |
| v0 | torch_flat_fp16 | 0.00 | 0.10 | 260.0 | 5,433.9 (exact ceiling) | 0.985 |
| v1 | torch_flat_fp16 | 0.00 | 0.10 | 260.0 | 5,435.0 (exact ceiling) | 0.984 |
| v1_5 | torch_flat_fp16 | 0.00 | 0.11 | 260.0 | 5,436.8 (exact ceiling) | 0.985 |
| v2 | torch_flat_fp16 | 0.00 | 0.11 | 260.0 | 5,436.6 (exact ceiling) | 0.988 |
| v3 | torch_flat_fp16 | 0.00 | 0.10 | 260.0 | 5,435.9 (exact ceiling) | 0.985 |
| v4 | torch_flat_fp16 | 0.00 | 0.11 | 260.0 | 5,436.3 (exact ceiling) | 0.987 |
| real | torch_flat_tf32 | 0.00 | 0.12 | 516.0 | 5,904.7 (exact ceiling) | 0.993 |
| v0 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,902.4 (exact ceiling) | 0.997 |
| v1 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,901.0 (exact ceiling) | 0.997 |
| v1_5 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,904.2 (exact ceiling) | 0.997 |
| v2 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,903.8 (exact ceiling) | 0.998 |
| v3 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,901.4 (exact ceiling) | 0.997 |
| v4 | torch_flat_tf32 | 0.00 | 0.10 | 516.0 | 5,905.1 (exact ceiling) | 0.997 |
