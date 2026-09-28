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
| real | cagra | 5.08 | 0.00 | 768.0 | 349,938.3 (floor @ recall 0.964) | 1.000 |
| v0 | cagra | 4.58 | 0.00 | 768.0 | 347,519.5 (floor @ recall 0.965) | 1.000 |
| v4 | cagra | 4.60 | 0.00 | 768.0 | 349,255.8 (floor @ recall 0.981) | 1.000 |
| real | cagra_iters | 4.81 | 0.00 | 768.0 | 613,181.4 | 0.966 |
| v0 | cagra_iters | 4.63 | 0.00 | 768.0 | 600,755.5 | 0.967 |
| v4 | cagra_iters | 4.64 | 0.00 | 768.0 | 729,427.0 | 0.982 |
| real | flat | 0.00 | 0.05 | 512.0 | 9,888.6 (exact ceiling) | 1.000 |
| v0 | flat | 0.00 | 0.05 | 512.0 | 9,883.3 (exact ceiling) | 1.000 |
| v4 | flat | 0.00 | 0.05 | 512.0 | 9,890.3 (exact ceiling) | 1.000 |
| real | ivf_flat | 1.33 | 0.00 | 512.0 | 113,182.5 | 0.999 |
| v0 | ivf_flat | 1.34 | 0.00 | 512.0 | 114,160.0 | 0.999 |
| v4 | ivf_flat | 1.33 | 0.00 | 512.0 | 124,712.7 | 0.999 |
| real | ivf_pq | 2.97 | 0.00 | 64.0 | not reached (peak recall 0.880) | 0.880 |
| v0 | ivf_pq | 2.93 | 0.00 | 64.0 | not reached (peak recall 0.879) | 0.879 |
| v4 | ivf_pq | 3.05 | 0.00 | 64.0 | not reached (peak recall 0.886) | 0.886 |
