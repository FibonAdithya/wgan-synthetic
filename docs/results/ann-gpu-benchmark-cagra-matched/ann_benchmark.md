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
| real | cagra | 7.16 | 0.00 | 768.0 | 273,276.0 (floor @ recall 0.965) | 1.000 |
| real | cagra_iters | 5.67 | 0.00 | 768.0 | 467,925.8 | 0.966 |
| real | flat | 0.00 | 0.05 | 512.0 | 7,123.5 (exact ceiling) | 1.000 |
| real | ivf_flat | 1.79 | 0.00 | 512.0 | 91,128.0 | 0.999 |
