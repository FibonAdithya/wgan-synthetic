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
| real | cagra | 7.57 | 0.00 | 768.0 | 250,469.0 (floor @ recall 0.963) | 1.000 |
| v0 | cagra | 7.13 | 0.00 | 768.0 | 231,341.0 (floor @ recall 0.966) | 1.000 |
| v1 | cagra | 7.19 | 0.00 | 768.0 | 235,761.6 (floor @ recall 0.963) | 1.000 |
| v1_5 | cagra | 7.02 | 0.00 | 768.0 | 237,524.3 (floor @ recall 0.963) | 1.000 |
| v2 | cagra | 7.31 | 0.00 | 768.0 | 245,221.3 (floor @ recall 0.997) | 1.000 |
| v3 | cagra | 7.24 | 0.00 | 768.0 | 269,465.0 (floor @ recall 0.999) | 1.000 |
| v4 | cagra | 7.05 | 0.00 | 768.0 | 241,335.6 (floor @ recall 0.989) | 1.000 |
| real | flat | 0.00 | 0.09 | 512.0 | 7,996.2 (exact ceiling) | 1.000 |
| v0 | flat | 0.00 | 0.09 | 512.0 | 7,948.1 (exact ceiling) | 1.000 |
| v1 | flat | 0.00 | 0.09 | 512.0 | 7,982.4 (exact ceiling) | 1.000 |
| v1_5 | flat | 0.00 | 0.11 | 512.0 | 7,961.5 (exact ceiling) | 1.000 |
| v2 | flat | 0.00 | 0.09 | 512.0 | 7,953.0 (exact ceiling) | 1.000 |
| v3 | flat | 0.00 | 0.09 | 512.0 | 7,955.7 (exact ceiling) | 1.000 |
| v4 | flat | 0.00 | 0.09 | 512.0 | 7,966.1 (exact ceiling) | 1.000 |
| real | ivf_flat | 1.44 | 0.00 | 512.0 | 73,062.9 | 0.999 |
| v0 | ivf_flat | 1.45 | 0.00 | 512.0 | 71,585.8 | 0.999 |
| v1 | ivf_flat | 1.43 | 0.00 | 512.0 | 74,450.1 | 0.999 |
| v1_5 | ivf_flat | 1.44 | 0.00 | 512.0 | 73,743.5 | 0.999 |
| v2 | ivf_flat | 1.59 | 0.00 | 512.0 | 128,793.3 | 1.000 |
| v3 | ivf_flat | 1.48 | 0.00 | 512.0 | 224,924.8 | 1.000 |
| v4 | ivf_flat | 1.43 | 0.00 | 512.0 | 87,644.2 | 0.999 |
| real | ivf_pq | 3.18 | 0.00 | 64.0 | not reached (peak recall 0.881) | 0.881 |
| v0 | ivf_pq | 3.69 | 0.00 | 64.0 | not reached (peak recall 0.877) | 0.877 |
| v1 | ivf_pq | 3.58 | 0.00 | 64.0 | not reached (peak recall 0.879) | 0.879 |
| v1_5 | ivf_pq | 3.37 | 0.00 | 64.0 | not reached (peak recall 0.879) | 0.879 |
| v2 | ivf_pq | 3.43 | 0.00 | 64.0 | 42,066.0 | 0.906 |
| v3 | ivf_pq | 3.50 | 0.00 | 64.0 | 129,841.8 | 0.916 |
| v4 | ivf_pq | 3.33 | 0.00 | 64.0 | not reached (peak recall 0.894) | 0.894 |
