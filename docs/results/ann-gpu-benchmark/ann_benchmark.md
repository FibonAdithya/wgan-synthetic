# GPU ANN benchmark (target recall@10 = 0.90)

All corpora are L2-normalized; see the design note. These figures are
not comparable with published SIFT1M results. Build time is train and
add phases, timed separately. The flat/exact index has no swept knob;
its row reports the single measured QPS as the exact-search ceiling,
not an interpolated value at the target recall.

| Corpus | Index | Train (s) | Add (s) | Index (MB) | QPS @ recall 0.90 | Peak recall |
|---|---|---|---|---|---|---|
| real | cagra | 8.39 | 0.00 | 768.0 | 250,985.4 | 0.972 |
| v0 | cagra | 7.02 | 0.00 | 768.0 | 231,091.4 | 0.967 |
| v1 | cagra | 8.54 | 0.00 | 768.0 | 230,819.7 | 0.969 |
| v1_5 | cagra | 8.13 | 0.00 | 768.0 | 226,023.8 | 0.968 |
| v2 | cagra | 6.46 | 0.00 | 768.0 | 252,608.1 | 0.966 |
| v3 | cagra | 7.47 | 0.00 | 768.0 | 261,483.5 | 0.958 |
| v4 | cagra | 7.86 | 0.00 | 768.0 | 240,321.7 | 0.968 |
| real | flat | 0.00 | 0.09 | 512.0 | 7,967.7 (exact ceiling) | 1.000 |
| v0 | flat | 0.00 | 0.09 | 512.0 | 7,973.1 (exact ceiling) | 1.000 |
| v1 | flat | 0.00 | 0.09 | 512.0 | 7,939.2 (exact ceiling) | 1.000 |
| v1_5 | flat | 0.00 | 0.09 | 512.0 | 7,946.1 (exact ceiling) | 1.000 |
| v2 | flat | 0.00 | 0.09 | 512.0 | 7,965.3 (exact ceiling) | 1.000 |
| v3 | flat | 0.00 | 0.09 | 512.0 | 7,931.9 (exact ceiling) | 1.000 |
| v4 | flat | 0.00 | 0.09 | 512.0 | 7,945.0 (exact ceiling) | 1.000 |
| real | ivf_flat | 1.44 | 0.00 | 512.0 | 61,694.5 | 0.971 |
| v0 | ivf_flat | 1.46 | 0.00 | 512.0 | 57,133.9 | 0.967 |
| v1 | ivf_flat | 1.46 | 0.00 | 512.0 | 57,646.7 | 0.968 |
| v1_5 | ivf_flat | 1.65 | 0.00 | 512.0 | 57,275.7 | 0.967 |
| v2 | ivf_flat | 1.60 | 0.00 | 512.0 | 92,681.6 | 0.965 |
| v3 | ivf_flat | 1.49 | 0.00 | 512.0 | 159,159.8 | 0.959 |
| v4 | ivf_flat | 1.46 | 0.00 | 512.0 | 68,265.7 | 0.967 |
| real | ivf_pq | 3.53 | 0.00 | 64.0 | 64,460.9 | 0.939 |
| v0 | ivf_pq | 3.39 | 0.00 | 64.0 | 65,461.4 | 0.940 |
| v1 | ivf_pq | 3.35 | 0.00 | 64.0 | 64,681.4 | 0.940 |
| v1_5 | ivf_pq | 3.28 | 0.00 | 64.0 | 65,458.9 | 0.940 |
| v2 | ivf_pq | 3.39 | 0.00 | 64.0 | 90,544.6 | 0.932 |
| v3 | ivf_pq | 3.51 | 0.00 | 64.0 | 133,596.5 | 0.919 |
| v4 | ivf_pq | 3.48 | 0.00 | 64.0 | 67,887.5 | 0.934 |
