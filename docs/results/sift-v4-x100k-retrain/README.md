# SIFT v4 at 100k, retrained

The 2026-08-10 run of `configs/sift/v4_sift1m_x100k.yaml` lost its checkpoints
when the box was rebuilt, and a gate cannot admit a generator that no longer
exists. This is the same config, retrained unchanged.

- Job `wgan-synthetic-20260917T114405Z-a772c5`, commit `50ca078`, script
  `scripts/sift_v4_x100k_job.sh`, RTX 3060 Ti. Exit 0, about 2 h 26 min wall
  from submit to the report.
- Corpus `/workspace/data-cache/sift_1m.npy`, sha256 `4921953796bc…cb768d`,
  the same hash as the 2026-08-19 refetch.
- `best_generator.pt` is step 86,000 (`select_on: cov_fro`, lowest `cov_fro`
  0.003402). Checkpoints stay on the box at
  `/workspace/sift-v4/v4_sift1m_x100k/`; hashes in `CHECKSUMS.txt`.
- `eda/summary.json`: 50,000 samples per checkpoint at seed 42, measured at
  the canonical conditions (N 20000, k 100, hubness k 10, nlist 256).

Against the mean of the ten real draws in `docs/datasets/sift_noise_floor.json`:

| Statistic | real mean (10-draw range) | `v4_best` | `v4_step100000` | 2026-08-10 `v4` |
|---|---|---|---|---|
| LID median | 17.6911 (17.6209-17.7458) | 16.4147 (-7.2%) | 16.4083 (-7.3%) | 16.3357 (-7.7%) |
| Relative contrast | 2.2599 (2.2556-2.2739) | 2.3231 (+2.8%) | 2.3252 (+2.9%) | 2.3265 (+2.9%) |
| Hubness skew | 1.9026 (1.8284-1.9652) | 1.7937 (-5.7%) | 1.7641 (-7.3%) | 1.7770 (-6.6%) |
| IVF Gini | 0.3039 (0.2866-0.3180) | 0.2987 (-1.7%) | 0.3107 (+2.2%) | 0.2999 (-1.3%) |
| Exact-zero fraction | 0.2298 (canonical row) | 0.2394 | 0.2427 | 0.2321 |

The retrain reproduces the August run to within 0.6 points on every gated
statistic, so the 100k result was not a one-off. All three pass
`gates/sift.yaml`; see `docs/datasets/sift.md`, `## Gate`.
