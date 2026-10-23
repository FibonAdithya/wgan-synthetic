# scout_kv v1 diagnostics: retrain noise and hub position

Two checks run on 2026-10-23 after v1's 12-layer result (`../v1/report.html`,
PR #67): v1 failed four of five bands, with large per-layer gains at some
layers and regressions at others. All numbers are MEASURED from the files in
this directory, on tig-gpu-alt (L40), at commit d1b46dc.

## 1. Retrain noise: v1 trained again at a second seed

`noise_train.sh` retrained v1 at training seed 1 for L19, L27, L35 and L43. The
configs were generated from `configs/scout_kv/v1/L{LL}.yaml` with only `seed`
(42 → 1) and `output_dir` changed, and are kept in `noise/configs/`.
`noise_measure.sh` gated them as v1 was gated: sampling seed 0, EMA, best and
final checkpoints. The seed-42 values come from `../v1/full/`, and v0 from
`../v1/full/v0_best_seed0.jsonl` (same box).

Best checkpoints. "Gap" is |seed 42 − seed 1|. "Ref spread" is max − min over
the five real reference draws at that layer.

| layer | statistic | real | v0 | v1 s42 | v1 s1 | gap | ref spread | v0→v1 same direction at both seeds |
|---|---|---|---|---|---|---|---|---|
| 19 | HNSW reach | 0.500–0.875 | 1.000 | 0.625 | 0.625 | 0.000 | 0.375 | yes |
| 19 | HNSW recall | 0.854–0.882 | 0.972 | 0.880 | 0.902 | 0.022 | 0.027 | yes |
| 19 | IVF ratio | 0.133–0.172 | 0.095 | 0.266 | 0.277 | 0.011 | 0.039 | yes (overshoots real) |
| 27 | HNSW reach | 0.750–1.000 | 0.750 | 0.625 | 0.625 | 0.000 | 0.250 | yes |
| 27 | HNSW recall | 0.839–0.867 | 0.793 | 0.808 | 0.715 | 0.094 | 0.028 | **no** |
| 27 | IVF ratio | 1.776–1.866 | 1.900 | 1.549 | 1.246 | 0.302 | 0.090 | yes (away from real) |
| 35 | HNSW reach | 0.125–0.500 | 0.625 | 0.875 | 0.375 | 0.500 | 0.375 | **no** |
| 35 | HNSW recall | 0.567–0.729 | 0.727 | 0.921 | 0.696 | 0.225 | 0.162 | **no** |
| 35 | IVF ratio | 1.896–2.525 | 1.041 | 1.539 | 1.631 | 0.091 | 0.629 | yes |
| 43 | HNSW reach | 0.000–0.125 | 0.375 | 0.125 | 0.750 | 0.625 | 0.125 | **no** |
| 43 | HNSW recall | 0.574–0.637 | 0.824 | 0.617 | 0.906 | 0.289 | 0.063 | **no** |
| 43 | IVF ratio | 2.974–3.480 | 0.760 | 2.702 | 3.016 | 0.315 | 0.507 | yes |

Key-side recall rises from v0 at every layer and both seeds (to 0.987–1.000).
IVF recall moves the same way at both seeds except at L35, where both moves are
under 0.01. The final checkpoints give the same picture.

What this shows:

- **IVF ratio and key-side recall respond to the flag reliably.** Both seeds
  move the same way, by more than the gap between them. The IVF ratio moves
  toward real at L35 and L43. At L19 it rises from below the real range to
  above it, and ends further from real than v0 (0.094–0.105 above the range
  against v0's 0.038 below). At L27 it moves away from real at both seeds.
  So the flag moves the IVF ratio reliably, but not always toward real.
- **Per-layer HNSW reach and recall are dominated by training noise at L35
  and L43, and HNSW recall also at L27.** The gap between seeds is larger
  than v1's move from v0, and the sign flips. At L43, seed 1 is worse than
  v0 on both (reach 0.750, recall 0.906). L19 reproduces closely.
- **The pilot's go was partly luck.** The pilot (`../v1/pilot/`) judged
  L19 and L43 at two sampling seeds of one training run, which does not
  capture training noise. Its L43 HNSW gains did not reproduce. A pilot for
  the next rung needs at least two training seeds, or should rest on the
  statistics that are stable across seeds.
- **Caveat:** two training seeds at four layers. This measures the size of
  retrain noise at those layers, not its distribution, and says nothing
  directly about the pooled 12-layer statistics the gate uses.

## 2. Where real h's hubs sit

`hub_position.py` takes 20,000 real body rows per layer (books 12–19,
positions 1–31,743, seed 42). It counts how often each row appears in the
other rows' 10 nearest neighbours (N_10, L2), and calls the top 1% hubs.
Rows are uniform over position, so each eighth of the body holds 12.5% of
rows. Results are in `hub_position.json`.

| layer | hubness skew | hubs in the first / last eighth | median hub position (all rows 15,822) | Spearman N_10 vs position / norm / cos to mean |
|---|---|---|---|---|
| 3 | 3.0 | 0.04 / 0.31 | 24,299 | 0.32 / −0.16 / 0.27 |
| 7 | 26.7 | 0.08 / 0.52 | 28,219 | 0.29 / 0.32 / 0.36 |
| 11 | 30.0 | 0.07 / 0.69 | 29,557 | 0.39 / 0.22 / 0.38 |
| 15 | 37.4 | 0.24 / 0.40 | 17,297 | 0.35 / 0.30 / 0.40 |
| 19 | 52.3 | 0.09 / 0.56 | 28,721 | 0.31 / 0.27 / 0.32 |
| 23 | 48.8 | 0.08 / 0.42 | 26,484 | 0.17 / 0.15 / 0.15 |
| 27 | 49.9 | 0.23 / 0.21 | 15,166 | 0.24 / 0.19 / 0.23 |
| 31 | 42.5 | 0.37 / 0.06 | 5,869 | 0.33 / 0.26 / 0.31 |
| 35 | 40.5 | 0.23 / 0.04 | 12,953 | 0.26 / 0.22 / 0.25 |
| 39 | 41.3 | 0.28 / 0.06 | 14,564 | 0.31 / 0.29 / 0.32 |
| 43 | 42.7 | 0.28 / 0.06 | 15,822 | 0.32 / 0.32 / 0.34 |
| 47 | 40.5 | 0.22 / 0.04 | 9,399 | 0.22 / 0.07 / 0.24 |

What this shows:

- **Hubs are concentrated by position at every layer**, and in every layer
  they come from all 8 books (largest single book 15–37%).
- **Where they concentrate depends on the layer.** They sit late in the body
  at L7, L11, L19 and L23, are split between the start and the end at L15
  and L27, and sit early or mid-body at L31–L47 (L35–L43 also have a second
  concentration near the middle; see `share_hubs_by_bin`).
- **Position is not the only correlate.** N_10 correlates with position,
  with norm and with alignment to the mean at similar strength (0.07–0.40 at L7–L47; at L3, where hubness is low (skew 3.0), the norm correlation is −0.16).
  These three may co-vary with each other; this check does not separate them.

For v2: v1's single body flag treats all of the body alike, which fits its
lack of hubs (v1 body skew 2.6–6.4 against real 26.7–52.3 at L7–L47). A
finer position condition (for example position bins) is supported by this
check. One generator per layer already lets each layer learn its own profile.
Whether position, norm or mean alignment is the better conditioning signal
is not settled here.

## Files

- `noise/measure/v1s1_{best,final}_L{19,27,35,43}.jsonl`: gate records of the
  seed-1 retrains.
- `noise/configs/`, `noise/training/L{LL}/`: generated configs, run metadata
  and training logs.
- `noise/*.log`: job logs. The first parallel launch of L35/L43
  (`v1_noise_B.log`) died on the trainer's per-card GPU lock after 1,800 s
  and was requeued serially (`v1_noise_B2.log`).
- `hub_position.json`, `v1_hub_position.log`.
- `noise/SHA256SUMS`: hashes listed on the box, verified locally after
  transfer.

Training times on the L40 with the card otherwise idle: 1,906–1,917 s per
layer.
