# v3 `structured_gated` — rescued run artifacts

Recovered 2026-08-05 from `/workspace/wgan-v3` on `tig-gpu`, a checkout with no
git history that was about to be reclaimed. The code for this track is on this
branch; these are its results, and they existed nowhere else.

Checkpoints (3.3GB across two runs) were **not** kept. `best_generator.pt` for
each run was moved to `/workspace/keep/` on the box; the intermediate
`checkpoint_step_*.pt` files were deleted. What is here is the metric and
configuration record, which is what a decision about this track actually needs.

## Runs

| Run | Steps | Config | Status |
|---|---|---|---|
| `sift_gan_v3` | 30k | `sift_gan_v3/run_config.yaml` | completed |
| `x100k_structured` | 100k | `x100k_structured/run_config.yaml` | completed, `rc=0` |

## ANN difficulty, 30k run vs real SIFT

Measured at the canonical settings — N=20000, k=100, hubness k=10, nlist=256 —
from `eda_v3_30k/summary.json`.

| Statistic | Real | v3 synthetic | |
|---|---|---|---|
| LID median | 17.74 | 10.63 | too low |
| Relative contrast median | 2.27 | 3.14 | too high |
| Hubness skew | 1.88 | 0.86 | too low |
| IVF cell-balance Gini | 0.304 | 0.261 | close |
| Exact-zero fraction | 0.230 | 0.179 | under |
| Duplicate row fraction | 0.0006 | 0.0000 | none produced |

**The result is negative, and consistently so.** Every difficulty statistic
points the same way: the synthetic corpus is *easier* to search than real SIFT.
Lower intrinsic dimension, higher relative contrast and flatter hubness all
mean nearest neighbours are better separated than they should be. An ANN
algorithm tuned against this corpus would look better than it is.

That the four move together is the useful part. It is not four independent
misses to be tuned away one at a time — it reads as one cause, most plausibly
that the generator is not reproducing the local density variation that makes
real descriptor neighbourhoods hard.

Note also that `duplicate_row_fraction` is exactly zero against real SIFT's
0.0006. Real SIFT descriptors are quantized to a lattice, so exact duplicates
occur; a continuous generator cannot produce them.

## Caveats

- The comparison above is the **30k** run. The 100k run
  (`x100k_structured`) completed but its ANN report was not captured before the
  directory was found, so only its `run_metadata.json` and final training-loop
  eval are here.
- These were measured under L2, which is correct for SIFT.
- `logs/` holds the driver scripts as well as their output, so the exact
  invocations are recoverable.
