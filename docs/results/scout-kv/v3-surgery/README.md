# scout_kv v3 real-data test: what makes real search hard

Status: design and decision rule, fixed on 2026-10-29 before the box run. Results will be
added below this section; the rule is not edited after the run.

## Why

Every generator so far (v0, v1, v2) makes HNSW search too easy. The largest gaps are at
L11, L15 and L43 (HNSW recall@10 at ef 176, median over 8 heads, from
`../v2/compare_layers_s*.json` and v1's `../v1/full/v1_final_seed0.jsonl`):

| layer | real, 5 draws | v1 | v2 s42 | v2 s1 |
|---|---|---|---|---|
| 11 | 0.521-0.718 | 0.907 | 0.935 | 0.948 |
| 15 | 0.462-0.715 | 0.818 | 0.921 | 0.907 |
| 43 | 0.574-0.637 | 0.617 | 0.862 | 0.928 |

The controls (`../controls/README.md`) showed that a Gaussian with real h's covariance is
very easy (0.97 at L11 and L43), and that taking the end-of-book queries away does not change L11.
So the hardness at L11 is in the distribution of real h beyond its covariance. This test
removes one property of real h at a time and measures which removal makes real search as
easy as generated search.

## Sets

Real h of books 12-19 at L11, L15 and L43, measured through the gate's real split
(`kv_gate.real_layer_records`, the code `measure-real` uses), so queries still come from
each book's last 1,024 tokens. Code: `src/kv/surgery.py`. Principal components are fitted
on the non-sink rows of the 8 books.

| set | kept | removed |
|---|---|---|
| `real` | everything; at seed 0 it must reproduce `../reference/seed0.jsonl` | nothing |
| `gaussian` | mean and covariance | all non-Gaussian structure, book and position means |
| `s0` | covariance, each component's values, the means of each (book, 1,024-position block) | the joint structure of the components within a token |
| `s32`, `s256` | as `s0`, plus the top 32 / 256 components unshuffled | the joint structure below rank 32 / 256 |
| `gz` | the rank dependence between components | heavy tails (each component mapped to Gaussian quantiles by rank) |

Each set is measured at seeds 0 and 1. The seed sets both the corpus draw (the reference's
`corpus_seed`) and the transform's random draws. Each record also carries neighbour fields
(`nbr_n_distinct_keys`, `nbr_hub_slot_share`, `nbr_hub_median_pos`, `nbr_own_book_share`,
`nbr_share_lag_le_1024`), and `intrinsic.jsonl` holds body and tail LID, contrast,
hubness and IVF Gini, measured as `../v2/intrinsic.py` measured the v2 runs.

`v2_neighbours.py` measures the same neighbour fields on v2's generated books at all 12
layers and both seeds, for comparison with the real values in `../controls/neighbours.jsonl`.

## Decision rule

The statistic is `hnsw_recall_k10_ef176`, median over the 8 heads, per layer. The
thresholds come from the real 5-draw maximum and the mean of v2's two seeds:

- **removes** the hardness: recall at or above `real_max + 0.5 * (v2_mean - real_max)`
  at both seeds;
- **keeps** the hardness: recall at or below `real_max + 0.05` at both seeds;
- **partial** otherwise.

| layer | real max | v2 mean | removes at | keeps at |
|---|---|---|---|---|
| 11 | 0.7184 | 0.9418 | >= 0.8301 | <= 0.7684 |
| 15 | 0.7148 | 0.9139 | >= 0.8143 | <= 0.7648 |
| 43 | 0.6366 | 0.8949 | >= 0.7658 | <= 0.6866 |

Sanity check, which stops the reading if it fails: `real` at seed 0 and seed 1 must land
within 0.01 of the reference draw of the same seed (`../reference/seed0.jsonl`,
`seed1.jsonl`) on this statistic at all three layers. The reference was measured on the A100
and this run is on the L40. The controls re-measured v0 across the two boxes and matched to
the fourth decimal.

How the outcomes are read:

- `gaussian` is expected to remove the hardness, as it did in the controls.
- `s0` removes and `gz` keeps: the hardness is in how components vary together within a
  token, and v3 has to model that joint structure.
- `s0` keeps and `gz` removes: heavy-tailed component values carry the hardness.
- `s32` / `s256`: the smallest kept rank at which the set keeps the hardness says how many
  leading components hold it.
- Anything else is reported as found, without a v3 recommendation.

## Run

`run.sh` on tig-gpu-alt from a clone at the pushed commit: `v2_neighbours.py`, then
`python -m src.kv.surgery --layers 11,15,43 --seeds 0,1`. ESTIMATE (unverified): about
2 h, from the controls' measured 110 s per (layer, set) gate measurement plus the
intrinsic and neighbour statistics.
