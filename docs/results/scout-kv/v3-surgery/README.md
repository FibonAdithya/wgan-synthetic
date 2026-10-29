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

## Results (2026-10-29)

Run on tig-gpu-alt (L40) at commit `ab3d171`, 11:06-13:27 UTC. All 27 result files matched their
box-side sha256 after transfer (`SHA256SUMS`). The box wrote one `gate.jsonl`; it is committed
split by layer as `gate_L{11,15,43}.jsonl` to keep each file under 1 MB, and
`cat gate_L11.jsonl gate_L15.jsonl gate_L43.jsonl` matches the box hash. `summary.json` is
built by `summarise.py`; `report.html` (self-contained, figures inlined) by `make_report.py`.

### Claim table

| claim | value | MEASURED / ESTIMATED | file | when |
|---|---|---|---|---|
| sanity: `real` reproduces reference seed 0 / 1 | passes, largest difference 0.0000 | MEASURED | `summary.json` | 2026-10-29 |
| `s0` (components shuffled within each token) | removes the hardness at L11, L15, L43 | MEASURED | `summary.json` | 2026-10-29 |
| `gz` (heavy tails removed) | keeps it at L11, L15; partial at L43 | MEASURED | `summary.json` | 2026-10-29 |
| `s256` (top 256 components kept) | removes at L11, L15; partial at L43 | MEASURED | `summary.json` | 2026-10-29 |
| `s32`, `gaussian` | remove at all three layers | MEASURED | `summary.json` | 2026-10-29 |
| body hubness, removing sets vs the others (split read after the run) | L11 ≤ 11.9 vs ≥ 20.1; L15 ≤ 18.0 vs ≥ 35.7; L43 ≤ 7.2 vs ≥ 32.3; real 30.0-42.7 | MEASURED | `intrinsic.jsonl` | 2026-10-29 |
| v1 / v2 body hubness at these layers | 2.4-6.2 | MEASURED | `../v1/full/intrinsic.jsonl`, `../v2/runs/*/intrinsic.jsonl` | 2026-10-29 |
| box time | 2.34 h (438 s v2 neighbours + 7,995 s surgery) | MEASURED | `run.log` | 2026-10-29 |

### Rule applied

HNSW recall@10 ef 176, median of 8 heads, seed 0 / seed 1:

| set | L11 | L15 | L43 |
|---|---|---|---|
| `real` | 0.565 / 0.618 keeps | 0.623 / 0.463 keeps | 0.574 / 0.617 keeps |
| `gaussian` | 0.971 / 0.946 removes | 0.987 / 0.989 removes | 0.971 / 0.966 removes |
| `s0` | 0.958 / 0.972 removes | 0.965 / 0.945 removes | 0.829 / 0.861 removes |
| `s32` | 0.955 / 0.957 removes | 0.941 / 0.944 removes | 0.884 / 0.875 removes |
| `s256` | 0.867 / 0.928 removes | 0.933 / 0.953 removes | 0.666 / 0.732 partial |
| `gz` | 0.557 / 0.622 keeps | 0.708 / 0.687 keeps | 0.734 / 0.800 partial |

Reading, as fixed above: `s0` removes and `gz` keeps at L11 and L15, so there the hardness is in
how the components vary together within a token, not in heavy-tailed component values. At L43
`s0` removes and `gz` is partial: the joint structure carries most of it and the heavy tails some.
The joint structure that matters is not confined to the top 256 components at L11 and L15
(`s256` removes); at L43 the top 256 hold most of it (`s256` partial).

### Observations beyond the rule

- **Body hubness separates the outcomes.** At each layer every removing set has lower body
  hubness than every other set (table above). The split was read off the results. The shuffles
  that remove hubs also raise body LID (L11: real 17.5, `s0` 50.1), so this test does not
  separate hubs from LID. v1 and v2 do at L11: their body LID (19.3, 19.0) is within 2.5 of real
  while their hubness is about 3 and their search is easy.
- **v1 at L43** is the one generated case inside the real range (0.617, hubness 5.9). v1
  retrained at seed 1 is easy there (0.906, hubness 6.2), as the v1 diagnostics found.
- **Where queries land in v2's books.** Real queries put their top-10 lists on 31 and 64
  distinct keys at L7 and L11; v2 spreads them over more than twice as many. From L15 on, real hub
  keys sit late in the books (median position above 25,000 at 8 of 9 layers, not L39); v2's sit
  more than 1,000 positions earlier at all 8 of those layers at both seeds.
- **IVF at L43** stays near the real range under `s0` (scan ratio 3.53 / 3.42, real
  2.97-3.48) while HNSW becomes easy, so the IVF statistics there depend on the block means
  (book and position), which `s0` keeps.

### For v3

Body hubs are the measured property that tracks the hardness, so they are the candidate target
for v3. This is co-occurrence, not proof that hubs alone cause the hardness: the hubness split
was read after the run, and the transforms that remove hubs also change LID and the rest of the
joint structure. Removing the within-token joint structure removes the hubs; matching each
component's distribution or the leading 256 components does not keep them at L11 and L15.
Hubness skew can be measured on a batch, so it can serve as a training signal or a selection
statistic. Open question: whether the components past rank 256 that matter at L11 and L15 are
the directions `W_k` reads most, since the key projection is not ordered by h variance.
