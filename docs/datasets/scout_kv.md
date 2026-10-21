# scout_kv

Attention inputs (`h`, 5,120 dimensions) captured from Llama-4-Scout-17B-16E-Instruct
running over PG-19 books. The family exists to train a generator whose samples,
projected with the model's own key and query weights, give key and query sets that
IVF and HNSW indexes find as hard to search as real ones. The design record is
`docs/ai/specs/2026-10-18-kv-cache-scout-design.md` (non-authoritative; this page
and the code win on any conflict).

This family departs from `AGENTS.md` invariant 1. Its gate is IVF and HNSW search
difficulty (`src/eval/kv_gate.py::evaluate`). The four intrinsic statistics of
`src/eval/ann_difficulty.py` are diagnostics here, measured on `h` itself.

## Status

| | |
|---|---|
| Ladder | `v0` trained for all 12 global layers (`docs/results/scout-kv/training/`); fails the gate on four of five statistics (`docs/results/scout-kv/v0/report.html`) |
| Gate bands | **set** 2026-10-19 by the owner, as a tolerance around the five-draw real reference; see `## Gate bands` |
| Gate question | **open** since 2026-10-19: real `h` split the way synthetic corpora are split is already outside the real range on v0's side (R1); v1 design waits on it. Evidence for keeping the reference: the held-out-book control (2026-10-21) reads *reachable*. See `## Controls: why v0 is too easy` |
| Capture | lives only on the GPU box (`tig-gpu-alt` since 2026-10-19), at `/workspace/kv/capture`. It is not in the repo and not fetchable with `src/data/fetch.py` |

## What a sample is

One sample is the input to a global-attention layer's `self_attn`: the output of
`input_layernorm`, a 5,120-dimensional vector for one token position. Keys and queries
are made from it afterwards with the model's weights (`src/kv/transforms.py`). The
generator never sees keys or queries.

- Global layers: `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47` (`src/data/kv_capture.py::GLOBAL_LAYERS`), 12 in all, no RoPE.
- 8 KV heads and 40 query heads, `head_dim` 128. Query head `j` reads KV head `j // 5`.
- One generator is trained per layer, so there are 12 `v0` configs, `configs/scout_kv/v0/L03.yaml` through `configs/scout_kv/v0/L47.yaml`. They differ only in `data.kv_layer` and `output_dir` (pinned by `tests/test_scout_kv_configs.py`).
- Position 0 (the attention sink) and the last 1,024 positions of a 32,768-token book are never indexed in a corpus. Training rows use positions 1 to 32,767, so 12 books give 393,204 rows per layer.

## Commands

The `measure-real` and `measure-run` subcommands run on the GPU box, where the capture lives.
`check` and `compare-layers` need only a records file.

```
python -m src.eval.kv_gate measure-real --capture-dir D --out real.jsonl [--books 12-19] [--layers 3,7,...] [--n-sink 1]
python -m src.eval.kv_gate measure-run  --capture-dir D --run-dir 3=runs/scout_kv/v0/L03 ... --out run.jsonl \
    [--checkpoint-name best_generator.pt] [--ema] [--n-keys 31736] [--layers 11,19,43]
python -m src.eval.kv_gate compare-layers --records run.jsonl [--reference ref.jsonl ...]
python -m src.eval.kv_gate check --records run.jsonl [--gate-file gates/scout_kv.yaml]
```

- `measure-real` profiles the real reference. `--books` takes a range (`12-19`) or a
  list (`12,13`). One draw over the 96 corpora took 1763 to 2443 s on a 15.36-core quota
  (MEASURED, `docs/results/scout-kv/reference/README.md`).
- `measure-run` needs one `--run-dir LAYER=PATH` for every one of the 12 global layers
  and refuses to start otherwise. `--ema` measures the checkpoint under its stored EMA
  weights; `best_generator.pt` already holds them, so the flag changes nothing there.
  With `--layers 11,19,43` it measures only those layers. Each must be global and have a
  run dir, and the list must be non-empty with no repeats. All of this is checked before the
  first measurement. That is for control experiments:
  `check` still fails such records, because the gate needs all 96 corpora.
- `compare-layers` pools each layer's records separately and reports, per statistic, the
  min, max and mean over the reference draws, the candidate's value, and whether it is
  `inside` (bounds inclusive), `above`, `below` or `none`. `--reference` is repeatable and
  defaults to `docs/results/scout-kv/reference/seed0.jsonl` to `seed4.jsonl` under the repo
  root. Each layer entry also gives its `n_reference_sets`. The command loads the gate
  file's `canonical` block. The top-level `condition_mismatches` list in the output names
  records whose counts (`n_keys`, `n_q_ood`, `n_q_id`, `n_sink`, `n_tail`) or index
  conditions differ from it. The list is reported, not enforced. It prints JSON and exits 0,
  or exits 1 on an unreadable input or a missing or malformed gate file.
- Both measuring commands also take `--n-tail` (1024), `--nlist` (256), `--hnsw-m` (48),
  `--ef-construction` (500), `--threads` (15) and `--seed` (0). The first four defaults
  are locked conditions, and `check` fails records measured under other values.
  `--seed` picks the draw.
- `check` prints a JSON report and exits 0 (pass), 1 (fail) or 2 (a band is null).

The capture itself is made by `scripts/kv/capture_scout.py --model-dir M --pg19-parquet P
--out-dir D [--first-book 0] [--n-books 4] [--ctx 32768]`. Before the long run it checks
that the projections reproduce the model's own q and k, on a `--smoke-ctx` (2048) prefix
and on one past the temperature floor, and exits non-zero if they do not.
`--gpu-weights` (30GiB) caps the weights held on the GPU.

## Book split

Books are numbered in capture order (`scripts/kv/capture_scout.py`) and split by
book, not by token, because tokens inside a book are not independent.

| split | books | use |
|---|---|---|
| train | 0-11 | generator training (`data.kv_books`) |
| reference | 12-19 | the gate's real reference and its noise floor |

Nothing the gate measures against is trained on.

## Locked conditions

Held in the `canonical` block of `gates/scout_kv.yaml` and checked against what a
measurement records (`src/eval/kv_gate.py::load_kv_gate`).

| | |
|---|---|
| Corpora | 96 = 12 global layers x 8 KV heads, pooled |
| Keys per corpus | 31,736 |
| Real-style queries | 5,120 (1,024 `h` x 5 query heads) |
| Key-side queries | 1,024 |
| IVF | IVFFlat, inner product, `nlist` 256, `nprobe` grid 1 to 256 |
| HNSW | Xbox reduction, `M` 48, `efConstruction` 500, `efSearch` grid 10 to 2048 |
| Recall target | 0.9 |
| Preprocess | `center: true`, `whiten: false`, `l2_normalize: false` |
| Generator output | `model.output_normalize: false`: norms carry meaning for inner-product search |

## Gate statistics

Each is pooled over the 96 corpora. There is one band per statistic and no combined
score. The `_id` statistic is a check, not a gap.

| statistic | definition |
|---|---|
| `hnsw_reach_k10` | share of corpora in which real-style queries reach recall@10 >= 0.9 within the efSearch sweep |
| `hnsw_recall_k10_ef176` | median recall@10 at `efSearch` 176, real-style queries |
| `ivf_ratio_k128` | median of (scan fraction at recall@128 = 0.9, real-style) / (same, key-side) |
| `ivf_recall_k128_np24` | median recall@128 at `nprobe` 24 of 256, real-style queries |
| `hnsw_recall_k10_ef176_id` | the HNSW recall statistic for key-side queries |

Diagnostics, not gated: the four intrinsic statistics on `h`; per-layer and per-head
versions of each gate statistic; HNSW work at recall 0.9.

The in-training selector (`training.select_on: gate`) is not this gate. It picks
`best_generator.pt` from LID and contrast gaps on `h`. The gate is run afterwards,
with `kv_gate measure-run`.

## Gate bands

Set 2026-10-19 by the owner, after the v0 gate run. Each band holds all five real draws
over books 12-19 (`docs/results/scout-kv/reference/`) with room on both sides, because
five draws understate the range. Bounds are inclusive. The comments in
`gates/scout_kv.yaml` give the reasoning per band. Real and v0 values are MEASURED, from
the committed `check_*.json` files.

| statistic | band | real, 5 draws | v0 best | v0 under the band |
|---|---|---|---|---|
| `hnsw_reach_k10` | 0.52 to 0.69 | 0.5625 to 0.6250 | 0.8750 | fail |
| `hnsw_recall_k10_ef176` | 0.78 to 0.88 | 0.7982 to 0.8526 | 0.9274 | fail |
| `ivf_ratio_k128` | 0.90 to 1.15 | 0.9403 to 1.0761 | 0.6841 | fail |
| `ivf_recall_k128_np24` | 0.80 to 0.865 | 0.8185 to 0.8432 | 0.8246 | pass |
| `hnsw_recall_k10_ef176_id` | >= 0.99 | 1.0000 in all five | 0.9757 | fail |

Unlike the sift, glove and nytimes gates, no generator was admitted when the bands were
set: they say what the next rung must reach. `tests/test_kv_gate.py` pins that every real
draw passes and that v0's best and final checkpoints fail on exactly the four statistics
above.

## Mixed-book reference: measured

A generator trained on several books emits independent `h`, so its corpus resembles a
mixture of books. The check compared 96 real mixed corpora (4 profiled books) with
the 4 single-book profiles. All numbers are MEASURED, from
`docs/results/kv-cache-profile/mixed-book/comparison.txt` (method in
`docs/results/kv-cache-profile/mixed-book/README.md`).

| statistic | single books (range of 4) | mixed | verdict |
|---|---|---|---|
| HNSW reach, recall@10 >= 0.9, real-style | 0.469 to 0.552 | 0.573 | above the range |
| median HNSW recall@10 at ef176, real-style | 0.696 to 0.804 | 0.806 | above the range |
| median HNSW recall@10 at ef176, key-side | 1.000 to 1.000 | 1.000 | inside |
| median IVF scan ratio, k=128 | 1.051 to 1.146 | 1.120 | inside |
| median IVF recall@128 at nprobe 24, real-style | 0.795 to 0.843 | 0.817 | inside |

- Mixed corpora are easier for HNSW than the easiest single book on reach@10: 0.573 against 0.552 (about 2 points). Median recall@10 at ef176 differs by 0.2 points (0.806 against 0.804). IVF is unchanged. The mixed-book check used 31,740 keys (4 books), not the gate's 31,736 (8 books). MEASURED, source `docs/results/kv-cache-profile/mixed-book/comparison.txt`. The decision rule fixed before the run (mixed inside the single-book range) failed for HNSW. The owner chose mixed corpora as the gate reference anyway.
- Own-book preference over all heads is none: the median share of a query's exact top-10 from its own book is 0.253, where 0.25 means no preference. By layer it is not uniform: the median is 0.34 for L27 and 0.33 for L31 and L35, and 0.21 to 0.28 for the other layers. MEASURED, source `docs/results/kv-cache-profile/mixed-book/comparison.txt` ("own-book share" lines).
- Run cost for 96 mixed corpora was 18.7 min (README above). The 5-draw 8-book noise floor took about 3.4 h in all on tig-gpu, MEASURED (`docs/results/scout-kv/reference/README.md`), against the spec's estimate of 1.6 h.

`data.metric: l2` only governs the in-training selector on `h`; the gate searches by inner product.

## Controls: why v0 is too easy

The controls plan (`docs/ai/plans/2026-10-19-scout-kv-controls.md`) measured real and
synthetic-like control sets on layers 11, 19 and 43 against the five-draw real range.
The write-up is `docs/results/scout-kv/controls/README.md` (figures in `report.html`,
every number in `summary.json`). All values below are MEASURED, from `summary.json`.

- **R1 fires at seeds 0, 1 and 2.** `pooled_real` is real `h` from books 12-19 with keys
  and queries drawn from one pool, as synthetic corpora are built. It is already outside
  the real range on v0's side. At L43 its IVF ratio is 1.12 to 1.15 against real 2.97 to
  3.48 (v0 0.76), and its recall@10 at ef176 is 0.685 to 0.770 against real 0.574 to
  0.637 (v0 0.824). Much of the gap comes from the reference taking queries from the
  last 1,024 positions of each book, which an unconditional generator cannot reproduce.
- The training books are not easier than the reference books (R2 does not fire).
- The plan's hard stop applies: no v1 design until the gate question is settled.
- The cross-book control (`docs/ai/plans/2026-10-21-scout-kv-cross-book.md`) used four key books. Its positive
  control failed (R6), so it gives no reading.
- The held-out-book control (`docs/ai/plans/2026-10-21-scout-kv-held-out.md`) keeps the reference's keys exactly
  (books 12-19) and takes the queries from the tails of eight books from 0-11. Its path check and positive control
  pass, and the held-out queries are on v0's side in 0 of 4 draws. The reading is **reachable**: tail queries from
  books outside the key set are not easier than the reference, so a position-conditioned generator is a viable v1
  target. At layer 19 they are harder, partly because books 0-11 differ from 12-19. Results:
  `docs/results/scout-kv/controls/README.md`.

## Not yet done

- The gate question above: keep the end-of-book reference and condition the generator on
  position, or re-base the reference on pooled sampling.
- A generator that passes the gate. v0 fails it (see `## Gate bands`).
- Whether the in-training selector (`select_on: gate`, LID and contrast on `h`, holdout 2% = 7,864 rows) picks good checkpoints under the search-difficulty gate. In v0 the best and final checkpoints differ only at layers 7 and 11, and both fail the gate on the same four statistics, so v0 does not settle it.
