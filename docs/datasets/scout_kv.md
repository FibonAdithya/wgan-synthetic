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
| Ladder | `v0` configs defined for all 12 global layers, none trained |
| Gate bands | **unset**. Every band in `gates/scout_kv.yaml` is `null`. Setting them is a human decision, made from the reference noise floor; until then `python -m src.eval.kv_gate check` exits 2 with verdict `unset` |
| Capture | lives only on `tig-gpu`, at `/workspace/kv/capture`. It is not in the repo and not fetchable with `src/data/fetch.py` |

## What a sample is

One sample is the input to a global-attention layer's `self_attn`: the output of
`input_layernorm`, a 5,120-dimensional vector for one token position. Keys and queries
are made from it afterwards with the model's weights (`src/kv/transforms.py`). The
generator never sees keys or queries.

- Global layers: `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47` (`src/data/kv_capture.py::GLOBAL_LAYERS`), 12 in all, no RoPE.
- 8 KV heads and 40 query heads, `head_dim` 128. Query head `j` reads KV head `j // 5`.
- One generator is trained per layer, so there are 12 `v0` configs, `configs/scout_kv/v0/L03.yaml` through `configs/scout_kv/v0/L47.yaml`. They differ only in `data.kv_layer` and `output_dir` (pinned by `tests/test_scout_kv_configs.py`).
- Position 0 (the attention sink) and the last 1,024 positions of a 32,768-token book are never indexed in a corpus. Training rows use positions 1 to 32,767, so 12 books give 393,204 rows per layer.

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
- Run cost for 96 mixed corpora was 18.7 min (README above). The 8-book reference and the 5-draw noise floor have not been measured; the spec's estimate of 1.6 h for 5 draws is ESTIMATE (unverified).

`data.metric: l2` only governs the in-training selector on `h`; the gate searches by inner product.

## Not yet done

- Bands: unset (see Status).
- Training: no `v0` run has been made. Training time per layer is unknown.
- Whether the in-training selector (`select_on: gate`, LID and contrast on `h`, holdout 2% = 7,864 rows) picks good checkpoints under the search-difficulty gate is untested. The gate is run afterwards on `best_generator.pt` and the final checkpoint.
