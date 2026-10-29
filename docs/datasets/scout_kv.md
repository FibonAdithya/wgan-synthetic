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
| Ladder | `v0` trained for all 12 global layers (`docs/results/scout-kv/training/`); fails the gate on four of five statistics (`docs/results/scout-kv/v0/report.html`); `v1` (v0 plus a body/tail position flag, `docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md`) trained for all 12 layers and measured 2026-10-22: it also fails four of five bands, a different four (key-side recall now passes at 1.000; IVF recall leaves its band at 0.870), and by the spec's rule it does not move every statistic toward real (`docs/results/scout-kv/v1/report.html`); a seed-1 retrain at four layers is in `docs/results/scout-kv/v1-diagnostics/`; `v2` (v1 plus a per-book drifting latent and a window critic, `docs/ai/specs/2026-10-28-scout-kv-v2-book-latent-design.md`) trained for all 12 layers at seeds 42 and 1 and measured 2026-10-29: it meets its aim (same-book share within 0.15 of real at both seeds on 11 of 12 layers, missing at L3; body LID at least v1 final minus 6 at both seeds on all 12), and still fails the search gate (it fails four of five bands at training seed 42 and two of five at seed 1), with L19 HNSW search too easy and body hubness far below real (`docs/results/scout-kv/v2/report.html`) |
| Gate bands | **set** 2026-10-19 by the owner, as a tolerance around the five-draw real reference; see `## Gate bands` |
| Gate question | decided 2026-10-22: position-aware `v1` (`docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md`). Background: real `h` split the way synthetic corpora are split is already outside the real range on v0's side (R1). See `## Controls: why v0 is too easy` |
| Capture | lives only on the GPU box (`tig-gpu-alt` since 2026-10-19), at `/workspace/kv/capture`. It is not in the repo and not fetchable with `src/data/fetch.py` |

## What a sample is

One sample is the input to a global-attention layer's `self_attn`: the output of
`input_layernorm`, a 5,120-dimensional vector for one token position. Keys and queries
are made from it afterwards with the model's weights (`src/kv/transforms.py`). The
generator never sees keys or queries.

- Global layers: `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47` (`src/data/kv_capture.py::GLOBAL_LAYERS`), 12 in all, no RoPE.
- 8 KV heads and 40 query heads, `head_dim` 128. Query head `j` reads KV head `j // 5`.
- One generator is trained per layer, so there are 12 `v0` configs, `configs/scout_kv/v0/L03.yaml` through `configs/scout_kv/v0/L47.yaml`. They differ only in `data.kv_layer` and `output_dir` (pinned by `tests/test_scout_kv_configs.py`). The 12 `v1` configs, `configs/scout_kv/v1/L03.yaml` through `configs/scout_kv/v1/L47.yaml`, equal v0 plus `data.kv_n_tail: 1024`, `data.kv_tail_holdout: 1024`, `model.n_flags: 2` and `training.tail_batch_share: 0.25`, with `output_dir: runs/scout_kv/v1/L{LL}`. All 12 were trained on 2026-10-22 (`docs/results/scout-kv/v1/training/`). The 12 `v2` configs, `configs/scout_kv/v2/L03.yaml` through `configs/scout_kv/v2/L47.yaml`, were trained on 2026-10-28/29 at seeds 42 and 1 (`docs/results/scout-kv/v2/`). They set `model.generator_type: book_latent`, `model.book_latent`, `model.window_critic`, `training.n_windows`, `training.tail_window_share`, `training.window_weight` and `training.lambda_gp: 10`, keep v1's `data.kv_n_tail` and `model.n_flags`, and drop the keys only the v0/v1 trainer reads (the holdout, `tail_batch_share` and the best-checkpoint selector). `output_dir` is `runs/scout_kv/v2/L{LL}`. v2 trains on whole books with `python -m src.train.train_book_latent --config configs/scout_kv/v2/L{LL}.yaml [--resume PATH]`, and `src.train.train_wgan_gp` refuses these configs. `--resume` restores the weights, optimizers and EMA but not the random-number state: it re-seeds from the config, so the resumed part replays the window, noise and latent draws from step 1 and is not identical to an uninterrupted run (`train_wgan_gp` resumes the same way). None of the 24 committed v2 runs was resumed. v2 is judged on the final EMA weights only.
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
python -m src.kv.judge pilot --v1 SEED=v1.jsonl ... --v0 SEED=v0.jsonl ... --out pilot_best.json
python -m src.kv.judge full  --v1 v1.jsonl --v0 SEED=v0.jsonl ... --out full_best.json
python -m src.train.train_book_latent --config configs/scout_kv/v2/L19.yaml [--resume PATH]
python -m src.sample.sample_books --run-dir R --capture-dir CAP --out-dir R/capture [--seed 0] [--checkpoint final.pt] [--book-ids 12-19]
python -m src.eval.kv_gate measure-real --capture-dir R/capture --books 12-19 --layers L --seed 0 --source v2 --out v2.jsonl
python -m src.kv.book_judge port    --runs-dir D --out port.json
python -m src.kv.book_judge verdict --runs-dir D --v1-seed1 F [--v1-seed1 F ...] [--layers 3,19,27] --out verdict.json
```

- `src.train.train_book_latent` trains one v2 layer and writes `last.pt`, renamed to `final.pt` at the end, beside `run_config.yaml`. `src.sample.sample_books` writes synthetic books 12-19 (or `--book-ids`) in the capture format, with the sequence length taken from the checkpoint, and refuses an existing `--out-dir`. The synthetic position 0 is the training mean. Pointing `measure-real` at that directory measures the synthetic books; `--source NAME` sets the `source` field of the records (default `real`), so v2 records are not labelled as real. `measure-run`, `src.sample.generate` and `evaluate_distribution` refuse a v2 run, because they build the generator through `build_generator` or read a v0/v1 run directory.
- The v2 go/no-go is judged by a pure-Python module, `src/kv/book_judge.py` (`port_check` for the L19/L27 port check against the probe, `verdict` for the two criteria and the aim), run as `python -m src.kv.book_judge port` or `verdict`; it reads the committed JSON/JSONL results, so the call is not made by eye.
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
- `src.kv.judge` applies the v1 "moves toward real" rule (spec: `docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md`,
  "How v1 is judged") to records files, in plain Python with no GPU. Each call judges one
  checkpoint kind (all best or all final). `pilot` compares v1 with v0 at layers 19 and 43,
  per layer, pairing each v1 sampling seed with the v0 file of the same seed. `full` compares
  pooled statistics against `gates/scout_kv.yaml`. `--out` writes the verdict as JSON; the
  module docstring in `src/kv/judge.py` has the details and the one-seed, mismatched-corpus
  and mixed-kind refusals.
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
- The plan's hard stop was: no v1 design until the gate question is settled. The question was
  settled on 2026-10-22 in favour of keeping the end-of-book reference and conditioning the
  generator on position (a body/tail flag). The design is
  `docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md`. `v1` was trained and measured on 2026-10-22; results in `docs/results/scout-kv/v1/report.html`.

## Not yet done

- A generator that passes the gate. v0 and v1 each fail four of the five bands (see `## Gate bands` and `docs/results/scout-kv/v1/report.html`). v2 fails four of five at training seed 42 and two of five at seed 1, with HNSW reach and recall failing at both and L19 HNSW search too easy; its body hubness (1.9-14.9 at layers 15-47) is as far below real as v1's (`docs/results/scout-kv/v2/README.md`). v1's pilot layers (19, 43) moved every statistic toward real, but the pilot used one training run, and at a second training seed the L43 HNSW gains did not reproduce: per-layer HNSW reach and recall are dominated by retrain noise at L35 and L43 (`docs/results/scout-kv/v1-diagnostics/README.md`). Pooled over 12 layers only HNSW reach (0.875 to 0.760, still above the band) and key-side recall (0.976 to 1.000, now inside) move. Real body `h` has strong hubs at layers 7-47 (skew 26.7-52.3) that v1's body samples lack (2.6-6.4 over best and final checkpoints), as v0's did. Those hubs are concentrated by position inside the body, at a layer-dependent place (same README, section 2).
- Whether the in-training selector (`select_on: gate`, LID and contrast on `h`, holdout 2% = 7,864 rows) picks good checkpoints under the search-difficulty gate. In v0 the best and final checkpoints differ only at layers 7 and 11, and both fail the gate on the same four statistics, so v0 does not settle it.
