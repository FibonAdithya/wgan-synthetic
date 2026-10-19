# scout_kv real reference and noise floor (B3)

Five independent draws of the real reference over books 12-19, plus one diagnostic draw
over books 12-15. Each draw profiles 96 corpora (12 global layers x 8 KV heads) with
`python -m src.eval.kv_gate measure-real --capture-dir /workspace/kv/capture --books 12-19 --seed {s}`,
and `kv_gate check` pools them into the five gate statistics. Run on tig-gpu on 2026-10-18
at commit 6c012e9. The files were copied off the box by tar stream, and all 12 sha256
hashes matched on both sides.

Files:

- `seed{0..4}.jsonl`: per-corpus search profiles (96 records each).
- `check_seed{0..4}.json`: `kv_gate check` output for each draw.
- `four_books.jsonl`, `check_four_books.json`: the diagnostic draw over books 12-15, seed 0.

## Pooled statistics per draw

All values are MEASURED. Each `check_seed*.json` reports verdict `unset` (exit 2) because
every band in `gates/scout_kv.yaml` is null, and none reports a condition failure.

| draw | verdict | hnsw_reach_k10 | hnsw_recall_k10_ef176 | ivf_ratio_k128 | ivf_recall_k128_np24 | hnsw_recall_k10_ef176_id |
|---|---|---|---|---|---|---|
| seed0 | unset | 0.6146 | 0.8292 | 1.0503 | 0.8185 | 1.0000 |
| seed1 | unset | 0.6042 | 0.8402 | 0.9403 | 0.8336 | 1.0000 |
| seed2 | unset | 0.5625 | 0.8358 | 1.0444 | 0.8384 | 1.0000 |
| seed3 | unset | 0.6250 | 0.8526 | 1.0761 | 0.8308 | 1.0000 |
| seed4 | unset | 0.6146 | 0.7982 | 0.9716 | 0.8432 | 1.0000 |
| four_books (12-15) | fail | 0.5625 | 0.8141 | 1.1358 | 0.8413 | 1.0000 |

## Spread over the five 8-book draws

MEASURED, computed from the five `check_seed*.json` files.

| statistic | mean | min | max | spread (max-min) |
|---|---|---|---|---|
| hnsw_reach_k10 | 0.6042 | 0.5625 | 0.6250 | 0.0625 |
| hnsw_recall_k10_ef176 | 0.8312 | 0.7982 | 0.8526 | 0.0544 |
| ivf_ratio_k128 | 1.0165 | 0.9403 | 1.0761 | 0.1358 |
| ivf_recall_k128_np24 | 0.8329 | 0.8185 | 0.8432 | 0.0247 |
| hnsw_recall_k10_ef176_id | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

`hnsw_reach_k10` is the fraction of the 96 corpora that reach recall 0.9, so it moves in
steps of 1/96 (about 0.0104). Its 0.0625 spread is six corpora.

`hnsw_recall_k10_ef176_id` is 1.0 in every draw, so it cannot separate a generator from
real data at this operating point. Whether it should carry a band is for the human to decide.

## The four-book draw

The draw over books 12-15 has 31,740 keys per corpus against the canonical 31,736, so
`check` fails it on that condition alone: all 96 of its reasons are
`n_keys is 31740, canonical 31736`. The mismatch is expected and comes from the different
book set. Its statistics are kept for comparison only. Its `ivf_ratio_k128` (1.1358) is
above the largest of the five 8-book draws (1.0761). Its `hnsw_reach_k10` (0.5625) equals
the smallest 8-book draw.

## Timing

MEASURED from `/workspace/logs/b3.log`: 2443, 2186, 2158, 2187 and 1763 s for seeds 0-4,
and 1641 s for the four-book draw, about 3.4 h in total. The plan estimated about 19 min per
draw. The box's CPU quota is 15.36 cores (`cpu.max` 1536000/100000), and the HNSW builds
are CPU-bound.

## Bands

The owner set the bands in `gates/scout_kv.yaml` from this table on 2026-10-19. Every band
holds all five 8-book draws, and `hnsw_recall_k10_ef176_id` carries a one-sided `min: 0.99`.
The `check_seed*.json` files above were made before that and still read `unset`; under the
set bands all five draws pass (`tests/test_kv_gate.py::test_shipped_gate_passes_every_real_draw`).
The four-book draw still fails, on its key count alone.
