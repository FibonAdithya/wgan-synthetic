# scout_kv cross-book control

**Goal.** Find out whether a generator that samples keys and queries independently, with queries
drawn from late positions, can in principle reach the real scout_kv gate target. If it can, v1 is a
position-conditioned generator. If it cannot, the hardness depends on each query's relation to
earlier text in its own book, and the gate question goes back to the owner.

**Background.** The controls plan (`2026-10-19-scout-kv-controls.md`, PR #66) found R1 fires at all
three seeds. Real `h` split the way synthetic corpora are split (one pool, no positions) is already
on v0's side of the real range. The real reference (`kv_gate.measure_real` →
`corpus.mixed_real_corpus`) takes keys from positions `n_sink .. T-n_tail-1` of each of books 12-19
and queries from the last `n_tail` positions of the same books. Its key-side queries (`q_id`) are
the k-projections of the *same* tail tokens as the real-style queries. Every book has T = 32768
rows (checked on the box: `doc12/h_L11.safetensors` has shape [32768, 5120]); the box has 141 GB RAM.

This plan runs two controls on real `h`, both through `kv_gate.measure_h_layer` (the path that
synthetic data is measured on):

- `same_book_tail` (positive control): keys from the bodies of books A, queries from the tails of the
  same books A. It reproduces the reference split with 4 key books instead of 8.
- `cross_book_tail`: keys from the bodies of books A (the **same rows** as `same_book_tail`), queries
  from the tails of the other four reference books B. Each query is unrelated to every key, which is
  what independent sampling produces.

The two controls differ only in which books the queries come from.

## Global constraints

- Locked index conditions: `nlist=256`, `hnsw_m=48`, `ef_construction=500` (passed explicitly).
- Layers 11, 19, 43. `n_keys=31736`, `n_tail=1024`, `n_sink=1`. Reference books 12-19.
- Row positions come from the seed alone, so the same positions are used at every layer, as in
  `pooled_split`.
- Partitions: A = (12, 13, 14, 15) with B = (16, 17, 18, 19); and A = (16, 17, 18, 19) with
  B = (12, 13, 14, 15). Seeds 0 and 1. That is 4 draws per control.
- `h_id` is the same tail rows as `h_query`, as in the reference.
- No change to `gates/`, the bands, `measure_real`, `mixed_real_corpus`, `CONTROLS`, `b7.sh`, or any
  committed result file from PR #66.
- No v1 design. The R1 hard stop from the controls plan is still in force.
- Python: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`. `make check` must pass.

## Task 1: `tail_split` in `src/kv/controls.py`

```python
def tail_split(key_books, query_books, n_keys, n_tail, seed, n_sink=1):
    """Keys from the bodies of `key_books`, queries from the tails of `query_books`.

    Every book has the same length T. The body is positions n_sink .. T-n_tail-1 and the tail is
    positions T-n_tail .. T-1, as in mixed_real_corpus. Each key book gives n_keys // len(key_books)
    rows and each query book gives n_tail // len(query_books) rows, sampled without replacement.
    Key positions come from default_rng([seed, 0]) and tail positions from default_rng([seed, 1]),
    so the key rows do not depend on query_books. h_id is h_query (the same rows).

    Returns (h_keys, h_query, h_id, key_pos, q_pos), where key_pos and q_pos are (n, 2) int arrays
    of (index into the books argument list, position).
    """
```

Books are passed as two lists of tensors (`key_books`, `query_books`). `same_book_tail` passes the
same list twice. Raise `ValueError` if the book lengths differ, if `n_keys % len(key_books)` or
`n_tail % len(query_books)` is non-zero, or if a body is shorter than the rows it must give.

Tests (`tests/test_scout_kv_controls.py`). Helper books encode their identity: `h[t] = [book_id, t,
noise...]`, T=64, 4 key books with ids 0-3 and 4 query books with ids 4-7. The ids and positions are
small integers, exact in float32.

| test | asserts | mutation it catches |
|---|---|---|
| T1 `test_tail_split_positions_and_books` | `n_keys=160`, `n_tail=16`: every key has id in 0-3 and position in [1, 48); every query has id in 4-7 and position in [48, 64); 40 keys and 4 queries per book; no duplicate (id, position) among keys or among queries; `key_pos[:,0]` equals the key id column and `q_pos[:,0] + 4` equals the query id column (both are list indices); `key_pos[:,1]`/`q_pos[:,1]` equal the position column | key region running to T (tail leaks into keys); tail region starting at T-n_tail-1; sink row included; rows taken from the wrong list; unequal per-book counts |
| T2 `test_tail_split_id_is_query_rows` | `h_id` equals `h_query` row for row | `h_id` drawn from different tail rows (as pooled_split does) |
| T3 `test_tail_split_key_rows_ignore_query_books` | the same seed with query books 4-7, then with one query book (a different number of tail draws): identical `h_keys` and `key_pos` | one rng shared by keys and queries |
| T4 `test_tail_split_seeds_and_errors` | the same seed gives identical output; seeds 0 and 1 differ in `key_pos` and in `q_pos`; `ValueError` for unequal lengths, `n_keys=161`, `n_tail=18`, and `n_keys=4*48` (each body has 47 rows); `n_keys=4*47` succeeds and uses every body row; one query book with `n_tail=16` takes all 16 tail rows (the tail region is `n_tail` long, so a single query book is the exact-capacity case) | unseeded rng; seed ignored by one stream; missing guards; capacity guard off by one (`>=` for `>`) |

## Task 2: wire the controls into `measure_controls` and the script

- Add `TAIL_CONTROLS = ("same_book_tail", "cross_book_tail")`. Leave `CONTROLS` unchanged, so the
  script's default `--controls` and `b7.sh` behave as before. Validation accepts names from either
  tuple.
- New keyword `key_books=(12, 13, 14, 15)`. It must be a non-empty subset of `ref_books` with fewer
  books than `ref_books`, or `ValueError` (before any output is written). Query books:
  `same_book_tail` → `key_books`; `cross_book_tail` → `ref_books` minus `key_books`, in
  `ref_books` order.
- Per layer, load each needed book once with `_book_rows`, call `tail_split(..., seed, n_sink=1)`
  once per control, move to the device as float, and `run(...)` with
  `extra={"control": name, "books": list(key_books), "query_books": list(qbooks)}`.
- `docs/results/scout-kv/controls/controls.py` gains `--key-books` (parsed with `parse_books`,
  default `12-15`).

Tests:

| test | asserts | mutation it catches |
|---|---|---|
| T5 `test_measure_controls_tail_controls_feed_the_right_rows` | tiny capture as in the existing end-to-end test, but each book's `h[t] = [book, t, 0...]` (bf16-exact), T=64, `n_keys=160`, `n_tail=16`, `key_books=(13,15,17,19)` (non-contiguous and not the default, so a hard-coded or first-four choice fails); monkeypatch `kv_gate.measure_h_layer` with a spy that records the tensors and `extra` and returns `[]`. For `same_book_tail`: key ids and query ids are {13,15,17,19}. For `cross_book_tail`: key ids {13,15,17,19}, query ids {12,14,16,18}. Key rows are identical between the two controls; `h_id` is `h_query`; `extra` matches | query books not swapped for cross; key_books ignored; seed or rng differing between the controls |
| T6 `test_measure_controls_tail_controls_records` | real `measure_h_layer` on that tiny capture: files `same_book_tail.jsonl` and `cross_book_tail.jsonl`, 2 records each, `n_keys=160`, `n_q_id=16`, `n_q_ood=80`, `books`/`query_books` as above; a second call raises `FileExistsError`; `key_books=(12, 99)` and `key_books=range(12, 20)` raise `ValueError` and write nothing | wrong output naming; missing guards |

## Task 3: rules and write-up code in `summarise.py`

Add to `summarise.py`:

- `XBOOK_DRAWS = ("a_s0", "a_s1", "b_s0", "b_s1")`. Inputs are
  `xbook/{draw}/{control}.jsonl` under the controls directory.
- `n_outside(cmp_)`: for one `compare_layers` output, the number of rule statistics with at least 2
  of the 3 layers outside the real range in *either* direction. Status `above`, `below` and `none`
  (a null statistic) count as outside; only `inside` does not.
- `xbook_verdict(outside_same, flatters_cross)`: two lists of 4 booleans. `outside_same[d]` is
  `n_outside(same_book_tail at draw d) >= 2`; `flatters_cross[d]` is R1's test. It returns
  `{"positive_control_valid": ..., "n_cross_flatter": ..., "reading": ...}` with the rules below.
- `main()` adds `summary["xbook"]`: per draw and control, the `compare_layers` output and the
  `rules()` result, then the verdict. It is written only when all 8 files exist.

Test T8 `test_n_outside_counts_both_directions` builds a fake `compare_layers` output. One
statistic has statuses `above`, `below` and `inside` (counted); one has `inside`, `inside` and
`below` (not counted); one has `none`, `above` and `inside` (counted). It expects 2. It catches
counting only one direction, counting `none` as inside, and a threshold of 1 or 3 layers.

Test T7 `test_xbook_verdict_boundaries` covers:

- same `[T,F,F,F]` with cross counts 0, 1, 2, 3 and 4, which give reachable, reachable,
  inconclusive, own_book and own_book;
- same `[T,T,F,F]`, which gives `positive_control_valid` False and `reading` None.

It catches off-by-one errors in either threshold and the positive-control gate being ignored.

## Decision rules (fixed before the run)

`flatters(control, draw)` is R1's test: `rules()[...]["statistics_with_2_of_3"] >= 2`, on the
rule statistics `hnsw_reach_k10`, `hnsw_recall_k10_ef176` and `ivf_ratio_k128`, with v0's direction
taken from `v0/best.jsonl`, and the real range from `reference/seed{0..4}.jsonl`.

- **R6 (positive control valid):** `same_book_tail` is outside the real range (`n_outside >= 2`,
  either direction) in at most 1 of the 4 draws. Either direction counts, because a construction
  that made the data *harder* would let `cross_book_tail` look reachable without earning it. If R6
  fails, the 4-key-book construction or the measurement path itself moves the statistics. R7 is then
  not read: report and stop.
- **R7 (cross-book reading)**, with `n` = the number of draws in which `cross_book_tail` flatters:
  - `n <= 1`: **reachable**. Unrelated tail queries keep the real hardness, so a
    position-conditioned generator is a viable v1 target.
  - `n >= 3`: **own_book**. The hardness needs queries related to their own book's keys.
  - `n == 2`: **inconclusive**.

Base rates under exchangeability, treating cells as independent (all ESTIMATE, unverified):

- A real-like draw lies beyond the 5-draw real range with probability 1/6 per side per cell.
- It flatters (R1's one-sided test) about 1.6% of the time.
- It is outside in either direction (R6's test) about 17% of the time.
- R6 wrongly declares a real-like positive control invalid about 13% of the time. Reach takes values
  in steps of 1/8 and the bounds are inclusive, so the true rates are lower.

The write-up also tables, per layer, the IVF ratio and recall@10 for both controls next to real,
`pooled_real` and v0. That table is descriptive and does not change R6 or R7.

## B9: box run on tig-gpu-alt

1. Push the branch. Stream the changed files to `/workspace/repo` on the box (tar plus sha256 at
   both ends) and record the hashes in the run log.
2. `docs/results/scout-kv/controls/xbook.sh` runs 4 steps, each with the `step` pattern from
   `b7.sh` and a `.done` marker:
   `controls.py --controls same_book_tail,cross_book_tail --key-books {12-15|16-19} --seed {0|1}
   --out-dir /workspace/kv/controls/xbook/{a|b}_s{0|1}`.
   Partition `a` has key books 12-15, and `b` has 16-19. The `step` function is copied verbatim from `b7.sh`, which
   `tests/test_b7_script.py` covers. `xbook.sh` itself has no test, and a mistake in it fails on the box rather than
   silently.
3. Expected time: about 110 s per control per layer, so about 45 minutes in total (ESTIMATE, based
   on `b7_seeds.log`). Launch detached with `setsid nohup` and poll.
4. Pull the 8 jsonl files and the log with tar, and check sha256 against the box.

## B10: write-up

- Run `summarise.py`, then add an "R6/R7: cross-book control" section to the controls `README.md`
  with the verdict, the descriptive table and claim-table rows.
- Add the same table to `report.html` via `make_report.py`, open it in headless Chrome and check it.
- Commit the results, and update the memory note `scout-kv-controls-r1-fires.md`.
