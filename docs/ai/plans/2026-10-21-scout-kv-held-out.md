# scout_kv held-out-book tail control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, with a valid positive control, whether real tail queries from books that are not in the key set keep
the real scout_kv search difficulty. If they do, a position-conditioned generator is a viable v1 target.

**Architecture:** The cross-book plan (`2026-10-21-scout-kv-cross-book.md`) failed its positive control (R6), because
its 4-key-book construction moved the statistics at L19. This plan keeps the keys exactly as the reference builds them:
`corpus.mixed_real_corpus` over books 12-19, per KV head, with the reference's per-head seed. Only the queries change.
They come from the tails of 8 books taken from 0-11, which are not in the key set. Two controls are measured per draw,
on the same keys and the same index seed:

- `rebuild`: the reference corpus itself (`mixed_real_corpus`, unchanged). At fresh seeds 5-8 it is exchangeable with
  the five committed reference draws, so it is the positive control.
- `held_out_tail`: the same keys, with the real-style and key-side queries taken from the tails of the held-out books.

**Tech Stack:** Python, numpy, torch, faiss (via `src/kv/search_profile.py`), pytest.

**Spec:** none. The authority is the controls plan `docs/ai/plans/2026-10-19-scout-kv-controls.md` (R1 hard stop), the
cross-book plan above (R6/R7 and their base rates), and the owner's decision of 2026-10-21 to redo the cross-book
control with 8 key books and held-out query books.

## Background facts (checked 2026-10-21)

- `mixed_real_corpus(h_books, wk, wq, g, *, n_sink, n_tail, seed)` draws, per book in order, the key positions and then
  the tail positions from one `np.random.default_rng(seed)`. Its `q_id` is the k-projection of the same tail rows as
  `q_ood`. `measure_real` seeds it and `profile_corpus` with `corpus_seed(layer, g, seed)`.
- Every capture book has T = 32768 rows. The box tig-gpu-alt holds `doc0`..`doc19` with all 12 global layers (checked
  with `ls /workspace/kv/capture/doc0 doc11`). It is an L40 (46 GB), idle, with 141 GB RAM.
- Books 0-7 measured on their own (`controls/train_books_real.jsonl`, seed 0) differ from the reference range: at
  L11 the IVF ratio is below (0.996 against 1.223-1.304) and IVF recall above; at L19 every rule statistic is on the
  hard side; at L43 HNSW recall@10 is above (0.641 against 0.574-0.637). MEASURED, `summary.json`
  `compare_layers.train_books_real`.
- Tiny-capture tests need T > 128 + n_tail + 1 keys per corpus, because `profile_corpus` takes k = 128 ground truth
  (see `tests/test_kv_gate.py::test_measure_real_end_to_end_on_a_tiny_capture`, which uses T = 400).

## Global Constraints

- Locked index conditions: `nlist=256`, `hnsw_m=48`, `ef_construction=500`, `n_sink=1`, `n_tail=1024`.
- Rule layers 11, 19, 43 (`summarise.LAYERS`). Reference books 12-19.
- Draws (fixed now, before any measurement):

  | draw | seed | held-out query books |
  |---|---|---|
  | `s5` | 5 | 0, 1, 2, 3, 4, 5, 6, 7 |
  | `s6` | 6 | 4, 5, 6, 7, 8, 9, 10, 11 |
  | `s7` | 7 | 0, 1, 2, 3, 8, 9, 10, 11 |
  | `s8` | 8 | 0, 1, 2, 4, 5, 6, 8, 10 |

  Eight query books give 128 tail rows each, as the reference does. Seeds 5-8 are not among the committed reference
  seeds 0-4, so no draw shares its keys with a draw that sets the real range.
- No change to `gates/`, the bands, `measure_real`, `mixed_real_corpus`, `measure_controls`, `tail_split`, `b7.sh`,
  `xbook.sh`, or any committed measurement file (`*.jsonl`, `*.log`). `summary.json`, `README.md`, `report.html` and
  `SHA256SUMS` are regenerated or extended in B12.
- No v1 design. The R1 hard stop is still in force.
- Python: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`. `make check` must pass.

## Review Focus

- A held-out query book that is also a reference book (for example 12 passed by mistake): own-book share would be
  silently non-zero. `measure_held_out` must refuse it before writing anything (Task 2 test H5).
- A record file from the wrong draw copied into `heldout/sN/`: summarise must refuse records whose `seed` or
  `query_books` do not match the draw table (Task 3 test S2).
- The rebuild path drifting from `measure_real` (a different seed, sink or field): Task 2 test H4 compares the two
  record for record.
- The pairing check passing when layers are missing: `--layers` restricts both sides, and a missing corpus inside
  those layers still fails (Task 3 test P1).
- Query labels colliding with key-book labels, which would make `own_book_share_top10` count held-out queries as
  own-book: labels start at `len(h_ref_books)` (Task 1 test H2; Task 2 test H4 checks the share is exactly 0).

---

### Task 1: `held_out_tail_corpus` in `src/kv/controls.py`

**Files:**
- Modify: `src/kv/controls.py`
- Test: `tests/test_scout_kv_held_out.py` (new)

**Interfaces:**
- Consumes: `src.kv.corpus.mixed_real_corpus`, `Corpus`; `src.kv.transforms.head_keys`, `head_queries`, `n_rep`.
- Produces:
  `held_out_tail_corpus(h_ref_books, h_query_books, wk, wq, g, *, n_sink, n_tail, seed) -> Corpus`.
  Keys, `key_book` and `key_pos` are those of `mixed_real_corpus(h_ref_books, wk, wq, g, n_sink=n_sink,
  n_tail=n_tail, seed=seed)`. Tail positions come from `np.random.default_rng([seed, 1])`: per query book in order,
  `np.sort(rng.choice(np.arange(T - n_tail, T), n_tail // Q, replace=False))`. `q_ood` is `head_queries` of those
  rows, `q_id` is `head_keys` of the same rows. Query book labels are `len(h_ref_books) + j` for query book `j`;
  `q_pos` is the tail positions repeated `n_rep` times, as in `mixed_real_corpus`.
  Raises `ValueError` if `h_query_books` is empty, if any query book's length differs from the reference books', or
  if `n_tail % len(h_query_books)` is non-zero.

Test fixture: `_books(n, T=400, d=16, seed)` gives n random float32 tensors; `_weights(d=16)` gives `wk` (2·128, d)
and `wq` (2·5·128, d) from a seeded rng. Reference: 4 books; query books: 2 books; `n_sink=1`, `n_tail=16`,
`seed=3`, `g=1`.

- [ ] **Step 1: Write the failing tests**

| test | asserts | mutation it catches |
|---|---|---|
| H1 `test_held_out_corpus_keeps_reference_keys` | `keys`, `key_book`, `key_pos` are `np.array_equal` to `mixed_real_corpus`'s at the same seed and `g` | keys drawn from a fresh rng or after the tails; keys taken from the query books; `g` ignored |
| H2 `test_held_out_corpus_queries_come_from_query_book_tails` | for each query book j: `q_id_book == 4 + j` on exactly 8 rows, its `q_id_pos` are distinct and in [384, 400), `q_id` rows equal `head_keys(hq[j][pos], wk, 1)` and the matching `q_ood` rows equal `head_queries(hq[j][pos], wq, 1, 5)` (exact); `q_book` is `np.repeat(q_id_book, 5)` and `q_pos` is `np.repeat(q_id_pos, 5)`; `q_book.min() == 4`. With one query book, `set(q_id_pos) == set(range(384, 400))` | queries taken from reference books; tail region off by one (the one-book case takes the whole region); labels starting at 0 (own-book collision); `q_id` from different rows than `q_ood` |
| H3 `test_held_out_corpus_seeds_and_errors` | same seed twice gives identical `q_id_pos`; seeds 3 and 4 give different `q_id_pos`; `ValueError` for no query books, for a query book of length 399, and for 3 query books (16 % 3 ≠ 0) | tail rng with a fixed seed; missing guards |

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_scout_kv_held_out.py -q` → FAIL (ImportError).
- [ ] **Step 3: Implement `held_out_tail_corpus`** as in Interfaces.
- [ ] **Step 4: Run** the same command → PASS. Mutation-check H1-H3 against the mutations in the table.
- [ ] **Step 5: Commit** `src/kv/controls.py tests/test_scout_kv_held_out.py`.

### Task 2: `measure_held_out` and the script

**Files:**
- Modify: `src/kv/controls.py`
- Create: `docs/results/scout-kv/controls/heldout.py`
- Test: `tests/test_scout_kv_held_out.py`

**Interfaces:**
- Consumes: Task 1; `kv_gate.load_books`, `load_layer_weights`, `HEAD_DIM`, `corpus_seed`, `profile_corpus`,
  `conditions`, `TARGET`, `_own_book_share_top10`, `_append`, `_device`.
- Produces:
  - `HELD_OUT_CONTROLS = ("rebuild", "held_out_tail")`.
  - `measure_held_out(capture_dir, layers, seed, out_dir, *, query_books, controls=HELD_OUT_CONTROLS,
    ref_books=range(12, 20), n_sink=1, n_tail=1024, nlist=256, hnsw_m=48, ef_construction=500, threads=15) -> None`.
    Writes `out_dir/{control}.jsonl`, one record per (layer, KV head). Before writing anything it raises `ValueError`
    for an unknown control, empty `query_books`, or `query_books` that share a book with `ref_books`, and
    `FileExistsError` if an output exists. Per layer it loads the reference books and the query books once
    (`load_books`, moved to the device as float). Per KV head, with `s = corpus_seed(layer, g, seed)`: `rebuild`
    uses `mixed_real_corpus(..., seed=s)`, `held_out_tail` uses `held_out_tail_corpus(..., seed=s)`, and both are
    profiled with `seed=s`. The query books are loaded only when `held_out_tail` is requested.
  - Record fields, in this order: `layer, kv_head, source, seed, books, n_keys, n_q_ood, n_q_id, n_sink, n_tail,
    conditions, own_book_share_top10, **profile, seconds, control, query_books`. `source` is `"control"`, `books`
    is `list(ref_books)`, and `query_books` is `list(ref_books)` for `rebuild` and `list(query_books)` for
    `held_out_tail`.
  - `heldout.py`: `--capture-dir` (required), `--layers` (`parse_books`, default `11,19,43`), `--seed` (int,
    required), `--query-books` (`parse_books`, required), `--controls` (default `rebuild,held_out_tail`),
    `--out-dir` (required), `--threads` (default 15).

Test fixture: a tiny capture with 20 books at layer 3, T = 400, d = 16, random float32 h, and 2 KV heads (as in the
`measure_real` tiny test). Common kwargs: `layers=[3]`, `seed=0`, `query_books=[0, 1]`, `n_tail=16`, `nlist=4`,
`hnsw_m=8`, `ef_construction=16`, `threads=1`.

- [ ] **Step 1: Write the failing tests**

| test | asserts | mutation it catches |
|---|---|---|
| H4 `test_measure_held_out_records` | the files are exactly `held_out_tail.jsonl` and `rebuild.jsonl`, 2 records each; every record has `books == [12..19]`, `n_keys == 8 * ((400 - 16 - 1) // 8)`, `n_q_ood == 80`, `n_q_id == 16`, `source == "control"`, `control` equal to the file stem; `query_books` is `[12..19]` for rebuild and `[0, 1]` for held_out; held_out `own_book_share_top10 == 0.0` exactly. `kv_gate.measure_real(cap, books=range(12, 20), layers=[3], seed=0, ...)` with the same sizes gives records equal to the rebuild records on every field except `source`, `seconds`, `control`, `query_books` | rebuild using a different seed, sink, or path from `measure_real`; profile seed not `corpus_seed`; held-out labels colliding with key books; `query_books` not recorded |
| H5 `test_measure_held_out_rejects_bad_arguments` | `ValueError` and no `out_dir` created for `query_books=[0, 12]`, `query_books=[]`, and `controls=["rebuild", "nope"]`; a second call on a finished `out_dir` raises `FileExistsError` and leaves the files' mtimes unchanged | missing guards; guards that run after the first write |

- [ ] **Step 2: Run** → FAIL (ImportError).
- [ ] **Step 3: Implement** `measure_held_out`, `HELD_OUT_CONTROLS` and `heldout.py`.
- [ ] **Step 4: Run** → PASS; mutation-check H4-H5. Also run
  `PYTHONPATH=. .venv/bin/python docs/results/scout-kv/controls/heldout.py --help` → exit 0.
- [ ] **Step 5: Commit** `src/kv/controls.py docs/results/scout-kv/controls/heldout.py tests/test_scout_kv_held_out.py`.

### Task 3: pairing check, rules, box script

**Files:**
- Modify: `docs/results/scout-kv/controls/check_pairing.py`, `docs/results/scout-kv/controls/summarise.py`
- Create: `docs/results/scout-kv/controls/heldout.sh`
- Test: `tests/test_check_pairing.py`, `tests/test_scout_kv_summarise.py`

**Interfaces:**
- `check_pairing.py`: `--neighbours` and the new `--records` form a required mutually exclusive group. With
  `--records`, the per-corpus share is the record's `own_book_share_top10`. New `--layers` (`parse_books`, default
  all) keeps only those layers on both sides before comparing. Everything else is unchanged.
- `summarise.py`:
  - `HELDOUT_DRAWS = {"s5": (5, (0, 1, 2, 3, 4, 5, 6, 7)), "s6": (6, (4, 5, 6, 7, 8, 9, 10, 11)),
    "s7": (7, (0, 1, 2, 3, 8, 9, 10, 11)), "s8": (8, (0, 1, 2, 4, 5, 6, 8, 10))}` (seed, query books).
  - `HELDOUT_CONTROLS = ("rebuild", "held_out_tail")`. Inputs: `heldout/{draw}/{control}.jsonl`.
  - `check_heldout_draw(records, seed, query_books, control)`: `SystemExit` unless every record has that `seed` and
    `control`, `books == list(range(12, 20))`, and `query_books` equal to `list(query_books)` for `held_out_tail` or
    `list(range(12, 20))` for `rebuild`.
  - `path_check_ok(report) -> bool`: True iff `report["n_compared"] == 24` and `missing`, `mismatched` and
    `meta_mismatch` are all empty (`report` is `pair_check.json`).
  - `heldout_verdict(path_ok, outside_rebuild, flatters_held_out)`: if `path_ok` is False, returns
    `{"path_ok": False, "positive_control_valid": None, "n_held_out_flatter": None, "reading": None}`. Otherwise
    `xbook_verdict(outside_rebuild, flatters_held_out)` with the key `n_cross_flatter` renamed `n_held_out_flatter`,
    the reading `"own_book"` renamed `"not_reachable"`, and `"path_ok": True` added.
  - `paired_diff(cmp_a, cmp_b) -> dict`: per rule layer and rule statistic, `a - b` of the pooled `value`, `None`
    when either value `is None` (0.0 is a legitimate value: real L43 reach is 0.0 in some draws).
  - `heldout(refs, v0cmp)`: like `xbook`, plus per draw `paired_diff(held_out, rebuild)`. `main()` writes
    `summary["heldout"]` and `summary["R6_R7_heldout"]` only when all 8 files and `heldout/pair_check.json` exist.
- `heldout.sh`: the `step` function copied verbatim from `xbook.sh`; `OUT` defaults to
  `/workspace/kv/controls/heldout`; `sha256sum` of the code files it runs. Steps, in order:
  1. `pair_s0`: `heldout.py --controls rebuild --seed 0 --query-books 0-7 --out-dir $OUT/pair_s0`.
  2. `pair_check`: `step pair_check pair_check.json -- bash -c "$PY check_pairing.py --records
     $OUT/pair_s0/rebuild.jsonl --layers 11,19,43 > $OUT/pair_check.json"`. The redirect is inside `bash -c`, so the
     step's own `DONE` line stays in the log. A non-zero exit stops the script (the `step` function exits 1).
  3. `s5`..`s8`: `heldout.py --seed N --query-books <table> --out-dir $OUT/sN`.

  It ends with `echo HELDOUT_COMPLETE`.

- [ ] **Step 1: Write the failing tests**

| test | asserts | mutation it catches |
|---|---|---|
| P1 `test_records_mode_with_layers` | records built from the committed seed0 reference: all 96 pass in `--records` mode; only the layer-11/19/43 records, with `--layers 11,19,43`, give `n_compared == 24` and exit 0; the same file without `--layers` exits 1 with 72 missing; layer-11 records with one share moved by 0.002 exit 1 with that key mismatched; dropping one layer-19 record with `--layers 11,19,43` exits 1 with it missing | `--records` reading the neighbours field; `--layers` filtering one side only; filtering hiding missing corpora |
| S1 `test_heldout_verdict_boundaries` | with `path_ok=True`: rebuild `[T,F,F,F]` with held-out counts 0, 1, 2, 3, 4 give reachable, reachable, inconclusive, not_reachable, not_reachable; rebuild `[T,T,F,F]` gives `positive_control_valid` False and `reading` None. With `path_ok=False` and otherwise-reachable inputs, `reading` is None. `path_check_ok` is True for a clean 24-corpus report and False for 23 compared, or one entry in any of the three lists | off-by-one in either threshold; the positive-control or path gate ignored; label not renamed; `n_compared` not checked |
| S2 `test_check_heldout_draw_refuses_mislabelled_records` | a matching held_out record set of 3 records passes; `SystemExit` when the **last** record has a wrong `seed`, a wrong `query_books`, or a wrong `control`, and for a rebuild record set carrying held-out `query_books` | draws not checked; only the first record checked |
| S3 `test_paired_diff_keeps_zero` | fake compare outputs where a = 0.0 and b = 0.125 give -0.125, a = 0.5 and b = 0.0 give 0.5, and a None gives None | truthiness instead of `is None` |

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the three files.
- [ ] **Step 4: Run** → PASS; mutation-check P1, S1, S2. Dry-run `heldout.sh` with `PY` set to a stub script that
  creates the expected output files and prints its arguments; check the step order and the arguments.
- [ ] **Step 5: Run** `make check` → pass. **Commit** the five files.

## Decision rules (fixed before the run)

`flatters(control, draw)` is R1's test: `rules()[...]["statistics_with_2_of_3"] >= 2` on `hnsw_reach_k10`,
`hnsw_recall_k10_ef176` and `ivf_ratio_k128`, v0's direction from `v0/best.jsonl`, real range from
`reference/seed{0..4}.jsonl`. `n_outside` is the cross-book plan's either-direction count.

- **R6a (path):** `pair_check` exits 0: the seed-0 rebuild reproduces `reference/seed0.jsonl` on all 24 rule-layer
  corpora (own-book share within 1e-3, equal metadata). If it fails, the code path differs from the reference: stop
  and report, read nothing else.
- **R6 (positive control valid):** `rebuild` has `n_outside >= 2` in at most 1 of the 4 draws. If it fails, the
  reference range from five draws is narrower than the draw-to-draw spread of the reference construction itself. R7
  is then not read: report and stop.
- **R7 (held-out reading)**, with `n` = the number of draws in which `held_out_tail` flatters:
  - `n <= 1`: **reachable**. Queries from books outside the key set keep the real difficulty.
  - `n >= 3`: **not_reachable**.
  - `n == 2`: **inconclusive**.

What R7 does and does not separate: the held-out books are the books a v1 generator would be trained on, so a
held-out tail query is what an ideal position-conditioned generator trained on books 0-11 would produce. `reachable`
supports that v1 target. `not_reachable` means such a generator would fail, but R7 does not say whether the cause is
the query's link to its own book or the difference between books 0-11 and 12-19 (see Background). The write-up
places the held-out deltas next to the `train_books_real` deltas cell by cell; that comparison is descriptive.

Base rates under exchangeability, treating cells as independent (ESTIMATE, unverified, carried over from the
cross-book plan): a real-like draw flatters about 1.6% of the time and is outside in either direction about 17% of
the time; R6 wrongly declares a valid positive control invalid about 13% of the time.

## B11: box run on tig-gpu-alt

1. Push the branch. Stream the changed files to `/workspace/repo` with `git archive` and tar, check sha256 at both
   ends, and record the hashes in the run log.
2. Launch `heldout.sh` with `setsid nohup`, log to `/workspace/kv/controls/heldout/b11.log`. Expected time: about
   110 s per control per layer, so `pair_s0` about 5.5 minutes and each draw about 11 minutes; about 50 minutes in
   total (ESTIMATE, from `b9.log`: 651-673 s per two-control draw).
3. Poll with a background loop that stops after 3 unreachable polls.
4. Pull the 9 jsonl files, `pair_check.json` and the log with tar, and check sha256 against the box.

## B12: write-up

- Run `summarise.py`. Add an "R6/R7: held-out-book control (2026-10-21)" section to the controls `README.md` with
  the R6a/R6/R7 results, the paired-difference table, the `train_books_real` comparison, and claim-table rows.
- Add the section to `report.html` via `make_report.py`, open it in headless Chrome and check it.
- Commit the results with a `SHA256SUMS` update, and update the memory note `scout-kv-controls-r1-fires.md`.
