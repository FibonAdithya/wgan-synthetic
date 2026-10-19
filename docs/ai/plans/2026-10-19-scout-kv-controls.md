# scout_kv controls: why v0 is too easy to search, before designing v1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Run the `audit-plan` skill on this file before dispatching any implementer; the `PreToolUse(Agent)` gate blocks dispatch until it is marked audited.

**Goal:** Find out, without training anything, why the scout_kv `v0` generators are easier to search than real data. Then a human can choose the direction for `v1` from measurements rather than guesses.

**Architecture:**
- **Code tasks 1-4** run locally, CPU only, under `make check`. They add:
  - a function that profiles any supplied `h` the way a trained run is profiled;
  - a per-layer subset mode for `measure-run`;
  - a `compare-layers` subcommand that places per-layer values against the five real draws;
  - a script that builds three control sets of `h` from the capture;
  - a neighbour-position diagnostic: where in its book a real query's exact top-10 keys sit (Task 4).
- **Box tasks B7-B8** are run by the orchestrator only, never by a subagent, because they need `ssh tig-gpu-alt` with the sandbox disabled. They measure the controls on layers 11, 19 and 43, run the neighbour diagnostic on all 12 layers, and write the results up.

**Tech Stack:** Python 3.12, numpy, torch (local CPU; box `2.14.1+cu126` in `/workspace/kvenv`), faiss-cpu 1.15.1, pyyaml, pytest, ruff.

**Background reading, in order:**
1. `docs/datasets/scout_kv.md`: the family page, which is authoritative.
2. `docs/results/scout-kv/v0/report.html`: what v0 gets wrong.
3. `docs/ai/specs/2026-10-18-kv-cache-scout-design.md`: the design record, non-authoritative.

There is no new spec. This plan only measures; it changes no gate and no ladder rung.

## Where things stand (2026-10-19)

- **Branch and PR.** The work so far is on branch `llama-kv-cache`, PR #65 on `origin` (`FibonAdithya/wgan-synthetic`, a public repo). When this plan was written, its checks were green at `fffd7ba` and another agent was finishing it. Start this plan on a new branch from `main` once #65 has merged. If it has not merged, start from `origin/llama-kv-cache`, and say so in the ledger.
- **Gate bands.** The owner set them on 2026-10-19 (`gates/scout_kv.yaml`). Every real draw passes. v0's best and final checkpoints both fail on 4 of 5 statistics; only `ivf_recall_k128_np24` passes.
- **v0, pooled over the 96 corpora** (MEASURED, `docs/results/scout-kv/v0/check_best.json`).
  - **HNSW:** v0 is too easy. Reach is 0.875 against real's 0.5625-0.6250, and recall@10 at ef176 is 0.927 against 0.798-0.853.
  - **IVF:** for v0's real-style queries, IVF scans a smaller share relative to key-side queries than it does for real data. The ratio is 0.684 against 0.940-1.076.
  - **Key-side recall:** slightly harder than real, 0.976 against 1.0.
- **Intrinsic statistics** (MEASURED, `docs/results/scout-kv/v0/intrinsic.jsonl`). Real `h` has strong hubs at layers 15-47, with hubness skew 38.3-47.2. v0 has 4.0-14.4.
- **Per layer, for the three layers this plan uses** (MEASURED; pooled per layer over 8 KV heads from the committed records; reach moves in steps of 1/8):

| layer | statistic | real, min-max of 5 draws | v0 best |
|---|---|---|---|
| 11 | reach | 0.5000-0.7500 | 1.0000 |
| 11 | rec10 | 0.5212-0.7184 | 0.9309 |
| 11 | ivf_ratio | 1.2230-1.3039 | 0.7594 |
| 11 | ivf_rec | 0.5928-0.6429 | 0.7074 |
| 11 | id_rec | 0.9964-0.9999 | 0.9950 |
| 19 | reach | 0.5000-0.8750 | 1.0000 |
| 19 | rec10 | 0.8543-0.8817 | 0.9722 |
| 19 | ivf_ratio | 0.1330-0.1718 | 0.0955 |
| 19 | ivf_rec | 0.9815-0.9875 | 0.9627 |
| 19 | id_rec | 1.0000-1.0000 | 0.9583 |
| 43 | reach | 0.0000-0.1250 | 0.3750 |
| 43 | rec10 | 0.5738-0.6366 | 0.8239 |
| 43 | ivf_ratio | 2.9738-3.4804 | 0.7606 |
| 43 | ivf_rec | 0.5163-0.5911 | 0.7656 |
| 43 | id_rec | 0.9999-1.0000 | 0.9672 |

- **Data, on `tig-gpu-alt`** (L40 46 GB, cgroup quota 23.04 cores):
  - capture: `/workspace/kv/capture`, books 0-19 plus `weights/`;
  - repo: `/workspace/repo`, at `43ce93e`;
  - v0 checkpoints: `/workspace/repo/runs/scout_kv/v0/L{LL}/`, each with `best_generator.pt` and `checkpoint_step_30000.pt`;
  - venv: `/workspace/kvenv`.
  All of it was sha256-verified against tig-gpu. `tig-gpu` is no longer needed; tearing it down is the owner's call.
- **Box gotchas, all measured:**
  - **Threads:** export `OMP_NUM_THREADS=8` for anything that runs torch. The default thread count on a cgroup-capped box made training about 5× slower.
  - **SSH:** it needs the sandbox disabled. Add `-o ClearAllForwardings=yes` if another session holds the LocalForward.
  - **authorized_keys:** the file has no trailing newline. Add one before appending a key.
  - **Transfers:** copy off the box by `tar` stream only, and verify sha256 on both sides.
  - **faiss:** `profile_corpus` builds HNSW deterministically only single-threaded. With `--threads` above 1, two runs of one seed can differ slightly.

### The asymmetry this plan tests first

Real and synthetic corpora are not built the same way. `src/kv/corpus.py::mixed_real_corpus` takes keys from positions 1 to `T - 1025` of each book, and takes the queries (real-style and key-side) from the **last 1,024 positions**. `src/kv/corpus.py::synthetic_corpus` draws keys and queries from **one pool** of generator samples, with no position at all.

So the real reference asks: how hard is it to find, among a book's earlier tokens, the neighbours of its final tokens? The synthetic side asks the same question with every token drawn from one distribution. If end-of-book `h` differs from the rest, real data has a key/query shift that no unconditional generator can reproduce. In that case v0's "too easy" verdict would say more about the gate than about the generator. Control `pooled_real` (Task 3) measures exactly this: real `h`, split the synthetic way.

(Query temperature scaling `s(pos)` is not part of the asymmetry. Neither corpus applies it, and a positive per-query scale does not change inner-product rankings.)

## Global Constraints

- Family `scout_kv`. Global layers `(3, 7, ..., 47)`, 8 KV heads, `head_dim` 128.
- **Layers for every control measurement in this plan: 11, 19, 43.** They are outside the real range on all five statistics in v0's per-layer values above.
- **The neighbour diagnostic (Task 4) is the exception: all 12 layers, books 12-19, seed 0.** It does no index build, so it is cheap. It must rebuild exactly the 96 corpora that `docs/results/scout-kv/reference/seed0.jsonl` profiled, so that each corpus's neighbour statistics pair with that corpus's measured hardness.
- **Nothing in `gates/scout_kv.yaml` changes.** No band, no canonical condition. `kv_gate check` keeps requiring all 96 corpora. Subset records must still fail `check`.
- **Locked conditions** for every profile: `n_keys` 31,736, `n_tail` 1,024, `nlist` 256, `hnsw_m` 48, `ef_construction` 500, target 0.9. These are the `measure-run` defaults; do not pass other values.
- **Book splits:**
  - Gaussian fit: books 0-11, the training split, which is what the generator saw.
  - `pooled_real` and `norm_equalised`: books 12-19, so they draw on the same `h` as the reference.
- **Existing behaviour unchanged.** `measure-run` without `--layers` still requires and measures all 12 layers. Every existing test passes unmodified.
- **Git:**
  - Stage explicit paths only; never `git add -A` or `git add .`.
  - Run `PATH=$PWD/.venv/bin:$PATH make check` before every commit. The venv is the main repo's `.venv`.
  - End commits with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Numbers:** every number in a report, README or commit is labelled MEASURED or ESTIMATE.
- **Seeds:** every random draw is seeded from a `seed` argument. No wall-clock dependence.

## Review Focus

1. **The refactor in Task 1 changes v0's records.** If `measure_h_layer` changes the per-head seed formula or a record field, new v0 measurements stop being comparable with `docs/results/scout-kv/v0/best.jsonl`. Task 1's equivalence test pins this.
2. **A subset slipping through `check`.** Records for 3 layers must still fail `check`, with the coverage reason.
3. **Overlapping key and query rows in `pooled_split`.** If a row is both a key and a query, recall is inflated, and that is exactly the error this plan is trying to detect. The test asserts the rows are disjoint.
4. **A Gaussian sampler that is wrong by a scale.** For example, using eigenvalues where their square roots belong, or rejecting a covariance that is not full rank. The 5120-dim covariance of real `h` may be rank-deficient. The test uses a rank-deficient fixture.
5. **`compare-layers` with its directions swapped**, or with bounds made exclusive. The test pins inclusive bounds and both directions, using committed real data.
6. **Task 4 changing the real reference.** Adding positions to `mixed_real_corpus` must not change which positions it draws. If one random draw moves, every committed reference record stops being reproducible. Test C2 pins the draw order, and the box acceptance check in B7 compares own-book shares with `seed0.jsonl` corpus by corpus.
7. **Query positions repeated in the wrong order.** `head_queries` emits the 5 query heads of one `h` row consecutively, so the OOD query positions are `np.repeat(tpos, 5)`, not `np.tile`. Test C1 checks every row.

---

### Task 1: `measure_h_layer` — profile any supplied `h`

**Files:**
- Modify: `src/eval/kv_gate.py`
- Test: `tests/test_kv_gate.py`

- [ ] **Step 1: Write the tests first.** Comparing `measure_run_layer` with `measure_h_layer` would be tautological once one calls the other. So pin the behaviour against independent facts instead.
  - **Fixture.** Use `_tiny_run` and `_TINY` (2 KV heads, D=16). Build fixed tensors `(h_keys, h_query, h_id)` from a seeded `torch.Generator`.
  - **Spy.** Extend the `_spy_keys` pattern into a spy that records `(keys, q_ood, q_id, seed)` for every `profile_corpus` call. `seed` arrives as a keyword argument.
  1. `measure_h_layer(h_keys, h_query, h_id, cap, 3, source="pooled_real", seed=2, nlist=..., hnsw_m=..., ef_construction=..., extra={"control": "x", "books": [12, 13]})` must:
     - call `profile_corpus` once per head, in head order;
     - pass seeds `[1009 * 3 + g + 7919 * 2 for g in (0, 1)]`;
     - pass keys equal to `head_keys(h_keys, wk, g)`, `q_ood` equal to `head_queries(h_query, wq, g, 5)` and `q_id` equal to `head_keys(h_id, wk, g)` (all to `rtol` 1e-6);
     - return records whose `source` is `"pooled_real"`, whose `control` is `"x"`, whose `books` is `[12, 13]` (so `extra` is merged after the fixed fields), and whose `n_tail` equals `len(h_query)`.
     Catches: a changed seed formula, a changed head order, keys or queries built from the wrong tensor (including `h_query` and `h_id` swapped, which the equal lengths allow), `source` or `extra` being ignored, and `extra` merged before the fixed fields.
  2. Records from `measure_run_layer` on `_tiny_run` must have exactly the key set of the first record in the committed `docs/results/scout-kv/v0/best.jsonl`. Catches a field dropped or renamed by the refactor, which would make new v0 measurements incomparable with the committed ones.
  3. `measure_h_layer` with `len(h_id) != len(h_query)` raises `ValueError`.
- [ ] **Step 2:** run the tests. Tests 1 and 3 must fail, because `measure_h_layer` does not exist yet. Test 2 passes before the refactor: it is the guard that the refactor keeps it passing.
- [ ] **Step 3: Implement.**
  - Move the per-head loop out of `measure_run_layer` into `measure_h_layer(h_keys, h_query, h_id, capture_dir, layer, *, source, seed, nlist, hnsw_m, ef_construction, extra=None) -> list[dict]`.
  - `n_tail` is not a parameter. It is `len(h_query)`, and the function raises if `len(h_id)` differs.
  - `source` is written to the record's `source` field. `extra` is merged into each record after the fixed fields, so a control's `books` replaces the synthetic default `[]`.
  - `measure_h_layer` does not set faiss threads. Its callers do, as `measure_run` already does.
  - `run_dir`, `checkpoint_name` and `generator_weights` move into `extra` for the trained-run path.
  - `measure_run_layer` becomes `synthetic_h(...)` followed by `measure_h_layer(..., source="synthetic", extra={...})`. Keep its signature and output unchanged.
- [ ] **Step 4:** run `tests/test_kv_gate.py`. The new tests and every existing test must pass, the existing ones without edits.
- [ ] **Step 5:** run `make check`, then commit: `kv_gate: measure_h_layer profiles any supplied h`.

### Task 2: `measure-run --layers` and `compare-layers`

**Files:**
- Modify: `src/eval/kv_gate.py`
- Test: `tests/test_kv_gate.py`
- Modify: `docs/datasets/scout_kv.md` (the Commands section)

- [ ] **Step 1: Write the tests first.**
  1. `measure_run(..., layers=[3])` on the tiny-run fixture, with run dirs supplied for both 3 and 7 (the same tiny run dir; the capture has weights for layer 3 only), writes records for layer 3 only, one per KV head. Catches ignoring `layers` and measuring every run dir supplied.
  2. `measure_run(..., layers=[3, 7])` with a run dir only for 3 raises `ValueError` naming layer 7, and the `out` file does not exist afterwards. Catches a missing run dir being silently skipped, and validation that runs after layer 3 has already been written.
  3. `measure_run(..., layers=[4])`, with run dirs supplied for 3 and 4, raises `ValueError` whose message says 4 is not a global layer. Supplying a run dir for 4 matters: without it, the missing-run-dir check would raise anyway and the test would catch nothing.
  4. `measure_run(...)` without `layers` and with a run dir missing still raises the existing "needs a run dir for every global layer" error. Catches the default changing to "whatever was supplied".
  5. `kv_gate.evaluate` on the 24 committed `v0/best.jsonl` records for layers 11, 19 and 43 gives verdict `fail`, with a reason that names the missing corpora. Catches a subset passing `check`.
  6. `compare_layers(records, references)` on the committed `v0/best.jsonl` (layers 11, 19, 43), against the five committed `reference/seed{0..4}.jsonl`, gives the following statuses (values from the table in "Where things stand"). Catches swapped `above`/`below` and wrong per-layer pooling. Only the two `ivf_recall_k128_np24` rows catch the pooling error: pooled over all layers, v0 is `inside` on that statistic, while every other row below gives the same status either way (checked against the committed records on 2026-10-19).

     | layer | statistic | status |
     |---|---|---|
     | 11 | `hnsw_reach_k10` | `above` |
     | 11 | `ivf_ratio_k128` | `below` |
     | 19 | `hnsw_recall_k10_ef176_id` | `below` |
     | 43 | `ivf_ratio_k128` | `below` |
     | 43 | `hnsw_recall_k10_ef176` | `above` |
     | 11 | `ivf_recall_k128_np24` | `above` |
     | 19 | `ivf_recall_k128_np24` | `below` |

  7. A value equal to a reference bound is `inside`. Use `_full()` fixtures in which the reference sets put `hnsw_recall_op` at 0.8 and 0.9 and the candidate puts it at exactly 0.9, then at exactly 0.8: both are `inside`. A candidate whose IVF scan fractions are all `None` gets `ivf_ratio_k128` status `none`. Catches exclusive bounds on either side, and a `None` compared as a number.
  8. CLI: `python -m src.eval.kv_gate compare-layers --records docs/results/scout-kv/v0/best.jsonl` exits 0 and prints JSON with one entry per layer present, with `n_reference_sets` 5. Run it with `cwd=tmp_path` and an absolute `--records` path, so that default references resolved against the working directory instead of the repo root fail.
- [ ] **Step 2:** run the tests. Tests 4 and 5 pass already: they are regression guards on behaviour that must not change. Every other test must fail.
- [ ] **Step 3: Implement.**
  - **`measure_run(..., layers=None)`.**
    - When `layers` is `None`, behaviour is unchanged.
    - Otherwise every entry must be in `GLOBAL_LAYERS` and have a run dir, and only those layers are measured, in the order given. Validate everything before the first measurement.
    - CLI: `--layers 11,19,43` on `measure-run`, parsed the same way as `measure-real`'s.
  - **`compare_layers(records, references) -> dict`.**
    - For each layer present in `records`, and each statistic in `STATISTICS`, compute `pooled_statistics` over that layer's records. Do the same per reference set. Report `{min, max, mean}` over the reference sets, the candidate `value`, and `status`, which is one of `inside` (bounds inclusive), `above`, `below`, or `none` (when the value is `None`).
    - Include `n_reference_sets` and, per layer, `n_records`.
  - **CLI subcommand `compare-layers`:**
    - `--records PATH` is required.
    - `--reference PATH` may be repeated. It defaults to `docs/results/scout-kv/reference/seed{0..4}.jsonl`, resolved relative to the repo root (as `GATES_FILE` is).
    - Prints JSON and exits 0. An input error prints to stderr and exits 1.
  - **Docs:** add both to the Commands section of `docs/datasets/scout_kv.md`.
- [ ] **Step 4:** run `tests/test_kv_gate.py`, then `make check`.
- [ ] **Step 5:** commit: `kv_gate: per-layer subset measurement and compare-layers`.

### Task 3: Control sets of `h`

**Files:**
- Create: `src/kv/controls.py`, holding the pure functions. It lives in `src/` because `docs/results/scout-kv/` contains a hyphen, so it is not importable. `make check` also lints only `src` and `tests`.
- Create: `docs/results/scout-kv/controls/controls.py`, a thin argparse script that calls `src.kv.controls.measure_controls`, following `docs/results/scout-kv/v0/intrinsic.py`. All logic lives in `src/kv/controls.py` so that it is tested.
- Test: `tests/test_scout_kv_controls.py`

The three controls. Each yields `(h_keys, h_query, h_id)` with sizes `(31736, 1024, 1024)` per layer, measured through `measure_h_layer`:

| `source` | built from | what it isolates |
|---|---|---|
| `pooled_real` | books 12-19, position 0 dropped, every remaining position pooled; one seeded permutation split into keys, queries and key-side queries, disjoint | the end-of-book key/query split: the same `h` as the reference, sampled the synthetic way |
| `norm_equalised` | `pooled_real`'s three tensors, each row rescaled to the mean row norm of the three tensors together; directions unchanged | how much the spread of `h` norms contributes |
| `gaussian` | mean and full covariance of books 0-11 (position 0 dropped), fitted in float64; samples drawn with a seeded generator | whether second-order structure alone is as easy as v0, or as hard as real |

A fourth control needs no code: **`train_books_real`**, the standard real path over books 0-7, measured with `measure-real --books 0-7 --layers 11,19,43` in B7. It checks whether the training books are as hard as the reference books.

- [ ] **Step 1: Write the tests first.** Use small synthetic arrays, not the capture.
  1. `pooled_split(books, n_keys, n_tail, seed, n_sink=1)`. `books` is a list of full-length per-book arrays with position 0 included. It returns `(h_keys, h_query, h_id, idx_keys, idx_query, idx_id)`, where the indices are into the concatenation of the books after the first `n_sink` rows of each are dropped.
     - The returned rows come from the pooled rows with position 0 of each book excluded. Give position 0 a sentinel value and assert it never appears. Also build the pool independently in the test as `np.concatenate([b[n_sink:] for b in books])` and assert `h_keys == pool[idx_keys]`, `h_query == pool[idx_query]` and `h_id == pool[idx_id]`. The sentinel alone catches only a sink that is not dropped; this also catches one row too many dropped.
     - Key, query and key-side rows are disjoint. Assert this on row indices, which the function returns alongside the tensors.
     - The sizes are exact.
     - The same seed gives identical output, and a different seed gives different output.
     - It raises if the pool has fewer than `n_keys + 2 * n_tail` rows.
     - Catches: a sink off-by-one, overlapping splits, and an unseeded permutation.
  2. `equalise_norms(h_keys, h_query, h_id)`:
     - Every output row's norm equals the mean input row norm, computed over all three tensors together, to 1e-5 relative.
     - The cosine between each output row and its input row is above 1 - 1e-6.
     - Fixture: the three tensors are drawn at different scales (for example `h_query` x 3 and `h_id` x 0.5), so that per-tensor normalisation gives a visibly different result.
     - Catches: normalising each tensor separately, and normalising to 1.
  3. `fit_gaussian(arrays)` followed by `sample_gaussian(fit, n, seed)`, on a 6-dimensional fixture of rank 4. Dimensions 0-3 are independent with variances 0.25, 1, 4 and 9, chosen away from 1 so that using eigenvalues where square roots belong changes the result. Dimensions 4 and 5 are exact linear combinations of 0-3.
     - The sample mean and covariance from 200,000 draws match the fit: the diagonal to 2% relative, and every entry of the full covariance to 2% of the largest diagonal entry, absolute. The full check catches a wrong rotation (`V.T` in place of `V`), which the diagonal alone may not.
     - The result has no NaN.
     - Fixed seed: identical draws. A different seed: different draws.
     - Catches: eigenvalues used where their square roots belong, a Cholesky that fails on a singular matrix, and an unseeded generator.
     - Implementation hint: `eigh`, negative eigenvalues clamped to 0, `x = mu + z @ (V * sqrt(lam)).T`.
  4. `fit_gaussian` accumulates the sums in chunks, in float64, so 12 books of 32,767 × 5,120 never need to be resident at once. The test feeds the fixture in two chunks and compares the result with a single-pass fit, to 1e-10. Catches the sums being reset per chunk.
  5. **`measure_controls` end to end on a tiny capture.** Build 20 books of `T` = 64, D = 16 for layer 3, plus layer-3 weights with 2 KV heads, using the same writers as `test_measure_real_end_to_end_on_a_tiny_capture` (`write_bf16_safetensor`, and a local copy of `_write_multi`; do not edit `tests/test_kv_gate.py`). Run `measure_controls(cap, layers=[3], controls=["pooled_real", "norm_equalised", "gaussian"], seed=0, out_dir=tmp, threads=1, train_books=range(12), ref_books=range(12, 20), n_keys=200, n_tail=16, nlist=4, hnsw_m=8, ef_construction=16)`. Expect:
     - exactly the files `pooled_real.jsonl`, `norm_equalised.jsonl` and `gaussian.jsonl`, each holding 2 records (one per KV head);
     - in every record, `source` and `control` equal to the file's name, `n_keys` 200, `n_q_id` 16, and `n_q_ood` 80;
     - `books` equal to `[12, ..., 19]` for `pooled_real` and `norm_equalised`, and `[0, ..., 11]` for `gaussian`;
     - a second call with the same `out_dir` raises `FileExistsError` before writing anything (file modification times unchanged).
     Catches: a control written under the wrong name, the reference and training books swapped, sizes not passed through, and an overwrite.
- [ ] **Step 2:** run the tests and see them fail.
- [ ] **Step 3: Implement.**
  - **Functions:** `pooled_split`, `equalise_norms`, `fit_gaussian` and `sample_gaussian` as pure functions, plus `measure_controls(capture_dir, layers, controls, seed, out_dir, *, threads=15, train_books=range(12), ref_books=range(12, 20), n_keys=31736, n_tail=1024, nlist=256, hnsw_m=48, ef_construction=500)`. The size and index defaults are the locked conditions; the box run never overrides them.
  - **`measure_controls` and the script:**
    - **Script arguments:** `--capture-dir`, `--layers` (default `11,19,43`), `--controls` (default all three), `--seed` (default 0), `--out-dir`, `--threads` (default 15). `measure_controls` checks every output file for existence before any work, then calls `faiss.omp_set_num_threads(threads)` once, before any profile.
    - **Output:** one JSONL per control, `<out-dir>/<source>.jsonl`. Refuse to overwrite an existing file.
    - **Layer loop:** per layer, load the books with `src/data/kv_capture.py::read_safetensor` (via `h_path`) one book at a time for the Gaussian fit. Then call `measure_h_layer(..., source=<name>, seed=seed, extra={"control": <name>, "books": [...]})`.
    - **Gaussian sampling:** run it on the GPU when one is present; the box has the L40.
    - **Logging:** print one line per layer per control, with seconds.
- [ ] **Step 4:** run `make check`.
- [ ] **Step 5:** commit: `results(scout_kv): control sets of h for the v0 diagnosis`.

### Task 4: Neighbour positions of real queries

**Files:**
- Modify: `src/kv/corpus.py` (`Corpus` gains position and key-side book fields; new `corpus_seed`)
- Modify: `src/eval/kv_gate.py` (every inline `1009 * layer + g + 7919 * seed` becomes `corpus_seed(layer, g, seed)`)
- Create: `src/kv/neighbours.py` (pure functions plus `measure_neighbours`)
- Create: `docs/results/scout-kv/controls/neighbours.py`, a thin argparse script calling `measure_neighbours`
- Test: `tests/test_kv_corpus.py` (C1-C3), `tests/test_kv_neighbours.py` (N1-N5, S1)

Run this task after Task 2. Both edit `src/eval/kv_gate.py`, so they must not run in parallel.

**The question.** Where in its book do a real-style query's exact top-10 keys sit? There are three readings, and each is structure that an unconditional generator cannot produce, because it draws no positions:
- **recency:** same-book neighbours concentrated at a small lag `q_pos - key_pos`, compared with the lags from the query to all keys of its own book;
- **early tokens:** neighbours concentrated at low positions (after the dropped sink), compared with the positions of all keys;
- **hubs:** a few keys taking many top-10 slots, and where those keys sit.

**Definitions** (fixed now; per corpus, computed separately for the query sets `ood` and `id`, with k = 10). Ground truth is exact inner-product top-10 computed exactly as `_own_book_share_top10` computes it: raw `q @ keys.T` in float32 numpy, then `np.argpartition(-sims, 9, axis=1)[:, :10]`. No centring.
- `own_book_share`: share of top-10 slots whose key is from the query's book.
- `edges = [1, 2, 4, ..., 32768]`: 16 edges, 15 log2 bins, used for both positions and lags. Use `np.histogram` with these edges. Every key position (1 to 31,743) and every lag (1 to 32,766) falls inside them.
- `nbr_pos_share`: histogram of `key_pos` over all top-10 slots, divided by the slot count. `key_pos_share`: histogram of `key_pos` over all keys, divided by `n_keys`.
- `nbr_lag_share`: histogram of `q_pos - key_pos` over the **same-book** slots only, divided by their count. It is `None` when there are no same-book slots. `base_lag_share`: for each query, the histogram of lags to **every key of its own book**, normalised to sum 1 per query, then averaged over queries.
- Scalars: `share_lag_le_1024` is the share of same-book slots with lag <= 1024, and is `None` when there are no same-book slots (test with `== 0`, not truthiness). `base_share_lag_le_1024` is, for each query, the share of its own-book keys with lag <= 1024, averaged over queries; `share_pos_lt_16` (all slots, `key_pos < 16`) and `base_share_pos_lt_16` (all keys).
- Hubs: `n_hub = max(1, n_keys // 100)`. Count slot occurrences per key. `hub_slot_share` is the sum of the `n_hub` largest counts divided by the total slots, which is invariant to ties. `hub_median_pos` is the median `key_pos` of the keys at `np.argsort(-counts, kind="stable")[:n_hub]`. `key_median_pos` is the median `key_pos` of all keys.

- [ ] **Step 1: Write the tests first.**
  - **C1. Positions reproduce the corpus.** Use the existing `_setup(2)` in `tests/test_kv_corpus.py` (`T` = 64, D = 16, 2 KV heads; columns 2 and up are random, so every row differs), with `n_sink` 1, `n_tail` 16 and `g` = 0. Do not add a second module-level `T`. For every row `i`, check each of the following to `rtol` 1e-5 and `atol` 1e-6:
    - `c.keys[i]` equals `head_keys(h[c.key_book[i]][c.key_pos[i]][None], wk, 0)[0]`;
    - `c.q_ood[i]` equals `head_queries(h[c.q_book[i]][c.q_pos[i]][None], wq, 0, 5)[i % 5]`;
    - `c.q_id[i]` equals `head_keys(h[c.q_id_book[i]][c.q_id_pos[i]][None], wk, 0)[0]`.
    Also check the ranges: `key_pos` lies in `[n_sink, T - n_tail)`, and `q_pos` and `q_id_pos` lie in `[T - n_tail, T)`. Catches: `np.tile` used for `np.repeat`, positions recorded off by the sink or tail offset, and the wrong book.
  - **C2. The draws are unchanged.** In the test, replicate the two draws per book in their current order with `np.random.default_rng(seed)`: `np.sort(rng.choice(np.arange(n_sink, Tk), per, replace=False))`, then `np.sort(rng.choice(np.arange(Tk, T), tail_per, replace=False))`. Assert that `key_pos` and `q_id_pos` equal them for both books. Catches: a draw added, removed or reordered, which would silently change every committed reference corpus.
  - **C3.** `synthetic_corpus` sets all four new arrays to -1, with lengths matching `keys`, `q_ood`, `q_id` and `q_id`.
  - **N1. A hand-built corpus with known neighbours.** Build a `Corpus` directly, in D = 2.
    - Keys: book 0 at positions 1 to 20, book 1 at positions 101 to 120 (40 keys). Key vector `[score, 0]`. Book-0 keys at positions 11 to 20 have score `100 + pos`, book-0 keys at positions 1 to 10 have score `pos`, and book-1 keys have score 0.5.
    - OOD queries, both `[1, 0]`: Q1 from book 0 at position 40, Q2 from book 1 at position 130.
    - Both top-10s are book-0 positions 11 to 20. Expected values:
      - `own_book_share` 0.5;
      - `nbr_lag_share`: 1.0 in bin [16, 32), 0 elsewhere, because Q1's lags are 20 to 29 and Q2 contributes nothing;
      - `base_lag_share`: 0.15 in [8, 16), 0.65 in [16, 32) and 0.20 in [32, 64), because Q1 gives 12/20 and 8/20, and Q2's own-book lags of 10 to 29 give 6/20 and 14/20;
      - `nbr_pos_share`: 0.5 in [8, 16) and 0.5 in [16, 32);
      - `key_pos_share` (divided by 40): 1 in [1, 2), 2 in [2, 4), 4 in [4, 8), 8 in [8, 16), 5 in [16, 32) and 20 in [64, 128);
      - `share_pos_lt_16` 0.5 and `base_share_pos_lt_16` 0.375.
    - Catches: other-book slots counted in the lag histogram (which halves [16, 32) to 0.5 and puts 0.5 in [64, 128), from Q2's lags of 110 to 119), a baseline over all keys instead of own-book keys (which gives negative lags for Q1), and the lag sign flipped.
  - **N2. Key-side queries use their own positions.** Add to N1 two key-side queries, both `[1, 0]`: one from book 0 at position 45 and one from book 1 at position 125. Expect `id.own_book_share` 0.5, and `id.nbr_lag_share` 0.7 in [16, 32) and 0.3 in [32, 64) (lags 25 to 34). Catches the `id` path reading the OOD positions, which gives lags 20 to 29, all in [16, 32).
  - **N3. Hubs.** Build 150 keys in D = 8, with `key_pos = 1 + index`. Key 7 is `[1000, 0, ..., 0]`. The other keys have coordinate 0 equal to 0 and the rest drawn N(0, 1), seeded. There are 50 queries with coordinate 0 equal to 1 and the rest N(0, 1), seeded. As a fixture precondition, assert in the test that no other key appears in all 50 top-10s. Expect `n_hub` 1, `hub_slot_share` exactly 0.1 and `hub_median_pos` 8. Catches: a share over queries instead of slots (which gives 1.0), `n_hub` taken from the query count, and the median taken over all keys.
  - The expected values in N1-N3 were checked against an independent numpy implementation when this task was written (2026-10-19).
  - **N4.** `neighbour_positions` raises `ValueError` on a corpus from `synthetic_corpus`, whose positions are -1.
  - **N5.** On a random `mixed_real_corpus` built from C1's capture (46 keys, 80 OOD queries), `ood.own_book_share` equals `kv_gate._own_book_share_top10(c)` exactly. Catches k other than 10, and the `id` queries used in place of `ood`.
  - **S1. It pairs with `measure_real`.** On the tiny capture of `test_measure_real_end_to_end_on_a_tiny_capture`, rebuilt in the new test file as a helper (do not edit the existing test), run `kv_gate.measure_real(..., seed=3)` and `measure_neighbours(..., seed=3)` with the same books, layer, `n_sink` and `n_tail`. For each KV head, the two records' own-book shares are equal. A second `measure_neighbours` call to the same `out` raises `FileExistsError`. Catches a seed formula that differs from `measure_real`'s, which would pair neighbour statistics with the wrong corpus. Seed 3 rather than 0 is deliberate, so that a formula that drops the `7919 * seed` term fails.
- [ ] **Step 2:** run the tests and see them fail. C2 passes before the change only if the positions already exist, so it must fail with an `AttributeError` here. That is expected.
- [ ] **Step 3: Implement.**
  - **`Corpus`:** add `key_pos`, `q_pos`, `q_id_pos` and `q_id_book` as the last four fields. Set them in both constructors. For the real corpus they are `kpos`, `np.repeat(tpos, r)`, `tpos` and `np.full(tail_per, b)`, concatenated over the books. Do not add, remove or reorder any `rng` call.
  - **`corpus_seed(layer, g, seed) -> int`:** returns `1009 * layer + g + 7919 * seed`. Replace every inline copy of the formula in `src/eval/kv_gate.py` with it.
  - **`neighbour_positions(c, k=10) -> dict`:** returns `{"ood": {...}, "id": {...}, "edges": [...], "n_hub": ...}` with the fields defined above. All values must be JSON-serialisable (lists and floats, not numpy arrays). It raises `ValueError` if any of `key_pos` is negative.
  - **`measure_neighbours(capture_dir, books, layers, seed, out, *, n_sink=1, n_tail=1024)`:**
    - It refuses to overwrite `out`.
    - It loads data exactly as `measure_real` does: `load_books`, then `.to(dev).float()`, then `load_layer_weights`.
    - Per KV head, it builds `mixed_real_corpus(..., seed=corpus_seed(layer, g, seed))` and appends `{layer, kv_head, source: "real", seed, books, n_keys, n_sink, n_tail, k: 10, **neighbour_positions(c), seconds}`.
    - It prints one line per head. It needs no faiss.
  - **Script:** `--capture-dir`, `--books` (default `12-19`, parsed with `kv_gate.parse_books`), `--layers` (default all of `GLOBAL_LAYERS`), `--seed` (default 0), `--out`, `--n-sink`, `--n-tail`.
- [ ] **Step 4:** run `tests/test_kv_corpus.py`, `tests/test_kv_neighbours.py` and `tests/test_kv_gate.py`. Every existing test passes without edits. Then run `make check`.
- [ ] **Step 5:** commit: `kv: neighbour positions of real queries`.

---

### Box tasks (orchestrator only)

### B7: Measure the controls on tig-gpu-alt

1. **Ship the code.** Bring `/workspace/repo` on the box to this branch's HEAD. Use `git bundle` streamed over ssh, or a `tar` stream of the changed files, and verify sha256 on both sides. Run `tests/test_kv_gate.py` and `tests/test_scout_kv_controls.py` there with `/workspace/kvenv`.
2. **Run, detached** (`setsid nohup ... &`), with `OMP_NUM_THREADS=8`, logging to `/workspace/logs/b7.log`. Record the PID.
   - `python -m src.eval.kv_gate measure-real --capture-dir /workspace/kv/capture --books 0-7 --layers 11,19,43 --seed 0 --out /workspace/kv/controls/train_books_real.jsonl`
   - `python docs/results/scout-kv/controls/controls.py --capture-dir /workspace/kv/capture --out-dir /workspace/kv/controls --seed 0`
   - `python -m src.eval.kv_gate measure-run --capture-dir /workspace/kv/capture --layers 11,19,43 --run-dir 11=runs/scout_kv/v0/L11 --run-dir 19=runs/scout_kv/v0/L19 --run-dir 43=runs/scout_kv/v0/L43 --ema --seed 0 --out /workspace/kv/controls/v0_best_subset.jsonl`. `--ema` matches `docs/results/scout-kv/v0/b6.sh`. It is a no-op on `best_generator.pt`. This re-measures v0 through the new path. It must match the matching rows of `docs/results/scout-kv/v0/best.jsonl` up to HNSW thread nondeterminism (see the gotchas). Report any difference.
   - **`--threads`:** use 15, the value every committed record used, so the timings compare.
   - First in the script, because it is the cheapest: `python docs/results/scout-kv/controls/neighbours.py --capture-dir /workspace/kv/capture --seed 0 --out /workspace/kv/controls/neighbours.jsonl`, which covers all 12 layers and books 12-19 by default.
   - **Acceptance check for the neighbour run, before writing anything up.** For every one of the 96 corpora, `ood.own_book_share` must be within 1e-3 of the `own_book_share_top10` in `docs/results/scout-kv/reference/seed0.jsonl` for the same (layer, kv_head). One flipped slot moves the share by 1/51,200, so 1e-3 allows for float32 ties flipping when the projection runs on the L40 instead of the A100. If any corpus is outside that, stop and report: the corpora are not the ones `seed0` profiled, and the pairing in R5 is void.
3. **Cost:** ESTIMATE (unverified) about 30-45 min in total.
   - Profiling layers 11, 19 and 43 took 328-460 s per run on tig-gpu (MEASURED from the `seconds` fields of the committed records).
   - There are 5 profiling runs, plus the neighbour run. The neighbour run is ESTIMATE (unverified) 10-20 min: 96 exact top-10 searches with no index, and about 32 GB of reads (12 layers x 8 books x 335 MB).
   - Reading 12 books per layer for the Gaussian fit adds about 4.0 GB of reads per layer. One `h_L11.safetensors` is 335,544,400 bytes (MEASURED, `ls -l` on tig-gpu-alt).
   - The L40 box's speed is not yet measured.
4. **Copy back** the five control JSONL files, `neighbours.jsonl` and the log, by `tar` stream, with sha256 verified on both sides.

### B8: Write it up

1. Run `compare-layers` for each of the five control JSONL files, plus v0 best. (`neighbours.jsonl` is not a gate record and does not go through `compare-layers`.)
2. Write `docs/results/scout-kv/controls/README.md`. Its neighbour section has a per-layer table of the median over the 8 heads of each scalar against its baseline (`own_book_share`, `share_lag_le_1024` against `base_share_lag_le_1024`, `share_pos_lt_16` against `base_share_pos_lt_16`, `hub_slot_share`, and `hub_median_pos` against `key_median_pos`), for `ood` and `id`, next to that layer's measured hardness from `seed0.jsonl`. `report.html` adds the enrichment per log2 bin (`nbr_lag_share / base_lag_share` and `nbr_pos_share / key_pos_share`) per layer. Beyond that, the README holds one table per layer: rows are the statistics, and columns are the real range, then `train_books_real`, `pooled_real`, `norm_equalised`, `gaussian` and v0. Label every value MEASURED.
3. Write a self-contained `report.html`, with charts as inline SVG and no external assets (the repo rule for analysis reports). Open it in headless Chromium and check that every chart renders before calling it done.
4. Apply the decision rules below as written. Report what they say, and do not reinterpret them after seeing the data.

## Decision rules (fixed before the run)

For each control, count how many of the three layers put it outside the real per-layer range **in the same direction as v0**. Do this for each of `hnsw_reach_k10`, `hnsw_recall_k10_ef176` and `ivf_ratio_k128`.

- **R1. Hard stop: does the gate's measurement path flatter synthetic data?** It does if `pooled_real` is outside in v0's direction on at least 2 of 3 layers, for at least 2 of those three statistics. In that case real `h` sampled the synthetic way is already "too easy", and part of v0's failure is the asymmetry, not the generator.
  - **Stop.** Do not design v1.
  - Report to the owner. The decision about the gate belongs to them. Options include generating queries with a position-conditioned generator, or re-basing the reference on pooled sampling.
- **One draw against five.** Each control is a single draw (seed 0), set against the range of five real draws. If R1's verdict hangs on one layer-statistic cell, so that flipping it would flip R1, re-measure `pooled_real` at seeds 1 and 2 before reporting, and apply R1 to every seed.
- **R2. Are the training books representative?** If `train_books_real` is outside the real range in v0's direction on 2 or more layers, for 2 or more of the three statistics, the generator learned from easier data than it is judged against. Report it as a finding next to R1.
- **R3 to R4: descriptive only.** No automatic conclusion follows from these; report them for the v1 design.
  - **R3:** where `gaussian` sits between v0 and real.
  - **R4:** whether `norm_equalised` moves `pooled_real` toward v0.
  - **R5: neighbour positions against hardness.** Over the 96 corpora, report the Spearman rho between each of four `ood` scalars and the same corpus's seed-0 hardness. The scalars are `share_lag_le_1024 - base_share_lag_le_1024`, `share_pos_lt_16 - base_share_pos_lt_16`, `hub_slot_share` and `own_book_share`. The hardness is `plain_ood_k10.hnsw_recall_op` and the per-corpus IVF ratio (`plain_ood_k128.ivf_scan_frac / plain_id_k128.ivf_scan_frac`, skipping corpora where either is unreached, as `kv_gate.ivf_ratios` does). Report rho and n for every pair. Call a pair "associated" only when |rho| >= 0.5. Per layer, a bin's enrichment is the median over the 8 heads of the per-corpus ratio (`nbr_lag_share / base_lag_share`, or `nbr_pos_share / key_pos_share`). Count a bin only where the baseline share is at least 0.01. Report every bin with enrichment >= 2 at 4 or more of the 12 layers. Compute Spearman from ranks with numpy (`scipy` is in the venv but is not a declared dependency). No conclusion about v1 follows automatically.

## Not in this plan

- **v1's design.** After B8, the owner picks the direction. Then follow the repo's design flow: brainstorm, write a spec under `docs/ai/specs/`, write an audited plan.
- **Any change to `gates/scout_kv.yaml`.**
- **Training anything.** Measuring training speed on the L40 is left for the first v1 run.
- **Tearing down `tig-gpu`.** That is the owner's call.
