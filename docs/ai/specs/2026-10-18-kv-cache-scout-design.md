# KV-cache track: synthetic attention inputs for Llama 4 Scout, gated on IVF and HNSW search

Design, 2026-10-18. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built. Scope: a new
family, `scout_kv`, its data, its gate and its `v0` rung. Later rungs are
not designed here.

## Problem

At decode time a long-context model scores the current query against every
cached key. Replacing that scan with an approximate maximum-inner-product
index (IVF, HNSW) is the setting of Zharkov, Singal, Padaki and Andoni 2026
(arXiv 2610.10135). Benchmarking those indexes needs KV caches, and real ones
are expensive to produce: a forward pass of a 109B-parameter model per
corpus. This track trains a generator whose samples, run through the model's
own projections, give key and query sets that IVF and HNSW find as hard, and
hard in the same way, as real ones.

The hard part, measured on real data (`docs/results/kv-cache-profile/`), is
that queries are out of distribution with respect to keys. HNSW reaches
recall@10 >= 0.9 for real queries in 52.9% of corpora and for key-like
queries in 100%. A generator that only matched the key distribution would
miss the property that makes the problem hard.

## Decisions already made

These were made by the project owner during the 2026-10-16 to 10-18
sessions. The spec records them; it does not reopen them.

| decision | choice |
|---|---|
| source of real data | Llama-4-Scout-17B-16E-Instruct, as a data source only (no fine-tuning, no model in the loop at generation time) |
| what the generator produces | the attention input `h` (5120 dims). Keys and queries are made afterwards with the model's own weights |
| gate | IVF and HNSW search difficulty, not the four intrinsic statistics of `src/eval/ann_difficulty.py` |
| which heads | all 12 global layers x 8 KV heads = 96 corpora, pooled, following the paper. Per-layer and per-head views are diagnostics only |
| real reference | corpora mixed from several books, not single books |
| compute | the rented Vast A100-80GB (`tig-gpu`), for capture and training |

The gate decision departs from `AGENTS.md` invariant 1, which names the four
intrinsic statistics as the gate. For this family only, the search
statistics below are the gate and the four intrinsic statistics, measured
on `h`, are diagnostics. `AGENTS.md` and `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate`
must say so when the family lands, or invariant 1 will read as violated.

## Facts the design rests on

All measured on the box 2026-10-16 to 10-18, unless marked ESTIMATE.

| fact | value | source |
|---|---|---|
| global layers | 3, 7, ..., 47 (12), no RoPE | `capture/meta.json` |
| GQA | query head `j` reads KV head `j // 5`; 8 KV heads, 40 query heads, `head_dim` 128 | `capture/meta.json` |
| `h` | input to `self_attn`, i.e. the output of `input_layernorm` (RMSNorm) | `modeling_llama4.py` lines 403-406, transformers 4.57.6 |
| generation recipe | `k = W_k h`, `q = s(pos) W_q h`, `s = 1 + 0.1 ln(1 + floor((pos+1)/8192))`; worst reconstruction error 0.23%, control without `s` >= 9.5% | report, capture table |
| `s(pos)` and search | a positive per-query scale does not change any query's ranking under flat, IVF (inner product) or Xbox-HNSW search | follows from the definitions; holds exactly |
| mixed vs single book | mixed corpora reach HNSW recall@10 >= 0.9 in 57.3% of corpora vs 46.9-55.2% for the four single books; IVF statistics inside the single-book range | `mix_check.jsonl` sha256 `bdeef572...` |
| own-book preference | median share of a query's exact top-10 from its own book 0.253 (0.25 = none) | same |
| profile cost | 18.7 min for 96 mixed corpora, plain workload, 15 threads | same |
| capture cost | 78-83 s forward per 32,768-token book, plus model load | `capture/meta.json` |
| disk | 77 GB free of 300 GB; model 203 GB; capture 16 GB for 4 books | `df`, `du` on the box |
| RAM | 2,003 GB total | `free -g` on the box |

## Data

**Capture 16 more books.** The capture script takes books in PG-19 test row
order, skipping any row shorter than 32,768 tokens; books 0-3 are rows 0-3.
Books 4-19 are the next 16 qualifying rows, first 32,768 tokens each, with
the existing reconstruction check. Each book's row and title are recorded
beside its capture, and the existing `meta.json` is not rewritten. That gives
20 books. At about 4 GB per book this uses 64 GB of the 77 GB free, which
leaves 13 GB. The model stays on disk so a recapture remains possible.

**Split by book, not by token.** Tokens within a book are not independent,
so a token-level holdout would leak.

| split | books (capture order) | use |
|---|---|---|
| train | 0-11 (12 books) | generator training |
| reference | 12-19 (8 books) | the gate's real reference and its noise floor |

Rows 0-3, the profiled books, are in train. Nothing gated on is ever trained on.

**Training rows.** Per layer: positions 1-32,767 of each training book
(position 0, the attention sink, excluded), 12 x 32,767 = 393,204 rows of
5,120 dims. Read straight from the capture's bf16 safetensors and converted
to float32 in memory (8 GB per layer). No `.npy` staging, because a float32
copy of all 12 layers would need 96 GB and the disk has 13 GB.

**Preprocessing.** `center: true`, `whiten: false`, `l2_normalize: false`.
Norms carry meaning for inner-product search, so per-vector normalisation is
not allowed. Centering is inverted at sample time through the existing
`invert_preprocess`.

**Generator output normalisation.** Found while planning: the trainer
L2-normalises generator output unconditionally, at both training sites and
in `sample_generator` (`src/train/train_wgan_gp.py`, `normalize_l2`),
whatever `preprocess.l2_normalize` says. A new model key
`output_normalize` (default `true`, so every existing config is unchanged)
turns that off; `scout_kv` configs set it to `false`.

## Synthetic corpus

For one layer's generator and one KV head `g`, a synthetic corpus is:

| part | construction | count (matches real) |
|---|---|---|
| keys | `W_k[g] h` for independent samples `h` | 31,736 |
| real-style queries | `W_q[j] h` for `j` in `5g ... 5g+4`, on 1,024 independent `h`; each `h` gives 5 queries, as in the model | 5,120 |
| key-side baseline queries | `W_k[g] h` on another 1,024 independent `h` | 1,024 |

The model's weights are the ones in `capture/weights/` (verified against
Meta's checkpoint hashes). `s(pos)` is not applied for search. It is
applied, at the reference corpora's tail positions, only for the attention
error diagnostic below.

## Real reference

A reference corpus has the same three parts and counts, drawn from the 8
reference books mixed. Keys come equally from each book's positions
1-31,743: 3,967 per book, 31,736 in total (8 x floor(31,743 / 8)). The
synthetic corpus uses the same count. Queries and key-side baseline queries come equally
from each book's last 1,024 positions. This matches what the generator
produces: independent draws from many books.

**Noise floor.** 5 reference draws with different sampling seeds over the
same 8 books, each run across all 96 corpora. The spread of each gate
statistic across the 5 draws is the noise floor. Once, also measure a
4-of-8-book mixture, so whoever sets the bands can see how much the number
of books in the mixture moves the statistics (4-book mixtures moved HNSW
reach by about 2 points on the profiled books). Cost: about 1.6 h for the 5
draws (ESTIMATE, from the 18.7 min measured per 96 corpora).

## Gate

Measured over all 96 (layer, KV head) corpora with the indexes and sweeps of
the profiling run: faiss IVFFlat inner product, `nlist` 256, `nprobe` swept
1-256; HNSW on the Xbox reduction, `M` 48, `efConstruction` 500, `efSearch`
swept 10-2048. The sink and the 1,024 most recent positions are never indexed.

| statistic | definition | why it is in the gate |
|---|---|---|
| `hnsw_reach_k10` | share of the 96 corpora in which real-style queries reach recall@10 >= 0.9 within the sweep | the out-of-distribution gap; 52.9% real vs 100% key-side |
| `hnsw_recall_k10_ef176` | median recall@10 at `efSearch` 176 (the paper's setting), real-style queries | the same gap at a fixed cost |
| `ivf_ratio_k128` | median over corpora of (scan fraction at recall@128 = 0.9, real-style) / (same, key-side) | IVF difficulty relative to the easy case |
| `ivf_recall_k128_np24` | median recall@128 at `nprobe` 24 of 256 (same share of lists as the paper's 96 of 1024), real-style queries | IVF at a fixed cost |
| `hnsw_recall_k10_ef176_id` | the HNSW recall statistic for key-side queries | a check, not a gap: real is 1.000, and a generator that broke key geometry would fall here |

A band per statistic, not a combined score, as in the existing gates. **The
band values are a human decision** (`AGENTS.md`, "What requires a human").
`gates/scout_kv.yaml` ships with every band `null`, and the checker exits 2
with verdict `unset` until a person fills them in from the noise floor.
Locked conditions (key count, query counts, `nlist`, `M`, `efConstruction`,
sweep grids) live in the gate file's `canonical` block and are checked
against what the measurement recorded, as `check_gate` does now.

**Diagnostics, not gated:** the four intrinsic statistics on `h` itself;
per-layer and per-head versions of every gate statistic; HNSW work at recall
0.9; the share of corpora with 20 or fewer distinct top-10 keys (sink-like
heads); SoftmaxLift and top-k attention error with flat, IVF and HNSW
retrieval (needs `W_v`, which the capture step adds to `capture/weights/`).

## Generator, `v0`

One generator per global layer, 12 in all, each a plain run of the existing
trainer. `v0` is the existing `mlp` generator and critic, widened for 5,120
outputs. It is the baseline, not a guess at the answer.

| key | value |
|---|---|
| `generator_type` | `mlp` |
| `latent_dim` | 128 |
| `generator_hidden_dims` | [1024, 2048, 4096] |
| `critic_hidden_dims` | [2048, 1024, 512] |
| `num_gen_steps` | 30,000 |
| everything else | the SIFT `v0` optimiser settings (lr 1e-4, betas [0.0, 0.9], `n_critic` 3, `lambda_gp` 5, EMA 0.999) |
| checkpoint selection | `select_on: gate` (LID and contrast on `h`, cheap) during training; the KV gate is run afterwards on `best_generator.pt` and on the final checkpoint |

The KV gate takes about 19 minutes per evaluation (ESTIMATE), too slow to run
inside the training loop, so selection uses the existing cheap selector and
the KV gate judges the result. The cheap selector's holdout is the trainer's
usual random 5% of rows, so it is drawn from the training books; that is
acceptable for ranking checkpoints of one run, and the KV gate uses only the
reference books. Whether that cheap selector picks good KV
checkpoints is itself measured: the gate runs on both checkpoints.

Training time per layer is not known. The plan's first training task
measures one layer before the other 11 are queued.

Possible later rungs, not part of `v0`: emit `h` as RMS-normalised output
times the layer's `input_layernorm` weight, which is the form `h` takes in
the model; a critic that also sees `W_k h` and `W_q h`.

## Code

| piece | where | tested by |
|---|---|---|
| capture (needs the 203 GB model) | `scripts/kv/capture_scout.py`, ported from the spike, plus `W_v` export | the reconstruction check it already runs; its pure helpers in pytest |
| `h` to keys / queries, GQA mapping, `s(pos)`, Xbox | `src/kv/transforms.py` | pytest on small random weights against a direct computation |
| recall, work at target, IVF / HNSW sweeps | `src/kv/search_profile.py` | pytest with planted answers (a corpus whose exact neighbours are known) |
| mixed corpora, real and synthetic | `src/kv/corpus.py` | pytest: counts, book balance, no sink, no tail keys, seed reproducibility |
| gate statistics and band check | `src/eval/kv_gate.py`, `gates/scout_kv.yaml` | pytest: unset bands exit 2; real reference rows pass once bands are set; a deliberately broken corpus (keys used as queries) fails `hnsw_reach_k10` |
| capture loader | `format: kv_capture` in `src/data/dataset.py`, no `safetensors` dependency | pytest on a tiny fixture written on the box by the real `safetensors` library |
| configs | `configs/scout_kv/v0/L03.yaml` ... `L47.yaml` | config-load test |
| family page | `docs/datasets/scout_kv.md` | existing docs contract tests |

The spike scripts under `docs/results/kv-cache-profile/spike/` stay as the
record of what ran. The mixed-book check script and its results are
committed under `docs/results/kv-cache-profile/mixed-book/`.

## Needs a human before implementation

1. **New pinned dependency.** `faiss-cpu==1.15.1` is not in
   `requirements.txt`. Adding a pin is a human decision under `AGENTS.md`.
   (`safetensors` is not needed: the capture files are a fixed layout, an
   8-byte header length, a JSON header and raw bf16, which the loader reads
   directly.)
2. **The doc change for invariant 1**, worded as above.
3. **Bands**, after the noise floor is measured. Nothing ships with a band set.

## Box budget

The box bills about $1.2/h whether or not it is working. In order:

| step | box time (ESTIMATE) |
|---|---|
| capture 16 books | about 45 min |
| reference noise floor (5 draws + one 4-book mixture) | about 2 h |
| `v0`, one layer, timed | unknown; measured |
| `v0`, remaining 11 layers | 11 x the measured time |
| KV gate on 2 checkpoints | about 40 min |

There is no persistent volume. Stopping keeps the disk; destroying or
recycling erases it, including the captures. Before any teardown, the
irreproducible results (JSON outputs, gate summaries, the chosen
checkpoints with their `run_config.yaml`) come off the box by `tar` stream
with sha256 verification. The captures are reproducible from the script and
are not transferred.

## Out of scope

Local (RoPE, chunked) layers; other models; contexts other than 32,768
tokens; latency benchmarking of the indexes; a single generator conditioned
on layer.
