> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.
>
> **The API below is out of date.** A 2026-08-13 code review found three
> measurement bugs in the shipped harness, and fixing them changed its shape:
> `benchmark_cell` was replaced by `run_repeat` plus an interleaved scheduler
> in `run_grid`, and `throughput_vectors_per_second` was split into
> `generate_vectors_per_second` and `end_to_end_vectors_per_second`. See
> `PROJECT_DOCUMENTATION.md` § "4b) Benchmark sampling cost" for the current
> interface and `docs/results/generation-timing/README.md` for which of this
> design's conclusions were withdrawn.

# Generation-Timing Benchmark

**Date:** 2026-08-11
**Branch:** `feat/generation-timing-benchmark`
**Status:** Design approved, pending implementation plan

## Problem

A challenge is being designed around this project's synthetic corpora, and it
draws a fresh corpus **per instance, at runtime, on a GPU**. That places
generation on the critical path of a per-instance deadline, so the design
needs to know what a draw of N vectors costs before it can choose N, choose an
architecture, or size the runtime.

Today the only available number is the pair of lines `src/sample/generate.py`
prints at the end of a run:

```
Generation timing: compute=1.234s total=5.678s throughput=81037.3 vectors/s
```

That is one architecture, one N, one repeat, and an opaque `total` that folds
CUDA context creation, checkpoint load, kernel warmup, the device-to-host copy
and `np.save` into a single figure. It cannot answer which of those a
per-instance runtime actually pays, and it does not cover the `gated` or
`structured_gated` generators at all unless a trained checkpoint for each
happens to be on disk.

This is a measurement exercise, not an optimisation one. Nothing here tunes
generation; the deliverable is a defensible table of numbers to design
against.

## Scope

**In scope.** A benchmark harness that sweeps corpus size against generator
architecture on the GPU, decomposes the cost into phases, and emits JSON plus
a markdown table.

**Out of scope.** Tuning generation throughput; CPU numbers; a CI performance
regression gate; changing `src/sample/generate.py`. Each was considered and
deliberately excluded — the driving question is what a challenge instance
costs, not how to make it cheaper or how to stop it regressing.

## Approach

Generators are built straight from configs with `build_generator`, at random
initialisation. **Generation time depends on architecture shape, not on weight
values**, so no trained checkpoint is required. This matters: only SIFT has a
trained ladder, so a checkpoint-driven benchmark could not cover all three
architectures today. `--checkpoint` remains available for the case where a
specific checkpoint's load cost is wanted.

The rejected alternative was to drive the existing `src/sample/generate.py`
CLI as a subprocess per grid cell and parse its stdout. That measures the
literal shipping path including interpreter startup and CUDA context creation,
which is genuinely what a per-instance runtime pays *if* it forks a process
per instance. It was rejected because stdout parsing is brittle, repeats
re-pay startup, and the resulting number cannot be decomposed. Reporting
`cuda_init` as its own phase recovers what that approach was worth: add the
phases your runtime actually pays.

## What gets measured

A per-instance total is not one number. The phases scale differently in N, and
which ones a runtime pays depends on decisions the challenge has not made yet,
so they are timed and reported separately.

| Phase | Scope | Rationale |
|---|---|---|
| `cuda_init` | once per process | Context creation is seconds-scale and paid once per process. Whether an instance pays it depends on whether the runtime forks. Measured by timing a first one-element device allocation before the grid starts, and reported in `environment`, not per cell. |
| `build` | per config | `build_generator(...).to(device)`, `.eval()`, plus `torch.load` when `--checkpoint` is given. |
| `warmup` | per config | One forward at the benchmark batch size, absorbing lazy initialisation and kernel autotune so they do not contaminate steady state. |
| `generate` | per cell | The batched loop: `randn` on device, forward, L2 normalize. Device-side only. |
| `to_host` | per cell | `.cpu().numpy()` per batch and assembly into one `(N, d)` array. Reported separately so a design that builds its index on-GPU can subtract it. |
| `save` | per cell, opt-in | `np.save`. Off unless `--save-dir` is given; not every design writes a file. |

**Every timed region is fenced with `torch.cuda.synchronize()`** (a no-op off
CUDA). CUDA kernel launches are asynchronous, so an unfenced region times the
launch queue rather than the execution, and the whole benchmark would be
fiction. This is the single most important correctness property in the
harness.

Each cell runs `--repeats` times (default 5) and reports **min, median and
p95** for the repeated phases. Min is the machine's ceiling, median is the
typical case, and p95 is what a per-instance deadline should be budgeted
against. `build` and `warmup` are once-per-config and reported as scalars.

Each cell also records `torch.cuda.max_memory_allocated()`, reset per cell.
How much VRAM an instance needs is the sibling question to how long it takes,
and the measurement is free at this point in the code. Off CUDA the counter
does not exist, so `peak_vram_bytes` is `null` — which is the case every test
runs under.

The harness preallocates `np.empty((N, d), np.float32)` and fills slices,
rather than appending to a list and calling `np.concatenate` the way
`generate.py` does. Concatenation holds both the list and the result at once,
doubling peak host memory at N = 1M. Whether `generate.py` should adopt the
same fill is a separate judgement and is not part of this work.

## The grid

- **N ladder:** 1 000, 10 000, 100 000, 1 000 000. Override with repeated
  `--num-samples`.
- **Architectures:** `configs/sift/v1.yaml` (`mlp`), `configs/sift/v2.yaml`
  (`gated`), `configs/sift/v4.yaml` (`structured_gated`). Override with
  repeated `--config`.
- **Batch size:** fixed at 4096, matching `generate.py`'s default. Recorded in
  the JSON, not swept. Sweeping it would triple the grid to answer a tuning
  question that is out of scope.
- **Descriptor dim:** 128 throughout the default grid.

The three default configs are a matched triple. All three declare
`latent_dim: 128`, `generator_hidden_dims: [512, 1024, 1024]`,
`negative_slope: 0.2` and a 128-dimensional SIFT output; they differ only in
`generator_type` and its gate hyperparameters. Architecture is therefore the
only varying factor, and a difference in the table is attributable to it.

Because the default grid is fixed at 128 dimensions, it does **not** speak to
the wider families — GIST at 960 dimensions and openai at 1536 will cost more
per vector. `--config` accepts any config, so those rows can be added when a
family needs one. Holding dimension fixed by default is what keeps
architecture unconfounded with descriptor width.

That is 12 cells at 5 repeats, comfortably one queued GPU job.

## Module design

`src/sample/benchmark.py`, invoked as `python -m src.sample.benchmark`. It
sits beside `generate.py` because it is the sampler's concern; `src/eval/`
holds ANN-difficulty and distribution diagnostics, which this is not.

Four units:

- **`benchmark_cell(generator, *, num_samples, batch_size, latent_dim, device,
  repeats, save_dir) -> dict`** — one config at one N. Owns the fenced timing,
  the batched loop and the preallocated fill. Takes a built module and knows
  nothing about configs, checkpoints or files, which is what makes it testable
  against a two-line `nn.Module` on CPU.
- **`run_grid(config_paths, num_samples, ...) -> list[dict]`** — loads and
  parses configs, calls `build_generator`, drives the cells, and releases each
  cell's array before the next so a 1M cell does not stack with its successor.
- **`format_markdown_table(cells) -> str`** — pure, no I/O.
- **`main()`** — argument parsing, environment capture, and the two writes.

L2 normalization is imported from `normalize_l2` in
`src/train/train_wgan_gp.py` rather than re-inlined. The import is
side-effect free. Adding a fifth copy of the rule would put the timed path out
of step with the trained path, which is exactly the failure the earlier
`l2_normalize` consolidation was meant to end.

### Command-line interface

| Flag | Default | Notes |
|---|---|---|
| `--config` | the matched triple | Repeatable. |
| `--num-samples` | 1000, 10000, 100000, 1000000 | Repeatable. |
| `--batch-size` | 4096 | Matches `generate.py`. |
| `--repeats` | 5 | Per cell. |
| `--device` | the config's `device` | Resolved through `src.device.resolve_device` without `strict`. |
| `--seed` | 42 | Matches `generate.py`. |
| `--output-dir` | required | Receives both output files. |
| `--save-dir` | none | When given, enables the `save` phase. |
| `--checkpoint` | none | Valid only with exactly one `--config`; otherwise the run errors rather than guessing which config a checkpoint pairs with. |

Timings are only trustworthy with exclusive access to the card. The queue
already provides that, and the `environment` block records the device so a
contended run is identifiable after the fact.

## Output

`--output-dir` receives `generation_benchmark.json` and
`generation_benchmark.md`.

```json
{
  "environment": {
    "device": "cuda:0",
    "gpu_name": "…",
    "torch_version": "…",
    "cuda_version": "…",
    "cuda_init_seconds": 0.0,
    "batch_size": 4096,
    "repeats": 5,
    "seed": 42,
    "timestamp": "2026-08-11T00:00:00Z"
  },
  "cells": [
    {
      "config": "configs/sift/v1.yaml",
      "generator_type": "mlp",
      "latent_dim": 128,
      "descriptor_dim": 128,
      "num_samples": 100000,
      "build_seconds": 0.0,
      "warmup_seconds": 0.0,
      "generate_seconds": {"min": 0.0, "median": 0.0, "p95": 0.0},
      "to_host_seconds": {"min": 0.0, "median": 0.0, "p95": 0.0},
      "save_seconds": null,
      "peak_vram_bytes": 0,
      "throughput_vectors_per_second": 0.0
    }
  ]
}
```

`save_seconds` is `null` when `--save-dir` was not given. Throughput is
derived from the median `generate` time.

The markdown is one row per cell: config, architecture, N, median `generate`,
median `to_host`, p95 `generate` + p95 `to_host` as a budgeting total,
throughput, and peak VRAM. That total deliberately excludes `cuda_init`,
`build`, `warmup` and `save`, because whether an instance pays those is the
open design question; they stay in the JSON to be added back as needed.

Curated results belong in `docs/results/generation-timing/` — the JSON, the
table, and the queue job spec that produced them — matching the shape of
`docs/results/v4-logratio/`. `runs/` is gitignored, so the script writes there
and the numbers are copied across once they look sane.

## Testing

`tests/test_benchmark_generation.py`, CPU-only, with tiny temporary configs
and a two-line `nn.Module`, so `make check` stays in seconds:

- `benchmark_cell` returns every documented phase key; all durations are
  finite and non-negative.
- Its output array is `(N, d)` and `float32`, and an `N` that is not a
  multiple of `batch_size` yields exactly `N` rows — the partial-final-batch
  case.
- Emitted rows are unit-norm. Normalization is on the timed path, so the test
  pins that the real rule is the one being timed.
- `run_grid` over two tiny configs and two tiny N values produces four cells
  with consistent keys, and the assembled result survives a `json.dumps`
  round-trip.
- `format_markdown_table` renders a header plus one row per cell.
- `--checkpoint` with two `--config` values exits with an error.
- The three default configs exist, parse, and declare three distinct
  `generator_type` values — cheap insurance against a config being renamed or
  repointed out from under the default grid.

**No test asserts an absolute duration.** Timing assertions are how a suite
becomes flaky, and the machine running `make check` is not the machine the
numbers come from. The tests pin structure and correctness; the GPU job
produces the numbers.

## Producing the numbers

One queued GPU job covering the whole 12-cell grid, since the lane serializes
and the grid is small. The job needs no dataset staged: random weights and no
real corpus, so it is a worktree, the venv, and the module. `--output-dir`
must be a declared artifact path outside `runs/`.
