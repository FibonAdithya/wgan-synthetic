# SIFT generation timing

Measured 2026-08-11 from commit `63974fa` on the exclusive `gpuq` GPU lane.
The machine had one NVIDIA GeForce RTX 4060, PyTorch 2.13.0+cu130, and CUDA
13.0. The benchmark used batch size 4096, seed 42, and five repeats per cell.
The generators were randomly initialized from the matched SIFT v1, v2, and v4
configs; these are architecture timing measurements, not quality comparisons.

`generation_benchmark.json` is the full phase-level record and
`generation_benchmark.md` is the budgeting view. `gpuq_job_spec.json` pins the
exact command, commit, lane, timeout, and exit status.

At N=1M, median device-side generation was 0.482 s for the MLP, 0.554 s for the
gated generator, and 0.779 s for the structured-gated generator. Device-to-host
copy and host-array assembly added about 0.59–0.60 s. The p95 budgeting totals
were 1.128 s, 1.203 s, and 1.441 s respectively. CUDA initialization was 0.270 s
once per process; build and warmup remain separate in the JSON.

The MLP's N=1,000 p95 contains a single startup-sized outlier even though the
configured warmup had already run: its median is 0.000633 s while its p95 is
0.051648 s. Retain the raw result, but do not use that tiny-cell p95 to infer
scaling or a deadline. The larger cells are stable and approximately linear.

These timings are operational diagnostics only. They do not measure or change
the ANN-difficulty gate.
