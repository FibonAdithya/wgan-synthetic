# scout_kv v2: results

v2 meets its aim. Criterion 1 holds at both training seeds on 11 of 12 layers, which clears the
10-layer requirement; it misses at L3. Criterion 2 holds at both seeds on all 12. The search gate
still fails at both seeds, as the plan expected. The largest remaining gaps are the L19 search
difficulty and body hubness.

Spec: `docs/ai/specs/2026-10-28-scout-kv-v2-book-latent-design.md`. Plan:
`docs/ai/plans/2026-10-28-scout-kv-v2.md`. Full report with figures: `report.html`, built by
`PYTHONPATH=. python docs/results/scout-kv/v2/make_report.py`.

- Criterion 1: same-book share of h within 0.15 of real.
- Criterion 2: body LID at least v1 final minus 6.
- Aim: criterion 1 at both seeds on at least 10 layers, and criterion 2 at both seeds on all 12.

## Claim table

Everything was run on tig-gpu-alt (L40) from commit c5a86a2. The pilot ran 2026-10-28 12:06–16:22 UTC
and the full run 2026-10-28 16:58 to 2026-10-29 05:50 UTC. 219 result files were copied off the
box, and all of them matched the box-side sha256 (56 pilot, 163 full).

| claim | value | MEASURED / ESTIMATED | file | when |
|---|---|---|---|---|
| aim | true | MEASURED | `verdict.json` | 2026-10-29 |
| criterion 1, both seeds | 11 of 12 layers (miss: L3) | MEASURED | `verdict.json` | 2026-10-29 |
| criterion 2, both seeds | 12 of 12 layers | MEASURED | `verdict.json` | 2026-10-29 |
| L3 same-book share, s42 / s1 / real | 0.210 / 0.184 / 0.409 | MEASURED | `verdict.json` | 2026-10-28 |
| tightest criterion-1 pass | L39 seed 1, gap 0.137 | MEASURED | `report.html` (from `verdict.json`) | 2026-10-29 |
| tightest criterion-2 pass | L27 seed 1, body LID 19.05 vs floor 19.03 | MEASURED | `verdict.json` | 2026-10-28 |
| port check against the probe | passed, 4 of 4 runs | MEASURED | `pilot/port.json` | 2026-10-28 |
| pooled gate bands passed, v2 s42 / v2 s1 / v1 final | 1 / 3 / 1 of 5 | MEASURED | `check_s42.json`, `check_s1.json`, `../v1/full/check_v1_final_seed0.json` | 2026-10-29 |
| HNSW recall@10 ef 176, pooled, s42 / s1 (band 0.78–0.88) | 0.942 / 0.919 | MEASURED | `check_s*.json` | 2026-10-29 |
| L19 HNSW recall@10 ef 176, s42 / s1 (reference max) | 0.988 / 0.898 (0.882) | MEASURED | `report.html` (from `gate_s*.jsonl`) | 2026-10-29 |
| body hubness skew, L15–L47: real / v2 / v1 final | 37.4–52.3 / 1.9–14.9 / 3.3–6.4 | MEASURED | `report.html` (from `runs/*/intrinsic.jsonl`, `../v1/full/intrinsic.jsonl`) | 2026-10-29 |
| compute, 24 runs | 17.1 h on one L40, 2512–2678 s per run | MEASURED | `runs/logs/*.log` | 2026-10-29 |

## Verdict

| layer | same-book real | v2 s42 | v2 s1 | body LID real | v2 s42 | v2 s1 | v1 final s42 |
|---|---|---|---|---|---|---|---|
| 3 | 0.409 | 0.210 ✗ | 0.184 ✗ | 9.68 | 13.06 | 12.80 | 13.86 |
| 7 | 0.431 | 0.305 | 0.357 | 14.62 | 16.40 | 15.83 | 15.19 |
| 11 | 0.458 | 0.457 | 0.423 | 17.55 | 19.04 | 19.19 | 19.28 |
| 15 | 0.530 | 0.646 | 0.617 | 28.23 | 23.70 | 22.65 | 23.67 |
| 19 | 0.645 | 0.735 | 0.753 | 29.25 | 23.68 | 24.00 | 25.89 |
| 23 | 0.816 | 0.896 | 0.888 | 32.74 | 21.17 | 21.58 | 25.15 |
| 27 | 0.895 | 0.926 | 0.934 | 29.96 | 20.23 | 19.05 | 25.03 |
| 31 | 0.877 | 0.917 | 0.909 | 27.72 | 21.04 | 19.51 | 24.54 |
| 35 | 0.865 | 0.926 | 0.900 | 26.15 | 19.56 | 17.92 | 23.16 |
| 39 | 0.770 | 0.841 | 0.906 | 26.43 | 21.66 | 21.45 | 24.07 |
| 43 | 0.786 | 0.873 | 0.894 | 25.58 | 18.55 | 17.41 | 21.44 |
| 47 | 0.841 | 0.929 | 0.923 | 23.30 | 14.63 | 14.62 | 18.48 |

v1 seed 1 exists at L19, L27, L35 and L43 (25.51, 25.03, 23.12 and 22.07). The seed-1 runs are
compared with those values there, and with v1 seed 42 elsewhere.

- v2 clusters by book less than real at L3–L11 and more than real from L15 on. Only L3 is
  outside the tolerance.
- From L19 on, v2 body LID is 1.51–5.98 below the v1 value each run is compared with. The spec
  allows a drop of up to 6. At L27 seed 1 the drop is 5.98, so the margin there is 0.02.
- The two seeds land in different places on the pooled gate: seed 42 passes 1 band and seed 1
  passes 3, the IVF ratio and IVF recall among them. On both IVF statistics the seeds are farther
  apart than either is from v1, so the difference reflects retrain noise rather than v2.

## Known gaps

- **L19 search is too easy.** The probe found that any generated token-to-token relation makes
  L19 HNSW search too easy, and v2 keeps that gap. The seed-1 run is close to the reference range,
  but seed 42 is far above it. HNSW reach and recall fail their pooled bands at both seeds.
- **Body hubness.** Real body hubness skew is 37–52 from L15 on. v2 reaches 2–15, about where v1
  was, so the latent does not create hubs.
- **Short-range similarity.** From L15 on, real same-book cosine drops by 0.119 from lag 1 to
  lag 16, and v2 drops by 0.015. v2 has the slow book-level component without the short-range one,
  and it keeps tokens of one book more similar than real at lag 4096.

## Files

- `verdict.json`: the verdict over 12 layers.
- `gate_s{42,1}.jsonl`, `check_s{42,1}.json`, `compare_layers_s{42,1}.json`: gate records, the
  pooled check (exit 1 at both seeds) and the per-layer comparison with the reference.
- `pilot/`: the pilot verdict, the port check and the pilot README written at the stop.
- `runs/L{LL}_s{SEED}/`: per-run config, training log, gate records, locality, hub position and
  intrinsic diagnostics, with a `SHA256SUMS`. Checkpoints and captures stay on the box under
  `/workspace/kv/v2/`.
- `runs/v1_seed1_intrinsic.jsonl`: v1 seed-1 body and tail LID at L19, L35 and L43.
- `runs/logs/`: the run.sh logs, with per-run seconds.
- `run.sh`, `seed_config.py`, `intrinsic.py`, `v1_seed1_intrinsic.sh`, `make_report.py`: the
  scripts that produced the above.
