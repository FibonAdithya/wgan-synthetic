# scout_kv v1 pilot (L19, L43): go/no-go

Pilot of the v1 rung (v0 + a body/tail position flag), per
`docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md` ("Pilot, then
the full run") and `docs/ai/plans/2026-10-22-scout-kv-v1.md` (B1-B4). Trained
and measured on tig-gpu-alt (L40) on 2026-10-22 at commit 25ca533. Per-layer
numbers are diagnostics, not a gate verdict: the gate verdict needs all 12
layers (B6).

## Verdict

The spec's rule: go to all 12 layers if, at both L19 and L43, `ivf_ratio_k128`,
`hnsw_reach_k10` and `hnsw_recall_k10_ef176` each move toward real at both
sampling seeds, for one checkpoint kind, and key-side recall is not lower than
v0's by more than v0's sampling-seed spread. v1 meets a criterion if either
kind meets it in full.

| checkpoint kind | `go` | file |
|---|---|---|
| best (`best_generator.pt`, EMA) | **True** | `verdict_best.json` |
| final (`checkpoint_step_30000.pt`, EMA) | False | `verdict_final.json` |

So the pilot reads **go**, from the best checkpoints. No cell is borderline
(no move closes the gap by less than 2x the reference spread while staying
outside the target).

All numbers below are MEASURED from the jsonl files in this directory with
`python -m src.kv.judge pilot` (commands at the end).

### Best checkpoints

| layer | statistic | target | v0 s0 / s1 | v1 s0 / s1 | moves |
|---|---|---|---|---|---|
| 19 | `hnsw_reach_k10` | 0.375-0.875 | 1.000 / 1.000 | 0.625 / 0.750 | yes (inside) |
| 19 | `hnsw_recall_k10_ef176` | 0.831-0.882 | 0.972 / 0.976 | 0.880 / 0.913 | yes |
| 19 | `ivf_ratio_k128` | 0.133-0.369 | 0.095 / 0.105 | 0.266 / 0.319 | yes (inside) |
| 19 | key-side recall | not below v0 − 0.002 | 0.958 / 0.960 | 1.000 / 1.000 | not lower |
| 43 | `hnsw_reach_k10` | 0.000-0.125 | 0.375 / 0.750 | 0.125 / 0.250 | yes |
| 43 | `hnsw_recall_k10_ef176` | 0.574-0.637 | 0.824 / 0.859 | 0.617 / 0.650 | yes |
| 43 | `ivf_ratio_k128` | 2.974-3.480 | 0.760 / 0.753 | 2.702 / 2.660 | yes |
| 43 | key-side recall | not below v0 − 0.005 | 0.967 / 0.972 | 1.000 / 1.000 | not lower |

L19 targets are widened to the training books' values (owner's decision, spec
"L19 exception"). At L19 seed 1 the HNSW recall (0.913) is still above the
target and moves only because it closes v0's gap by more than the spread. At
L43 seed 1, reach (0.250) and recall (0.650) sit just outside the target. The
IVF ratio moves from about 0.76 to about 2.7 against a real 2.97-3.48. Key-side
recall reaches 1.000 at both layers; v0 fails that band (>= 0.99) pooled.

### Final checkpoints

Identical to best at L43, where the selector picked step 30,000. At L19 the
final checkpoint fails one cell: `ivf_ratio_k128` at seed 1 is 0.387, past the
widened target's top (0.369) by 0.018. v0's distance there is 0.028, so the gap
closes by 0.010, less than the spread 0.039. Overshoot counts as distance, so
the cell does not move, and `go` is False for this kind.

## Checkpoint selection (B1 and the training logs)

| layer | selected step | body score at 30k | tail score over training |
|---|---|---|---|
| 19 | 25,000 (0.592; 20,000 scored 0.593) | 0.316 | 0.314 → 0.293 |
| 43 | 30,000 (score still falling: 0.528 at 25k, 0.494 at 30k) | 0.167 | 0.415 → 0.327 |

Selector noise floor (`../selector_noise.json`, five disjoint real/real pairs
at the holdout sizes): tail 0.021-0.104 at L19 and 0.017-0.139 at L43; body
0.006-0.035 at L19 and 0.011-0.063 at L43. The tail half of the score is above
its floor throughout training, so it is not noise. But it changes little
between evaluations, so the body half does most of the ranking. L43's score
still falling at the last step suggests it is not fully trained at 30,000 steps:
ESTIMATE (unverified), one run.

## v0 on this box

v0 best was re-measured on the L40 at seeds 0 and 1 (plan B3), so v0's seed
spread does not include the A100 → L40 change. The L40 seed-0 file matches the
committed A100 `../../v0/best.jsonl` at L19/L43 on reach, HNSW recall and
key-side recall exactly. IVF ratio differs in the 4th decimal (L19 0.0955 vs
0.0953, L43 0.7606 vs 0.7604). The judge uses the L40 files.

## Cost of the full run (B5 + B6)

- Training on the L40: L19 2,091 s, L43 1,888 s (MEASURED,
  `../training/pilot_train.log`). v0 took 2,463-2,756 s per layer on the A100.
- Measurement: 228-236 s per two-layer `measure-run` (MEASURED,
  `pilot_measure.log`).
- Full run, the other 10 layers: about 5.5 h of training, then about 1.2 h to
  measure v1 best, v1 final and v0 best at seed 1 on all 12 layers.
  ESTIMATE (unverified), from the timings above.
- During measurement another process outside this container used 2.5-3.8 GB
  of the card (`nvidia-smi` listed PIDs that `ps` here cannot see). That
  affects time only, not the statistics.

## Inputs

| file | run_dir |
|---|---|
| `v1_{best,final}_seed{0,1}.jsonl` | `/workspace/repo_v1/runs/scout_kv/v1/L19`, `.../L43` |
| `v0_best_seed{0,1}.jsonl` | `/workspace/repo/runs/scout_kv/v0/L19`, `.../L43` |

Every file here was sha256-listed on the box, tar-streamed and verified
locally (`../STAGE_SHA256SUMS`).

```
python -m src.kv.judge pilot --v1 0=pilot/v1_best_seed0.jsonl --v1 1=pilot/v1_best_seed1.jsonl \
    --v0 0=pilot/v0_best_seed0.jsonl --v0 1=pilot/v0_best_seed1.jsonl --out pilot/verdict_best.json
python -m src.kv.judge pilot --v1 0=pilot/v1_final_seed0.jsonl --v1 1=pilot/v1_final_seed1.jsonl \
    --v0 0=pilot/v0_best_seed0.jsonl --v0 1=pilot/v0_best_seed1.jsonl --out pilot/verdict_final.json
```

(paths relative to `docs/results/scout-kv/v1/`; run from the repo root with
the `docs/results/scout-kv/v1/` prefix).
