# scout_kv v2 probe: token-by-token vs drifting latent

A throwaway spike, not a candidate generator. It compares two ways of making
tokens within a book depend on each other, against a position-only control.
Open `report.html` for the tables and figure, and see `summary.html` for the
findings it embeds.

- **A, token by token:** h_t = dec(z_t, s_{t-1}, pos_t), with
  s_t = GRU(enc(h_t), s_{t-1}).
- **B, drifting latent:** h_t = dec(z_t, u_book(t), pos_t), with u a random
  Fourier-feature function of position drawn fresh per book (64 channels,
  length scales 2–16,384 tokens).
- **C, position only:** h_t = dec(z_t, pos_t).

The three arms share v1's decoder trunk and one critic. The critic scores
8-token windows at strides 1–1024. Each arm was trained 30k steps with
λ_gp 10, at layers 27 and 19, with seeds 42 and 1.

## Files

| path | what |
|---|---|
| `probe.py` | the three arms, training, sampling to capture format, intrinsic statistics |
| `probe_run.sh` | train → sample → gate → locality → hub_position → intrinsic, per run |
| `compare.py` | builds `summary.json` and `report.html` from `runs/` and committed v0/v1/real records |
| `cross_book_lag.py`, `control/` | real h, same-book vs different-book cosine by lag |
| `runs/L{19,27}/{A,B,C}_s{42,1}/` | small result files copied from the box. Checkpoints and synthetic books stay on tig-gpu-alt under `/workspace/kv/v2probe/` |
| `runs/SHA256SUMS_L{19,27}` | box-side hashes; `cd runs && sha256sum -c SHA256SUMS_L19 SHA256SUMS_L27` gives 84 OK |

Rebuild the report with:

```
PYTHONPATH=$PWD python docs/results/scout-kv/v2-probe/compare.py \
  --runs docs/results/scout-kv/v2-probe/runs --out docs/results/scout-kv/v2-probe \
  --layers 27,19 --summary-html docs/results/scout-kv/v2-probe/summary.html
```

## Claim table

Every value comes from this session (2026-10-26). The box ran commit
`bbe4c28` (the `START` line of `/workspace/logs/v2probe.log`), and
`git diff --stat bbe4c28 HEAD -- probe.py probe_run.sh src` is empty.

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| h-NN same-book share, real / B / A / C, L27 | 0.895 / 0.909, 0.904 / 0.615, 0.291 / 0.125 | MEASURED | `summary.json` (locality.py) | 2026-10-26 |
| same, L19 | 0.645 / 0.729, 0.656 / 0.225, 0.286 / 0.124 | MEASURED | `summary.json` | 2026-10-26 |
| key-NN same-book share, real / B, L27 and L19 | 0.653 / 0.517, 0.520; 0.327 / 0.300, 0.249 | MEASURED | `summary.json` | 2026-10-26 |
| training seconds, A / B / C (30k steps) | 2,240–2,290 / 1,380–1,510 / 1,430–1,450 | MEASURED | `train_log.jsonl` last row, `*.train.log` | 2026-10-26 |
| L27 HNSW reach, real range / v1 / B | 0.75–1.0 / 0.625 / 0.875, 0.750 | MEASURED | `summary.json` (kv_gate measure-real) | 2026-10-26 |
| L27 IVF recall, real range / v1 / B | 0.7896–0.8245 / 0.768–0.775 / 0.7895, 0.818 | MEASURED | `summary.json` | 2026-10-26 |
| L19 HNSW recall, real range / v1 / arms | 0.854–0.882 / 0.869–0.902 / 0.949–0.984 | MEASURED | `summary.json` | 2026-10-26 |
| L19 IVF ratio, A and B seed pairs | 0.167, 0.337; 0.259, 0.137 | MEASURED | `summary.json` | 2026-10-26 |
| C cosine at lag 256 vs real different-book lag 256, L27 / L19 | 0.620, 0.616 vs 0.430 / 0.475, 0.464 vs 0.329 | MEASURED | `summary.json`, `control/cross_book_lag.json` | 2026-10-26 |
| body LID, real / v1 / arms, L27 | 30.0 / 25.0 / 16.2–20.8 | MEASURED | `summary.json` (intrinsic) | 2026-10-26 |
| body LID, real / v1 / arms, L19 | 29.2 / 26.0 / 19.8–23.7 | MEASURED | `summary.json` | 2026-10-26 |
| body hubness, real / v1 / arms, L27 and L19 | 49.9, 52.3 / 5.5, 5.6 / 1.5–5.0 | MEASURED | `summary.json` | 2026-10-26 |

## Caveats

- Two layers and two seeds only. Per-layer HNSW statistics were shown to be
  retrain-noise-dominated (PR #68), so the search rows show direction, not a
  ranking.
- The windows' end positions are uniform, so the tail's share of training is
  about 3% (ESTIMATE (unverified) from 1,024 tail positions out of 32,768).
  v1 conditioned on an explicit body/tail flag instead. This is one of three changes from
  v1 that could explain the lower LID. The other two are the window critic
  and the position features.

## Ablation: which change cost LID (2026-10-26)

`ablation/report.html` covers this run. Arm C at L27 was rerun with each probe change from v1 undone:
the per-token critic (`Ctok`), the tail bit as the only position input (`Cflag`), 25% tail windows
(`Ctail`), and all three together (`Cv1`). Each variant used seeds 42 and 1 at commit 571cd65. v1 seed 1's
L27 LID is in `runs/v1s1_intrinsic_L27.jsonl`. `runs/SHA256SUMS_ablation` covers the 57 copied files.

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| body LID: real / v1 s42, s1 / C / Ctok / Cv1 | 29.96 / 25.03, 25.03 / 19.76, 19.82 / 24.73, 24.78 / 25.37, 25.35 | MEASURED | `ablation/summary.json` | 2026-10-26 |
| body LID: Ctail / Cflag | 18.78, 18.59 / 22.93, 22.24 | MEASURED | `ablation/summary.json` | 2026-10-26 |
| body hubness: v1 / Ctok / Cv1 / C | 5.47, 7.30 / 5.48, 6.55 / 7.72, 6.38 / 2.51, 2.27 | MEASURED | `ablation/summary.json` | 2026-10-26 |
| cosine at lag 256: C / Ctok / real same book / real different books | 0.620, 0.616 / 0.529, 0.530 / 0.498 / 0.430 | MEASURED | `ablation/summary.json`, `control/cross_book_lag.json` | 2026-10-26 |
| Cflag final W-distance and GP vs C | 631, 623 and 25.3, 23.2 vs 75.8, 75.1 and 2.3, 2.1 | MEASURED | `train_log.jsonl` last rows | 2026-10-26 |
| tail LID: real / C / Ctail | 12.91 / 16.00, 15.95 / 14.86, 15.03 | MEASURED | `ablation/summary.json` | 2026-10-26 |

The window critic causes the LID loss. Undoing it alone (Ctok) recovers v1's body LID, contrast and
hubness, and undoing all three changes (Cv1) reproduces v1. The window critic differs from v1's critic
in three ways, and the run does not separate them: a smaller per-token trunk, a gradient penalty taken
jointly over 8 tokens, and the window head.

## Two critics (2026-10-27)

`two-critics/report.html` covers this run. Arms B and C were trained with v1's per-token critic and the window critic side
by side, with the generator loss = token score + 0.1 × window score (`--critic both`, names `Bboth`, `Cboth`). Runs
are at L27 and L19, seeds 42 and 1, commit 7460b2e. `runs/SHA256SUMS_both` covers the 56 copied files.

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| body LID L27: real / v1 / Cboth / Bboth / B | 29.96 / 25.03, 25.03 / 25.10, 25.26 / 19.62, 19.05 / 16.19, 16.67 | MEASURED | `two-critics/summary.json` | 2026-10-27 |
| body LID L19: real / v1 s42 / Cboth / Bboth / B | 29.25 / 25.98 / 25.66, 25.65 / 22.94, 23.77 / 20.05, 19.79 | MEASURED | `two-critics/summary.json` | 2026-10-27 |
| h-NN same-book L27, L19: real / Bboth | 0.895, 0.645 / 0.889, 0.900 and 0.744, 0.532 | MEASURED | `two-critics/summary.json` | 2026-10-27 |
| key-NN same-book L27: real / B / Bboth | 0.653 / 0.517, 0.520 / 0.405, 0.499 | MEASURED | `two-critics/summary.json` | 2026-10-27 |
| L19 HNSW recall: real / v1 / Bboth / Cboth | 0.854–0.882 / 0.869–0.902 / 0.980, 0.976 / 0.941, 0.965 | MEASURED | `two-critics/summary.json` | 2026-10-27 |
| window/token per-token gradient ratio at the final step | 0.35–0.50 | MEASURED | `runs/L*/[BC]both_s*/train_log.jsonl` | 2026-10-27 |
| two-critic training time, 30k steps | 2,200–2,290 s | MEASURED | `train_log.jsonl` last rows | 2026-10-27 |

With two critics, the position-only generator keeps v1's LID, so the down-weighted window term costs nothing. Adding the
drifting latent still costs 2.5–5.5 LID points against Cboth, though it keeps the book clustering. compare.py now builds the
figure legend from every panel.

**Retraction (2026-10-27, latent/position run below).** This section's report said the sinusoidal position features
"likely make L19 recall too easy". Bflag removes them, keeping only the tail bit as v1 does, and its L19 HNSW recall is still
0.989 / 0.969 (real 0.854–0.882). The position features are not the cause. The note is repeated in
`two-critics/summary.html`.

## Latent and position inputs (2026-10-27)

`latent-position/report.html` covers this run. Both variants are Bboth with one change each, at L27 and L19, seeds 42 and 1,
commit f409ae6 (the `START` line of `/workspace/logs/v2probe_lat.log`). `runs/SHA256SUMS_lat` covers the 56 copied files
(`cd runs && sha256sum -c SHA256SUMS_lat` gives 56 OK).

- **Bflag** (`--pos flag`): the tail bit is the only position input, as in v1.
- **Bsmall** (`--cond 16 --max-len 1024`): 16 latent channels instead of 64, and length scales 2–1,024 instead of 2–16,384.

Rebuild with:

```
PYTHONPATH=$PWD python docs/results/scout-kv/v2-probe/compare.py \
  --runs docs/results/scout-kv/v2-probe/runs --out docs/results/scout-kv/v2-probe/latent-position \
  --layers 27,19 --arms B,Bboth,Bflag,Bsmall,Cboth \
  --summary-html docs/results/scout-kv/v2-probe/latent-position/summary.html \
  --title "scout_kv v2 probe: latent and position inputs"
```

| claim | value | MEASURED / ESTIMATED | source | when |
|---|---|---|---|---|
| body LID L27: real / v1 / Bboth / Bflag / Bsmall / Cboth | 29.96 / 25.03, 25.03 / 19.62, 19.04 / 21.02, 20.35 / 22.59, 23.07 / 25.10, 25.26 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| body LID L19: real / v1 s42 / Bboth / Bflag / Bsmall / Cboth | 29.25 / 25.98 / 22.94, 23.77 / 23.12, 23.14 / 24.54, 25.19 / 25.66, 25.65 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| h-NN same-book L27: real / Bboth / Bflag / Bsmall / Cboth | 0.895 / 0.889, 0.900 / 0.935, 0.902 / 0.644, 0.575 / 0.124, 0.124 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| h-NN same-book L19: real / Bboth / Bflag / Bsmall | 0.645 / 0.744, 0.532 / 0.761, 0.698 / 0.333, 0.229 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| key-NN same-book L27: real / Bboth / Bflag / Bsmall | 0.653 / 0.405, 0.500 / 0.558, 0.520 / 0.347, 0.292 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| L19 HNSW recall: real range / v1 / Bboth / Bflag / Bsmall / Cboth | 0.854–0.882 / 0.869–0.902 / 0.980, 0.976 / 0.989, 0.969 / 0.933, 0.979 / 0.941, 0.965 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| L27 HNSW recall: real range / v1 / Bflag / Bsmall | 0.839–0.867 / 0.808, 0.715 / 0.905, 0.751 / 0.804, 0.748 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| body hubness: real / Bflag / Bsmall (L27; L19) | 49.9; 52.3 / 3.25, 3.43; 3.51, 8.47 / 6.47, 5.57; 4.79, 4.11 | MEASURED | `latent-position/summary.json` | 2026-10-27 |
| lag-256 cosine L27: real same-book / real cross-book / Bflag / Bboth | 0.498 / 0.430 / 0.379, 0.565 / 0.609, 0.624 | MEASURED | `latent-position/summary.json`, `control/cross_book_lag.json` | 2026-10-27 |
| final window W-distance L27: Bboth / Bflag / Bsmall | 28.3, 33.1 / 43.4, 51.2 / 38.1, 43.4 | MEASURED | `runs/L27/*_s*/train_log.jsonl` last rows | 2026-10-27 |
| training time, 30k steps | 2,190–2,260 s | MEASURED | `train_log.jsonl` last rows | 2026-10-27 |

Removing the position features costs no LID (+1.3–1.4 at L27, unchanged at L19) and keeps the clustering, but does not fix
L19 search. Every probe run has a window critic and makes L19 search too easy, but so does v0 (0.972), which has none.
v1 is the only generator near real (0.869–0.902). The cause is not identified, and Ctok was not run at L19. The short latent recovers most of the LID (0.5–2.5 below Cboth) but loses most of the book clustering, so book
identity lives in the latent's slow components.
