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
