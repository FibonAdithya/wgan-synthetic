# scout_kv v2 pilot (L3, L19, L27 at seeds 42 and 1)

Result: the port check passes, and L19 and L27 meet both criteria at both seeds. L3 misses
criterion 1 (same-book share within 0.15 of real) at both seeds, so under plan step B3 the full
run (B4) is on hold until the owner decides.

Run on tig-gpu-alt (L40), commit c5a86a2, `run.sh 19:42 27:42 3:42 19:1 27:1 3:1`, from
2026-10-28T12:06:05Z to 16:21:56Z. Files are in `../runs/L{LL}_s{SEED}/`, each with a
`SHA256SUMS`; all 56 transferred files matched the hashes taken on the box.

## Claim table

| claim | value | MEASURED / ESTIMATED | command or file | when |
|---|---|---|---|---|
| port check | passed (4/4 runs inside both windows) | MEASURED | `book_judge port` → `port.json` | 2026-10-28 |
| L19 port windows | same_book 0.642–0.866, body LID 21.22–25.17 | MEASURED | `port.json` | 2026-10-28 |
| L27 port windows | same_book 0.802–1.035, body LID 18.85–22.52 | MEASURED | `port.json` | 2026-10-28 |
| L3 same_book, s42 / s1 | 0.210 / 0.184 | MEASURED | `verdict.json` | 2026-10-28 |
| L3 real same_book | 0.409 | MEASURED | `verdict.json` (from `../../locality/locality.json`) | 2026-10-28 |
| L3 chance same_book | 0.125 | MEASURED | `../runs/L03_s42/locality.json` `chance.same_book` | 2026-10-28 |
| L3 body LID, s42 / s1 | 13.06 / 12.80 (v1 final 13.86) | MEASURED | `verdict.json` | 2026-10-28 |
| L19 same_book, s42 / s1 | 0.735 / 0.753 (real 0.645) | MEASURED | `verdict.json` | 2026-10-28 |
| L19 body LID, s42 / s1 | 23.68 / 24.00 (v1 final 25.89 / 25.51) | MEASURED | `verdict.json` | 2026-10-28 |
| L27 same_book, s42 / s1 | 0.926 / 0.934 (real 0.895) | MEASURED | `verdict.json` | 2026-10-28 |
| L27 body LID, s42 / s1 | 20.23 / 19.05 (v1 final 25.03 / 25.03) | MEASURED | `verdict.json` | 2026-10-28 |
| L3 h_l2 same_book_within_2048, real / s42 / s1 | 0.160 / 0.059 / 0.043 | MEASURED | `locality.json` files | 2026-10-28 |
| L3 median position gap of same-book neighbours, real / s42 / s1 | 3223 / 5047 / 6470 | MEASURED | `locality.json` files | 2026-10-28 |
| seconds per run (train + sample + score) | 2512–2600, mean 2558.5 | MEASURED | `../runs/logs/v2_b2_pilot.log` `DONE` lines | 2026-10-28 |
| full run (18 runs) | about 12.8 h | ESTIMATE (unverified) | 18 × 2558.5 s | 2026-10-28 |

## Verdict per layer

| layer | criterion 1, both seeds | criterion 2, both seeds |
|---|---|---|
| 3 | no (gap to real 0.199 and 0.224; limit 0.15) | yes |
| 19 | yes | yes |
| 27 | yes | yes |

v1 seed 1 has no L3 checkpoint, so the L3 seed-1 run is compared with v1 seed 42 there, as the
spec states. Criterion 2 passes at L3 either way.

## What the L3 miss looks like

At L3, v2 clusters tokens by book less than real does: same-book share is about halfway from
chance to real. The same-book neighbours it does find sit farther apart in position than real
ones (median gap 5047 and 6470 tokens against 3223), and the share within 2048 positions is
0.059 and 0.043 against 0.160. At L19 and L27 the same generator slightly over-clusters. One plausible
reading is that the latent's book-level component is too weak relative to token noise at the
early layer, where the real books separate less in absolute terms. That reading is untested.

The aim needs criterion 1 at both seeds on at least 10 of 12 layers. L3 alone does not decide
it, but L7 and L11 sit close to L3 in real same-book share (0.431 and 0.458), so they may miss
the same way. The full run would show this.

## Owner decision

Options: run B4 as planned and accept L3 as one of the two allowed misses; or stop and revise
the generator for early layers first. The box is idle while this is open.
