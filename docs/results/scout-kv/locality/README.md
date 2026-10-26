# Is real h sequence-local?

`locality.py` (2026-10-25, on real h of books 12–19) measured four things:
consecutive similarity by lag, nearest-neighbour locality in h and in key
space, and where real tail queries' hits sit. Results are in `locality.json`
and `locality.log`.

## Correction (2026-10-26): the lag curve is mostly a position effect

**Retracted:** "consecutive tokens' h are alike from L15 on, and still alike
4,096 tokens apart (0.55–0.74 at L31–L43), against 0.02–0.07 for random
same-book pairs". That reading is wrong. `locality.py` compared lagged
same-book pairs only with random same-book pairs, which are typically ~10k
tokens apart. It had no control for absolute position.

The v2 probe's position-only arm (independent tokens, conditioned on
position) produced a lag curve as high as real h. So
`../v2-probe/cross_book_lag.py` measured the missing control: the same lags
between *different* books. Results are in `../v2-probe/control/`. MEASURED,
50,000 pairs per lag, seed 42. Median cosine of centred h:

| layer | same book lag 1 | lag 16 | lag 256 | lag 4096 | different books, same position | different books, lag 4096 |
|---|---|---|---|---|---|---|
| 7 | 0.244 | 0.136 | 0.114 | 0.081 | 0.106 | 0.075 |
| 15 | 0.534 | 0.332 | 0.284 | 0.206 | 0.264 | 0.186 |
| 19 | 0.634 | 0.431 | 0.366 | 0.240 | 0.336 | 0.212 |
| 23 | 0.719 | 0.567 | 0.497 | 0.309 | 0.462 | 0.288 |
| 27 | 0.720 | 0.570 | 0.495 | 0.308 | 0.438 | 0.281 |
| 31 | 0.835 | 0.751 | 0.706 | 0.547 | 0.675 | 0.528 |
| 35 | 0.875 | 0.810 | 0.777 | 0.635 | 0.752 | 0.618 |
| 43 | 0.883 | 0.832 | 0.812 | 0.736 | 0.797 | 0.726 |
| 47 | 0.794 | 0.718 | 0.682 | 0.590 | 0.649 | 0.570 |

(L3 is ~0 throughout; L11 and L39 are in `control/cross_book_lag.json`.)

What this shows:

- **Tokens at similar absolute positions are alike in every book.** Most of
  the lag curve is this effect, and it decays with the distance in position,
  not with the distance within one book. That is why random same-book pairs,
  ~10k positions apart, score near 0.
- **The within-book part is small and short-range.** It is the gap between
  the same-book and different-book columns. At L27 it is about +0.28 at lag 1,
  +0.13 at lag 16, +0.06 at lag 256 and +0.03 at lag 4096. At L35–L43 it is
  +0.09–0.12 at lag 1 and ≤0.03 beyond lag 16.
- **The nearest-neighbour results stand.** h-space same-book shares of
  0.41–0.90 and key-space shares of 0.49–0.65 at L23–L35 (chance 0.125)
  compare candidates that already share the position effect. A small
  within-book excess is enough to decide which neighbour wins.

## For v2

- Position conditioning accounts for the lag curve.
- The within-book structure to reproduce is the nearest-neighbour clustering
  and the sharp similarity over the first ~16 tokens.
