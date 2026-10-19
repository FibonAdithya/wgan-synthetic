"""Compare mixed-book corpora (mix_check.jsonl) against the 4 single-book v2 profiles.

Decision rule fixed before the run: mixing is harmless for a gate statistic if the mixed
value, pooled over the 96 (layer, kv_head) pairs, lies inside the range of the 4 books.
"""

import json
import sys
from statistics import median

summary, mixed_path = sys.argv[1], sys.argv[2]
single = json.load(open(summary))["v2"]
mixed = [json.loads(line) for line in open(mixed_path)]
keys = {(r["layer"], r["kv_head"]) for r in mixed}
assert len(mixed) == len(keys), "duplicate (layer, kv_head) rows in mixed run"
print(f"mixed corpora: {len(mixed)}")


def ratio(r, k):
    o, i = r[f"plain_ood_k{k}"]["ivf_scan_frac"], r[f"plain_id_k{k}"]["ivf_scan_frac"]
    return o / i


STATS = {
    "HNSW reach recall@10>=0.9, OOD": lambda rs: sum(r["plain_ood_k10"]["hnsw_reached"] for r in rs) / len(rs),
    "HNSW reach recall@128>=0.9, OOD": lambda rs: sum(r["plain_ood_k128"]["hnsw_reached"] for r in rs) / len(rs),
    "median HNSW recall@10 at ef176, OOD": lambda rs: median(r["plain_ood_k10"]["hnsw_recall_op"] for r in rs),
    "median HNSW recall@10 at ef176, ID": lambda rs: median(r["plain_id_k10"]["hnsw_recall_op"] for r in rs),
    "median IVF OOD/ID scan ratio, k=128": lambda rs: median(ratio(r, 128) for r in rs),
    "median IVF recall@128 at nprobe24, OOD": lambda rs: median(r["plain_ood_k128"]["ivf_recall_op"] for r in rs),
}

print(f"\n{'statistic':42s} {'book0':>7s} {'book1':>7s} {'book2':>7s} {'book3':>7s} | {'mixed':>7s}  verdict")
for name, f in STATS.items():
    per_book = [f([r for r in single if r["doc"] == d and (r["layer"], r["head"]) in keys]) for d in range(4)]
    m = f(mixed)
    lo, hi = min(per_book), max(per_book)
    verdict = "inside" if lo <= m <= hi else ("BELOW" if m < lo else "ABOVE")
    print(f"{name:42s} " + " ".join(f"{v:7.3f}" for v in per_book) + f" | {m:7.3f}  {verdict} [{lo:.3f}, {hi:.3f}]")

inside = below = above = 0
for r in mixed:
    b = [s["plain_ood_k10"]["hnsw_recall_op"] for s in single if s["layer"] == r["layer"] and s["head"] == r["kv_head"]]
    m = r["plain_ood_k10"]["hnsw_recall_op"]
    if m < min(b) - 1e-9:
        below += 1
    elif m > max(b) + 1e-9:
        above += 1
    else:
        inside += 1
print(f"\nper head, HNSW recall@10 at ef176 (OOD), mixed vs range of 4 books: "
      f"inside {inside}, below (harder) {below}, above (easier) {above}")

own = sorted(r["own_doc_share_top10"] for r in mixed)
q = lambda p: own[min(len(own) - 1, int(p * len(own)))]  # noqa: E731
print(f"own-book share of exact top-10 (0.25 = no book preference, 1.0 = only own book): "
      f"min {own[0]:.3f}  q25 {q(.25):.3f}  median {q(.5):.3f}  q75 {q(.75):.3f}  max {own[-1]:.3f}")
print("by layer (median own-book share):",
      ", ".join(f"L{L} {median(r['own_doc_share_top10'] for r in mixed if r['layer'] == L):.2f}"
                for L in sorted({r['layer'] for r in mixed})))
