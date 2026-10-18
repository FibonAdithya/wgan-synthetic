# Mixed-book check (2026-10-18)

Question: a generator trained on several books pooled emits independent `h`, so each
synthetic corpus resembles a mixture of books. Is a real mixed-book corpus as hard to
search as a single-book one?

`mix_check.py` builds, for each of the 96 (global layer, KV head) pairs, one real corpus
mixing the 4 profiled PG-19 books (7,935 keys per book, 31,740 total; 256 tail positions
per book x 5 query heads = 5,120 queries; 1,024 key-side queries) and runs the same
plain-workload IVF / Xbox-HNSW sweeps as `../spike/profile_v2.py`, whose helpers it
imports. It ran on the A100 box next to `profile_v2.py` (sha256 `d25335f9...`, the version
in the README one directory up) in 18.7 min.

`mix_compare.py` compares it with the 4 single-book profiles in `../profile_summary.json`;
its output is `comparison.txt`, regenerable with

    python -I mix_compare.py ../profile_summary.json mix_check.jsonl

Decision rule fixed before the run: mixing is harmless for a statistic if the mixed value,
pooled over the 96 pairs, lies inside the range of the 4 books. Result: IVF statistics are
inside; HNSW statistics are about 2 points easier than the easiest book, so the rule fails
there. The owner chose mixed corpora as the gate reference anyway (see
`docs/ai/specs/2026-10-18-kv-cache-scout-design.md`).

SHA-256:

    0775f9f58d022b8c45c4448105eb3e0e8a8228027dccaecd202188787d19f4bf  mix_check.py
    bdeef57201b1f3e86941da5a1bcfad742535eae4f721f61a60a087b3b8f69b67  mix_check.jsonl
