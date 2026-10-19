# Spike scripts behind the KV-cache head-profiling report

Throwaway code from the 2026-10-16/17 probe, kept so `../report.html` has its provenance
in git. Not production code: nothing imports it and it has no tests. Port into `src/`
with tests if the track goes ahead.

| script | what it did | run as (on the A100 box) |
|---|---|---|
| `capture_scout.py` | captured the attention input `h` at Llama 4 Scout's 12 global layers over 4 PG-19 books, and verified `k = W_k h`, `q = s(pos) W_q h` against the model | `python capture_scout.py --model-dir /workspace/models/scout --pg19-parquet <pg19-test parquet> --out-dir capture --gpu-weights 24GiB` |
| `profile_heads.py` | first profiling run: sink kept, HNSW on raw inner product. **Its HNSW result is withdrawn**, see the report | `python profile_heads.py --capture-dir capture --out profile.jsonl` |
| `profile_v2.py` | second run: sink dropped, Xbox HNSW, k in {10, 128}, SoftmaxLift, attention error | `python profile_v2.py --capture-dir capture --model-dir /workspace/models/scout --out profile_v2.jsonl` |

The copies here differ from the ones that ran only by ruff fixes (import order, one
unused import, one `noqa`, formatting), so `make check` stays green. SHA-256 of the
versions that actually ran:

    82f40b71978d9c8228029258d309addb5bf5050f72c06fdd0f91fcd90eff66a9  capture_scout.py
    0463f0cd2f1ee1604c84642533ddfc89927fd7e87460818317d036c0a2bcd65d  profile_heads.py
    d25335f9c09fe899864fb846044c8eb63aa6cdac823e81eb0aac51ca54776b65  profile_v2.py

Inputs:

- Model: `unsloth/Llama-4-Scout-17B-16E-Instruct` at commit
  `afd8e498c87bda51c7ea8ec68ea2f7c066e6340b`. All 50 weight shards are byte-identical
  to `meta-llama/Llama-4-Scout-17B-16E-Instruct` at `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3`
  (local `sha256sum` matched Meta's LFS hashes). Llama 4 Community License.
- Text: `emozilla/pg19-test` at `c5e39bf32e33f9111323aa68d7d9000d22722035`, rows 0-3,
  first 32,768 tokens of each.

Raw outputs (`profile.jsonl` 1.2 MB, `profile_v2.jsonl` 3.3 MB) are not committed;
`../profile_summary.json` is the reduction the report renders from, made by
`scripts/kv_cache_profile_report.py compact`. Raw-file SHA-256:

    0ae1f3aba1ffcd12113c6ab1056292be0d613015bc0fdde065da4a6c0fcc4e58  profile.jsonl
    a5a75d8e57fdb81d976555384e524e4add52ab99ad3cd669d1909d4962bf7fb0  profile_v2.jsonl
    f78f360b2d06dda90a882c21205198318139a29ffa28969627d06700993a6205  capture/meta.json
