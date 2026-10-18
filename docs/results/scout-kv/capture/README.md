# scout_kv capture, books 4-19

Per-book records written by `scripts/kv/capture_scout.py` on tig-gpu, 2026-10-18,
run with `--first-book 4 --n-books 16` at commit 25ced67. The `h` tensors themselves
stay on the box at `/workspace/kv/capture` (76 GB for 20 books); they are reproducible
from the script and are not copied.

MEASURED on the box:

- every book's worst q/k reconstruction error is 0.00228-0.00229 (acceptance limit 0.005);
- wall time 24.5 min for 16 books including model load (log start 16:16:35 UTC, last write 16:41:07 UTC);
- forward pass 76.8-94.2 s per 32,768-token book;
- the 4-book `meta.json` is unchanged (sha256 `f78f360b2d06dda9...`);
- `weights/layerXX.safetensors` now also hold `v_proj` (BF16).

Book numbers follow capture order, not PG-19 row numbers: short rows are skipped, so
book 4 is row 5 and book 19 is row 25. Each `book.json` records its row and title.
Files were copied by tar stream; the 16 sha256 hashes matched on both sides.
