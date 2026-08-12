"""The benchmark grid: every corpus against every index against every knob.

Nothing here names cuVS. The runner reaches indexes only through
`IndexAdapter`, which is what makes the whole loop -- timing, fencing, failure
handling, incremental output -- testable on a CPU-only box with fake adapters.

Timing discipline: `adapter.sync()` is called immediately before the clock
starts and immediately before it stops. cuVS calls are asynchronous, so an
unfenced region times the launch queue rather than the work and every number
in the table would be fiction. This is the single most important correctness
property in the harness.

Warmup: each cell issues one untimed, discarded search before its timed
repeats. Measured on the box, CAGRA's first search over 1M vectors costs
126.2 ms against an 82.2 ms steady state, and the very first cuVS search in a
process pays seconds of one-time initialization on top of that. Without a
warmup, that cold call would inflate every summary statistic -- including
`min` -- with a cost that has nothing to do with steady-state throughput. The
warmup is fenced the same way the timed repeats are, so its (discarded) work
has fully landed before the first timed region's clock starts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.eval.ann_benchmark import metrics
from src.eval.ann_benchmark.corpora import Corpus
from src.eval.ann_benchmark.indexes import IndexAdapter


@dataclass(frozen=True)
class BuildRecord:
    corpus: str
    index: str
    train_seconds: float | None
    add_seconds: float | None
    index_bytes: int | None
    params: dict[str, object]
    peak_vram_bytes: int | None = None
    failed: str | None = None


@dataclass(frozen=True)
class SearchRecord:
    corpus: str
    index: str
    param_name: str
    param_value: int | None
    recall: float | None
    qps_min: float | None
    qps_median: float | None
    qps_p95: float | None
    num_queries: int
    failed: str | None = None


def _flush(
    records_path: Path,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
) -> None:
    """Rewrite the records file after every cell.

    Rewriting rather than appending keeps the file valid JSON at all times, so
    a job killed mid-grid leaves something readable rather than a truncated
    array. The grid is a few hundred small records; the write cost is noise
    next to a CAGRA build.
    """
    payload = {
        "builds": [asdict(b) for b in builds],
        "searches": [asdict(s) for s in searches],
    }
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _time_search(
    adapter: IndexAdapter,
    built,
    queries: np.ndarray,
    k: int,
    param: int | None,
    repeats: int,
) -> tuple[np.ndarray, list[float]]:
    """Run one untimed warmup, then `repeats` timed searches.

    The whole query set goes in one call per repeat. GPU indexes are
    throughput devices; issuing one query at a time would measure launch
    latency rather than the index.

    The warmup's result is discarded and never timed. It is fenced with
    `sync()` on both sides regardless -- once before it runs, so it does not
    race whatever fenced the previous cell, and once after, so its (async,
    on-device) work has actually finished before the first timed clock
    starts. Skipping that second fence would let the warmup's tail overlap
    the first timed call and corrupt exactly the number this exists to fix.
    """
    adapter.sync()
    adapter.search(built, queries, k, param)  # warmup: discarded, untimed
    adapter.sync()

    seconds: list[float] = []
    distances = None
    for _ in range(repeats):
        adapter.sync()
        started = time.perf_counter()
        distances, _ = adapter.search(built, queries, k, param)
        adapter.sync()
        seconds.append(time.perf_counter() - started)
    return distances, seconds


def run_grid(
    corpora_list: Sequence[Corpus],
    adapters: Sequence[IndexAdapter],
    *,
    k: int,
    repeats: int,
    records_path: Path,
) -> tuple[list[BuildRecord], list[SearchRecord]]:
    """Build every index over every corpus and sweep its search knob."""
    records_path = Path(records_path)
    builds: list[BuildRecord] = []
    searches: list[SearchRecord] = []

    for corpus in corpora_list:
        vectors = np.load(corpus.vectors_path)
        queries = np.load(corpus.queries_path)
        truth = np.load(corpus.truth_distances_path)

        for adapter in adapters:
            try:
                built = adapter.build(vectors)
            except Exception as exc:  # noqa: BLE001 - one bad cell, not the grid
                builds.append(
                    BuildRecord(
                        corpus=corpus.name,
                        index=adapter.name,
                        train_seconds=None,
                        add_seconds=None,
                        index_bytes=None,
                        params=adapter.describe(),
                        failed=f"{type(exc).__name__}: {exc}",
                    )
                )
                _flush(records_path, builds, searches)
                continue

            builds.append(
                BuildRecord(
                    corpus=corpus.name,
                    index=adapter.name,
                    train_seconds=built.train_seconds,
                    add_seconds=built.add_seconds,
                    index_bytes=built.index_bytes,
                    params=adapter.describe(),
                    peak_vram_bytes=built.peak_vram_bytes,
                )
            )
            _flush(records_path, builds, searches)

            for param in adapter.sweep_params():
                try:
                    distances, seconds = _time_search(
                        adapter, built, queries, k, param, repeats
                    )
                    recall = metrics.recall_at_k(distances, truth)
                    throughput = [metrics.qps(queries.shape[0], s) for s in seconds]
                    summary = metrics.summarize(throughput)
                    searches.append(
                        SearchRecord(
                            corpus=corpus.name,
                            index=adapter.name,
                            param_name=adapter.param_name,
                            param_value=param,
                            recall=recall,
                            qps_min=summary["min"],
                            qps_median=summary["median"],
                            qps_p95=summary["p95"],
                            num_queries=int(queries.shape[0]),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    searches.append(
                        SearchRecord(
                            corpus=corpus.name,
                            index=adapter.name,
                            param_name=adapter.param_name,
                            param_value=param,
                            recall=None,
                            qps_min=None,
                            qps_median=None,
                            qps_p95=None,
                            num_queries=int(queries.shape[0]),
                            failed=f"{type(exc).__name__}: {exc}",
                        )
                    )
                _flush(records_path, builds, searches)

    return builds, searches
