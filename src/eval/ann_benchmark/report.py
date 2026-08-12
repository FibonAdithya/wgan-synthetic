"""Render benchmark records as JSON, a markdown table and an HTML report.

The HTML inlines plotly the same way `src.eval.eda.figures` does -- the
script tag holds `plotly.offline.get_plotlyjs()` verbatim, no CDN reference
-- so the report is readable from a checkout with no network. That is how it
will be read, since it lands in `docs/results/` and is opened from disk.

`peak_vram_bytes` reaches the JSON through `asdict` but is deliberately not a
markdown or HTML table column. It is a card-wide delta rather than a
per-index allocation (see `indexes._device_used_bytes`), so a column of it
beside exact per-index byte counts would read as more precise than it is.

The Flat/exact index has no swept knob and sits at recall 1.0 by
construction: it never goes through `metrics.qps_at_recall`, because
"unreachable" and "exact" are different facts and collapsing them into one
QPS-at-target column would blur that. Its row instead carries `exact_qps`,
its single measured throughput, labelled as the exact-search ceiling.

A build or search failure is carried through as an explicit row rather than
dropped. A table that only ever shows successful cells reads as complete
when it is not -- and a corpus/index pair a build failed on, or one whose
every search attempt failed, is exactly the kind of result this benchmark
exists to surface.
"""

from __future__ import annotations

import html as html_module
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from src.eval.ann_benchmark import metrics
from src.eval.ann_benchmark.runner import BuildRecord, SearchRecord

NOT_REACHED = "not reached"
SEARCH_FAILED = "search failed"
BUILD_FAILED = "build failed"
EXACT_CEILING = "exact ceiling"

REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 1180px; padding: 24px; color: #1a202c; }
h1 { margin-bottom: 4px; } h2 { margin-top: 36px; border-bottom: 1px solid #e2e8f0;
       padding-bottom: 6px; }
.meta { color: #4a5568; font-size: 14px; margin-bottom: 24px; }
.note { background: #f7fafc; border-left: 3px solid #2b6cb0; padding: 10px 14px;
        margin: 12px 0; font-size: 14px; }
table { border-collapse: collapse; font-size: 14px; margin: 12px 0; }
th, td { border: 1px solid #e2e8f0; padding: 6px 12px; text-align: right; }
thead th { background: #f7fafc; } tbody th { text-align: left; }
"""


def _is_exact_cell(index: str, cell_searches: Sequence[SearchRecord]) -> bool:
    """An "exact" cell has no swept knob: every search record has no param.

    Falls back to the `flat` index name when the cell has no search records
    at all -- whether because the build failed before any search ran, or
    because a successful build simply produced zero search records -- since
    there is then nothing to inspect.
    """
    if cell_searches:
        return all(s.param_name == "" for s in cell_searches)
    return index == "flat"


def headline_rows(
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    *,
    target_recall: float,
) -> list[dict[str, object]]:
    """One row per (corpus, index): build cost and QPS at the target recall.

    Every attempted cell gets a row, including ones whose build failed --
    `run_grid` always appends a `BuildRecord` even on failure, so iterating
    `builds` already covers every cell that was tried, not just the ones
    that produced a search curve.
    """
    build_by_cell = {(b.corpus, b.index): b for b in builds}
    searches_by_cell: dict[tuple[str, str], list[SearchRecord]] = defaultdict(list)
    for s in searches:
        searches_by_cell[(s.corpus, s.index)].append(s)

    rows: list[dict[str, object]] = []
    for (corpus, index), build in build_by_cell.items():
        cell_searches = searches_by_cell.get((corpus, index), [])
        successful = [s for s in cell_searches if s.recall is not None]
        points = [(s.recall, s.qps_median) for s in successful]
        recalls = [r for r, _ in points]
        peak_recall = max(recalls) if recalls else None

        build_seconds = None
        if build.train_seconds is not None and build.add_seconds is not None:
            build_seconds = build.train_seconds + build.add_seconds

        is_exact = _is_exact_cell(index, cell_searches)
        search_failed = bool(cell_searches) and not successful

        qps_at_target = None
        exact_qps = None
        if build.failed is None and not search_failed:
            if is_exact:
                exact_qps = successful[0].qps_median if successful else None
            else:
                qps_at_target = metrics.qps_at_recall(points, target_recall)

        rows.append(
            {
                "corpus": corpus,
                "index": index,
                "train_seconds": build.train_seconds,
                "add_seconds": build.add_seconds,
                "build_seconds": build_seconds,
                "index_bytes": build.index_bytes,
                "peak_vram_bytes": build.peak_vram_bytes,
                "is_exact": is_exact,
                "qps_at_target": qps_at_target,
                "exact_qps": exact_qps,
                "peak_recall": peak_recall,
                "failed": build.failed,
                "search_failed": search_failed,
            }
        )
    return rows


def write_json(
    path: Path,
    *,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    environment: dict[str, object],
) -> None:
    """Every cell, unaggregated, plus what produced it."""
    payload = {
        "environment": environment,
        "builds": [asdict(b) for b in builds],
        "searches": [asdict(s) for s in searches],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _escape_markdown_cell(text: str) -> str:
    """Escape characters that would break a GFM table cell.

    A raw `|` splits the cell into extra columns and misaligns the whole
    table; a raw newline ends the row early. Both are realistic in a cuVS/
    CUDA failure message (C++ template types, `<unnamed>` frames), so a
    failure string is escaped before it is ever interpolated into a row --
    the same string that is this table's most important diagnostic must not
    be the thing that corrupts the table it's reported in.
    """
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")


def _qps_cell(row: dict[str, object], *, escape) -> str:
    if row["failed"] is not None:
        return f"{BUILD_FAILED}: {escape(str(row['failed']))}"
    if row["search_failed"]:
        return SEARCH_FAILED
    if row["is_exact"]:
        return f"{_fmt(row['exact_qps'])} ({EXACT_CEILING})"
    if row["qps_at_target"] is None:
        peak = _fmt(row["peak_recall"], 3)
        return f"{NOT_REACHED} (peak recall {peak})"
    return _fmt(row["qps_at_target"])


def _recall_cell(row: dict[str, object]) -> str:
    if row["failed"] is not None or row["search_failed"]:
        return "—"
    return _fmt(row["peak_recall"], 3)


def write_markdown(
    path: Path, rows: Sequence[dict[str, object]], *, target_recall: float
) -> None:
    """The headline table.

    A cell whose curve never reached the target prints "not reached" and its
    peak recall, rather than the nearest measured point. Substituting a
    number there would hide the most interesting result the table can carry.
    A cell whose build failed, or whose every search attempt failed, prints
    that fact instead of a blank -- a blank reads as a formatting glitch, not
    a result.
    """
    lines = [
        f"# GPU ANN benchmark (target recall@10 = {target_recall:.2f})",
        "",
        "All corpora are L2-normalized; see the design note. These figures are",
        "not comparable with published SIFT1M results. Build time is train and",
        "add phases, timed separately. The flat/exact index has no swept knob;",
        "its row reports the single measured QPS as the exact-search ceiling,",
        "not an interpolated value at the target recall.",
        "",
        "| Corpus | Index | Train (s) | Add (s) | Index (MB) | "
        f"QPS @ recall {target_recall:.2f} | Peak recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r["index"]), str(r["corpus"]))):
        megabytes = (
            None if row["index_bytes"] is None else float(row["index_bytes"]) / 1e6
        )
        lines.append(
            f"| {row['corpus']} | {row['index']} | "
            f"{_fmt(row['train_seconds'], 2)} | {_fmt(row['add_seconds'], 2)} | "
            f"{_fmt(megabytes)} | "
            f"{_qps_cell(row, escape=_escape_markdown_cell)} | "
            f"{_recall_cell(row)} |"
        )
    lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plotlyjs_script() -> str:
    """Inline plotly.js as a `<script>` body, matching `eda.figures` "inline" mode."""
    return f"<script>{pyo.get_plotlyjs()}</script>"


def write_html(
    path: Path,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    *,
    target_recall: float,
) -> None:
    """Recall-vs-QPS curves, one facet per index, one trace per corpus.

    Failed cells do not have a curve to plot, so they are listed underneath
    the figure instead of silently missing from it.
    """
    index_names = sorted({s.index for s in searches} | {b.index for b in builds})
    corpus_names = sorted({s.corpus for s in searches} | {b.corpus for b in builds})

    figure = make_subplots(
        rows=1,
        cols=max(len(index_names), 1),
        subplot_titles=index_names or ["no data"],
        shared_yaxes=True,
    )
    for column, index in enumerate(index_names, start=1):
        for corpus in corpus_names:
            points = sorted(
                (s.recall, s.qps_median)
                for s in searches
                if s.corpus == corpus and s.index == index and s.recall is not None
            )
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[r for r, _ in points],
                    y=[q for _, q in points],
                    mode="lines+markers",
                    name=corpus,
                    legendgroup=corpus,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_vline(x=target_recall, line_dash="dot", row=1, col=column)
        figure.update_xaxes(title_text="recall@10", row=1, col=column)

    figure.update_yaxes(title_text="queries/second", type="log", row=1, col=1)
    figure.update_layout(
        title=(
            "GPU ANN benchmark: recall vs throughput "
            "(L2-normalized corpora; not comparable with published SIFT1M)"
        ),
        template="plotly_white",
        height=520,
    )

    rows = headline_rows(builds, searches, target_recall=target_recall)
    table_rows = "".join(
        "<tr>"
        f"<th>{row['corpus']}</th><td>{row['index']}</td>"
        f"<td>{_fmt(row['train_seconds'], 2)}</td>"
        f"<td>{_fmt(row['add_seconds'], 2)}</td>"
        f"<td>{_fmt(None if row['index_bytes'] is None else float(row['index_bytes']) / 1e6)}</td>"
        f"<td>{_qps_cell(row, escape=html_module.escape)}</td>"
        f"<td>{_recall_cell(row)}</td>"
        "</tr>"
        for row in sorted(rows, key=lambda r: (str(r["index"]), str(r["corpus"])))
    )
    failed_lines = [
        f"{b.corpus}/{b.index}: {BUILD_FAILED} -- {html_module.escape(b.failed)}"
        for b in builds
        if b.failed
    ] + [
        f"{s.corpus}/{s.index} ({s.param_name}={s.param_value}): "
        f"search failed -- {html_module.escape(s.failed)}"
        for s in searches
        if s.failed
    ]
    failed_html = (
        "<h2>Failed cells</h2><pre>" + "\n".join(failed_lines) + "</pre>"
        if failed_lines
        else ""
    )
    vram_note = (
        '<div class="note">Peak VRAM (in the JSON, not shown here) is a '
        "card-wide delta measured around each build, not a per-index "
        "allocation.</div>"
    )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>GPU ANN benchmark</title>"
        f"<style>{REPORT_CSS}</style>"
        f"{_plotlyjs_script()}"
        "</head><body>"
        "<h1>GPU ANN benchmark</h1>"
        f'<div class="meta">target recall@10 = {target_recall:.2f}</div>'
        f"{vram_note}"
        + figure.to_html(full_html=False, include_plotlyjs=False)
        + "<h2>Headline table</h2>"
        "<table><thead><tr><th>Corpus</th><th>Index</th><th>Train (s)</th>"
        "<th>Add (s)</th><th>Index (MB)</th>"
        f"<th>QPS @ recall {target_recall:.2f}</th><th>Peak recall</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
        f"{failed_html}"
        "</body></html>"
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
