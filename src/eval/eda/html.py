"""Turning finished figures into a page on disk.

Everything here works on `(title, note, figure)` triples and knows nothing
about what a panel means. Static PNG export is best-effort: kaleido v1 shells
out to Chrome, which is not installed everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def format_stat(value: float | int | None) -> str:
    """Render one statistics-table cell.

    Counts stay counts: `format(1200000, '.6g')` is `1.2e+06`, which reads as
    a measurement rather than a tally, so integer-valued statistics (row
    counts, discarded queries, the measured k and nlist) are formatted as
    plain integers and only the genuinely continuous ones get `.6g`.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return format(int(value), "d")
    return format(value, ".6g")


def stats_table_html(stats: list[dict]) -> str:
    keys = [k for k in stats[0] if k != "name"]
    header = "".join(f"<th>{s['name']}</th>" for s in stats)
    rows = []
    for k in keys:
        cells = "".join(f"<td>{format_stat(s[k])}</td>" for s in stats)
        rows.append(f"<tr><th>{k}</th>{cells}</tr>")
    return (
        "<table><thead><tr><th>statistic</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


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

CDN_SRC = "https://cdn.plot.ly/plotly-3.7.0.min.js"


def plotlyjs_head(mode: str, out_dir: Path) -> str:
    """Return the <script> tag(s) that make Plotly available to the page."""
    import plotly.offline as pyo

    if mode == "cdn":
        return f'<script charset="utf-8" src="{CDN_SRC}"></script>'
    if mode == "directory":
        asset = out_dir / "plotly.min.js"
        if not asset.exists():
            asset.write_text(pyo.get_plotlyjs(), encoding="utf-8")
        return '<script charset="utf-8" src="plotly.min.js"></script>'
    return f"<script>{pyo.get_plotlyjs()}</script>"


def build_report(
    sections: list[tuple[str, str, go.Figure]],
    meta_html: str,
    head_script: str,
    heading: str = "Descriptor EDA",
) -> str:
    """Assemble the page. `heading` names the corpus in the tab and the h1.

    Defaulted rather than required because `compare_variants` calls this
    too. It used to be the literal string "SIFT descriptor EDA", which
    titled every family's report after the one family that had a ladder --
    an openai report announcing itself as SIFT is the kind of wrong that
    survives review because nobody reads their own report's header.
    """
    body = []
    for title, note, fig in sections:
        body.append(f"<h2>{title}</h2>")
        if note:
            body.append(f'<div class="note">{note}</div>')
        body.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{heading}</title>"
        f"<style>{REPORT_CSS}</style>"
        f"{head_script}"
        "</head><body>"
        f"<h1>{heading}</h1>"
        f"{meta_html}" + "".join(body) + "</body></html>"
    )


def export_pngs(sections: list[tuple[str, str, go.Figure]], out_dir: Path) -> list[str]:
    """Static export is best-effort: kaleido v1 shells out to Chrome."""
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, (title, _, fig) in enumerate(sections, start=1):
        slug = title.lower().replace(" ", "_").replace("/", "-")
        path = png_dir / f"{idx:02d}_{slug}.png"
        fig.write_image(str(path), width=1200, height=fig.layout.height or 460, scale=2)
        written.append(str(path))
    return written
