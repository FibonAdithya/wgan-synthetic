"""Orchestration: load, compute, walk the panel registry, write the report.

`run` takes an `argparse.Namespace` because `compare_variants` hand-builds
one; everything below it takes an `EdaConfig`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval import ann_difficulty
from src.eval.eda import html, metrics, panels
from src.eval.eda.config import EdaConfig
from src.eval.eda.series import load_series


def build_context(cfg: EdaConfig) -> panels.Context:
    """Load the data and compute everything every panel might need."""
    sets = load_series(cfg)
    ann_metrics = {
        s.name: ann_difficulty.compute(
            s.x,
            k=cfg.ann_k,
            k_hub=cfg.ann_hub_k,
            nlist=cfg.ivf_nlist,
            max_rows=cfg.ann_max_rows,
            seed=cfg.seed,
        )
        for s in sets
    }
    return panels.Context(
        config=cfg,
        series=sets,
        ann_metrics=ann_metrics,
        divergence=(
            metrics.dimension_divergence(sets, cfg.top_divergent)
            if len(sets) > 1
            else None
        ),
    )


def run(args: argparse.Namespace) -> Path:
    """Write the report. Takes a Namespace because compare_variants builds one."""
    cfg = EdaConfig.from_args(args)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(cfg)
    series = ctx.series
    ann_metrics = ctx.ann_metrics
    has_synth = len(series) > 1
    stats = [
        metrics.summary_stats(
            s,
            cfg.knn,
            cfg.num_pairs,
            cfg.seed,
            cfg.knn_max_rows,
            ann_metrics[s.name],
        )
        for s in series
    ]

    sections = [
        (panel.resolve_title(ctx), panel.resolve_note(ctx), fig)
        for panel in panels.PANELS
        if (fig := panel.build(ctx)) is not None
    ]
    worst_dims = ctx.divergence.worst if ctx.divergence else {}

    synth_desc = (
        " &middot; ".join(f"{s.name}" for s in series if not s.is_real)
        if has_synth
        else "no synthetic overlay"
    )
    meta_html = (
        f'<div class="meta">real: <code>{cfg.real_path}</code>'
        f" &middot; overlays: {synth_desc}"
        f" &middot; preprocess: <code>{cfg.preprocess}</code>"
        f" &middot; subsampled to {series[0].x.shape[0]} rows</div>"
        + html.stats_table_html(stats)
    )

    report_html = html.build_report(
        sections,
        meta_html,
        html.plotlyjs_head(cfg.plotlyjs, out_dir),
        heading=f"Descriptor EDA: {Path(cfg.real_path).stem}",
    )
    report_path = out_dir / "eda_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    png_paths: list[str] = []
    png_error = None
    if not cfg.no_png:
        try:
            png_paths = html.export_pngs(sections, out_dir)
        except Exception as exc:  # kaleido needs a Chrome binary
            png_error = str(exc)

    summary = {
        "real_path": cfg.real_path,
        "synthetic_paths": cfg.synthetic_path,
        "preprocess": cfg.preprocess,
        "seed": cfg.seed,
        "ann_settings": {
            "k": cfg.ann_k,
            "k_hub": cfg.ann_hub_k,
            "max_rows": cfg.ann_max_rows,
            "nlist": cfg.ivf_nlist,
        },
        "knn_settings": {"k": cfg.knn, "max_rows": cfg.knn_max_rows},
        "stats": stats,
        "worst_dimensions": worst_dims,
        "report_html": str(report_path),
        "png_files": png_paths,
    }
    if png_error:
        summary["png_error"] = png_error
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Wrote {report_path}")
    if png_paths:
        print(f"Wrote {len(png_paths)} PNGs to {out_dir / 'png'}")
    elif png_error:
        print(f"PNG export skipped: {png_error}")
    print(f"Wrote {out_dir / 'summary.json'}")

    return report_path
