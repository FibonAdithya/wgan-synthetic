"""Exploratory data analysis report for descriptor corpora.

The CLI entrypoint is `src/eval/eda_report.py`, which is what
`python -m src.eval.eda_report` runs. This package holds the implementation,
split by responsibility:

    config    parsed settings as a typed value object
    cli       argparse
    series    loading and preprocessing the sets being compared
    metrics   the numbers
    figures   the plotly figures
    glyphs    the descriptor glyph panel, shared with plot_descriptor_grid
    notes     computed prose fragments for the ANN panels
    panels    the report's panel registry
    html      page assembly and static export
    pipeline  orchestration

This module deliberately re-exports nothing: every name has exactly one
import path, so a reader who sees `figures.fig_pca_spectrum` knows where it
lives without checking.
"""
