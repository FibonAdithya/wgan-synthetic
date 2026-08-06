import numpy as np

from src.eval.eda import html


def test_format_stat_renders_counts_as_integers_not_scientific_notation():
    assert html.format_stat(1200000) == "1200000"
    assert html.format_stat(np.int64(1200000)) == "1200000"
    assert html.format_stat(None) == "n/a"
    assert html.format_stat(0.5) == "0.5"


def test_stats_table_renders_a_large_discarded_count_as_a_tally():
    html_out = html.stats_table_html(
        [{"name": "real", "lid_discarded_queries": 1200000, "lid_median": 12.5}]
    )
    assert "1200000" in html_out
    assert "1.2e+06" not in html_out
