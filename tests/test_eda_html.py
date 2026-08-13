import numpy as np

from src.eval.eda import html


def test_format_stat_renders_counts_as_integers_not_scientific_notation():
    assert html.format_stat(1200000) == "1200000"
    assert html.format_stat(np.int64(1200000)) == "1200000"
    assert html.format_stat(None) == "n/a"
    assert html.format_stat(0.5) == "0.5"


def test_build_report_names_the_corpus_in_the_title_and_heading():
    """Every family's report used to be titled after SIFT.

    The heading is the only thing on the page that says which corpus was
    measured, so a hardcoded one mislabels five of the six families.
    """
    out = html.build_report([], "", "", heading="Descriptor EDA: openai_250k")

    assert "<title>Descriptor EDA: openai_250k</title>" in out
    assert "<h1>Descriptor EDA: openai_250k</h1>" in out
    assert "SIFT" not in out


def test_stats_table_renders_a_large_discarded_count_as_a_tally():
    html_out = html.stats_table_html(
        [{"name": "real", "lid_discarded_queries": 1200000, "lid_median": 12.5}]
    )
    assert "1200000" in html_out
    assert "1.2e+06" not in html_out
