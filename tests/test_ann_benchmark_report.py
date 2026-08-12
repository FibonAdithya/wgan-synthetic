"""Tests for benchmark report rendering."""

import json

import pytest

from src.eval.ann_benchmark import report
from src.eval.ann_benchmark.runner import BuildRecord, SearchRecord


def _builds():
    return [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=12.0,
            add_seconds=0.0,
            index_bytes=1024,
            params={"n_lists": 4096},
        ),
        BuildRecord(
            corpus="v2",
            index="ivf_flat",
            train_seconds=9.0,
            add_seconds=0.0,
            index_bytes=1024,
            params={"n_lists": 4096},
        ),
    ]


def _searches():
    def rec(corpus, param, recall, q):
        return SearchRecord(
            corpus=corpus,
            index="ivf_flat",
            param_name="n_probes",
            param_value=param,
            recall=recall,
            qps_min=q * 0.9,
            qps_median=q,
            qps_p95=q * 1.1,
            num_queries=10,
        )

    return [
        rec("real", 1, 0.80, 400.0),
        rec("real", 2, 0.95, 100.0),
        rec("v2", 1, 0.50, 900.0),
        rec("v2", 2, 0.70, 500.0),
    ]


def test_headline_interpolates_qps_at_the_target_recall():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.875)
    real = next(r for r in rows if r["corpus"] == "real")
    assert real["qps_at_target"] == pytest.approx(200.0)


def test_headline_reports_none_when_the_target_is_unreachable():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    v2 = next(r for r in rows if r["corpus"] == "v2")
    assert v2["qps_at_target"] is None
    assert v2["peak_recall"] == pytest.approx(0.70)


def test_headline_carries_build_time_through():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    real = next(r for r in rows if r["corpus"] == "real")
    assert real["build_seconds"] == pytest.approx(12.0)


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "out.json"
    report.write_json(
        path,
        builds=_builds(),
        searches=_searches(),
        environment={"gpu": "test-gpu"},
    )
    payload = json.loads(path.read_text())
    assert payload["environment"]["gpu"] == "test-gpu"
    assert len(payload["searches"]) == 4
    assert payload["builds"][0]["corpus"] == "real"


def test_markdown_marks_an_unreachable_target_rather_than_inventing_a_number(
    tmp_path,
):
    path = tmp_path / "out.md"
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    assert "not reached" in text
    assert "| real |" in text
    assert "0.90" in text


def test_html_is_self_contained(tmp_path):
    path = tmp_path / "out.html"
    report.write_html(path, _builds(), _searches(), target_recall=0.90)
    text = path.read_text()
    assert text.lstrip().startswith("<")
    # Inlined plotly, not a CDN reference: the report has to be readable from
    # a checkout with no network. Plotly 6.9.0's own bundled JS embeds the
    # literal string "cdn.plot.ly" as a default topojsonURL config value (a
    # mapbox tileset default we never use), so a bare substring check would
    # fail against every self-contained embedding, CDN or not. What actually
    # matters -- no externally-loaded script tag -- is what's checked here.
    assert 'src="https://cdn.plot.ly' not in text
    assert 'script src="http' not in text
    assert "recall" in text.lower()


# --- Additional coverage: the flat/exact index and failure visibility. ---
# These are hard requirements from the task, not covered by the brief's
# sample tests above: an exact index has no swept knob and must never be
# routed through `qps_at_recall`, and a failed build or search must show up
# as a row rather than being silently dropped.


def _flat_build(corpus="real", failed=None):
    return BuildRecord(
        corpus=corpus,
        index="flat",
        train_seconds=0.0 if failed is None else None,
        add_seconds=1.0 if failed is None else None,
        index_bytes=2048 if failed is None else None,
        params={},
        failed=failed,
    )


def _flat_search(corpus="real", recall=1.0, q=50.0, failed=None):
    return SearchRecord(
        corpus=corpus,
        index="flat",
        param_name="",
        param_value=None,
        recall=None if failed else recall,
        qps_min=None if failed else q * 0.9,
        qps_median=None if failed else q,
        qps_p95=None if failed else q * 1.1,
        num_queries=10,
        failed=failed,
    )


def test_headline_reports_exact_ceiling_for_the_flat_index_not_a_target_qps():
    builds = [_flat_build()]
    searches = [_flat_search()]
    rows = report.headline_rows(builds, searches, target_recall=0.90)
    row = rows[0]
    assert row["is_exact"] is True
    assert row["qps_at_target"] is None
    assert row["exact_qps"] == pytest.approx(50.0)


def test_markdown_labels_the_flat_row_as_an_exact_ceiling(tmp_path):
    path = tmp_path / "out.md"
    builds = [_flat_build()]
    searches = [_flat_search()]
    rows = report.headline_rows(builds, searches, target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    assert "exact ceiling" in text
    assert "50.0" in text


def test_headline_row_survives_a_failed_build():
    builds = [_flat_build(failed="RuntimeError: boom")]
    rows = report.headline_rows(builds, [], target_recall=0.90)
    assert len(rows) == 1
    assert rows[0]["failed"] == "RuntimeError: boom"
    assert rows[0]["qps_at_target"] is None
    assert rows[0]["exact_qps"] is None


def test_markdown_shows_a_failed_build_instead_of_a_blank_row(tmp_path):
    path = tmp_path / "out.md"
    builds = [_flat_build(failed="RuntimeError: boom")]
    rows = report.headline_rows(builds, [], target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    assert "build failed" in text
    assert "boom" in text


def test_headline_distinguishes_search_failure_from_unreachable_target():
    builds = [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=1.0,
            add_seconds=0.0,
            index_bytes=64,
            params={"n_lists": 8},
        )
    ]
    searches = [
        SearchRecord(
            corpus="real",
            index="ivf_flat",
            param_name="n_probes",
            param_value=1,
            recall=None,
            qps_min=None,
            qps_median=None,
            qps_p95=None,
            num_queries=10,
            failed="RuntimeError: cuvs search failed",
        )
    ]
    rows = report.headline_rows(builds, searches, target_recall=0.90)
    assert rows[0]["search_failed"] is True
    assert rows[0]["qps_at_target"] is None


def test_markdown_shows_search_failed_not_not_reached(tmp_path):
    path = tmp_path / "out.md"
    builds = [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=1.0,
            add_seconds=0.0,
            index_bytes=64,
            params={"n_lists": 8},
        )
    ]
    searches = [
        SearchRecord(
            corpus="real",
            index="ivf_flat",
            param_name="n_probes",
            param_value=1,
            recall=None,
            qps_min=None,
            qps_median=None,
            qps_p95=None,
            num_queries=10,
            failed="RuntimeError: cuvs search failed",
        )
    ]
    rows = report.headline_rows(builds, searches, target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    assert "search failed" in text


def test_html_lists_failed_cells_instead_of_omitting_them(tmp_path):
    path = tmp_path / "out.html"
    builds = _builds() + [_flat_build(corpus="v3", failed="RuntimeError: OOM")]
    searches = _searches()
    report.write_html(path, builds, searches, target_recall=0.90)
    text = path.read_text()
    assert "Failed cells" in text
    assert "OOM" in text


def test_write_json_carries_peak_vram_but_markdown_does_not_column_it(tmp_path):
    md_path = tmp_path / "out.md"
    json_path = tmp_path / "out.json"
    builds = [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=1.0,
            add_seconds=0.0,
            index_bytes=64,
            params={"n_lists": 8},
            peak_vram_bytes=123456,
        )
    ]
    report.write_json(json_path, builds=builds, searches=[], environment={})
    payload = json.loads(json_path.read_text())
    assert payload["builds"][0]["peak_vram_bytes"] == 123456

    rows = report.headline_rows(builds, [], target_recall=0.90)
    report.write_markdown(md_path, rows, target_recall=0.90)
    text = md_path.read_text()
    assert "123456" not in text


# --- Fix round 1: failure messages must not corrupt the medium they render
# into. A raw cuVS/CUDA error string can contain `|` (breaks a GFM table
# cell), `<`/`>` (parsed as HTML tags) and `&` (starts an HTML entity) --
# exactly the characters C++ template-type error text carries.


def test_markdown_escapes_a_pipe_in_the_failure_message_so_columns_stay_aligned(
    tmp_path,
):
    path = tmp_path / "out.md"
    malicious = "RuntimeError: bad | alloc <T> & <unnamed>"
    builds = [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=None,
            add_seconds=None,
            index_bytes=None,
            params={},
            failed=malicious,
        )
    ]
    rows = report.headline_rows(builds, [], target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    row_line = next(line for line in text.splitlines() if line.startswith("| real |"))
    # The header/row format is 7 columns -> 8 unescaped `|` delimiters. If the
    # injected `|` were left raw, this row would have 9 and every column
    # after it would be shifted for the rest of the table.
    unescaped_pipes = row_line.replace("\\|", "").count("|")
    assert unescaped_pipes == 8
    assert "\\|" in row_line
    # The message is escaped, not dropped -- it must still be legible.
    assert "bad" in row_line and "alloc" in row_line


def test_html_escapes_failure_messages_in_the_table_and_the_failed_cells_list(
    tmp_path,
):
    path = tmp_path / "out.html"
    build_failure = "std::bad_alloc<template<float>> & <unnamed>"
    search_failure = "cuvs search failed: <script>alert(1)</script> & boom"
    builds = [
        BuildRecord(
            corpus="real",
            index="flat",
            train_seconds=None,
            add_seconds=None,
            index_bytes=None,
            params={},
            failed=build_failure,
        ),
        BuildRecord(
            corpus="v2",
            index="ivf_flat",
            train_seconds=1.0,
            add_seconds=0.0,
            index_bytes=64,
            params={"n_lists": 8},
        ),
    ]
    searches = [
        SearchRecord(
            corpus="v2",
            index="ivf_flat",
            param_name="n_probes",
            param_value=1,
            recall=None,
            qps_min=None,
            qps_median=None,
            qps_p95=None,
            num_queries=10,
            failed=search_failure,
        )
    ]
    report.write_html(path, builds, searches, target_recall=0.90)
    text = path.read_text()

    # Neither raw failure string appears unescaped: a live `<script>` tag or
    # a stray `<`/`>` pair would corrupt the DOM rather than just look ugly.
    assert "<template<float>>" not in text
    assert "<script>alert(1)</script>" not in text
    # The escaped forms are present -- the message survives, legibly.
    assert "&lt;template&lt;float&gt;&gt;" in text
    assert "&amp;" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
