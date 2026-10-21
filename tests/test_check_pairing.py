"""B7 pairing check against the committed seed0 reference."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "docs/results/scout-kv/controls/check_pairing.py"
REF = REPO / "docs/results/scout-kv/reference/seed0.jsonl"
META = ("seed", "books", "n_sink", "n_tail", "n_keys")


def _ref():
    return [json.loads(ln) for ln in REF.read_text().splitlines() if ln.strip()]


def _nb(ref_rows):
    return [
        {
            "layer": r["layer"],
            "kv_head": r["kv_head"],
            **{k: r[k] for k in META},
            "ood": {"own_book_share": r["own_book_share_top10"]},
        }
        for r in ref_rows
    ]


def _check(tmp_path, rows):
    p = tmp_path / "nb.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--neighbours", str(p), "--reference", str(REF)],
        env=env,
        capture_output=True,
        text=True,
    )
    return r.returncode, json.loads(r.stdout), r


def test_exact_copy_passes(tmp_path):
    """Catches blank lines crashing and n_exact miscounting."""
    rc, rep, r = _check(tmp_path, _nb(_ref()))
    assert rc == 0, r.stderr
    assert rep["n_exact"] == 96 and rep["n_compared"] == 96
    assert (
        rep["missing"] == [] and rep["mismatched"] == [] and rep["meta_mismatch"] == []
    )


def test_shifted_corpus_is_mismatched_only(tmp_path):
    rows = _nb(_ref())
    rows[5]["ood"]["own_book_share"] += 0.01
    key = [rows[5]["layer"], rows[5]["kv_head"]]
    rc, rep, _ = _check(tmp_path, rows)
    assert rc == 1
    assert rep["mismatched"] == [key]
    assert rep["missing"] == [] and rep["meta_mismatch"] == []


def test_removed_corpus_is_missing_only(tmp_path):
    rows = _nb(_ref())
    gone = rows.pop(7)
    rc, rep, _ = _check(tmp_path, rows)
    assert rc == 1
    assert rep["missing"] == [[gone["layer"], gone["kv_head"]]]
    assert rep["mismatched"] == [] and rep["meta_mismatch"] == []


def test_differing_seed_is_meta_mismatch(tmp_path):
    rows = _nb(_ref())
    rows[3]["seed"] = 1
    rc, rep, _ = _check(tmp_path, rows)
    assert rc == 1
    assert rep["meta_mismatch"] == [[rows[3]["layer"], rows[3]["kv_head"]]]
    assert rep["mismatched"] == [] and rep["missing"] == []


# --- held-out plan, Task 3: --records and --layers ---


def _run(tmp_path, rows, *extra):
    p = tmp_path / "recs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--records", str(p), "--reference", str(REF)]
        + list(extra),
        env=env,
        capture_output=True,
        text=True,
    )
    return r.returncode, json.loads(r.stdout), r


def test_records_mode_with_layers(tmp_path):
    """Catches --records reading the neighbours field, --layers filtering only one
    side, and the filter hiding corpora missing inside the chosen layers."""
    ref = _ref()
    rc, rep, r = _run(tmp_path, ref)
    assert rc == 0 and rep["n_compared"] == 96, r.stderr
    rc, rep, r = _run(tmp_path, ref, "--layers", "11,19,43")
    assert rc == 0 and rep["n_compared"] == 24 and rep["missing"] == [], r.stderr
    rule = [x for x in ref if x["layer"] in (11, 19, 43)]
    rc, rep, r = _run(tmp_path, rule, "--layers", "11,19,43")
    assert rc == 0 and rep["n_compared"] == 24, r.stderr
    rc, rep, _ = _run(tmp_path, rule)
    assert rc == 1 and len(rep["missing"]) == 72
    moved = [dict(x) for x in rule]
    i = next(k for k, x in enumerate(moved) if x["layer"] == 11)
    moved[i]["own_book_share_top10"] += 0.002
    rc, rep, _ = _run(tmp_path, moved, "--layers", "11,19,43")
    assert rc == 1 and rep["mismatched"] == [[11, moved[i]["kv_head"]]]
    j = next(k for k, x in enumerate(rule) if x["layer"] == 19)
    dropped = rule[:j] + rule[j + 1 :]
    rc, rep, _ = _run(tmp_path, dropped, "--layers", "11,19,43")
    assert rc == 1 and rep["missing"] == [[19, rule[j]["kv_head"]]]
