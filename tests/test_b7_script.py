"""B7 box script: a rerun after a failure resumes instead of dying (no GPU, no faiss)."""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B7 = REPO / "docs/results/scout-kv/controls/b7.sh"

FAKE_PY = """#!/bin/bash
echo "$*" >> "$CALLS"
out=""; outdir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out=$2; shift;;
    --out-dir) outdir=$2; shift;;
  esac
  shift
done
if [ -n "$out" ]; then echo '{}' > "$out"; fi
if [ -n "$outdir" ]; then
  for n in pooled_real norm_equalised gaussian; do echo '{}' > "$outdir/$n.jsonl"; done
fi
case " $ORIG_ARGS " in *"$FAIL_ON"*) [ -n "$FAIL_ON" ] && exit 1;; esac
exit 0
"""


def _fake(tmp_path):
    # ORIG_ARGS is rebuilt inside the script from "$@" before the loop consumes it.
    fake = tmp_path / "fakepy"
    fake.write_text(
        FAKE_PY.replace(
            'echo "$*" >> "$CALLS"', 'ORIG_ARGS="$*"\necho "$*" >> "$CALLS"'
        )
    )
    fake.chmod(0o755)
    return fake


def _run(tmp_path, fake, calls, fail_on):
    env = dict(os.environ)
    env.update(
        REPO_DIR=str(REPO),
        PY=str(fake),
        CAP=str(tmp_path / "cap"),
        OUT=str(tmp_path / "out"),
        CALLS=str(calls),
        FAIL_ON=fail_on,
    )
    return subprocess.run(
        ["bash", str(B7)], env=env, capture_output=True, text=True, timeout=120
    )


def test_failure_quarantines_outputs_and_marks_nothing(tmp_path):
    """Catches markers touched on failure and partial outputs left in $OUT."""
    fake = _fake(tmp_path)
    calls = tmp_path / "calls1.log"
    r = _run(tmp_path, fake, calls, "controls.py")
    out = tmp_path / "out"
    assert r.returncode == 1, r.stdout + r.stderr
    assert (out / ".done.neighbours").exists()
    assert (out / ".done.train_books_real").exists()
    assert not (out / ".done.controls").exists()
    failed = list(out.glob("failed-controls-*"))
    assert len(failed) == 1
    for n in ("pooled_real", "norm_equalised", "gaussian"):
        assert (failed[0] / f"{n}.jsonl").exists()
        assert not (out / f"{n}.jsonl").exists()
    assert (
        "v0_best_subset" not in calls.read_text()
        and "measure-run" not in calls.read_text()
    )


def test_rerun_resumes_and_skips_done_steps(tmp_path):
    """Catches no skip: the second run would call neighbours.py again."""
    fake = _fake(tmp_path)
    _run(tmp_path, fake, tmp_path / "calls1.log", "controls.py")
    calls2 = tmp_path / "calls2.log"
    r = _run(tmp_path, fake, calls2, "")
    out = tmp_path / "out"
    assert r.returncode == 0, r.stdout + r.stderr
    text = calls2.read_text()
    assert "neighbours.py" not in text and "measure-real" not in text
    assert "controls.py" in text and "measure-run" in text
    for s in ("neighbours", "train_books_real", "controls", "v0_best_subset"):
        assert (out / f".done.{s}").exists()
    assert "SKIP neighbours" in r.stdout


def test_logs_source_hashes(tmp_path):
    """Catches a run that does not record which code produced it."""
    r = _run(tmp_path, _fake(tmp_path), tmp_path / "calls.log", "")
    lines = [ln for ln in r.stdout.splitlines() if "src/kv/neighbours.py" in ln]
    assert any(len(ln.split()[0]) == 64 for ln in lines), r.stdout
