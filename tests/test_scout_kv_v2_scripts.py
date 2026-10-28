"""scout_kv v2 box scripts: seed_config.py and run.sh (no GPU, no faiss; a fake $PY)."""

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "docs/results/scout-kv/v2"
RUN_SH = V2 / "run.sh"
SEED_CONFIG = V2 / "seed_config.py"

# The fake logs argv, writes --out / --out-dir / --run-dir outputs where the real step would,
# and runs the real interpreter for seed_config.py (a pure-python step).
FAKE_PY = """#!/bin/bash
echo "$*" >> "$CALLS"
case "$1" in
  *seed_config.py) exec "$REAL_PY" "$@";;
esac
args="$*"
out=""; outdir=""; rundir=""; cfg=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out=$2; shift;;
    --out-dir) outdir=$2; shift;;
    --run-dir) rundir=$2; shift;;
    --config) cfg=$2; shift;;
  esac
  shift
done
if [ -n "$outdir" ]; then mkdir -p "$outdir"; touch "$outdir/partial"; fi
if [ -n "$FAIL_ON" ]; then case "$args" in *"$FAIL_ON"*) exit 1;; esac; fi
if [ -n "$cfg" ]; then
  d=$("$REAL_PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$cfg")
  mkdir -p "$d"; echo x > "$d/final.pt"
fi
if [ -n "$out" ]; then echo '{}' > "$out"; fi
exit 0
"""


def _setup(tmp_path):
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    return fake


def _run(tmp_path, fake, specs, fail_on="", name="calls.log"):
    calls = tmp_path / name
    env = dict(os.environ)
    env.update(
        REPO=str(REPO),
        PY=str(fake),
        REAL_PY=sys.executable,
        CAP=str(tmp_path / "cap"),
        RUNS=str(tmp_path / "runs"),
        CALLS=str(calls),
        FAIL_ON=fail_on,
        DISK_MIN_GB="0",
    )
    r = subprocess.run(
        ["bash", str(RUN_SH), *specs],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = calls.read_text().splitlines() if calls.exists() else []
    return r, lines


def test_seed_config_changes_only_seed_and_output_dir(tmp_path):
    """Catches seed-1 configs drifting from the committed ones, or an output dir other than the
    one run.sh reads: seed == 1, output_dir == /x/L19_s1, and the rest of the config is equal."""
    src = REPO / "configs/scout_kv/v2/L19.yaml"
    out = tmp_path / "o.yaml"
    subprocess.run(
        [sys.executable, str(SEED_CONFIG), str(src), "1", "/x/L19_s1", str(out)],
        check=True,
    )
    a, b = yaml.safe_load(src.read_text()), yaml.safe_load(out.read_text())
    assert b["seed"] == 1 and b["output_dir"] == "/x/L19_s1"
    assert a["seed"] != 1 or a["output_dir"] != "/x/L19_s1"
    for c in (a, b):
        c.pop("seed")
        c.pop("output_dir")
    assert a == b


def test_run_sh_trains_into_the_run_dir_it_reads(tmp_path):
    """Catches the trainer writing to the committed output_dir while run.sh looks in $R: the
    config passed to train_book_latent has output_dir == $R."""
    r, lines = _run(tmp_path, _setup(tmp_path), ["19:1"])
    assert r.returncode == 0, r.stdout + r.stderr
    train = [ln for ln in lines if "train_book_latent" in ln][0]
    cfg = Path(train.split("--config ")[1].split()[0])
    got = yaml.safe_load(cfg.read_text())
    assert got["output_dir"] == str(tmp_path / "runs/L19_s1")
    assert got["seed"] == 1


def test_run_sh_runs_the_steps_in_order(tmp_path):
    """Catches a step skipped or out of order, a dropped --source v2 or --books 12-19."""
    r, lines = _run(tmp_path, _setup(tmp_path), ["19:42"])
    assert r.returncode == 0, r.stdout + r.stderr
    keys = [
        "seed_config.py",
        "train_book_latent",
        "sample_books",
        "measure-real",
        "locality.py",
        "hub_position.py",
        "intrinsic.py",
    ]
    idx = [next(i for i, ln in enumerate(lines) if k in ln) for k in keys]
    assert idx == sorted(idx) and len(lines) == len(keys)
    gate = lines[idx[3]]
    assert "--source v2 " in gate + " " and "--books 12-19" in gate
    assert "--layers 19" in gate
    assert "--source v2_s42" in lines[idx[6]] and "--layer 19" in lines[idx[6]]
    for tok in ("START", "TRAINED", "DONE", "V2_COMPLETE"):
        assert tok in r.stdout
    assert "DONE " + str(tmp_path / "runs/L19_s42") in r.stdout


def test_run_sh_pads_the_layer_and_reads_it_as_decimal(tmp_path):
    """Catches a bare layer reaching the config path unpadded (L3.yaml) or a leading zero read as
    octal ('08' fails): '3:42' uses L03 and --layers 3; '07:1' uses L07 and --layers 7."""
    r, lines = _run(tmp_path, _setup(tmp_path), ["3:42", "07:1"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "runs/L03_s42").is_dir() and (tmp_path / "runs/L07_s1").is_dir()
    assert any("configs/scout_kv/v2/L03.yaml 42" in ln for ln in lines)
    assert any("configs/scout_kv/v2/L07.yaml 1" in ln for ln in lines)
    gates = [ln for ln in lines if "measure-real" in ln]
    assert "--layers 3 " in gates[0] + " " and "--layers 7 " in gates[1] + " "
    # no committed layer is octal-invalid, so pin the arithmetic run.sh uses on 08 and 09 too
    out = subprocess.run(
        ["bash", "-c", "L=08; echo $((10#$L)) $(printf %02d $((10#09)))"],
        capture_output=True,
        text=True,
    ).stdout
    assert out.split() == ["8", "09"]
    assert "10#$L" in RUN_SH.read_text()


def test_run_sh_rerun_skips_finished_steps(tmp_path):
    """Catches a rerun redoing any finished step (retraining, resampling, rescoring): the second
    invocation makes no PY call at all."""
    fake = _setup(tmp_path)
    _run(tmp_path, fake, ["19:42"])
    r, lines = _run(tmp_path, fake, ["19:42"], name="calls2.log")
    assert r.returncode == 0, r.stdout + r.stderr
    assert lines == []
    assert "V2_COMPLETE" in r.stdout


def test_run_sh_discards_a_partial_capture_and_resamples(tmp_path):
    """Catches sampling straight into capture/ (a killed sample leaves a partial capture that a
    rerun treats as done): a leftover capture.tmp is removed, sampling reruns, and the end
    state is capture/ without the leftover."""
    fake = _setup(tmp_path)
    R = tmp_path / "runs/L19_s42"
    (R / "capture.tmp").mkdir(parents=True)
    (R / "capture.tmp/stale").write_text("x")
    r, lines = _run(tmp_path, fake, ["19:42"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert any("sample_books" in ln and "capture.tmp" in ln for ln in lines)
    assert (R / "capture").is_dir() and (R / "capture/partial").exists()
    assert not (R / "capture.tmp").exists()
    assert not (R / "capture/stale").exists()


def test_run_sh_failed_sample_leaves_no_capture(tmp_path):
    """Catches a failed sampler leaving capture/ behind: FAILED sample, exit 1, no capture/,
    no gate call."""
    r, lines = _run(tmp_path, _setup(tmp_path), ["19:42"], fail_on="sample_books")
    assert r.returncode == 1
    assert "FAILED sample " + str(tmp_path / "runs/L19_s42") in r.stdout
    assert not (tmp_path / "runs/L19_s42/capture").exists()
    assert not any("measure-real" in ln for ln in lines)


def test_v1_seed1_intrinsic_creates_the_output_directory(tmp_path):
    """Catches a missing mkdir of the output's parent: the fake tool records whether the
    directory existed when it was called (the real tool only appends and would crash)."""
    fake = tmp_path / "fakepy"
    fake.write_text(
        '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --out ] && o=$2; shift; done\n'
        '[ -d "$(dirname "$o")" ] && echo "$o" > "$MARK"\n'
    )
    fake.chmod(0o755)
    out = tmp_path / "fresh/dir/v1.jsonl"
    env = dict(os.environ, PY=str(fake), OUT=str(out), MARK=str(tmp_path / "mark"))
    r = subprocess.run(
        ["bash", str(V2 / "v1_seed1_intrinsic.sh")],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mark").read_text().strip() == str(out)


def test_run_sh_stops_on_a_failed_step(tmp_path):
    """Catches a failure that does not stop the run: FAILED gate, exit 1, no locality call, no
    gate.jsonl left behind, and the second run spec not started."""
    r, lines = _run(
        tmp_path, _setup(tmp_path), ["19:42", "27:42"], fail_on="measure-real"
    )
    assert r.returncode == 1
    assert "FAILED gate " + str(tmp_path / "runs/L19_s42") in r.stdout
    assert not any("locality.py" in ln for ln in lines)
    assert not any("L27" in ln for ln in lines)
    assert not (tmp_path / "runs/L19_s42/gate.jsonl").exists()
    assert "V2_COMPLETE" not in r.stdout


def test_run_sh_stops_when_the_disk_is_short(tmp_path):
    """Catches training on a full disk: DISK_MIN_GB above the free space gives DISK_SHORT, exit 3."""
    fake = _setup(tmp_path)
    env = dict(
        os.environ,
        REPO=str(REPO),
        PY=str(fake),
        REAL_PY=sys.executable,
        CAP=str(tmp_path / "cap"),
        RUNS=str(tmp_path / "runs"),
        CALLS=str(tmp_path / "c.log"),
        FAIL_ON="",
        DISK_MIN_GB="999999999",
    )
    r = subprocess.run(
        ["bash", str(RUN_SH), "19:42"], env=env, capture_output=True, text=True
    )
    assert r.returncode == 3 and "DISK_SHORT" in r.stdout
    assert (
        not (tmp_path / "c.log").exists()
        or "train_book_latent" not in (tmp_path / "c.log").read_text()
    )
