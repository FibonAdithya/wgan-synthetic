"""Tests for the scout_kv v2 judge (src/kv/book_judge.py)."""

import json
from pathlib import Path

import pytest

from src.kv.book_judge import (
    ALL_LAYERS,
    body_lid,
    load_v2,
    main,
    port_check,
    port_ranges,
    same_book_share,
    v1_final_lid,
    v1_lid_table,
    verdict,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "docs/results/scout-kv"
REAL_LOCALITY = RESULTS / "locality/locality.json"
PROBE_RUNS = RESULTS / "v2-probe/runs"
V1_FULL = RESULTS / "v1/full/intrinsic.jsonl"
V1S1_L27 = RESULTS / "v2-probe/runs/v1s1_intrinsic_L27.jsonl"
LAYERS12 = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)


def _write_intrinsic(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(layer, source, lid, flag="body"):
    return {
        "layer": layer,
        "source": source,
        "flag": flag,
        "n_rows": 5,
        "lid_median": lid,
    }


def _write_locality(path, shares):
    layers = [{"layer": L, "h_l2": {"same_book": s}} for L, s in shares.items()]
    path.write_text(json.dumps({"layers": layers}))


def _v2(sb=0.5, lid=20.0, layers=LAYERS12):
    return {(L, s): {"same_book": sb, "body_lid": lid} for L in layers for s in (42, 1)}


def _real(v=0.5, layers=LAYERS12):
    return {L: v for L in layers}


def _v1(v=20.0, layers=LAYERS12):
    return {(L, s): v for L in layers for s in (42, 1)}


def test_real_same_book_matches_the_spec():
    """Pins the spec's per-layer real values (criterion 1's targets)."""
    want = {
        3: 0.409,
        7: 0.431,
        11: 0.458,
        15: 0.530,
        19: 0.645,
        23: 0.816,
        27: 0.895,
        31: 0.877,
        35: 0.865,
        39: 0.770,
        43: 0.786,
        47: 0.841,
    }
    for L, v in want.items():
        assert same_book_share(REAL_LOCALITY, L) == pytest.approx(v, abs=5e-4)


def test_same_book_share_missing_layer_raises():
    """Catches a silent default for an absent layer."""
    with pytest.raises(KeyError):
        same_book_share(REAL_LOCALITY, 4)


def test_body_lid_selects_flag_and_source_and_rejects_ambiguity(tmp_path):
    """Catches reading the tail row, ignoring the source filter, or taking the first of several."""
    p = tmp_path / "i.jsonl"
    _write_intrinsic(
        p, [_row(3, "a", 1.0), _row(3, "a", 9.0, "tail"), _row(3, "b", 2.0)]
    )
    assert body_lid(p, 3, "a") == 1.0
    assert body_lid(p, 3, "b") == 2.0
    with pytest.raises(ValueError):
        body_lid(p, 3)  # two body rows
    with pytest.raises(ValueError):
        body_lid(p, 7, "a")  # none


def test_port_ranges_match_the_spec():
    """Pins the spec's port-check windows from the committed probe runs."""
    r = port_ranges(PROBE_RUNS)
    assert r[19]["body_lid"] == pytest.approx((22.72 - 1.5, 23.67 + 1.5), abs=0.01)
    assert r[19]["same_book"] == pytest.approx((0.742 - 0.1, 0.766 + 0.1), abs=0.001)
    assert r[27]["body_lid"] == pytest.approx((20.35 - 1.5, 21.02 + 1.5), abs=0.01)
    assert r[27]["same_book"] == pytest.approx((0.902 - 0.1, 0.935 + 0.1), abs=0.001)


def _probe_values():
    out = {}
    for L, name in ((19, "Bflag25"), (27, "Bflag")):
        for s in (42, 1):
            d = PROBE_RUNS / f"L{L}/{name}_s{s}"
            out[(L, s)] = {
                "same_book": same_book_share(d / "locality.json", L),
                "body_lid": body_lid(d / "intrinsic.jsonl", L),
            }
    return out


def test_the_probe_runs_pass_their_own_port_check():
    """Catches an inverted comparison: Bflag25 (L19) and Bflag (L27) measured values pass."""
    res = port_check(_probe_values(), port_ranges(PROBE_RUNS))
    assert res["passed"] is True
    assert all(v["same_book_ok"] and v["lid_ok"] for v in res["runs"].values())


def test_port_check_fails_outside_either_window_and_names_which():
    """Catches same_book and lid checks being swapped or merged into one flag."""
    ranges = port_ranges(PROBE_RUNS)
    v = _probe_values()
    v[(19, 42)] = dict(v[(19, 42)], body_lid=ranges[19]["body_lid"][1] + 0.5)
    v[(27, 1)] = dict(v[(27, 1)], same_book=ranges[27]["same_book"][0] - 0.01)
    res = port_check(v, ranges)
    assert res["passed"] is False
    assert res["runs"][(19, 42)] == {"same_book_ok": True, "lid_ok": False}
    assert res["runs"][(27, 1)] == {"same_book_ok": False, "lid_ok": True}
    assert res["runs"][(19, 1)] == {"same_book_ok": True, "lid_ok": True}


def test_port_check_window_edge_is_inclusive():
    """Catches a strict comparison at the port-window edge (EPS dropped)."""
    ranges = {
        L: {"same_book": (0.1 + 0.2, 0.7), "body_lid": (10.0, 20.0)} for L in (19, 27)
    }
    v = {
        (L, s): {"same_book": 0.3, "body_lid": 20.0} for L in (19, 27) for s in (42, 1)
    }
    # 0.1 + 0.2 == 0.30000000000000004 > 0.3: only EPS lets 0.3 pass
    assert port_check(v, ranges)["passed"] is True


def test_port_check_missing_run_raises():
    """Catches a missing (layer, seed) being skipped instead of failing."""
    v = _probe_values()
    del v[(27, 1)]
    with pytest.raises(ValueError):
        port_check(v, port_ranges(PROBE_RUNS))


def test_v1_final_lid_uses_the_final_row_where_it_exists():
    """Catches v1's best checkpoint being used (spec: L7 best 18.51 vs final 15.19)."""
    assert v1_final_lid(V1_FULL, 7) == pytest.approx(15.19, abs=0.01)
    assert v1_final_lid(V1_FULL, 27) == pytest.approx(
        25.03, abs=0.01
    )  # best is step 30,000


def test_c1_boundary_is_inclusive():
    """Catches a strict < at the threshold: real 0.5, v2 0.65 passes (0.65 - 0.5 is 0.15000000000000002)."""
    v = _v2(sb=0.65, layers=(3,))
    r = verdict(v, {3: 0.5}, _v1(layers=(3,)), (3,))
    assert r["runs"][(3, 42)]["c1"] is True
    v = _v2(sb=0.651, layers=(3,))
    assert verdict(v, {3: 0.5}, _v1(layers=(3,)), (3,))["runs"][(3, 42)]["c1"] is False
    v = _v2(sb=0.349, layers=(3,))  # below real, also two-sided
    assert verdict(v, {3: 0.5}, _v1(layers=(3,)), (3,))["runs"][(3, 42)]["c1"] is False


def test_c2_is_one_sided():
    """Catches c2 being two-sided (LID far above v1 must pass) or its 6.0 edge being strict."""

    def c2(lid):
        v = _v2(lid=lid, layers=(3,))
        return verdict(v, _real(layers=(3,)), _v1(20.0, layers=(3,)), (3,))["runs"][
            (3, 42)
        ]["c2"]

    assert c2(30.0) is True
    assert c2(14.0) is True  # exactly 6 below
    assert c2(13.99) is False
    r = verdict(
        _v2(lid=13.99, layers=(3,)), _real(layers=(3,)), _v1(20.0, layers=(3,)), (3,)
    )
    assert (
        r["runs"][(3, 42)]["v1_lid"] == 20.0 and r["runs"][(3, 42)]["body_lid"] == 13.99
    )


def test_aim_needs_ten_c1_layers_and_twelve_c2_layers():
    """Catches the aim rule's counts being off by one (MIN_C1_LAYERS 9 or 11, or c2 needing 11)."""

    def aim(n_c1_fail, c2_fail_at=None):
        v = _v2()
        for L in LAYERS12[:n_c1_fail]:
            for s in (42, 1):
                v[(L, s)] = dict(v[(L, s)], same_book=0.9)
        if c2_fail_at is not None:
            v[(c2_fail_at, 1)] = dict(v[(c2_fail_at, 1)], body_lid=1.0)
        return verdict(v, _real(), _v1(), LAYERS12)["aim"]

    assert aim(0) is True
    assert aim(2) is True  # c1 at exactly 10 layers
    assert aim(3) is False  # c1 at 9
    assert aim(0, c2_fail_at=47) is False  # one c2 miss at one seed


def test_a_layer_counts_only_if_both_seeds_pass():
    """Catches c1_both being 'either seed' or only the first seed."""
    v = _v2()
    v[(3, 1)] = dict(v[(3, 1)], same_book=0.9)
    r = verdict(v, _real(), _v1(), LAYERS12)
    assert r["layers"][3]["c1_both"] is False
    assert r["layers"][7]["c1_both"] is True
    v = _v2()
    v[(3, 42)] = dict(v[(3, 42)], body_lid=1.0)
    r = verdict(v, _real(), _v1(), LAYERS12)
    assert r["layers"][3]["c2_both"] is False


def test_seed1_uses_v1_seed1_only_where_it_exists(tmp_path):
    """Catches v2 seed 1 compared with v1 seed 42 at L19 (or with a missing v1 seed-1 value at L3)."""
    s1a = tmp_path / "a.jsonl"
    s1b = tmp_path / "b.jsonl"
    _write_intrinsic(s1a, [_row(27, "v1_best", 11.0)])
    _write_intrinsic(
        s1b,
        [
            _row(19, "v1_best", 12.0),
            _row(35, "v1_final", 13.0),
            _row(35, "v1_best", 99.0),
            _row(43, "v1_best", 14.0),
        ],
    )
    t = v1_lid_table(V1_FULL, [s1a, s1b], (3, 19, 27, 35, 43))
    s42 = {L: v1_final_lid(V1_FULL, L) for L in (3, 19, 27, 35, 43)}
    assert t[(19, 1)] == 12.0 and t[(19, 42)] == s42[19]
    assert t[(27, 1)] == 11.0
    assert t[(35, 1)] == 13.0  # final row preferred
    assert t[(43, 1)] == 14.0
    assert t[(3, 1)] == s42[3] == t[(3, 42)]
    assert t[(19, 1)] != t[(19, 42)]


def test_seed1_layer_missing_from_every_file_raises(tmp_path):
    """Catches a silent fallback to seed 42 where v1 seed 1 should exist."""
    s1 = tmp_path / "a.jsonl"
    _write_intrinsic(s1, [_row(27, "v1_best", 11.0)])
    with pytest.raises(ValueError):
        v1_lid_table(V1_FULL, [s1], (19, 27))
    # but a layer outside V1_SEED1_LAYERS never needs the file
    assert v1_lid_table(V1_FULL, [s1], (3,))[(3, 1)] == v1_final_lid(V1_FULL, 3)


def test_real_v1_seed1_file_loads():
    """Catches body_lid failing on the committed seed-1 file (real + v1_best rows)."""
    assert v1_final_lid(V1S1_L27, 27) > 0


def test_pilot_layers_give_no_aim():
    """Catches aim being computed (True/False) on a partial layer set."""
    pl = (3, 19, 27)
    r = verdict(_v2(layers=pl), _real(layers=pl), _v1(layers=pl), pl)
    assert r["aim"] is None
    assert set(r["layers"]) == set(pl)
    assert (19, 1) in r["runs"]


def test_missing_run_raises():
    """Catches a (layer, seed) absent from v2 being skipped."""
    v = _v2(layers=(3, 19))
    del v[(19, 1)]
    with pytest.raises(ValueError):
        verdict(v, _real(), _v1(), (3, 19))


def test_all_layers_constant():
    """Catches a wrong layer list, which would silently turn a full run into a pilot."""
    assert tuple(ALL_LAYERS) == LAYERS12


def _make_run(runs, L, s, sb, lid):
    d = runs / f"L{L:02d}_s{s}"
    d.mkdir(parents=True)
    _write_locality(d / "locality.json", {L: sb})
    _write_intrinsic(
        d / "intrinsic.jsonl",
        [_row(L, f"v2_s{s}", lid), _row(L, f"v2_s{s}", 1.0, "tail")],
    )


def test_load_v2_reads_zero_padded_dirs_and_the_seed_source(tmp_path):
    """Catches unpadded dir names (L3_s42) or a wrong source label / flag."""
    _make_run(tmp_path, 3, 42, 0.4, 17.0)
    _make_run(tmp_path, 3, 1, 0.41, 18.0)
    v = load_v2(tmp_path, (3,))
    assert v[(3, 42)] == {"same_book": pytest.approx(0.4), "body_lid": 17.0}
    assert v[(3, 1)]["body_lid"] == 18.0
    with pytest.raises(ValueError):
        load_v2(tmp_path, (3, 7))


def test_cli_port_and_verdict_write_json(tmp_path):
    """Catches CLI wiring: defaults from the repo root, run-dir layout, output file."""
    runs = tmp_path / "runs"
    for L, name in ((19, "Bflag25"), (27, "Bflag")):
        for s in (42, 1):
            d = PROBE_RUNS / f"L{L}/{name}_s{s}"
            _make_run(
                runs,
                L,
                s,
                same_book_share(d / "locality.json", L),
                body_lid(d / "intrinsic.jsonl", L),
            )
    out = tmp_path / "port.json"
    assert main(["port", "--runs-dir", str(runs), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True

    vruns = tmp_path / "vruns"
    for s in (42, 1):
        _make_run(vruns, 19, s, 0.645, 20.0)
        _make_run(vruns, 27, s, 0.895, 20.0)
    _write_intrinsic(
        tmp_path / "s1.jsonl", [_row(19, "v1_best", 20.0), _row(27, "v1_best", 20.0)]
    )
    vout = tmp_path / "verdict.json"
    assert (
        main(
            [
                "verdict",
                "--runs-dir",
                str(vruns),
                "--v1-seed1",
                str(tmp_path / "s1.jsonl"),
                "--layers",
                "19,27",
                "--out",
                str(vout),
            ]
        )
        == 0
    )
    res = json.loads(vout.read_text())
    assert res["aim"] is None
    assert res["layers"]["19"]["c1_both"] is True
