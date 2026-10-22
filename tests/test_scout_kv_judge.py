"""Tests for src/kv/judge.py, the "moves toward real" rule of the scout_kv v1 spec
(docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md, "How v1 is judged").

Each test's docstring names the mutation of judge.py it catches."""

import json
from pathlib import Path

import pytest

from src.eval import kv_gate
from src.kv.judge import (
    distance,
    full_verdict,
    is_borderline,
    main,
    moves_toward,
    pilot_verdict,
    train_books_targets,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs/results/scout-kv"
TRAIN_BOOKS_PATH = RESULTS / "controls/train_books_real.jsonl"
REFERENCE_PATHS = [RESULTS / f"reference/seed{s}.jsonl" for s in range(5)]
V0_BEST = RESULTS / "v0/best.jsonl"
GATE_FILE = REPO / "gates/scout_kv.yaml"

PILOT_STATS = ("ivf_ratio_k128", "hnsw_reach_k10", "hnsw_recall_k10_ef176")
KEY_SIDE = "hnsw_recall_k10_ef176_id"


def _read(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _layer(
    layer,
    *,
    reach=0.5,
    recall=0.8,
    ratio=1.0,
    ivf_recall=0.83,
    id_recall=1.0,
    seed=0,
    checkpoint="best_generator.pt",
):
    """Eight records (one per KV head) whose pooled statistics are exactly the
    given values: reach is a share of 8 heads, the rest are constant per head."""
    n_reached = round(reach * 8)
    assert n_reached / 8 == reach, "reach must be a multiple of 1/8"
    out = []
    for g in range(8):
        out.append(
            {
                "layer": layer,
                "kv_head": g,
                "seed": seed,
                "checkpoint": checkpoint,
                "plain_ood_k10": {
                    "hnsw_reached": g < n_reached,
                    "hnsw_recall_op": recall,
                },
                "plain_id_k10": {"hnsw_reached": True, "hnsw_recall_op": id_recall},
                # ratio * 0.5 / 0.5 is exact in binary floating point.
                "plain_ood_k128": {
                    "ivf_scan_frac": ratio * 0.5,
                    "ivf_recall_op": ivf_recall,
                },
                "plain_id_k128": {"ivf_scan_frac": 0.5, "ivf_recall_op": 1.0},
            }
        )
    return out


def _pilot_set(l19=None, l43=None, seed=0, checkpoint="best_generator.pt"):
    return _layer(19, seed=seed, checkpoint=checkpoint, **(l19 or {})) + _layer(
        43, seed=seed, checkpoint=checkpoint, **(l43 or {})
    )


# Synthetic pilot inputs. Per-layer reference ranges (two draws):
#   L19 reach 0.5-0.75, recall 0.85-0.88, ratio 0.13-0.17, key-side 1.0
#   L43 reach 0.0-0.125, recall 0.57-0.64, ratio 2.9-3.5, key-side 1.0
REFS = [
    _pilot_set(
        dict(reach=0.5, recall=0.85, ratio=0.13),
        dict(reach=0.0, recall=0.57, ratio=2.9),
    ),
    _pilot_set(
        dict(reach=0.75, recall=0.88, ratio=0.17),
        dict(reach=0.125, recall=0.64, ratio=3.5),
    ),
]
# Train books inside the L19 reference range, so the widening changes nothing.
TRAIN = _layer(19, reach=0.5, recall=0.86, ratio=0.15)
# v0: too easy at both layers, key-side below real.
V0_L19 = dict(reach=1.0, recall=0.97, ratio=0.09, id_recall=0.97)
V0_L43 = dict(reach=0.375, recall=0.82, ratio=0.76, id_recall=0.97)
# v1 that lands inside every target at both layers.
GOOD_L19 = dict(reach=0.625, recall=0.86, ratio=0.15, id_recall=0.97)
GOOD_L43 = dict(reach=0.0, recall=0.60, ratio=3.0, id_recall=0.97)


def _v0(seeds=(0, 1)):
    return {s: _pilot_set(V0_L19, V0_L43, seed=s) for s in seeds}


# ---------------------------------------------------------------------------
# distance / moves_toward / is_borderline


def test_distance_is_zero_inside_and_gap_outside():
    """Catches distance using the far edge, an exclusive bound, or treating a
    None bound as 0 instead of open."""
    assert distance(0.5, 0.0, 1.0) == 0.0
    assert distance(1.0, 0.0, 1.0) == 0.0
    assert distance(0.0, 0.0, 1.0) == 0.0
    assert distance(1.25, 0.0, 1.0) == pytest.approx(0.25)
    assert distance(-0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert distance(0.95, 0.99, None) == pytest.approx(0.04)
    assert distance(5.0, 0.99, None) == 0.0
    assert distance(-5.0, None, 1.0) == 0.0


def test_v0_whole_gap_smaller_than_spread_can_still_move_by_entering():
    """Catches the rule the builder check rejected: pooled hnsw_recall_k10_ef176 v0 0.9274,
    band top 0.88, spread 0.0544 -- a v1 at 0.86 moves; a v1 at 0.90 does not.
    Mutation caught: dropping the "inside" clause from moves_toward."""
    assert moves_toward(0.86, 0.9274, 0.78, 0.88, 0.0544)
    assert not moves_toward(0.90, 0.9274, 0.78, 0.88, 0.0544)


def test_overshoot_counts_as_distance():
    """Catches raw displacement being scored: L43 ivf_ratio range 2.974-3.480, v0 0.761;
    v1 at 6.0 is 2.52 past the top, v0 is 2.213 below -> not a move.
    Mutation caught: scoring |v1 - v0| or distance to the near edge on v0's side only."""
    assert not moves_toward(6.0, 0.761, 2.974, 3.480, 0.5066)
    # A v1 at 2.5 closes 1.739 of v0's 2.213 gap, more than the spread.
    assert moves_toward(2.5, 0.761, 2.974, 3.480, 0.5066)


def test_moves_needs_the_gap_to_close_by_more_than_the_spread():
    """Catches `<=` in place of `<`, and the spread being ignored. v0 distance
    0.5, spread 0.25 (all values exact in binary): closing exactly 0.25 is not
    a move, closing 0.125 is not, closing 0.375 is."""
    assert not moves_toward(1.25, 1.5, 0.0, 1.0, 0.25)
    assert not moves_toward(1.375, 1.5, 0.0, 1.0, 0.25)
    assert moves_toward(1.125, 1.5, 0.0, 1.0, 0.25)


def test_borderline_is_flagged_but_still_moves():
    """Catches borderline using 1 x spread, being set on non-moves, or being set
    on a v1 inside the target. v0 distance 0.5, spread 0.1."""
    # Gap closed by 1.5 x spread: moves, borderline.
    assert moves_toward(1.35, 1.5, 0.0, 1.0, 0.1)
    assert is_borderline(1.35, 1.5, 0.0, 1.0, 0.1)
    # Gap closed by 2.5 x spread: moves, not borderline.
    assert moves_toward(1.25, 1.5, 0.0, 1.0, 0.1)
    assert not is_borderline(1.25, 1.5, 0.0, 1.0, 0.1)
    # Inside the target: moves, never borderline.
    assert not is_borderline(0.95, 1.5, 0.0, 1.0, 0.1)
    # Gap closed by 0.5 x spread: no move, so not borderline.
    assert not is_borderline(1.45, 1.5, 0.0, 1.0, 0.1)


# ---------------------------------------------------------------------------
# train books


def test_train_books_targets_match_the_committed_file():
    """Pins the L19 numbers the spec quotes. Catches reading the wrong layer or
    file, or pooling across layers."""
    t = train_books_targets(TRAIN_BOOKS_PATH, layer=19)
    assert t["hnsw_reach_k10"] == 0.375
    assert t["hnsw_recall_k10_ef176"] == pytest.approx(0.8311, abs=1e-4)
    assert t["ivf_ratio_k128"] == pytest.approx(0.3687, abs=1e-4)


def test_train_books_targets_refuses_a_layer_not_in_the_file():
    """Catches an absent layer giving NaN/None statistics instead of an error."""
    with pytest.raises(ValueError, match="layer 3"):
        train_books_targets(TRAIN_BOOKS_PATH, layer=3)


# ---------------------------------------------------------------------------
# pilot_verdict


def test_l19_target_is_widened_to_the_training_books():
    """Catches the owner's L19 exception being dropped: L19 ivf_ratio v1 0.30 is outside the
    reference range 0.133-0.172 but inside 0.133-0.369, so it moves. Uses the committed
    reference draws, train-books file and v0 best.
    Mutation caught: removing the L19 widening (v1 distance 0.128 > v0 distance 0.038)."""
    refs = [_read(p) for p in REFERENCE_PATHS]
    train = _read(TRAIN_BOOKS_PATH)
    v0 = _read(V0_BEST)
    v1 = _pilot_set(dict(ratio=0.30), dict(ratio=3.0))
    out = pilot_verdict({0: v1}, {0: v0}, refs, train)
    cell = out["layers"][19]["statistics"]["ivf_ratio_k128"]
    assert cell["reference_min"] == pytest.approx(0.1330, abs=1e-4)
    assert cell["reference_max"] == pytest.approx(0.1718, abs=1e-4)
    assert cell["target_min"] == pytest.approx(0.1330, abs=1e-4)
    assert cell["target_max"] == pytest.approx(0.3687, abs=1e-4)
    # The spread stays the reference's, not the widened interval's.
    assert cell["spread"] == pytest.approx(0.1718 - 0.1330, abs=2e-4)
    assert cell["seeds"][0]["v1_distance"] == 0.0
    assert cell["moves"] is True
    # L19 reach and recall widen downward to the train-books values.
    reach = out["layers"][19]["statistics"]["hnsw_reach_k10"]
    assert (reach["target_min"], reach["target_max"]) == (0.375, 0.875)
    # L43 is not widened.
    l43 = out["layers"][43]["statistics"]["ivf_ratio_k128"]
    assert l43["target_min"] == l43["reference_min"]
    assert l43["target_max"] == l43["reference_max"]
    assert l43["target_max"] == pytest.approx(3.4804, abs=1e-4)


def test_pilot_go_when_everything_moves():
    """Catches a judge that always says no-go: the base case for the go tests below."""
    v1 = {s: _pilot_set(GOOD_L19, GOOD_L43, seed=s) for s in (0, 1)}
    out = pilot_verdict(v1, _v0(), REFS, TRAIN)
    assert out["go"] is True
    for layer in (19, 43):
        for name in PILOT_STATS:
            assert out["layers"][layer]["statistics"][name]["moves"] is True
        assert out["layers"][layer]["key_side"]["not_lower"] is True


def test_pilot_go_needs_every_seed_and_both_layers():
    """Catches `moves` being any-seed instead of every-seed, and `go` being
    computed from one layer: one seed failing at L43 -> go False."""
    bad_l43 = dict(GOOD_L43, ratio=0.76)  # v0's L43 IVF ratio: no move
    v1 = {
        0: _pilot_set(GOOD_L19, GOOD_L43, seed=0),
        1: _pilot_set(GOOD_L19, bad_l43, seed=1),
    }
    out = pilot_verdict(v1, _v0(), REFS, TRAIN)
    cell = out["layers"][43]["statistics"]["ivf_ratio_k128"]
    assert cell["seeds"][0]["moves"] is True
    assert cell["seeds"][1]["moves"] is False
    assert cell["moves"] is False
    assert out["layers"][19]["statistics"]["ivf_ratio_k128"]["moves"] is True
    assert out["go"] is False


def test_pilot_pairs_seeds():
    """Catches v1 at every seed being compared with v0 seed 0: build v0 seed 1 records far
    from real and v0 seed 0 near real; a v1 seed-1 value that beats v0 seed 1 but not v0
    seed 0 moves under pairing and would not move against seed 0."""
    # L43 ratio target 2.9-3.5, spread 0.6.
    v0 = {
        0: _pilot_set(V0_L19, dict(V0_L43, ratio=2.5), seed=0),  # distance 0.4
        1: _pilot_set(V0_L19, dict(V0_L43, ratio=0.5), seed=1),  # distance 2.4
    }
    v1 = {
        0: _pilot_set(GOOD_L19, GOOD_L43, seed=0),  # inside
        # distance 1.4: closes 1.0 > 0.6 against seed 1, but is farther than seed 0
        1: _pilot_set(GOOD_L19, dict(GOOD_L43, ratio=1.5), seed=1),
    }
    out = pilot_verdict(v1, v0, REFS, TRAIN)
    cell = out["layers"][43]["statistics"]["ivf_ratio_k128"]
    assert cell["seeds"][1]["v0"] == pytest.approx(0.5)
    assert cell["seeds"][1]["v0_distance"] == pytest.approx(2.4)
    assert cell["seeds"][1]["moves"] is True
    assert cell["moves"] is True
    assert out["go"] is True


def test_pilot_refuses_unpaired_seeds():
    """Catches the seed sets not being checked: v1 at seeds {0, 1}, v0 at {0}."""
    v1 = {s: _pilot_set(GOOD_L19, GOOD_L43, seed=s) for s in (0, 1)}
    with pytest.raises(ValueError, match="seed"):
        pilot_verdict(v1, _v0(seeds=(0,)), REFS, TRAIN)


def test_pilot_refuses_records_labelled_with_another_seed():
    """Catches a mislabelled --v1 SEED=PATH: records measured at seed 0 passed as
    seed 1 would pair with the wrong v0 draw."""
    v1 = {
        0: _pilot_set(GOOD_L19, GOOD_L43, seed=0),
        1: _pilot_set(GOOD_L19, GOOD_L43, seed=0),
    }
    with pytest.raises(ValueError, match="seed"):
        pilot_verdict(v1, _v0(), REFS, TRAIN)


def test_pilot_refuses_mixed_checkpoint_kinds():
    """Catches best and final checkpoints being mixed across layers, which the
    spec forbids ("all-best or all-final, never mixed across layers")."""
    mixed = _layer(19, seed=0, **GOOD_L19) + _layer(
        43, seed=0, checkpoint="checkpoint_step_30000.pt", **GOOD_L43
    )
    with pytest.raises(ValueError, match="checkpoint"):
        pilot_verdict({0: mixed}, _v0(seeds=(0,)), REFS, TRAIN)


def test_key_side_drop_beyond_v0_seed_spread_blocks_go():
    """Catches key-side recall being left out of `go`, and its allowance being
    the zero reference spread instead of v0's seed spread. v0 key-side 0.97 and
    0.96 at L19 (allowance 0.01): v1 0.955 at seed 0 is below 0.96 -> no-go;
    v1 0.961 at seed 0 is within the allowance -> go."""
    v0 = {
        0: _pilot_set(V0_L19, V0_L43, seed=0),
        1: _pilot_set(dict(V0_L19, id_recall=0.96), V0_L43, seed=1),
    }

    def v1_at(id0):
        return {
            0: _pilot_set(dict(GOOD_L19, id_recall=id0), GOOD_L43, seed=0),
            1: _pilot_set(dict(GOOD_L19, id_recall=0.96), GOOD_L43, seed=1),
        }

    low = pilot_verdict(v1_at(0.955), v0, REFS, TRAIN)
    ks = low["layers"][19]["key_side"]
    assert ks["v0_spread"] == pytest.approx(0.01)
    assert ks["seeds"][0]["not_lower"] is False
    assert ks["not_lower"] is False
    for name in PILOT_STATS:
        assert low["layers"][19]["statistics"][name]["moves"] is True
    assert low["go"] is False

    within = pilot_verdict(v1_at(0.961), v0, REFS, TRAIN)
    assert within["layers"][19]["key_side"]["not_lower"] is True
    assert within["go"] is True


def test_v1_none_does_not_move_and_blocks_go():
    """Catches a None v1 statistic raising, or being counted as a move. A v1
    whose L19 IVF ratio has no qualifying corpus (all key-side scan fractions 0)
    reads None."""
    v1 = _pilot_set(GOOD_L19, GOOD_L43, seed=0)
    for r in v1:
        if r["layer"] == 19:
            r["plain_id_k128"]["ivf_scan_frac"] = 0.0
    out = pilot_verdict({0: v1}, _v0(seeds=(0,)), REFS, TRAIN)
    cell = out["layers"][19]["statistics"]["ivf_ratio_k128"]
    assert cell["seeds"][0]["v1"] is None
    assert cell["moves"] is False
    assert out["go"] is False


def test_none_in_reference_or_v0_raises_naming_statistic_and_layer():
    """Catches broken inputs being judged: a None in a reference draw or in v0
    must raise ValueError naming the statistic and the layer."""
    refs = [list(REFS[0]), [dict(r) for r in REFS[1]]]
    for r in refs[1]:
        if r["layer"] == 43:
            r["plain_ood_k10"] = dict(r["plain_ood_k10"], hnsw_recall_op=None)
    v1 = {0: _pilot_set(GOOD_L19, GOOD_L43, seed=0)}
    with pytest.raises(
        ValueError, match=r"hnsw_recall_k10_ef176.*43|43.*hnsw_recall_k10_ef176"
    ):
        pilot_verdict(v1, _v0(seeds=(0,)), refs, TRAIN)

    v0 = _v0(seeds=(0,))
    for r in v0[0]:
        if r["layer"] == 19:
            r["plain_id_k10"] = dict(r["plain_id_k10"], hnsw_recall_op=None)
    with pytest.raises(
        ValueError, match=r"hnsw_recall_k10_ef176_id.*19|19.*hnsw_recall_k10_ef176_id"
    ):
        pilot_verdict(v1, v0, REFS, TRAIN)


# ---------------------------------------------------------------------------
# full_verdict


BANDS = kv_gate.load_kv_gate(GATE_FILE)["statistics"]
# Pooled reference draws inside every band: spreads are reach 0.125,
# recall 0.05, ratio 0.1, ivf_recall 0.02, key-side 0.
FULL_REFS = [
    _layer(3, reach=0.5, recall=0.80, ratio=0.95, ivf_recall=0.82),
    _layer(3, reach=0.625, recall=0.85, ratio=1.05, ivf_recall=0.84),
]
V0_FULL = dict(
    reach=0.875, recall=0.9274, ratio=0.684, ivf_recall=0.8246, id_recall=0.9757
)


def test_full_key_side_allowance_is_v0_seed_spread():
    """Catches the zero reference spread being used for key-side recall in full_verdict:
    v0 pooled 0.9757 (seed 0) and 0.9700 (seed 1), allowance 0.0057; v1 0.980 does not move
    (gap closed by 0.0043), v1 0.985 does."""
    v0 = {
        0: _layer(3, seed=0, **V0_FULL),
        1: _layer(3, seed=1, **dict(V0_FULL, id_recall=0.9700)),
    }

    def key_side(v1_id):
        v1 = _layer(3, **dict(V0_FULL, id_recall=v1_id))
        return full_verdict(v1, v0, FULL_REFS, BANDS)["statistics"][KEY_SIDE]

    slow = key_side(0.980)
    assert slow["spread"] == pytest.approx(0.0057)
    assert slow["target_min"] == 0.99 and slow["target_max"] is None
    assert slow["moves"] is False
    assert key_side(0.985)["moves"] is True


def test_full_uses_band_targets_and_pooled_reference_spread():
    """Catches full_verdict using the per-layer reference range or a v0 seed
    other than 0. Pooled recall: band 0.78-0.88, spread 0.05, v0 seed 0 0.9274
    (distance 0.0474): v1 0.89 closes 0.0374 (no move), v1 0.875 is inside."""
    v0 = {
        0: _layer(3, seed=0, **V0_FULL),
        1: _layer(3, seed=1, **dict(V0_FULL, recall=0.99)),
    }

    def recall(v):
        out = full_verdict(_layer(3, **dict(V0_FULL, recall=v)), v0, FULL_REFS, BANDS)
        return out["statistics"]["hnsw_recall_k10_ef176"]

    cell = recall(0.89)
    assert (cell["target_min"], cell["target_max"]) == (0.78, 0.88)
    assert cell["spread"] == pytest.approx(0.05)
    assert cell["v0"] == pytest.approx(0.9274)
    assert cell["moves"] is False
    assert cell["band_pass"] is False
    inside = recall(0.875)
    assert inside["moves"] is True and inside["band_pass"] is True


def test_full_ivf_recall_must_stay_inside_the_band():
    """Catches ivf_recall_k128_np24 being judged by the toward rule: v0 at 0.95
    (distance 0.085), v1 at 0.87 closes 0.08 > spread 0.02, but is outside the
    0.80-0.865 band, so it does not meet its requirement."""
    v0 = {0: _layer(3, seed=0, **dict(V0_FULL, ivf_recall=0.95))}
    out = full_verdict(
        _layer(3, **dict(V0_FULL, ivf_recall=0.87)), v0, FULL_REFS, BANDS
    )
    cell = out["statistics"]["ivf_recall_k128_np24"]
    assert cell["requirement"] == "inside"
    assert cell["moves"] is True
    assert cell["meets"] is False
    assert out["all_meet"] is False


def test_full_meets_all_when_every_statistic_lands_in_its_band():
    """Catches a full verdict that can never succeed."""
    v0 = {0: _layer(3, seed=0, **V0_FULL)}
    v1 = _layer(3, reach=0.625, recall=0.83, ratio=1.0, ivf_recall=0.83, id_recall=1.0)
    out = full_verdict(v1, v0, FULL_REFS, BANDS)
    assert out["all_bands_pass"] is True
    assert out["all_meet"] is True


def test_full_needs_v0_seed_0():
    """Catches v1 being compared with whichever v0 seed comes first."""
    v0 = {1: _layer(3, seed=1, **V0_FULL)}
    with pytest.raises(ValueError, match="seed 0"):
        full_verdict(_layer(3, **V0_FULL), v0, FULL_REFS, BANDS)


# ---------------------------------------------------------------------------
# committed v0 records against themselves, and the CLI


def test_v0_records_judged_against_themselves_do_not_move():
    """Catches a judge that always says go: v1 := v0 records -> go False, and in
    the full verdict only ivf_recall_k128_np24 (which v0 passes) meets its
    requirement."""
    refs = [_read(p) for p in REFERENCE_PATHS]
    v0 = _read(V0_BEST)
    pilot = pilot_verdict({0: v0}, {0: v0}, refs, _read(TRAIN_BOOKS_PATH))
    assert pilot["go"] is False
    for layer in (19, 43):
        for name in PILOT_STATS:
            assert pilot["layers"][layer]["statistics"][name]["moves"] is False
        assert pilot["layers"][layer]["key_side"]["not_lower"] is True
    assert pilot["layers"][19]["statistics"]["hnsw_recall_k10_ef176"]["seeds"][0][
        "v1"
    ] == pytest.approx(0.9722, abs=1e-4)

    full = full_verdict(v0, {0: v0}, refs, BANDS)
    meets = {n for n, c in full["statistics"].items() if c["meets"]}
    assert meets == {"ivf_recall_k128_np24"}
    failing = {n for n, c in full["statistics"].items() if not c["band_pass"]}
    assert failing == {
        "hnsw_reach_k10",
        "hnsw_recall_k10_ef176",
        "ivf_ratio_k128",
        "hnsw_recall_k10_ef176_id",
    }
    assert full["statistics"]["hnsw_recall_k10_ef176"]["spread"] == pytest.approx(
        0.0544, abs=1e-4
    )
    assert full["all_meet"] is False


def test_cli_writes_sorted_json_with_inputs(tmp_path, capsys):
    """Catches the CLI dropping the input paths, the default reference/train-books/
    gate paths not resolving from the repo root, or unsorted output keys."""
    out = tmp_path / "pilot.json"
    assert (
        main(
            ["pilot", "--v1", f"0={V0_BEST}", "--v0", f"0={V0_BEST}", "--out", str(out)]
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["go"] is False
    assert data["inputs"]["v1"] == {"0": str(V0_BEST)}
    assert data["inputs"]["references"] == [str(p) for p in REFERENCE_PATHS]
    assert data["inputs"]["train_books"] == str(TRAIN_BOOKS_PATH)
    assert text == json.dumps(data, sort_keys=True, indent=2) + "\n"

    out_full = tmp_path / "full.json"
    assert (
        main(
            [
                "full",
                "--v1",
                str(V0_BEST),
                "--v0",
                f"0={V0_BEST}",
                "--out",
                str(out_full),
            ]
        )
        == 0
    )
    data = json.loads(out_full.read_text(encoding="utf-8"))
    assert data["inputs"]["gate_file"] == str(GATE_FILE)
    assert data["all_meet"] is False
