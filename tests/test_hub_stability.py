import json

import numpy as np
import pytest

from src.eval import hub_stability


def test_draws_are_disjoint_when_the_pool_can_afford_it():
    draws, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)

    assert disjoint is True
    assert len(draws) == 10
    assert all(d.shape == (100,) for d in draws)
    combined = np.concatenate(draws)
    assert combined.size == np.unique(combined).size


def test_draws_overlap_and_say_so_when_the_pool_cannot():
    draws, disjoint = hub_stability.allocate_draws(1000, 400, 10, seed=42)

    assert disjoint is False
    assert len(draws) == 10
    # Each draw is still internally without replacement.
    assert all(np.unique(d).size == 400 for d in draws)


def test_the_exact_boundary_where_the_pool_is_used_up_is_still_disjoint():
    _, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)
    assert disjoint is True
    _, one_more = hub_stability.allocate_draws(999, 100, 10, seed=42)
    assert one_more is False


def test_draw_indices_are_sorted_so_the_subsample_preserves_corpus_order():
    draws, _ = hub_stability.allocate_draws(1000, 100, 3, seed=7)
    for d in draws:
        np.testing.assert_array_equal(d, np.sort(d))


def test_allocation_is_reproducible_under_the_same_seed():
    first, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    second, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_a_draw_larger_than_the_pool_is_an_error():
    with pytest.raises(hub_stability.HubStabilityError, match="pool"):
        hub_stability.allocate_draws(50, 100, 2, seed=42)


def _draw(rows: int = 300, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, dim)).astype(np.float32)


def test_measure_draw_returns_every_statistic_as_a_finite_number():
    values = hub_stability.measure_draw(
        _draw(), k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024
    )

    assert sorted(values) == sorted(hub_stability.STATISTICS)
    assert all(np.isfinite(v) for v in values.values())


def test_measure_draw_measures_every_row_it_is_given():
    # max_rows must be disabled inside: the caller has already drawn the
    # rows, and a second subsample would silently shrink the draw.
    big = hub_stability.measure_draw(
        _draw(rows=400),
        k=10,
        k_hub=5,
        nlist=4,
        seed=42,
        backend="sklearn",
        chunk_rows=1024,
    )
    small = hub_stability.measure_draw(
        _draw(rows=400)[:200],
        k=10,
        k_hub=5,
        nlist=4,
        seed=42,
        backend="sklearn",
        chunk_rows=1024,
    )
    assert big["lid_median"] != small["lid_median"]


def test_measure_draw_is_deterministic():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024)
    first = hub_stability.measure_draw(_draw(), **kwargs)
    second = hub_stability.measure_draw(_draw(), **kwargs)
    assert first == second


def test_measure_draw_agrees_between_backends():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, chunk_rows=1024)
    sk = hub_stability.measure_draw(_draw(), backend="sklearn", **kwargs)
    torch_ = hub_stability.measure_draw(_draw(), backend="torch", **kwargs)
    for name in hub_stability.STATISTICS:
        assert sk[name] == pytest.approx(torch_[name], rel=1e-4, abs=1e-6), name


def _spread(mean: float, low: float, high: float) -> dict:
    return {
        "mean": mean,
        "std": 0.0,
        "min": low,
        "max": high,
        "range_pct_of_mean": (high - low) / mean * 100.0,
        "cv_pct": 0.0,
    }


def test_a_stable_and_discriminating_statistic_qualifies():
    # 5% range, and the synthetic mean sits two ranges away.
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=1.1, draws_disjoint=True
    )
    assert verdict["stable"] is True
    assert verdict["discriminating"] is True
    assert verdict["verdict"] == "qualified"


def test_hubness_skews_measured_instability_is_rejected():
    # The real numbers from docs/datasets/glove_noise_floor.json.
    verdict = hub_stability.evaluate_rule(
        _spread(4.4976, 3.4630, 8.3308), synthetic_mean=1.695891, draws_disjoint=True
    )
    assert verdict["stable"] is False
    assert verdict["verdict"] == "rejected"


def test_a_stable_statistic_that_cannot_separate_is_rejected():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.95, 1.05), synthetic_mean=1.02, draws_disjoint=True
    )
    assert verdict["stable"] is True
    assert verdict["discriminating"] is False
    assert verdict["verdict"] == "rejected"


def test_overlapping_draws_downgrade_a_pass_to_provisional():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=1.1, draws_disjoint=False
    )
    assert verdict["verdict"] == "provisional"


def test_exactly_ten_percent_range_is_stable_because_the_bound_is_inclusive():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.95, 1.05), synthetic_mean=2.0, draws_disjoint=True
    )
    assert verdict["range_pct_of_mean"] == pytest.approx(10.0)
    assert verdict["stable"] is True
    assert verdict["verdict"] == "qualified"


def test_exactly_one_range_of_separation_discriminates():
    spread = _spread(1.0, 0.975, 1.025)
    verdict = hub_stability.evaluate_rule(
        spread, synthetic_mean=1.05, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] == pytest.approx(1.0)
    assert verdict["discriminating"] is True
    assert verdict["verdict"] == "qualified"


def test_a_corpus_with_no_synthetic_series_gets_a_stability_only_verdict():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=None, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] is None
    assert verdict["discriminating"] is None
    assert verdict["verdict"] == "stable"


def test_a_zero_range_cannot_be_divided_into_and_does_not_discriminate():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 1.0, 1.0), synthetic_mean=5.0, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] is None
    assert verdict["verdict"] == "rejected"


def _sweep_kwargs(**overrides):
    kwargs = dict(
        ns=[60],
        draws=3,
        k=8,
        k_hub=4,
        nlist=4,
        seed=42,
        backend="sklearn",
        chunk_rows=1024,
    )
    kwargs.update(overrides)
    return kwargs


def test_sweep_reports_one_cell_per_n_with_every_raw_draw():
    real = _draw(rows=400, seed=1)

    result = hub_stability.sweep(real, {}, **_sweep_kwargs(ns=[60, 100]))

    assert [c["n"] for c in result["cells"]] == [60, 100]
    for cell in result["cells"]:
        assert len(cell["real"]["per_draw"]) == 3
        assert sorted(cell["real"]["spread"]) == sorted(hub_stability.STATISTICS)


def test_sweep_records_the_pool_and_whether_the_draws_were_disjoint():
    real = _draw(rows=400, seed=1)

    result = hub_stability.sweep(real, {}, **_sweep_kwargs(ns=[60, 200]))

    assert result["pool_rows"] == 400
    disjoint = {c["n"]: c["draws_disjoint"] for c in result["cells"]}
    assert disjoint[60] is True  # 3 x 60 = 180 <= 400
    assert disjoint[200] is False  # 3 x 200 = 600 > 400
    assert result["cells"][0]["pool_to_n"] == pytest.approx(400 / 60)


def test_sweep_without_synthetic_series_gives_stability_only_verdicts():
    result = hub_stability.sweep(_draw(rows=400, seed=1), {}, **_sweep_kwargs())

    verdicts = result["cells"][0]["verdicts"]
    assert sorted(verdicts) == sorted(hub_stability.STATISTICS)
    assert all(v["verdict"] in {"stable", "unstable"} for v in verdicts.values())


def test_sweep_with_synthetic_series_measures_each_once_and_judges_the_mean():
    real = _draw(rows=400, seed=1)
    synthetic = {
        "v0_seed42": _draw(rows=400, seed=2),
        "v0_seed43": _draw(rows=400, seed=3),
    }

    result = hub_stability.sweep(real, synthetic, **_sweep_kwargs())

    cell = result["cells"][0]
    assert sorted(cell["synthetic"]["per_series"]) == ["v0_seed42", "v0_seed43"]
    assert sorted(cell["synthetic"]["mean"]) == sorted(hub_stability.STATISTICS)
    assert all(
        v["verdict"] in {"qualified", "provisional", "rejected"}
        for v in cell["verdicts"].values()
    )


def test_sweep_refuses_an_n_larger_than_the_corpus():
    with pytest.raises(hub_stability.HubStabilityError, match="pool"):
        hub_stability.sweep(_draw(rows=100, seed=1), {}, **_sweep_kwargs(ns=[500]))


def test_sweep_is_reproducible():
    real = _draw(rows=400, seed=1)
    first = hub_stability.sweep(real, {}, **_sweep_kwargs())
    second = hub_stability.sweep(real, {}, **_sweep_kwargs())
    assert first == second


def test_cli_writes_a_json_that_holds_its_own_evidence(tmp_path, monkeypatch, capsys):
    real_path = tmp_path / "real.npy"
    np.save(real_path, _draw(rows=400, seed=1))
    output = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "hub_stability",
            "--real-path",
            str(real_path),
            "--n",
            "60",
            "--draws",
            "3",
            "--k",
            "8",
            "--k-hub",
            "4",
            "--nlist",
            "4",
            "--output",
            str(output),
        ],
    )
    hub_stability.main()

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["real_path"] == str(real_path)
    assert written["rule"]["stable_max_range_pct"] == 10.0
    assert len(written["cells"][0]["real"]["per_draw"]) == 3
    assert json.loads(capsys.readouterr().out) == written


def test_cli_rejects_a_malformed_synthetic_path(tmp_path, monkeypatch):
    real_path = tmp_path / "real.npy"
    np.save(real_path, _draw(rows=200, seed=1))

    monkeypatch.setattr(
        "sys.argv",
        [
            "hub_stability",
            "--real-path",
            str(real_path),
            "--synthetic-path",
            "no-equals-sign",
            "--n",
            "50",
            "--draws",
            "2",
            "--k",
            "8",
            "--k-hub",
            "4",
            "--nlist",
            "4",
        ],
    )
    with pytest.raises(SystemExit):
        hub_stability.main()
