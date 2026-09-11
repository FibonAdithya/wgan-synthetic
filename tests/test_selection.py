import math

import numpy as np
import pytest

from src.train.selection import SELECTORS, gate_statistics, prefixed, selection_score


def test_selectors_are_the_two_documented_choices():
    assert SELECTORS == ("cov_fro", "gate")


def test_score_is_the_sum_of_normalised_lid_and_contrast_gaps():
    real = {"lid_median": 50.0, "relative_contrast_median": 2.0}
    fake = {"lid_median": 40.0, "relative_contrast_median": 3.0}
    # |40-50|/50 + |3.0-2.0|/2.0 = 0.2 + 0.5
    assert selection_score(fake, real) == pytest.approx(0.7)
    # Not symmetric in its arguments: swapping which side is "real" must
    # change the answer, or the function is silently just a distance.
    # |50-40|/40 + |2.0-3.0|/3.0 = 0.25 + 1/3
    assert selection_score(real, fake) == pytest.approx(0.25 + 1 / 3)


def test_score_prefers_the_checkpoint_nearer_real_on_both():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27}
    near = {"lid_median": 50.0, "relative_contrast_median": 1.30}
    far = {"lid_median": 14.0, "relative_contrast_median": 2.08}
    assert selection_score(near, real) < selection_score(far, real)


def test_score_ignores_hubness_and_gini():
    real = {
        "lid_median": 56.0,
        "relative_contrast_median": 1.27,
        "hubness_skew": 2.5,
        "ivf_gini": 0.8,
    }
    a = {
        "lid_median": 50.0,
        "relative_contrast_median": 1.30,
        "hubness_skew": 9.0,
        "ivf_gini": 0.1,
    }
    b = {
        "lid_median": 50.0,
        "relative_contrast_median": 1.30,
        "hubness_skew": 2.5,
        "ivf_gini": 0.8,
    }
    assert selection_score(a, real) == selection_score(b, real)


@pytest.mark.parametrize(
    "fake, real",
    [
        (
            {"lid_median": None, "relative_contrast_median": 1.3},
            {"lid_median": 56.0, "relative_contrast_median": 1.27},
        ),
        (
            {"lid_median": 50.0, "relative_contrast_median": 1.3},
            {"lid_median": 0.0, "relative_contrast_median": 1.27},
        ),
        ({"lid_median": 50.0}, {"lid_median": 56.0, "relative_contrast_median": 1.27}),
    ],
)
def test_score_is_infinite_when_a_statistic_is_unmeasurable(fake, real):
    assert selection_score(fake, real) == math.inf


def test_gate_statistics_returns_the_four_statistics_and_the_discard_count():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    stats = gate_statistics(x, metric="angular", seed=0)
    assert set(stats) == {
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
        "lid_discarded_queries",
        "zero_rows",
        "measured_rows",
    }
    assert stats["lid_median"] > 0 and stats["relative_contrast_median"] > 1.0
    assert stats["lid_discarded_queries"] == 0
    assert stats["zero_rows"] == 0
    assert stats["measured_rows"] == 300


def test_gate_statistics_drops_exact_zero_rows_and_reports_the_counts():
    rng = np.random.default_rng(2)
    n, dim, k = 200, 16, 7
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x_with_zeros = np.concatenate([x, np.zeros((k, dim), dtype=np.float32)], axis=0)

    with_zeros = gate_statistics(x_with_zeros, metric="angular", seed=5)
    without_zeros = gate_statistics(x, metric="angular", seed=5)

    assert with_zeros["zero_rows"] == k
    assert with_zeros["measured_rows"] == n
    assert without_zeros["zero_rows"] == 0
    assert without_zeros["measured_rows"] == n
    for key in (
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
        "lid_discarded_queries",
    ):
        assert with_zeros[key] == without_zeros[key]


def test_gate_statistics_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    assert gate_statistics(x, metric="l2", seed=3) == gate_statistics(
        x, metric="l2", seed=3
    )


def test_score_is_infinite_when_fake_discarded_more_than_half_its_queries():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27}
    fake = {
        "lid_median": 50.0,
        "relative_contrast_median": 1.30,
        "lid_discarded_queries": 60,
        "measured_rows": 100,
    }
    assert selection_score(fake, real) == math.inf


def test_score_is_finite_when_fake_discarded_less_than_half_its_queries():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27}
    fake = {
        "lid_median": 50.0,
        "relative_contrast_median": 1.30,
        "lid_discarded_queries": 40,
        "measured_rows": 100,
    }
    plain = {"lid_median": 50.0, "relative_contrast_median": 1.30}
    assert selection_score(fake, real) == pytest.approx(selection_score(plain, real))


def test_score_is_infinite_when_fake_has_zero_measured_rows():
    real = {"lid_median": 56.0, "relative_contrast_median": 1.27}
    fake = {
        "lid_median": 50.0,
        "relative_contrast_median": 1.30,
        "lid_discarded_queries": 0,
        "measured_rows": 0,
    }
    assert selection_score(fake, real) == math.inf


def test_prefixed_renames_every_key():
    assert prefixed({"lid_median": 1.0, "ivf_gini": 0.5}, "gate_fake_") == {
        "gate_fake_lid_median": 1.0,
        "gate_fake_ivf_gini": 0.5,
    }
