import math

import numpy as np
import pytest

from src.train.selection import SELECTORS, gate_statistics, prefixed, selection_score


def test_selectors_are_the_two_documented_choices():
    assert SELECTORS == ("cov_fro", "gate")


def test_score_is_the_sum_of_normalised_lid_and_contrast_gaps():
    real = {"lid_median": 50.0, "relative_contrast_median": 2.0}
    fake = {"lid_median": 40.0, "relative_contrast_median": 2.5}
    # |40-50|/50 + |2.5-2.0|/2.0 = 0.2 + 0.25
    assert selection_score(fake, real) == pytest.approx(0.45)


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
    }
    assert stats["lid_median"] > 0 and stats["relative_contrast_median"] > 1.0
    assert stats["lid_discarded_queries"] == 0


def test_gate_statistics_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    assert gate_statistics(x, metric="l2", seed=3) == gate_statistics(
        x, metric="l2", seed=3
    )


def test_prefixed_renames_every_key():
    assert prefixed({"lid_median": 1.0, "ivf_gini": 0.5}, "gate_fake_") == {
        "gate_fake_lid_median": 1.0,
        "gate_fake_ivf_gini": 0.5,
    }
