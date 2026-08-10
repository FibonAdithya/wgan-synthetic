"""The openai structural measurements.

Every test here runs on small synthetic data with a known answer, because
the point of these functions is to make a claim about a corpus and a
function that measures the wrong thing still returns a plausible float.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from src.eval import openai_structure as st


def _unit(rows: int, dim: int, seed: int = 0, cone: float = 0.0) -> np.ndarray:
    """`rows` unit vectors. `cone` shifts them all toward one direction."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, dim))
    if cone:
        x += cone * np.ones(dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_sample_rows_is_not_a_prefix():
    """openai_250k.npy ascends by original corpus index, so a prefix is the

    first slice of DBpedia in its own order rather than a sample of it.
    """
    x = np.arange(1000, dtype=np.float32).reshape(1000, 1)

    drawn = st.sample_rows(x, 50, seed=42).ravel()

    assert drawn.size == 50
    assert drawn.max() > 500, "sample never reached the second half"
    assert not np.array_equal(drawn, np.arange(50, dtype=np.float32))


def test_sample_rows_is_deterministic_and_returns_all_when_asked_for_more():
    x = np.arange(100, dtype=np.float32).reshape(100, 1)

    a = st.sample_rows(x, 10, seed=7)
    b = st.sample_rows(x, 10, seed=7)

    np.testing.assert_array_equal(a, b)
    assert st.sample_rows(x, 1000, seed=7).shape[0] == 100


def test_disjoint_draws_do_not_share_rows():
    """Draws that shared rows would share their hubs and duplicates, which

    is exactly the correlation the noise floor is trying to measure across.
    """
    x = np.arange(300, dtype=np.float32).reshape(300, 1)

    draws = st.disjoint_draws(x, count=3, rows=100, seed=0)

    assert [d.shape[0] for d in draws] == [100, 100, 100]
    seen = np.concatenate([d.ravel() for d in draws])
    assert len(np.unique(seen)) == 300


def test_disjoint_draws_refuses_a_corpus_too_small_to_supply_them():
    x = np.zeros((100, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="noise floor needs"):
        st.disjoint_draws(x, count=10, rows=20, seed=0)


# --------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------


def test_centering_changes_the_spectrum_on_anisotropic_data():
    """Regression test for a real bug: sklearn's PCA always subtracts the

    mean, so building this comparison on PCA returned identical numbers
    twice and the centering recommendation rested on a no-op.
    """
    x = _unit(2000, 16, seed=1, cone=1.5)

    raw, raw_ratio = st.spectrum_facts(x, "raw", center=False)
    centered, centered_ratio = st.spectrum_facts(x, "centered", center=True)

    assert raw["raw_top_component_share"] > centered["centered_top_component_share"], (
        "uncentered spectrum must be dominated by the shared mean direction"
    )
    assert not np.allclose(raw_ratio, centered_ratio)


def test_centering_barely_moves_an_already_centred_corpus():
    """The converse: with no mean direction there is nothing for centering to

    remove, so the two spectra should nearly coincide.
    """
    x = _unit(4000, 16, seed=2)

    raw, _ = st.spectrum_facts(x, "raw", center=False)
    centered, _ = st.spectrum_facts(x, "centered", center=True)

    assert raw["raw_top_component_share"] == pytest.approx(
        centered["centered_top_component_share"], abs=0.02
    )


def test_participation_ratio_recovers_the_dimension_of_isotropic_data():
    """Flat spectrum over d directions -> participation ratio near d."""
    x = _unit(8000, 16, seed=3)

    facts, _ = st.spectrum_facts(x, "raw", center=True)

    assert facts["raw_participation_ratio"] == pytest.approx(16, rel=0.15)


def test_participation_ratio_collapses_when_one_direction_dominates():
    rng = np.random.default_rng(4)
    x = np.zeros((2000, 16), dtype=np.float32)
    x[:, 0] = rng.normal(size=2000)
    x[:, 1:] = rng.normal(size=(2000, 15)) * 0.001

    facts, _ = st.spectrum_facts(x, "raw", center=True)

    assert facts["raw_participation_ratio"] < 1.5


# --------------------------------------------------------------------------
# norms and anisotropy
# --------------------------------------------------------------------------


def test_norm_facts_confirm_unit_norm_data():
    facts = st.norm_facts(_unit(500, 32, seed=5))

    assert facts["norm_mean"] == pytest.approx(1.0, abs=1e-5)
    assert facts["norm_max_abs_deviation_from_1"] < 1e-5


def test_anisotropy_separates_a_cone_from_a_sphere():
    """The mean vector's norm is ~0 on an isotropic sphere and approaches 1

    as every vector points the same way. This is the mlp-vs-spherical
    evidence, so it has to actually discriminate.
    """
    sphere, _ = st.anisotropy_facts(_unit(4000, 32, seed=6))
    cone, _ = st.anisotropy_facts(_unit(4000, 32, seed=6, cone=2.0))

    assert sphere["mean_vector_norm"] < 0.1
    assert cone["mean_vector_norm"] > 0.7
    assert cone["cos_to_mean_median"] > sphere["cos_to_mean_median"]


# --------------------------------------------------------------------------
# angular vs L2
# --------------------------------------------------------------------------


def test_l2_and_cosine_agree_on_neighbours_for_unit_norm_data():
    """||a-b||^2 = 2 - 2cos(a,b) is strictly monotone on the unit sphere, so

    the two metrics must induce identical neighbour sets. This is what makes
    hubness survive phase (c) unchanged.
    """
    x = _unit(600, 32, seed=7)

    facts = st.angular_vs_l2(x, seed=42, k=20, k_hub=5)

    assert facts["neighbour_set_agreement"] == pytest.approx(1.0)
    assert facts["hubness_skew_l2"] == pytest.approx(facts["hubness_skew_cosine"])


def test_cosine_lid_is_half_the_l2_lid_for_unit_norm_data():
    """Cosine distance is L2 squared over two, and the MLE is a ratio of

    logs of distances, so squaring doubles every log-ratio and halves the
    estimate. A constant factor means the recorded profile survives phase
    (c) as a rescaling rather than needing re-measurement.
    """
    x = _unit(600, 32, seed=8)

    facts = st.angular_vs_l2(x, seed=42, k=20, k_hub=5)

    assert facts["lid_median_cosine"] == pytest.approx(
        facts["lid_median_l2"] / 2.0, rel=0.02
    )


def test_angular_vs_l2_records_the_k_each_statistic_used():
    """LID is a k=100 statistic and hubness a k=10 one. Reporting both under

    one k produced numbers that were not the gate's numbers.
    """
    facts = st.angular_vs_l2(_unit(400, 16, seed=9), seed=42, k=15, k_hub=4)

    assert facts["angular_k"] == 15
    assert facts["angular_k_hub"] == 4


# --------------------------------------------------------------------------
# gate statistics and the noise floor
# --------------------------------------------------------------------------


def test_gate_statistics_reject_a_degenerate_draw():
    """summary() returns None for the medians when every query is discarded.

    numpy would turn that into a silent nan and the noise floor would report
    a nan spread as though it had measured something.
    """
    duplicates = np.ones((300, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="came back None"):
        st.gate_statistics(duplicates, seed=0)


def test_noise_floor_reports_a_spread_per_statistic():
    draws = [_unit(400, 16, seed=s) for s in (10, 11, 12)]

    floor = st.noise_floor(draws, seed=0)

    assert set(floor) == set(st.GATE_STATS)
    for stat, f in floor.items():
        assert f["min"] <= f["median"] <= f["max"], stat
        assert f["spread"] == pytest.approx(f["max"] - f["min"]), stat


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def test_histogram_draws_a_constant_series_instead_of_raising():
    """A corpus that really is exactly unit-norm makes np.histogram raise

    "Too many bins for data range". That is the family page's claim coming
    out true, so it must render, not crash.
    """
    fig = st.fig_histogram(np.ones(500), 80, "L2 norm", "norm", "#2b6cb0")

    assert isinstance(fig, go.Figure)
    assert "constant at 1" in fig.layout.title.text


def test_histogram_rejects_non_finite_values():
    values = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="nan or inf"):
        st.fig_histogram(values, 10, "x", "x", "#2b6cb0")
