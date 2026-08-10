"""Tests for the seed-to-seed noise floor.

The numbers here are hand-computed, not measured from anything: these tests
are about the arithmetic, not about GloVe.
"""

import pytest

from src.eval import noise_floor


def test_summarize_spread_reports_hand_computed_values():
    # mean 2.0; sample std (ddof=1) 1.0; range 2.0 -> 100% of mean; cv 50%.
    result = noise_floor.summarize_spread([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(1.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(3.0)
    assert result["range_pct_of_mean"] == pytest.approx(100.0)
    assert result["cv_pct"] == pytest.approx(50.0)


def test_summarize_spread_uses_sample_not_population_std():
    """Pins ddof=1, the convention docs/datasets/glove_noise_floor.json used.

    Population std of these values is 0.5; sample std is 0.5773502692. A file
    written under one convention and read under the other would silently
    understate the floor.
    """
    result = noise_floor.summarize_spread([1.0, 2.0])
    assert result["std"] == pytest.approx(0.7071067811865476)


def test_summarize_spread_rejects_a_single_value():
    """One draw has no spread; reporting 0.0 would read as 'perfectly stable'."""
    with pytest.raises(noise_floor.NoiseFloorError, match="at least two"):
        noise_floor.summarize_spread([1.0])
