import numpy as np
import pytest

from src.data.dataset import PreprocessConfig, PreprocessState


def test_metric_defaults_to_l2():
    assert PreprocessConfig().metric == "l2"


def test_metric_accepts_angular():
    assert PreprocessConfig(metric="angular").metric == "angular"


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        PreprocessConfig(metric="cosine")


def test_metric_survives_serialization_round_trip():
    state = PreprocessState(
        descriptor_dim=8, config=PreprocessConfig(metric="angular")
    )
    payload = state.to_serializable()
    assert payload["config"]["metric"] == "angular"
    assert PreprocessState.from_serializable(payload).config.metric == "angular"
