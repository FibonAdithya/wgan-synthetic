"""`data.preprocess.drop_zero_rows` removes exact-zero rows before the
train/holdout split, so both sides and the gate reference are clean."""

import numpy as np

from src.data.dataset import PreprocessConfig, PreprocessState, build_training_data


def _corpus(tmp_path, n=10, dim=4, zero_rows=(3, 7)):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    for i in zero_rows:
        x[i] = 0.0
    path = tmp_path / "corpus.npy"
    np.save(path, x)
    return path


def _build(path, drop):
    return build_training_data(
        descriptor_path=str(path),
        file_format="npy",
        descriptor_dim=4,
        holdout_fraction=0.2,
        preprocess_cfg=PreprocessConfig(drop_zero_rows=drop),
        seed=0,
    )


def test_flag_defaults_to_false():
    assert PreprocessConfig().drop_zero_rows is False


def test_drop_zero_rows_removes_exactly_the_zero_rows_before_the_split(tmp_path):
    x_train, x_holdout, state = _build(_corpus(tmp_path), drop=True)
    assert x_train.shape[0] + x_holdout.shape[0] == 8
    assert state.dropped_zero_rows == 2
    # Nothing left on either side has zero norm.
    assert (np.linalg.norm(x_train, axis=1) > 0).all()
    assert (np.linalg.norm(x_holdout, axis=1) > 0).all()


def test_flag_off_keeps_every_row_and_reports_zero(tmp_path):
    x_train, x_holdout, state = _build(_corpus(tmp_path), drop=False)
    assert x_train.shape[0] + x_holdout.shape[0] == 10
    assert state.dropped_zero_rows == 0


def test_dropped_count_survives_serialisation_round_trip():
    state = PreprocessState(
        descriptor_dim=4,
        config=PreprocessConfig(drop_zero_rows=True),
        dropped_zero_rows=2,
    )
    payload = state.to_serializable()
    assert payload["dropped_zero_rows"] == 2
    assert payload["config"]["drop_zero_rows"] is True
    restored = PreprocessState.from_serializable(payload)
    assert restored.dropped_zero_rows == 2
    assert restored.config.drop_zero_rows is True


def test_old_payload_without_the_field_loads_as_zero():
    payload = {
        "descriptor_dim": 4,
        "config": {"center": False, "whiten": False, "l2_normalize": True},
        "mean": None,
        "whitening_matrix": None,
    }
    assert PreprocessState.from_serializable(payload).dropped_zero_rows == 0
