# tests/test_nytimes_configs.py
"""v2 is v1 plus exactly two stated changes. Pin them, so a drift in any
other key is caught before it burns 40 GPU-minutes."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent / "configs" / "nytimes"


def _load(name):
    with (ROOT / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


V2_DELTA = {
    "model.critic_type": "neighbourhood",
    "model.critic_k": 20,
    "model.critic_distance_floor": 0.01,
    "data.preprocess.drop_zero_rows": True,
}


def test_v2_is_v1_plus_the_four_stated_changes():
    """Catches any v2 key drifting from v1 beyond the four stated changes."""
    v1 = _flatten(_load("v1.yaml"))
    v2 = _flatten(_load("v2.yaml"))
    for key, value in V2_DELTA.items():
        assert v2.pop(key) == value, key
    assert v2.pop("output_dir") == "runs/nytimes/v2"
    v1.pop("output_dir")
    assert v2 == v1


def test_v2_seed42_is_v2_with_an_absolute_real_path_and_its_own_output_dir():
    """Catches the instrument config drifting from v2 beyond path and output dir."""
    v2 = _flatten(_load("v2.yaml"))
    inst = _flatten(_load("v2_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2.pop("output_dir")
    v2.pop("data.real_path")
    assert inst == v2


def test_v2_job_script_runs_the_v2_seed42_config():
    """Catches the job script pointing at another config or run dir."""
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2_seed42_job.sh").read_text()
    assert "configs/nytimes/v2_seed42.yaml" in script
    assert "runs/nytimes/v2_seed42" in script


V2B_DELTA = {
    "model.critic_type": "neighbourhood_bank",
    "model.critic_bank_size": 16384,
}


def test_v2b_is_v2_plus_the_neighbour_source():
    """Catches any v2b key drifting from v2 beyond the bank: the spec says
    v2b against v2 is one change, the neighbour source."""
    v2 = _flatten(_load("v2.yaml"))
    v2b = _flatten(_load("v2b.yaml"))
    for key, value in V2B_DELTA.items():
        assert v2b.pop(key) == value, key
    assert v2b.pop("output_dir") == "runs/nytimes/v2b"
    v2.pop("output_dir")
    assert v2.pop("model.critic_type") == "neighbourhood"
    assert v2b == v2


def test_v2b_seed42_is_v2b_with_an_absolute_real_path_and_its_own_output_dir():
    """Catches the instrument config drifting from v2b beyond path and output dir."""
    v2b = _flatten(_load("v2b.yaml"))
    inst = _flatten(_load("v2b_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2b_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2b.pop("output_dir")
    v2b.pop("data.real_path")
    assert inst == v2b


def test_v2b_job_script_runs_the_v2b_seed42_config():
    """Catches the job script pointing at another config or run dir."""
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2b_seed42_job.sh").read_text()
    assert "configs/nytimes/v2b_seed42.yaml" in script
    assert "runs/nytimes/v2b_seed42" in script
