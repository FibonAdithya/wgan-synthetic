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


def test_v2c_is_v2_plus_the_set_critic():
    v2 = _flatten(_load("v2.yaml"))
    v2c = _flatten(_load("v2c.yaml"))
    assert v2c.pop("model.critic_type") == "neighbourhood_set"
    assert v2c.pop("model.critic_edge_dim") == 128
    assert v2c.pop("model.critic_edge_pool") == "max"
    assert v2c.pop("output_dir") == "runs/nytimes/v2c"
    v2.pop("model.critic_type")
    v2.pop("output_dir")
    assert v2c == v2


def test_v2c_seed42_is_v2c_with_an_absolute_real_path_and_its_own_output_dir():
    v2c = _flatten(_load("v2c.yaml"))
    inst = _flatten(_load("v2c_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2c_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2c.pop("output_dir")
    v2c.pop("data.real_path")
    assert inst == v2c


def test_v2c_job_script_runs_the_v2c_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2c_seed42_job.sh").read_text()
    assert "configs/nytimes/v2c_seed42.yaml" in script
    assert "runs/nytimes/v2c_seed42" in script


def test_v2c_requires_amp_off():
    """The set critic requires `training.amp: false`; the equality-with-v2
    test above only pins it transitively."""
    assert _flatten(_load("v2c.yaml"))["training.amp"] is False
    assert _flatten(_load("v2c_seed42.yaml"))["training.amp"] is False


def test_v2c_seed42_100k_is_v2c_seed42_with_the_budget_raised():
    """The continuation is the same instrument with only the budget and the
    output directory moved; any other key drifting (amp, the critic, the
    selector) would make the resume measure something else."""
    inst = _flatten(_load("v2c_seed42.yaml"))
    cont = _flatten(_load("v2c_seed42_100k.yaml"))
    assert cont.pop("output_dir") == "runs/nytimes/v2c_seed42_100k"
    assert cont.pop("training.num_gen_steps") == 100000
    inst.pop("output_dir")
    assert inst.pop("training.num_gen_steps") == 30000
    assert cont == inst


def test_v2c_100k_job_script_resumes_the_30k_run_into_the_100k_config():
    script = (
        ROOT.parent.parent / "scripts" / "nytimes_v2c_seed42_100k_job.sh"
    ).read_text()
    assert "configs/nytimes/v2c_seed42_100k.yaml" in script
    assert "runs/nytimes/v2c_seed42_100k" in script
    assert "--resume" in script
    assert "/workspace/nytimes-v2/v2c_seed42/checkpoint_step_30000.pt" in script
    assert "checkpoint_step_100000.pt" in script
    # The restored best_score may never be beaten; the script must then carry
    # the 30k run's selection over instead of failing on a missing file.
    assert "/workspace/nytimes-v2/v2c_seed42/best_generator.pt" in script


V3_GENERATOR_KEYS = {
    "model.generator_type": "spherical",
    "model.tangent_hidden_dim": 512,
    "model.radius_init": 0.95,
    "model.radius_min": 0.2,
    "model.radius_max": 1.5,
}


def test_v3_is_v2c_with_the_generator_swapped():
    v2c = _flatten(_load("v2c.yaml"))
    v3 = _flatten(_load("v3.yaml"))
    for key, value in V3_GENERATOR_KEYS.items():
        assert v3.pop(key) == value, key
    assert v3.pop("output_dir") == "runs/nytimes/v3"
    assert v3["model.skip_dim"] == 256
    for key in (
        "model.generator_type",
        "model.skip_init",
        "model.skip_init_gain",
        "output_dir",
    ):
        v2c.pop(key)
    assert v3 == v2c


def test_v3_seed42_is_v3_with_an_absolute_real_path_and_its_own_output_dir():
    v3 = _flatten(_load("v3.yaml"))
    inst = _flatten(_load("v3_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v3_seed42"
    assert inst.pop("data.real_path") == "/workspace/data-cache/nytimes_250k.npy"
    v3.pop("output_dir")
    v3.pop("data.real_path")
    assert inst == v3


def test_v3_job_script_runs_the_v3_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v3_seed42_job.sh").read_text()
    assert "configs/nytimes/v3_seed42.yaml" in script
    assert "runs/nytimes/v3_seed42" in script
    assert "/workspace/nytimes-v3/v3_seed42" in script
    assert "REAL=/workspace/data-cache/nytimes_250k.npy" in script


def test_v3_requires_amp_off():
    assert _flatten(_load("v3.yaml"))["training.amp"] is False
    assert _flatten(_load("v3_seed42.yaml"))["training.amp"] is False


def test_v4_is_v3_plus_the_regulariser():
    """Catches any v4 key drifting from v3 beyond the three lid_reg keys: the
    spec says v4 against v3 is one change, the regulariser."""
    v3 = _flatten(_load("v3.yaml"))
    v4 = _flatten(_load("v4.yaml"))
    assert v4.pop("training.lid_reg_k") == 20
    assert v4.pop("training.lid_reg_max_points") == 256
    alpha = v4.pop("training.lid_reg_alpha")
    assert isinstance(alpha, float) and alpha > 0.0, alpha
    assert v4.pop("output_dir") == "runs/nytimes/v4"
    v3.pop("output_dir")
    assert v4 == v3


def test_v4_keeps_distance_reg_off():
    """distance_reg penalises a global scalar that already matches -- median
    pairwise distance is 1.4038 real against 1.4054 for v3's selection -- while
    the radius probe put the deficit at 1-NN. Turning it on would make v4 a
    two-change rung."""
    assert _flatten(_load("v4.yaml"))["training.distance_reg_alpha"] == 0.0


def test_v4_records_where_its_alpha_came_from():
    """An alpha nobody can trace is an invented number. The comment beside it
    must name the probe that produced it and the committed result.

    This cannot catch a pasted value that kept the comment; what it catches is
    the provenance being dropped, which is how an invented number gets in."""
    text = (ROOT / "v4.yaml").read_text(encoding="utf-8")
    assert "lid_reg_scale_probe" in text
    assert "docs/results/nytimes-v4-lid-reg-scale" in text


def test_v4_seed42_is_v4_with_an_absolute_real_path_and_its_own_output_dir():
    v4 = _flatten(_load("v4.yaml"))
    inst = _flatten(_load("v4_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v4_seed42"
    assert inst.pop("data.real_path") == "/workspace/data-cache/nytimes_250k.npy"
    v4.pop("output_dir")
    v4.pop("data.real_path")
    assert inst == v4


def test_v4_job_script_runs_the_v4_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v4_seed42_job.sh").read_text()
    assert "configs/nytimes/v4_seed42.yaml" in script
    assert "runs/nytimes/v4_seed42" in script


def test_v4_job_script_documents_a_submit_without_vram():
    """--vram-mb lets the scheduler admit a second job onto the card, which
    the per-card lock then blocks until it dies with GpuBusyError.

    Limitation, stated so nobody reads more into this than it says: the flag
    is passed at `gpuq submit` time and never appears in the script body, so
    this cannot police the actual invocation. What it does police is the
    submit command the script documents in its header, which is the line a
    human copies.

    It reads that command specifically rather than searching the whole file,
    because the script also warns about the flag by name -- a blanket "string
    absent" check cannot tell a warning from a recommendation, and forbidding
    the string would mean forbidding the warning.
    """
    script = (ROOT.parent.parent / "scripts" / "nytimes_v4_seed42_job.sh").read_text()
    lines = script.splitlines()
    starts = [i for i, ln in enumerate(lines) if "gpuq submit" in ln]
    assert starts, "the script must document how to submit it"
    documented = []
    for start in starts:
        i = start
        while True:
            documented.append(lines[i])
            if not lines[i].rstrip().endswith("\\"):
                break
            i += 1
    command = "\n".join(documented)
    assert "--vram-mb" not in command, command
    assert "--lane gpu" in command


def test_v4_requires_amp_off():
    """The set critic requires amp off, and lid_reg's pairwise numerics are
    run outside autocast for the same reason; v4 inherits both from v3."""
    assert _flatten(_load("v4.yaml"))["training.amp"] is False
    assert _flatten(_load("v4_seed42.yaml"))["training.amp"] is False
