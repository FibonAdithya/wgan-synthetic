"""Tests for the sampling CLI that produces the project's primary deliverable.

`src.sample.generate` has no importable `run()` seam -- `main()` reads
`sys.argv` directly -- so these tests drive it the way the pipeline does, by
patching `sys.argv` and calling `main()`. That keeps the tests honest about
the surface operators actually use, including argparse's own validation.

Most tests build a checkpoint directly from `build_generator` +
`save_checkpoint`, which is a real state dict written by real production code
and costs milliseconds. One test runs the actual training loop end to end, so
the contract between what `train()` writes and what `generate` reads is pinned
by something other than a hand-assembled fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.models.critic import Critic
from src.models.generator import build_generator
from src.sample import generate
from src.train.train_wgan_gp import save_checkpoint, train

DESCRIPTOR_DIM = 8
LATENT_DIM = 4

MLP_MODEL_CFG = {
    "latent_dim": LATENT_DIM,
    "generator_hidden_dims": [6],
    "negative_slope": 0.2,
    "generator_type": "mlp",
}

GATED_MODEL_CFG = {
    "latent_dim": LATENT_DIM,
    "generator_hidden_dims": [6],
    "negative_slope": 0.2,
    "generator_type": "gated",
    "gate_temperature": 0.5,
    "logit_clamp": 4.0,
}


def _write_run(
    run_dir: Path,
    model_cfg: dict,
    descriptor_dim: int = DESCRIPTOR_DIM,
    l2_normalize: bool = True,
) -> Path:
    """Write a real checkpoint plus its run_config.yaml, and return the ckpt.

    Seeded because the weights come from torch's global RNG at construction
    time; without this the checkpoint -- and so every assertion about the
    vectors it produces -- would differ run to run. `main()` reseeds from
    `--seed` before sampling, so this does not leak into the code under test.
    """
    torch.manual_seed(0)
    generator = build_generator(model_cfg, output_dim=descriptor_dim)
    critic = Critic(input_dim=descriptor_dim, hidden_dims=[6], negative_slope=0.2)
    save_checkpoint(
        generator,
        critic,
        torch.optim.Adam(generator.parameters(), lr=1e-4),
        torch.optim.Adam(critic.parameters(), lr=1e-4),
        out_dir=run_dir,
        step=1,
        best=True,
        generator_weights="ema",
    )
    run_config = {
        "device": "cpu",
        "model": model_cfg,
        "data": {
            "descriptor_dim": descriptor_dim,
            "preprocess": {
                "center": False,
                "whiten": False,
                "l2_normalize": l2_normalize,
            },
        },
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config))
    return run_dir / "best_generator.pt"


def _run_cli(monkeypatch: pytest.MonkeyPatch, **options: object) -> None:
    """Invoke `generate.main()` as the CLI, with keyword-to-flag translation."""
    argv = ["generate.py"]
    for key, value in options.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    monkeypatch.setattr(sys, "argv", argv)
    generate.main()


@pytest.fixture
def mlp_run(tmp_path: Path) -> Path:
    return _write_run(tmp_path / "runs" / "mlp", MLP_MODEL_CFG)


@pytest.fixture
def gated_run(tmp_path: Path) -> Path:
    return _write_run(tmp_path / "runs" / "gated", GATED_MODEL_CFG)


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    out_path: Path,
    num_samples: int = 10,
    **extra: object,
) -> np.ndarray:
    _run_cli(
        monkeypatch,
        checkpoint=checkpoint,
        config=checkpoint.parent / "run_config.yaml",
        num_samples=num_samples,
        output_path=out_path,
        **extra,
    )
    return np.load(out_path.with_suffix(".npy"))


def test_get_device_returns_the_explicitly_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert generate.get_device("cpu") == torch.device("cpu"), (
        "an explicit device config must win over accelerator autodetection"
    )


def test_get_device_falls_back_to_cpu_when_auto_finds_no_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert generate.get_device("auto") == torch.device("cpu")


def test_get_device_prefers_cuda_over_mps_when_both_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert generate.get_device("auto") == torch.device("cuda")


def test_generate_writes_exactly_the_requested_number_of_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    out = _generate(monkeypatch, mlp_run, tmp_path / "synthetic.npy", num_samples=13)
    assert out.shape == (13, DESCRIPTOR_DIM)


def test_generated_vectors_are_saved_as_float32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    """float32 is load-bearing downstream: the eval and ANN tooling reads these
    files straight into float32 buffers, and a float64 file silently doubles
    the size of the deliverable."""
    out = _generate(monkeypatch, mlp_run, tmp_path / "synthetic.npy")
    assert out.dtype == np.float32


def test_generated_vectors_use_the_descriptor_dim_from_the_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write_run(
        tmp_path / "runs" / "wide", MLP_MODEL_CFG, descriptor_dim=32
    )
    out = _generate(monkeypatch, checkpoint, tmp_path / "synthetic.npy")
    assert out.shape[1] == 32, (
        "output width must come from the run config, not a hardcoded default"
    )


@pytest.mark.parametrize("run_fixture", ["mlp_run", "gated_run"])
def test_generated_vectors_are_unit_norm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    run_fixture: str,
) -> None:
    checkpoint = request.getfixturevalue(run_fixture)
    out = _generate(monkeypatch, checkpoint, tmp_path / "synthetic.npy", num_samples=32)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"vectors must land on the unit sphere; got norms in "
        f"[{norms.min():.6f}, {norms.max():.6f}]"
    )


def test_generated_vectors_are_unit_norm_even_when_the_run_config_disabled_l2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the current, deliberate coupling: `generate` normalises its output
    unconditionally, exactly as `evaluate_distribution.sample_fake` and
    `train.sample_generator` do, and does *not* consult
    `data.preprocess.l2_normalize`. A run trained without L2 normalisation
    therefore yields samples in a different space from its own training data.
    Anyone changing this should change all three call sites together."""
    checkpoint = _write_run(
        tmp_path / "runs" / "unnormalised", MLP_MODEL_CFG, l2_normalize=False
    )
    out = _generate(monkeypatch, checkpoint, tmp_path / "synthetic.npy", num_samples=16)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_generation_is_deterministic_for_a_fixed_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    first = _generate(monkeypatch, mlp_run, tmp_path / "a.npy", num_samples=16, seed=7)
    second = _generate(monkeypatch, mlp_run, tmp_path / "b.npy", num_samples=16, seed=7)
    assert np.array_equal(first, second), (
        "the same checkpoint and seed must reproduce the deliverable exactly"
    )


def test_different_seeds_produce_different_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    first = _generate(monkeypatch, mlp_run, tmp_path / "a.npy", num_samples=16, seed=7)
    second = _generate(monkeypatch, mlp_run, tmp_path / "b.npy", num_samples=16, seed=8)
    assert not np.array_equal(first, second)


def test_the_requested_count_is_honoured_when_it_is_not_a_multiple_of_the_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    """The batching loop trims with `min(batch_size, remaining)`; an off-by-one
    there would over- or under-deliver on every non-multiple request."""
    out = _generate(
        monkeypatch, mlp_run, tmp_path / "synthetic.npy", num_samples=7, batch_size=3
    )
    assert out.shape == (7, DESCRIPTOR_DIM)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_a_batch_size_larger_than_the_request_still_yields_one_short_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    out = _generate(
        monkeypatch, mlp_run, tmp_path / "synthetic.npy", num_samples=5, batch_size=4096
    )
    assert out.shape == (5, DESCRIPTOR_DIM)


def test_generate_creates_missing_parent_directories_for_the_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    out_path = tmp_path / "deeply" / "nested" / "synthetic.npy"
    _generate(monkeypatch, mlp_run, out_path)
    assert out_path.exists()


def test_an_output_path_without_a_suffix_gains_the_npy_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    """`np.save` appends `.npy` when the path has no suffix, so the file an
    operator gets is not the path they asked for. Downstream tooling globs for
    `*.npy`, so this is the behaviour to keep -- but it must stay explicit."""
    requested = tmp_path / "synthetic"
    _run_cli(
        monkeypatch,
        checkpoint=mlp_run,
        config=mlp_run.parent / "run_config.yaml",
        num_samples=4,
        output_path=requested,
    )
    assert not requested.exists()
    assert (tmp_path / "synthetic.npy").exists()


def test_generate_fails_when_the_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        _run_cli(
            monkeypatch,
            checkpoint=mlp_run.parent / "does_not_exist.pt",
            config=mlp_run.parent / "run_config.yaml",
            num_samples=4,
            output_path=tmp_path / "synthetic.npy",
        )


def test_generate_fails_when_the_run_config_beside_the_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    (mlp_run.parent / "run_config.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        _run_cli(
            monkeypatch,
            checkpoint=mlp_run,
            config=mlp_run.parent / "run_config.yaml",
            num_samples=4,
            output_path=tmp_path / "synthetic.npy",
        )


def test_the_cli_refuses_to_run_without_a_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_run: Path
) -> None:
    """A checkpoint is only loadable beside its run_config.yaml, because
    `generator_type` is recorded in the config and not in the checkpoint. The
    CLI must therefore make `--config` mandatory rather than guessing."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(mlp_run),
            "--num-samples",
            "4",
            "--output-path",
            str(tmp_path / "synthetic.npy"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        generate.main()
    assert excinfo.value.code == 2


def test_a_gated_checkpoint_cannot_be_loaded_through_an_mlp_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gated_run: Path
) -> None:
    """The load-bearing half of the same invariant: nothing in the checkpoint
    says which architecture wrote it, so pairing it with the wrong config must
    fail loudly at `load_state_dict` rather than sample from a fresh, untrained
    network of the wrong type."""
    wrong_config = tmp_path / "mlp_run_config.yaml"
    wrong_config.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "model": MLP_MODEL_CFG,
                "data": {"descriptor_dim": DESCRIPTOR_DIM},
            }
        )
    )
    with pytest.raises(RuntimeError):
        _run_cli(
            monkeypatch,
            checkpoint=gated_run,
            config=wrong_config,
            num_samples=4,
            output_path=tmp_path / "synthetic.npy",
        )


def test_generate_consumes_a_checkpoint_written_by_a_real_training_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The only test here that does not hand-build its checkpoint. It pins the
    seam between `train()` -- which writes best_generator.pt and
    run_config.yaml into output_dir -- and the sampler that reads both back."""
    output_dir = tmp_path / "run"
    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(output_dir),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 16,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 8,
            "generator_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 4,
            "n_critic": 2,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "ema_decay": 0.9,
            "num_workers": 0,
            "distance_reg_alpha": 0.1,
            "distance_reg_max_points": 16,
            "amp": False,
            "log_every": 1,
            "eval_every": 2,
            "save_every": 4,
        },
    }
    checkpoint_path, _ = train(config)

    out_path = tmp_path / "synthetic.npy"
    _run_cli(
        monkeypatch,
        checkpoint=checkpoint_path,
        config=output_dir / "run_config.yaml",
        num_samples=20,
        output_path=out_path,
    )

    out = np.load(out_path)
    assert out.shape == (20, 16)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
