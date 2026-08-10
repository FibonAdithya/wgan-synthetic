"""Tests for the checkpoint-free file-to-file evaluation entry point.

Unlike `evaluate_distribution.py`, this script never loads a generator and
never reads `run_metadata.json`: it compares two descriptor files directly.
That makes it the path an agent reaches for when it only has vectors on disk,
so the tests below pin the metric names it emits, how it caps sample counts,
its two file formats, and the preprocessing it applies by default.

`main()` has no injectable-argv seam -- `parse_args()` reads `sys.argv`
directly -- so the CLI is driven by patching `sys.argv` at that boundary.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import evaluate_file_to_file as eff

# The metric names `evaluate_distribution.py` writes. The file-to-file variant
# must not drift from them: a downstream comparison of the two reports keys off
# these strings.
DISTRIBUTION_METRIC_NAMES = frozenset(
    {
        "mean_l2",
        "var_l2",
        "cov_fro",
        "mmd_rbf",
        "pairwise_hist_l1",
        "knn_recall",
        "ann_proxy_recall",
    }
)


def write_npy(path: Path, arr: np.ndarray) -> Path:
    np.save(path, arr.astype(np.float32))
    return path


def write_fvecs(path: Path, arr: np.ndarray) -> Path:
    """Write `arr` in the fvecs layout: an int32 dim header per row, then floats."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    header = np.full((arr.shape[0], 1), arr.shape[1], dtype=np.int32)
    rows = np.hstack([header.view(np.float32), arr])
    rows.astype(np.float32).tofile(path)
    return path


def run_cli(monkeypatch, out_dir: Path, **flags) -> dict:
    """Invoke `main()` with `--flag value` pairs and return the parsed metrics."""
    argv = ["evaluate_file_to_file.py", "--output-dir", str(out_dir)]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    eff.main()
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))


def assert_metrics_close(actual: dict, expected: dict, message: str) -> None:
    """Compare two reports allowing only float32 round-off between them."""
    assert set(actual) == set(expected), message
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, rel=1e-6, abs=1e-9), (
            f"{message} (metric {name!r})"
        )


@pytest.fixture
def descriptor_pair(tmp_path: Path):
    """A real/synthetic `.npy` pair large enough for every metric to be defined."""
    rng = np.random.default_rng(20240805)
    real = write_npy(tmp_path / "real.npy", rng.normal(size=(200, 8)))
    synthetic = write_npy(tmp_path / "synthetic.npy", rng.normal(size=(60, 8)))
    return real, synthetic


def test_it_writes_every_metric_evaluate_distribution_writes_under_the_same_names(
    tmp_path: Path, descriptor_pair, monkeypatch
):
    real, synthetic = descriptor_pair

    metrics = run_cli(
        monkeypatch,
        tmp_path / "out",
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.2,
    )

    assert set(metrics) == DISTRIBUTION_METRIC_NAMES | {"num_samples_used"}, (
        f"metric names drifted from evaluate_distribution.py: {sorted(metrics)}"
    )
    assert all(math.isfinite(metrics[name]) for name in DISTRIBUTION_METRIC_NAMES)


def test_it_needs_no_checkpoint_or_run_metadata_beside_the_inputs(
    tmp_path: Path, descriptor_pair, monkeypatch
):
    """The whole point of this script: two files in, one metrics.json out."""
    real, synthetic = descriptor_pair
    out_dir = tmp_path / "nested" / "out"

    run_cli(
        monkeypatch,
        out_dir,
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.2,
    )

    assert sorted(p.name for p in out_dir.iterdir()) == ["metrics.json"]
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("run_metadata.json"))


def test_it_creates_a_missing_output_directory_including_parents(
    tmp_path: Path, descriptor_pair, monkeypatch
):
    real, synthetic = descriptor_pair
    out_dir = tmp_path / "a" / "b" / "c"

    run_cli(
        monkeypatch,
        out_dir,
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.2,
    )

    assert (out_dir / "metrics.json").exists()


def test_it_refuses_to_compare_files_whose_descriptor_dimensions_disagree(
    tmp_path: Path, monkeypatch
):
    rng = np.random.default_rng(1)
    real = write_npy(tmp_path / "real.npy", rng.normal(size=(50, 8)))
    synthetic = write_npy(tmp_path / "synthetic.npy", rng.normal(size=(50, 6)))

    with pytest.raises(ValueError, match=r"Dimension mismatch: real dim=8.*dim=6"):
        run_cli(
            monkeypatch,
            tmp_path / "out",
            real_path=real,
            synthetic_path=synthetic,
        )

    assert not (tmp_path / "out" / "metrics.json").exists(), (
        "a rejected comparison must not leave a metrics file behind"
    )


def test_num_samples_used_is_capped_by_the_smaller_of_holdout_and_synthetic(
    tmp_path: Path, monkeypatch
):
    rng = np.random.default_rng(2)
    real = write_npy(tmp_path / "real.npy", rng.normal(size=(200, 8)))
    synthetic = write_npy(tmp_path / "synthetic.npy", rng.normal(size=(12, 8)))

    metrics = run_cli(
        monkeypatch,
        tmp_path / "out",
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.1,
        num_samples=5000,
    )

    # 20 real holdout rows, 12 synthetic rows, a 5000 request: the file wins.
    assert metrics["num_samples_used"] == 12
    assert isinstance(metrics["num_samples_used"], int)


def test_num_samples_flag_caps_the_evaluation_when_it_is_the_tightest_bound(
    tmp_path: Path, monkeypatch
):
    rng = np.random.default_rng(3)
    real = write_npy(tmp_path / "real.npy", rng.normal(size=(200, 8)))
    synthetic = write_npy(tmp_path / "synthetic.npy", rng.normal(size=(200, 8)))

    metrics = run_cli(
        monkeypatch,
        tmp_path / "out",
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.25,
        num_samples=7,
    )

    assert metrics["num_samples_used"] == 7


def test_fvecs_and_npy_inputs_holding_the_same_vectors_score_identically(
    tmp_path: Path, monkeypatch
):
    """Format is a container detail; it must not move a single metric."""
    rng = np.random.default_rng(4)
    real_arr = rng.normal(size=(120, 8)).astype(np.float32)
    synthetic_arr = rng.normal(size=(40, 8)).astype(np.float32)

    from_npy = run_cli(
        monkeypatch,
        tmp_path / "npy_out",
        real_path=write_npy(tmp_path / "real.npy", real_arr),
        synthetic_path=write_npy(tmp_path / "synthetic.npy", synthetic_arr),
        holdout_fraction=0.25,
    )
    from_fvecs = run_cli(
        monkeypatch,
        tmp_path / "fvecs_out",
        real_path=write_fvecs(tmp_path / "real.fvecs", real_arr),
        synthetic_path=write_fvecs(tmp_path / "synthetic.fvecs", synthetic_arr),
        holdout_fraction=0.25,
    )

    assert_metrics_close(
        from_fvecs, from_npy, "fvecs and npy inputs must produce equal metrics"
    )


def test_explicit_format_flags_override_extension_based_detection(
    tmp_path: Path, monkeypatch
):
    """`auto` keys off the suffix, so a misnamed file needs the explicit flag."""
    rng = np.random.default_rng(5)
    real_arr = rng.normal(size=(120, 8)).astype(np.float32)
    synthetic_arr = rng.normal(size=(40, 8)).astype(np.float32)
    real = write_fvecs(tmp_path / "real.bin", real_arr)
    synthetic = write_fvecs(tmp_path / "synthetic.bin", synthetic_arr)

    with pytest.raises(ValueError, match="auto format detection"):
        run_cli(
            monkeypatch,
            tmp_path / "auto_out",
            real_path=real,
            synthetic_path=synthetic,
            holdout_fraction=0.25,
        )

    metrics = run_cli(
        monkeypatch,
        tmp_path / "explicit_out",
        real_path=real,
        synthetic_path=synthetic,
        real_format="fvecs",
        synthetic_format="fvecs",
        holdout_fraction=0.25,
    )

    assert metrics["num_samples_used"] == 30


def test_it_l2_normalizes_both_files_by_default_so_scale_cannot_move_the_metrics(
    tmp_path: Path, monkeypatch
):
    rng = np.random.default_rng(6)
    real_arr = rng.normal(size=(120, 8)).astype(np.float32)
    synthetic_arr = rng.normal(size=(40, 8)).astype(np.float32)
    real = write_npy(tmp_path / "real.npy", real_arr)

    unscaled = run_cli(
        monkeypatch,
        tmp_path / "unscaled",
        real_path=real,
        synthetic_path=write_npy(tmp_path / "syn.npy", synthetic_arr),
        holdout_fraction=0.25,
    )
    scaled = run_cli(
        monkeypatch,
        tmp_path / "scaled",
        real_path=real,
        synthetic_path=write_npy(tmp_path / "syn_scaled.npy", synthetic_arr * 10.0),
        holdout_fraction=0.25,
    )

    assert_metrics_close(
        scaled,
        unscaled,
        "default L2 normalization must make the report scale-invariant",
    )


def test_skip_l2_normalize_lets_raw_vector_scale_reach_the_metrics(
    tmp_path: Path, monkeypatch
):
    rng = np.random.default_rng(7)
    real_arr = rng.normal(size=(120, 8)).astype(np.float32)
    synthetic_arr = rng.normal(size=(40, 8)).astype(np.float32)
    real = write_npy(tmp_path / "real.npy", real_arr)

    unscaled = run_cli(
        monkeypatch,
        tmp_path / "unscaled",
        real_path=real,
        synthetic_path=write_npy(tmp_path / "syn.npy", synthetic_arr),
        holdout_fraction=0.25,
        skip_l2_normalize=True,
    )
    scaled = run_cli(
        monkeypatch,
        tmp_path / "scaled",
        real_path=real,
        synthetic_path=write_npy(tmp_path / "syn_scaled.npy", synthetic_arr * 10.0),
        holdout_fraction=0.25,
        skip_l2_normalize=True,
    )

    assert scaled["mean_l2"] > unscaled["mean_l2"], (
        "--skip-l2-normalize must leave the 10x scale visible in mean_l2"
    )


def test_the_run_is_reproducible_for_a_fixed_seed_and_shifts_when_the_seed_changes(
    tmp_path: Path, monkeypatch
):
    """The seed drives both the holdout split and the evaluation subsample."""
    rng = np.random.default_rng(8)
    real = write_npy(tmp_path / "real.npy", rng.normal(size=(200, 8)))
    synthetic = write_npy(tmp_path / "synthetic.npy", rng.normal(size=(60, 8)))
    common = dict(
        real_path=real, synthetic_path=synthetic, holdout_fraction=0.25, num_samples=20
    )

    first = run_cli(monkeypatch, tmp_path / "a", seed=11, **common)
    repeat = run_cli(monkeypatch, tmp_path / "b", seed=11, **common)
    other = run_cli(monkeypatch, tmp_path / "c", seed=12, **common)

    assert first == repeat, "same seed must reproduce the report exactly"
    assert first["mean_l2"] != other["mean_l2"], (
        "a different seed must draw a different holdout/evaluation subsample"
    )


def test_mmd_rbf_reports_squared_mmd_despite_its_name(tmp_path: Path, monkeypatch):
    """Pinning a known wart: the key says `mmd_rbf` but the value is MMD^2.

    Two point masses one unit apart give the closed form
    ``k_xx + k_yy - 2 k_xy = 2 - 2 exp(-gamma)``. The square root of that is
    the actual MMD, and it is *not* what lands in the report -- so anything
    comparing this number against a published MMD must square it first.
    """
    real = write_npy(tmp_path / "real.npy", np.zeros((40, 2)))
    synthetic = write_npy(tmp_path / "synthetic.npy", np.tile([1.0, 0.0], (40, 1)))

    metrics = run_cli(
        monkeypatch,
        tmp_path / "out",
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.25,
        gamma=1.0,
        skip_l2_normalize=True,
    )

    mmd_squared = 2.0 - 2.0 * math.exp(-1.0)
    assert metrics["mmd_rbf"] == pytest.approx(mmd_squared, rel=1e-6)
    assert metrics["mmd_rbf"] != pytest.approx(math.sqrt(mmd_squared), rel=1e-6)


def test_gamma_flag_reaches_the_rbf_kernel_behind_mmd(tmp_path: Path, monkeypatch):
    real = write_npy(tmp_path / "real.npy", np.zeros((40, 2)))
    synthetic = write_npy(tmp_path / "synthetic.npy", np.tile([1.0, 0.0], (40, 1)))
    common = dict(
        real_path=real,
        synthetic_path=synthetic,
        holdout_fraction=0.25,
        skip_l2_normalize=True,
    )

    metrics = run_cli(monkeypatch, tmp_path / "out", gamma=4.0, **common)

    assert metrics["mmd_rbf"] == pytest.approx(2.0 - 2.0 * math.exp(-4.0), rel=1e-6)


# The three normalisation tests that lived here moved to
# tests/test_normalisation_is_shared.py when this module stopped carrying its
# own copy of the rule. The `--skip-l2-normalize` behaviour, which is this
# module's own, is still covered above.


def test_random_sample_returns_the_whole_array_when_n_is_not_smaller():
    rng = np.random.default_rng(11)
    x = np.arange(20, dtype=np.float32).reshape(10, 2)

    np.testing.assert_array_equal(eff.random_sample(x, n=10, rng=rng), x)
    np.testing.assert_array_equal(eff.random_sample(x, n=99, rng=rng), x)


def test_random_sample_draws_without_replacement_when_it_subsamples():
    rng = np.random.default_rng(12)
    x = np.arange(200, dtype=np.float32).reshape(100, 2)

    drawn = eff.random_sample(x, n=30, rng=rng)

    assert drawn.shape == (30, 2)
    assert len({tuple(row) for row in drawn}) == 30, "rows must not repeat"
    assert {tuple(row) for row in drawn} <= {tuple(row) for row in x}


def test_random_sample_is_a_function_of_the_generator_state_alone():
    x = np.arange(200, dtype=np.float32).reshape(100, 2)

    first = eff.random_sample(x, n=10, rng=np.random.default_rng(13))
    second = eff.random_sample(x, n=10, rng=np.random.default_rng(13))
    third = eff.random_sample(x, n=10, rng=np.random.default_rng(14))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_parse_args_defaults_match_the_documented_cli_contract(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_file_to_file.py",
            "--real-path",
            "r.npy",
            "--synthetic-path",
            "s.npy",
            "--output-dir",
            "out",
        ],
    )

    args = eff.parse_args()

    assert args.real_format == "auto"
    assert args.synthetic_format == "auto"
    assert args.num_samples == 5000
    assert args.holdout_fraction == 0.05
    assert args.gamma == 1.0
    assert args.seed == 42
    assert args.skip_l2_normalize is False


@pytest.mark.parametrize("missing", ["--real-path", "--synthetic-path", "--output-dir"])
def test_parse_args_requires_both_input_paths_and_an_output_directory(
    monkeypatch, missing: str
):
    full = {
        "--real-path": "r.npy",
        "--synthetic-path": "s.npy",
        "--output-dir": "out",
    }
    argv = ["evaluate_file_to_file.py"]
    for flag, value in full.items():
        if flag != missing:
            argv.extend([flag, value])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        eff.parse_args()


def test_parse_args_rejects_a_file_format_it_cannot_load(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_file_to_file.py",
            "--real-path",
            "r.hdf5",
            "--synthetic-path",
            "s.npy",
            "--output-dir",
            "out",
            "--real-format",
            "hdf5",
        ],
    )

    with pytest.raises(SystemExit):
        eff.parse_args()
