from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from src.sample.benchmark import (
    DEFAULT_CONFIGS,
    build_schedule,
    config_slug,
    format_markdown_table,
    main,
    model_param_bytes,
    parse_args,
    run_grid,
    run_repeat,
    warmup_generator,
)


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class DoubleGenerator(nn.Module):
    """Emits float64 so the host-copy path has to narrow the dtype itself."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.double().repeat(1, 2)


class BufferedGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z) * self.scale


def _write_config(path: Path, generator_type: str = "mlp") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "data": {"descriptor_dim": 4},
                "model": {
                    "latent_dim": 3,
                    "generator_hidden_dims": [5],
                    "generator_type": generator_type,
                    "negative_slope": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )


def _empty_buffer(num_samples: int, descriptor_dim: int) -> np.ndarray:
    return np.empty((num_samples, descriptor_dim), dtype=np.float32)


def test_run_repeat_normalizes_and_fills_partial_final_batch() -> None:
    samples = _empty_buffer(7, 4)
    result = run_repeat(
        TinyGenerator(),
        num_samples=7,
        batch_size=3,
        latent_dim=3,
        device=torch.device("cpu"),
        samples=samples,
    )
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-6)
    assert result["save_seconds"] is None
    assert result["incremental_peak_vram_bytes"] is None
    for phase in ("generate_seconds", "to_host_seconds"):
        assert math.isfinite(result[phase]) and result[phase] >= 0


def test_run_repeat_narrows_a_float64_generator_into_the_float32_buffer() -> None:
    samples = _empty_buffer(4, 6)
    run_repeat(
        DoubleGenerator(),
        num_samples=4,
        batch_size=3,
        latent_dim=3,
        device=torch.device("cpu"),
        samples=samples,
    )
    assert samples.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-6)


def test_warmup_runs_the_normalization_it_times() -> None:
    warmed = warmup_generator(
        TinyGenerator(), batch_size=5, latent_dim=3, device=torch.device("cpu")
    )
    with torch.inference_mode():
        norms = torch.linalg.vector_norm(warmed, dim=1).tolist()
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_schedule_visits_every_cell_once_per_round() -> None:
    schedule = build_schedule(4, 3, np.random.default_rng(0))
    assert len(schedule) == 12
    for start in range(0, 12, 4):
        assert sorted(schedule[start : start + 4]) == [0, 1, 2, 3]


def test_schedule_reorders_cells_between_rounds() -> None:
    schedule = build_schedule(8, 4, np.random.default_rng(0))
    rounds = [schedule[start : start + 8] for start in range(0, 32, 8)]
    assert any(other != rounds[0] for other in rounds[1:])


def test_model_param_bytes_counts_parameters_and_buffers() -> None:
    # Linear(3, 4) is 16 floats of weight+bias, the buffer adds 4 more.
    assert model_param_bytes(BufferedGenerator()) == 20 * 4


def test_config_slug_distinguishes_equal_stems_in_different_families() -> None:
    assert config_slug(Path("configs/sift/v1.yaml")) != config_slug(
        Path("configs/deep/v1.yaml")
    )


def test_cell_reports_generate_and_end_to_end_throughput(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    (cell,) = run_grid([config], [8], batch_size=3, repeats=2, seed=0)
    generate = cell["generate_seconds"]["median"]
    host = cell["to_host_seconds"]["median"]
    assert cell["generate_vectors_per_second"] == pytest.approx(8 / generate)
    assert cell["end_to_end_vectors_per_second"] == pytest.approx(8 / (generate + host))
    assert cell["end_to_end_vectors_per_second"] < cell["generate_vectors_per_second"]


def test_cell_peak_vram_is_the_worst_repeat_not_an_average(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    (cell,) = run_grid([config], [8], batch_size=3, repeats=3, seed=0)
    # CPU has no allocator stats, but the key must exist and stay unaveraged.
    assert "incremental_peak_vram_bytes" in cell
    assert cell["model_param_bytes"] > 0


def test_run_grid_is_json_serializable(tmp_path: Path) -> None:
    configs = [tmp_path / "one.yaml", tmp_path / "two.yaml"]
    for path in configs:
        _write_config(path)
    cells = run_grid(configs, [2, 5], batch_size=3, repeats=1, seed=0)
    assert len(cells) == 4
    assert all("samples" not in cell for cell in cells)
    json.loads(json.dumps(cells))


def test_save_dir_keeps_one_corpus_file_per_cell(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    save_dir = tmp_path / "saved"
    cells = run_grid([config], [5], batch_size=3, repeats=3, seed=0, save_dir=save_dir)
    written = list(save_dir.rglob("*.npy"))
    assert len(written) == 1
    assert set(cells[0]["save_seconds"]) == {"min", "median", "p95"}


def test_save_dir_separates_configs_that_share_a_stem(tmp_path: Path) -> None:
    sift = tmp_path / "sift"
    deep = tmp_path / "deep"
    sift.mkdir()
    deep.mkdir()
    configs = [sift / "v1.yaml", deep / "v1.yaml"]
    for path in configs:
        _write_config(path)
    save_dir = tmp_path / "saved"
    run_grid(configs, [4], batch_size=3, repeats=1, seed=0, save_dir=save_dir)
    assert len(list(save_dir.rglob("*.npy"))) == 2


def test_format_markdown_table_has_one_row_per_cell(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    cells = run_grid([config], [2, 5], batch_size=3, repeats=1, seed=0)
    table = format_markdown_table(cells)
    assert len(table.strip().splitlines()) == 4


def test_main_persists_results_before_raising_on_non_finite_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    output_dir = tmp_path / "out"

    def _nan_grid(*args: object, **kwargs: object) -> list[dict[str, object]]:
        cells = run_grid([config], [4], batch_size=3, repeats=1, seed=0)
        cells[0]["to_host_seconds"]["median"] = float("nan")
        return cells

    monkeypatch.setattr("src.sample.benchmark.run_grid", _nan_grid)
    with pytest.raises(RuntimeError):
        main(
            [
                "--config",
                str(config),
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
            ]
        )
    written = json.loads((output_dir / "generation_benchmark.json").read_text())
    assert written["non_finite_timings"] is True
    assert (output_dir / "generation_benchmark.md").exists()


def test_checkpoint_rejects_multiple_configs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--config",
                "one.yaml",
                "--config",
                "two.yaml",
                "--checkpoint",
                "model.pt",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_default_configs_are_matched_architecture_triple() -> None:
    kinds = []
    for path_text in DEFAULT_CONFIGS:
        config = yaml.safe_load(Path(path_text).read_text(encoding="utf-8"))
        kinds.append(config["model"].get("generator_type", "mlp"))
    assert kinds == ["mlp", "gated", "structured_gated"]
