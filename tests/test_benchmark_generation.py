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
    benchmark_cell,
    format_markdown_table,
    parse_args,
    run_grid,
)


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


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


def test_benchmark_cell_returns_normalized_partial_batch() -> None:
    cell = benchmark_cell(
        TinyGenerator(),
        num_samples=7,
        batch_size=3,
        latent_dim=3,
        descriptor_dim=4,
        device=torch.device("cpu"),
        repeats=2,
    )
    assert cell["samples"].shape == (7, 4)
    assert cell["samples"].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(cell["samples"], axis=1), 1.0, atol=1e-6)
    assert cell["save_seconds"] is None
    assert cell["peak_vram_bytes"] is None
    for phase in ("generate_seconds", "to_host_seconds"):
        assert set(cell[phase]) == {"min", "median", "p95"}
        assert all(
            math.isfinite(value) and value >= 0 for value in cell[phase].values()
        )


def test_run_grid_is_json_serializable(tmp_path: Path) -> None:
    configs = [tmp_path / "one.yaml", tmp_path / "two.yaml"]
    for path in configs:
        _write_config(path)
    cells = run_grid(configs, [2, 5], batch_size=3, repeats=1)
    assert len(cells) == 4
    assert all("samples" not in cell for cell in cells)
    json.loads(json.dumps(cells))


def test_format_markdown_table_has_one_row_per_cell(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    _write_config(config)
    cells = run_grid([config], [2, 5], batch_size=3, repeats=1)
    table = format_markdown_table(cells)
    assert len(table.strip().splitlines()) == 4


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
