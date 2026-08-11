"""Benchmark generator sampling phases across architectures and corpus sizes."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from src.device import resolve_device
from src.models.generator import build_generator
from src.train.train_wgan_gp import normalize_l2

DEFAULT_CONFIGS = (
    "configs/sift/v1.yaml",
    "configs/sift/v2.yaml",
    "configs/sift/v4.yaml",
)
DEFAULT_NUM_SAMPLES = (1_000, 10_000, 100_000, 1_000_000)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def benchmark_cell(
    generator: nn.Module,
    *,
    num_samples: int,
    batch_size: int,
    latent_dim: int,
    descriptor_dim: int,
    device: torch.device,
    repeats: int,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """Time one generator/N cell and return timings plus the final draw."""
    if num_samples <= 0 or batch_size <= 0 or repeats <= 0:
        raise ValueError("num_samples, batch_size, and repeats must be positive")

    generate_times: list[float] = []
    to_host_times: list[float] = []
    save_times: list[float] = []
    samples = np.empty((num_samples, descriptor_dim), dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator.eval()
    with torch.inference_mode():
        for repeat in range(repeats):
            generated = 0
            generate_seconds = 0.0
            to_host_seconds = 0.0
            while generated < num_samples:
                current = min(batch_size, num_samples - generated)
                _synchronize(device)
                started = time.perf_counter()
                z = torch.randn(current, latent_dim, device=device)
                batch = normalize_l2(generator(z))
                _synchronize(device)
                generate_seconds += time.perf_counter() - started

                started = time.perf_counter()
                host_batch = batch.cpu().numpy().astype(np.float32, copy=False)
                samples[generated : generated + current] = host_batch
                _synchronize(device)
                to_host_seconds += time.perf_counter() - started
                generated += current

            generate_times.append(generate_seconds)
            to_host_times.append(to_host_seconds)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                np.save(save_dir / f"samples_repeat_{repeat}.npy", samples)
                save_times.append(time.perf_counter() - started)

    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    generate_summary = _summary(generate_times)
    return {
        "num_samples": num_samples,
        "generate_seconds": generate_summary,
        "to_host_seconds": _summary(to_host_times),
        "save_seconds": _summary(save_times) if save_times else None,
        "peak_vram_bytes": peak_vram,
        "throughput_vectors_per_second": num_samples
        / max(generate_summary["median"], 1.0e-12),
        "samples": samples,
    }


def run_grid(
    config_paths: list[Path],
    num_samples: list[int],
    *,
    batch_size: int,
    repeats: int,
    device_override: str | None = None,
    checkpoint: Path | None = None,
    save_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build each configured generator and benchmark every requested N."""
    cells: list[dict[str, Any]] = []
    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model_cfg = config["model"]
        data_cfg = config["data"]
        device = resolve_device(device_override or config["device"])

        started = time.perf_counter()
        generator = build_generator(
            model_cfg, output_dim=int(data_cfg["descriptor_dim"])
        ).to(device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device)
            generator.load_state_dict(state["generator_state_dict"])
        generator.eval()
        _synchronize(device)
        build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        with torch.inference_mode():
            generator(
                torch.randn(batch_size, int(model_cfg["latent_dim"]), device=device)
            )
        _synchronize(device)
        warmup_seconds = time.perf_counter() - started

        for count in num_samples:
            cell_save_dir = (
                save_dir / config_path.stem / str(count) if save_dir else None
            )
            cell = benchmark_cell(
                generator,
                num_samples=count,
                batch_size=batch_size,
                latent_dim=int(model_cfg["latent_dim"]),
                descriptor_dim=int(data_cfg["descriptor_dim"]),
                device=device,
                repeats=repeats,
                save_dir=cell_save_dir,
            )
            cell.pop("samples")
            cell.update(
                {
                    "config": str(config_path),
                    "generator_type": model_cfg.get("generator_type", "mlp"),
                    "latent_dim": int(model_cfg["latent_dim"]),
                    "descriptor_dim": int(data_cfg["descriptor_dim"]),
                    "build_seconds": build_seconds,
                    "warmup_seconds": warmup_seconds,
                }
            )
            cells.append(cell)
    return cells


def format_markdown_table(cells: list[dict[str, Any]]) -> str:
    """Render the budgeting view of benchmark cells."""
    lines = [
        "| Config | Architecture | N | Generate median (s) | To host median (s) | Budget p95 (s) | Vectors/s | Peak VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        peak = cell["peak_vram_bytes"]
        peak_text = "n/a" if peak is None else f"{peak / (1024**2):.1f} MiB"
        budget = cell["generate_seconds"]["p95"] + cell["to_host_seconds"]["p95"]
        lines.append(
            f"| {cell['config']} | {cell['generator_type']} | "
            f"{cell['num_samples']:,} | {cell['generate_seconds']['median']:.6f} | "
            f"{cell['to_host_seconds']['median']:.6f} | {budget:.6f} | "
            f"{cell['throughput_vectors_per_second']:.1f} | {peak_text} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--num-samples", action="append", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    args.config = args.config or [Path(path) for path in DEFAULT_CONFIGS]
    args.num_samples = args.num_samples or list(DEFAULT_NUM_SAMPLES)
    if args.checkpoint is not None and len(args.config) != 1:
        parser.error("--checkpoint requires exactly one --config")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    first_config = yaml.safe_load(args.config[0].read_text(encoding="utf-8"))
    device = resolve_device(args.device or first_config["device"])

    cuda_init_seconds = 0.0
    if device.type == "cuda":
        started = time.perf_counter()
        torch.empty(1, device=device)
        _synchronize(device)
        cuda_init_seconds = time.perf_counter() - started

    cells = run_grid(
        args.config,
        args.num_samples,
        batch_size=args.batch_size,
        repeats=args.repeats,
        device_override=str(device),
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
    )
    environment = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_init_seconds": cuda_init_seconds,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "seed": args.seed,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    result = {"environment": environment, "cells": cells}
    if not all(
        math.isfinite(value)
        for cell in cells
        for phase in ("generate_seconds", "to_host_seconds")
        for value in cell[phase].values()
    ):
        raise RuntimeError("benchmark produced a non-finite timing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generation_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generation_benchmark.md").write_text(
        format_markdown_table(cells), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
