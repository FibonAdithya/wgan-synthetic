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


def config_slug(path: Path) -> str:
    """Name a config by family and stem so `sift/v1` and `deep/v1` stay distinct."""
    parts = path.with_suffix("").parts
    return "_".join(parts[-2:]) if len(parts) > 1 else parts[-1]


def model_param_bytes(module: nn.Module) -> int:
    """Exact parameter and buffer bytes, free of one-time allocator workspace."""
    tensors = [*module.parameters(), *module.buffers()]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def build_schedule(
    cell_count: int, repeats: int, rng: np.random.Generator
) -> list[int]:
    """Order repeats as re-randomized rounds over every cell.

    Running a cell's repeats back-to-back lets slow-varying machine state alias
    onto whichever axis is being fitted. One fresh permutation per round spreads
    each cell's repeats across the whole run instead.
    """
    if cell_count <= 0 or repeats <= 0:
        raise ValueError("cell_count and repeats must be positive")
    schedule: list[int] = []
    for _ in range(repeats):
        schedule.extend(int(index) for index in rng.permutation(cell_count))
    return schedule


def warmup_generator(
    generator: nn.Module, *, batch_size: int, latent_dim: int, device: torch.device
) -> torch.Tensor:
    """Run the exact expression the timed loop runs.

    Warming only the forward pass leaves the normalization kernels to initialize
    lazily inside the first measured cell, which shows up as a startup-sized p95.
    """
    generator.eval()
    with torch.inference_mode():
        z = torch.randn(batch_size, latent_dim, device=device)
        warmed = normalize_l2(generator(z))
    _synchronize(device)
    return warmed


def run_repeat(
    generator: nn.Module,
    *,
    num_samples: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    samples: np.ndarray,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """Time one pass over `num_samples`, writing straight into `samples`."""
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")

    allocated_before = None
    reserved_before = None
    if device.type == "cuda":
        _synchronize(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

    generate_seconds = 0.0
    to_host_seconds = 0.0
    generator.eval()
    with torch.inference_mode():
        generated = 0
        while generated < num_samples:
            current = min(batch_size, num_samples - generated)
            _synchronize(device)
            started = time.perf_counter()
            z = torch.randn(current, latent_dim, device=device)
            batch = normalize_l2(generator(z))
            _synchronize(device)
            generate_seconds += time.perf_counter() - started

            started = time.perf_counter()
            destination = torch.from_numpy(samples[generated : generated + current])
            destination.copy_(batch)
            _synchronize(device)
            to_host_seconds += time.perf_counter() - started
            generated += current

    incremental_peak = None
    incremental_peak_reserved = None
    if device.type == "cuda":
        _synchronize(device)
        incremental_peak = (
            int(torch.cuda.max_memory_allocated(device)) - allocated_before
        )
        incremental_peak_reserved = (
            int(torch.cuda.max_memory_reserved(device)) - reserved_before
        )

    save_seconds = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        np.save(save_path, samples)
        save_seconds = time.perf_counter() - started

    return {
        "generate_seconds": generate_seconds,
        "to_host_seconds": to_host_seconds,
        "save_seconds": save_seconds,
        "incremental_peak_vram_bytes": incremental_peak,
        "incremental_peak_vram_reserved_bytes": incremental_peak_reserved,
        "resident_allocated_bytes": allocated_before,
    }


def _summarize_cell(
    records: list[dict[str, Any]], *, num_samples: int, context: dict[str, Any]
) -> dict[str, Any]:
    generate = _summary([record["generate_seconds"] for record in records])
    to_host = _summary([record["to_host_seconds"] for record in records])
    saves = [
        record["save_seconds"]
        for record in records
        if record["save_seconds"] is not None
    ]

    def _worst(key: str) -> int | None:
        values = [record[key] for record in records if record[key] is not None]
        return max(values) if values else None

    itemsize = np.dtype(np.float32).itemsize
    budget = generate["median"] + to_host["median"]
    return {
        "num_samples": num_samples,
        "generate_seconds": generate,
        "to_host_seconds": to_host,
        "save_seconds": _summary(saves) if saves else None,
        "incremental_peak_vram_bytes": _worst("incremental_peak_vram_bytes"),
        "incremental_peak_vram_reserved_bytes": _worst(
            "incremental_peak_vram_reserved_bytes"
        ),
        "resident_allocated_bytes": _worst("resident_allocated_bytes"),
        "model_param_bytes": context["model_param_bytes"],
        "host_output_bytes": int(num_samples * context["descriptor_dim"] * itemsize),
        "generate_vectors_per_second": num_samples / max(generate["median"], 1.0e-12),
        "end_to_end_vectors_per_second": num_samples / max(budget, 1.0e-12),
        "config": str(context["config_path"]),
        "generator_type": context["generator_type"],
        "latent_dim": context["latent_dim"],
        "descriptor_dim": context["descriptor_dim"],
        "build_seconds": context["build_seconds"],
        "warmup_seconds": context["warmup_seconds"],
    }


def _prepare_context(
    config_path: Path,
    *,
    batch_size: int,
    device_override: str | None,
    checkpoint: Path | None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    device = resolve_device(device_override or config["device"])
    latent_dim = int(model_cfg["latent_dim"])
    descriptor_dim = int(data_cfg["descriptor_dim"])

    started = time.perf_counter()
    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        generator.load_state_dict(state["generator_state_dict"])
    generator.eval()
    _synchronize(device)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warmup_generator(
        generator, batch_size=batch_size, latent_dim=latent_dim, device=device
    )
    warmup_seconds = time.perf_counter() - started

    return {
        "config_path": config_path,
        "slug": config_slug(config_path),
        "generator": generator,
        "device": device,
        "latent_dim": latent_dim,
        "descriptor_dim": descriptor_dim,
        "generator_type": model_cfg.get("generator_type", "mlp"),
        "model_param_bytes": model_param_bytes(generator),
        "build_seconds": build_seconds,
        "warmup_seconds": warmup_seconds,
    }


def run_grid(
    config_paths: list[Path],
    num_samples: list[int],
    *,
    batch_size: int,
    repeats: int,
    seed: int = 0,
    device_override: str | None = None,
    checkpoint: Path | None = None,
    save_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Benchmark every config/N cell on an interleaved, re-randomized schedule."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    slugs = [config_slug(path) for path in config_paths]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"config paths do not have distinct slugs: {', '.join(slugs)}")

    contexts = [
        _prepare_context(
            path,
            batch_size=batch_size,
            device_override=device_override,
            checkpoint=checkpoint,
        )
        for path in config_paths
    ]

    # One host buffer per descriptor width, sliced per cell, so interleaving
    # does not multiply the host footprint by the number of cells.
    largest = max(num_samples)
    buffers: dict[int, np.ndarray] = {}
    for context in contexts:
        width = context["descriptor_dim"]
        if width not in buffers:
            buffers[width] = np.empty((largest, width), dtype=np.float32)

    cells = [(index, count) for index in range(len(contexts)) for count in num_samples]
    records: list[list[dict[str, Any]]] = [[] for _ in cells]
    for cell_index in build_schedule(len(cells), repeats, np.random.default_rng(seed)):
        context_index, count = cells[cell_index]
        context = contexts[context_index]
        save_path = (
            save_dir / context["slug"] / str(count) / "samples.npy"
            if save_dir is not None
            else None
        )
        records[cell_index].append(
            run_repeat(
                context["generator"],
                num_samples=count,
                batch_size=batch_size,
                latent_dim=context["latent_dim"],
                device=context["device"],
                samples=buffers[context["descriptor_dim"]][:count],
                save_path=save_path,
            )
        )

    return [
        _summarize_cell(
            records[cell_index],
            num_samples=count,
            context=contexts[context_index],
        )
        for cell_index, (context_index, count) in enumerate(cells)
    ]


def format_markdown_table(cells: list[dict[str, Any]]) -> str:
    """Render the budgeting view of benchmark cells."""
    lines = [
        "| Config | Architecture | N | Generate median (s) | To host median (s) | "
        "Budget p95 (s) | Generate vectors/s | End-to-end vectors/s | Model params | "
        "Incremental peak VRAM | Host output |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        peak = cell["incremental_peak_vram_bytes"]
        peak_text = "n/a" if peak is None else f"{peak / (1024**2):.1f} MiB"
        params_text = f"{cell['model_param_bytes'] / (1024**2):.2f} MiB"
        host_text = f"{cell['host_output_bytes'] / (1024**2):.1f} MiB"
        budget = cell["generate_seconds"]["p95"] + cell["to_host_seconds"]["p95"]
        lines.append(
            f"| {cell['config']} | {cell['generator_type']} | "
            f"{cell['num_samples']:,} | {cell['generate_seconds']['median']:.6f} | "
            f"{cell['to_host_seconds']['median']:.6f} | {budget:.6f} | "
            f"{cell['generate_vectors_per_second']:.1f} | "
            f"{cell['end_to_end_vectors_per_second']:.1f} | {params_text} | "
            f"{peak_text} | {host_text} |"
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    first_config = yaml.safe_load(args.config[0].read_text(encoding="utf-8"))
    device = resolve_device(args.device or first_config["device"])
    # Fail on an unwritable output path now, not after the grid has run.
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        seed=args.seed,
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
        "schedule": "interleaved-rounds",
        "resident_generators": len(args.config),
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    non_finite = not all(
        math.isfinite(value)
        for cell in cells
        for phase in ("generate_seconds", "to_host_seconds")
        for value in cell[phase].values()
    )
    result = {
        "environment": environment,
        "cells": cells,
        "non_finite_timings": non_finite,
    }
    (args.output_dir / "generation_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generation_benchmark.md").write_text(
        format_markdown_table(cells), encoding="utf-8"
    )
    if non_finite:
        raise RuntimeError(
            "benchmark produced a non-finite timing; the run was still written to "
            f"{args.output_dir}"
        )


if __name__ == "__main__":
    main()
