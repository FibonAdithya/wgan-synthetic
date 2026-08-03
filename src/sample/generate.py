from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.device import resolve_device
from src.models.generator import build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic descriptors from trained generator.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = resolve_device(config["device"])
    model_cfg = config["model"]
    data_cfg = config["data"]

    generator = build_generator(
        model_cfg, output_dim=int(data_cfg["descriptor_dim"])
    ).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    print(f"Using device: {device}")

    out = []
    generated = 0
    generation_start = time.perf_counter()
    with torch.no_grad():
        while generated < args.num_samples:
            cur = min(args.batch_size, args.num_samples - generated)
            z = torch.randn(cur, int(model_cfg["latent_dim"]), device=device)
            x = generator(z)
            x = x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1.0e-8)
            x = x.cpu().numpy().astype(np.float32, copy=False)
            out.append(x)
            generated += cur
    generation_seconds = time.perf_counter() - generation_start

    synthetic = np.concatenate(out, axis=0)[: args.num_samples]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, synthetic)
    total_seconds = time.perf_counter() - start_time
    throughput = synthetic.shape[0] / max(generation_seconds, 1.0e-12)
    print(f"Saved {synthetic.shape[0]} synthetic vectors to {output_path}")
    print(
        "Generation timing: "
        f"compute={generation_seconds:.3f}s "
        f"total={total_seconds:.3f}s "
        f"throughput={throughput:.1f} vectors/s"
    )


if __name__ == "__main__":
    main()
