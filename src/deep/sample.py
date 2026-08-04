"""Sample a deep variant's checkpoint back into original DEEP coordinates.

src/sample/generate.py cannot be reused here: it L2-normalizes unconditionally
and never inverts the preprocessing transform, so a whitened run would silently
emit vectors in whitened space and be compared against real DEEP in original
space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.sift1m_dataset import PreprocessState
from src.deep.dataset import invert_preprocess
from src.models.generator import build_generator
from src.train.train_wgan_gp import get_device, sample_generator

CHECKPOINT_NAME = "best_generator.pt"
RUN_CONFIG_NAME = "run_config.yaml"
RUN_METADATA_NAME = "run_metadata.json"


def load_preprocess_state(run_dir: Path) -> PreprocessState:
    """Read the transform train() fitted, so sampling can undo it."""
    path = Path(run_dir) / RUN_METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {RUN_METADATA_NAME} in {run_dir}. It records the preprocessing "
            "state, without which samples cannot be returned to original "
            "coordinates. Copy it from the training box alongside the checkpoint."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PreprocessState.from_serializable(payload["preprocess_state"])


def sample_variant(
    run_dir: Path,
    num_samples: int,
    *,
    batch_size: int = 4096,
    seed: int = 42,
    checkpoint_name: str = CHECKPOINT_NAME,
) -> np.ndarray:
    """Draw `num_samples` vectors in original DEEP coordinates.

    Note: this calls `torch.manual_seed(seed)`, mutating the *global* torch
    RNG state with no save/restore of what it was beforehand. A caller
    looping over several variants (e.g. a report generator) with the same
    `seed` each iteration gets reproducible draws per call, but should not
    assume the global RNG is left as it found it.
    """
    run_dir = Path(run_dir)
    state = load_preprocess_state(run_dir)
    if state.mean is not None and state.config.l2_normalize:
        raise ValueError(
            f"{run_dir} was fitted with both centering and l2_normalize. "
            "sample_generator L2-normalizes its raw output, and "
            "invert_preprocess only exactly recovers directions when there "
            "is no centering step: with a mean subtracted, its relative "
            "contribution to `(x - mean) / c` varies per generated vector "
            "(c differs per row), so re-normalizing after inversion yields "
            "systematically wrong directions with no error otherwise. "
            "Retrain this variant with `center: false`, or compare it with "
            "a metric that does not depend on angular exactness."
        )
    config = yaml.safe_load((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))

    device = get_device(config["device"])
    model_cfg = config["model"]
    descriptor_dim = int(config["data"]["descriptor_dim"])

    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    torch.manual_seed(seed)
    x = sample_generator(
        generator,
        num_samples=num_samples,
        latent_dim=int(model_cfg["latent_dim"]),
        batch_size=batch_size,
        device=device,
    )
    return invert_preprocess(x, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = sample_variant(
        Path(args.run_dir),
        args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x)
    print(f"Saved {x.shape[0]} vectors to {out}")


if __name__ == "__main__":
    main()
