"""Does the linear-skip generator's hubness come from its trunk?

Across the logged evaluations of NYTimes v2 and v2c, holdout hubness skew
tracks the trunk's share of the output energy (Spearman -0.52 and -0.66
against skip share, measured 2026-09-15 from the committed run_metadata).
That is a correlation across training steps, where everything moves at
once. This probe isolates the mechanism on a fixed checkpoint: the same
latent draw is decoded once, then the trunk term is rescaled before the
skip term is added and the sum normalised, so the only thing that changes
between rows of the output table is how much of the trunk reaches the
sphere.

    x(s) = normalise(s * trunk(z_t) + skip(z_s))        s in --scales

`s = 0` is the skip term alone (a linear map of a Gaussian on the sphere)
and `trunk_only` is the trunk alone. If hubness rises monotonically with s
while the s = 0 row reads near the Gaussian's, the trunk-plus-skip
decomposition is the hub source; if it does not move with s, the hubs come
from somewhere else and the generator is not the lever.

Every set is measured with `ann_difficulty.compute` at the canonical
NYTimes conditions (20,000 rows, k 100, hub k 10, nlist 256, angular), the
same call the gate uses, plus effective rank and the median 1-NN distance
on the measured rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.ann_difficulty import compute, summary  # noqa: E402
from src.eval.eda.metrics import effective_rank  # noqa: E402
from src.models.generator import LinearSkipGenerator, build_generator  # noqa: E402

CANON = {"k": 100, "k_hub": 10, "nlist": 256, "max_rows": 20000, "seed": 42}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a checkpoint to probe; repeatable",
    )
    p.add_argument("--config", required=True, help="run_config.yaml beside them")
    p.add_argument("--real-path", required=True, help="cleaned, unit-norm real rows")
    p.add_argument("--output", required=True, help="JSON to write")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scales", default="0,0.25,0.5,1,2,4")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def normalise(x: torch.Tensor) -> np.ndarray:
    x = x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1.0e-8)
    return x.cpu().numpy().astype(np.float32, copy=False)


def measure(x: np.ndarray) -> dict[str, float | int | None]:
    m = compute(x, metric="angular", **CANON)
    out = dict(summary(m))
    out["median_nn1_distance"] = float(np.median(m.nearest_distance))
    out["effective_rank"] = effective_rank(x)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = yaml.safe_load(Path(args.config).read_text())
    scales = [float(s) for s in args.scales.split(",")]
    latent_dim = int(cfg["model"]["latent_dim"])
    out_dim = int(cfg["data"]["descriptor_dim"])

    real = np.load(args.real_path, mmap_mode="r")
    result: dict = {
        "config": args.config,
        "real_path": args.real_path,
        "num_samples": args.num_samples,
        "sample_seed": args.seed,
        "conditions": {**CANON, "metric": "angular"},
        "scales": scales,
        "real": measure(np.asarray(real, dtype=np.float32)),
        "checkpoints": {},
    }
    print(f"real: {json.dumps(result['real'])}", flush=True)

    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint wants NAME=PATH, got {spec!r}")
        gen = build_generator(cfg["model"], output_dim=out_dim).to(device)
        if not isinstance(gen, LinearSkipGenerator):
            raise SystemExit(f"{name}: generator_type is not linear_skip")
        state = torch.load(path, map_location=device)
        gen.load_state_dict(state["generator_state_dict"])
        gen.eval()

        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z = torch.randn(args.num_samples, latent_dim, generator=g).to(device)
        t = gen.trunk_latent_dim
        with torch.no_grad():
            trunk = gen.trunk(z[:, :t]).float()
            skip = gen.skip(z[:, t:]).float()
        trunk_energy = float((trunk * trunk).sum(dim=1).mean())
        skip_energy = float((skip * skip).sum(dim=1).mean())
        rows: dict[str, dict] = {}
        for s in scales:
            rows[f"scale_{s:g}"] = measure(normalise(s * trunk + skip))
            print(f"{name} scale {s:g}: {json.dumps(rows[f'scale_{s:g}'])}", flush=True)
        rows["trunk_only"] = measure(normalise(trunk))
        print(f"{name} trunk_only: {json.dumps(rows['trunk_only'])}", flush=True)
        result["checkpoints"][name] = {
            "path": path,
            "step": int(state.get("step", -1)),
            "trunk_energy": trunk_energy,
            "skip_energy": skip_energy,
            "skip_share": skip_energy / (trunk_energy + skip_energy),
            "rows": rows,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
