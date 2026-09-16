"""Measure `lid_reg`'s actual magnitude, so `lid_reg_alpha` can be sized.

`lid_reg_alpha` cannot be carried over from `distance_reg_alpha`. That penalty
is a single scalar, `|dist_real - dist_fake|`; `log_ratio_penalty` is an L1
*sum* over `lid_reg_k - 1` components -- 19 of them at the default k -- so it
sits on a different scale entirely and grows with k.

This probe measures three numbers at the config's actual k:

  floor   real vs real. Two independent draws from the same distribution, so
          this is the penalty's noise floor -- the value alpha multiplies even
          when the generator is perfect. Anything at or below it is not signal.
  v2      the trained v2 generator vs real. The gap a generator that is
          genuinely wrong about local structure produces; what the penalty has
          to work with.
  v3init  untrained v3 vs real. The gap at step 0, which is what alpha
          actually multiplies for the first few hundred steps.

alpha is then sized so the penalty contributes a stated fraction of |adv_loss|
at launch rather than dominating or vanishing.

Run on the GPU box, where data/sift_base.npy lives:

    /venv/main/bin/python tools/probes/lid_reg_scale_probe.py \
        --config configs/sift/v4.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.device import resolve_device
from src.models.generator import build_generator
from src.train.log_ratio import LogRatioTarget, batch_log_ratio_profile, log_ratio_penalty


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1e-8)


def gap(sampler, real_pool, k, max_points, batch, trials, device, seed):
    """Mean penalty over `trials` independent batch pairs, on a fresh target."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    target = LogRatioTarget()
    vals = []
    for _ in range(trials):
        ri = torch.randint(0, real_pool.shape[0], (batch,), generator=g)
        real = real_pool[ri].to(device)
        fake = sampler(batch)
        with torch.no_grad():
            vals.append(
                float(log_ratio_penalty(fake, real, k=k, max_points=max_points, target=target))
            )
    # Drop the first few: the EMA target is still converging onto the real
    # profile from its first observation, so early values understate the gap.
    warm = min(20, len(vals) // 4)
    return float(np.mean(vals[warm:])), float(np.std(vals[warm:]))


def resolve_probe_device(cfg: dict, device_arg: str | None) -> torch.device:
    """The device to run on: `--device` if given, else the config's.

    `resolve_device` rather than `torch.device` because every NYTimes config
    says `device: auto`, which `torch.device` refuses. `strict` stays off:
    this probe is short and read-only, so making it demand a GPU claim would
    be friction, and the `--device cpu` path is what keeps a cpu-lane job off
    the card.
    """
    return resolve_device(device_arg if device_arg is not None else cfg["device"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--v2-checkpoint", default=None, help="Trained v2 generator .pt")
    ap.add_argument("--v2-config", default="configs/sift/v2.yaml")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--target-fraction", type=float, default=0.05,
                    help="Share of |adv_loss| the penalty should contribute at launch.")
    ap.add_argument("--device", default=None,
                    help="Override the config's device. Pass cpu for a cpu-lane job.")
    ap.add_argument("--output", default=None, help="Write the result JSON here.")
    ap.add_argument("--adv-loss", type=float, default=None,
                    help="Typical |adv_loss| from an existing run's metrics.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).open())
    device = resolve_probe_device(cfg, args.device)
    print(f"device: {device}")
    t = cfg["training"]
    k, max_points, batch = int(t["lid_reg_k"]), int(t["lid_reg_max_points"]), int(t["batch_size"])
    dim = int(cfg["data"]["descriptor_dim"])

    raw = np.load(cfg["data"]["real_path"], mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = rng.choice(raw.shape[0], size=min(200000, raw.shape[0]), replace=False)
    real_pool = l2_normalize(torch.from_numpy(np.asarray(raw[np.sort(idx)], dtype=np.float32)))
    print(f"real pool {tuple(real_pool.shape)}  k={k}  max_points={max_points}  batch={batch}")

    out = {"k": k, "max_points": max_points, "batch": batch, "trials": args.trials}

    # floor: real vs real
    pool_a, pool_b = real_pool[: len(real_pool) // 2], real_pool[len(real_pool) // 2 :]
    gb = torch.Generator(device="cpu").manual_seed(1)
    out["floor"], out["floor_std"] = gap(
        lambda n: pool_b[torch.randint(0, pool_b.shape[0], (n,), generator=gb)].to(device),
        pool_a, k, max_points, batch, args.trials, device, seed=1)
    print(f"floor  (real vs real) : {out['floor']:.4f} +/- {out['floor_std']:.4f}")

    # v3 at init
    torch.manual_seed(0)
    v3 = build_generator(cfg["model"], dim).to(device).eval()
    latent = int(cfg["model"]["latent_dim"])
    with torch.no_grad():
        out["v3init"], out["v3init_std"] = gap(
            lambda n: l2_normalize(v3(torch.randn(n, latent, device=device))),
            real_pool, k, max_points, batch, args.trials, device, seed=2)
    print(f"v3init (untrained v3) : {out['v3init']:.4f} +/- {out['v3init_std']:.4f}")

    # trained v2
    if args.v2_checkpoint:
        v2cfg = yaml.safe_load(Path(args.v2_config).open())
        v2 = build_generator(v2cfg["model"], dim).to(device)
        ck = torch.load(args.v2_checkpoint, map_location=device, weights_only=False)
        v2.load_state_dict(ck["generator_state_dict"])
        v2.eval()
        v2latent = int(v2cfg["model"]["latent_dim"])
        with torch.no_grad():
            out["v2"], out["v2_std"] = gap(
                lambda n: l2_normalize(v2(torch.randn(n, v2latent, device=device))),
                real_pool, k, max_points, batch, args.trials, device, seed=3)
        print(f"v2     (trained v2)   : {out['v2']:.4f} +/- {out['v2_std']:.4f}")

    signal = out.get("v2", out["v3init"])
    print(f"\nsignal-to-floor ratio : {signal / max(out['floor'], 1e-9):.1f}x")
    if args.adv_loss:
        alpha = args.target_fraction * abs(args.adv_loss) / max(signal, 1e-9)
        out["suggested_alpha"] = alpha
        print(f"suggested lid_reg_alpha: {alpha:.4g}"
              f"  ({args.target_fraction:.0%} of |adv_loss|={args.adv_loss:.3g} at gap {signal:.4f})")
    print("\n" + json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
