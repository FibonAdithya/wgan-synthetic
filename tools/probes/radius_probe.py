"""Is v3's geometry a function of the shared radius, or did the trunk collapse?

Across NYTimes v3's 30 logged evaluations the radius climbs monotonically,
0.976 at step 1,000 to 1.448 at step 30,000, and never settles; every gate
statistic is a smooth unimodal function of it, best near r = 1.20 to 1.24
(measured 2026-09-16 from the committed `docs/results/nytimes-v3/`
run_metadata). That is a correlation across training steps, where the radius
and the weights move together. This probe breaks the confound on frozen
checkpoints: the same latent draw is decoded once per checkpoint, and the
shared angle is then overridden before the two components are recombined.

    u, t = generator.components(z)          # unit, orthogonal
    x(r) = cos(r) u + sin(r) t              # r in --radii, plus the learned r

Because `u` and `t` are computed once and reused, every row of a checkpoint's
table is the *same* latent draw seen at a different angle: the comparison
across r is paired and exact, and the only thing that changes between rows is
the angle.

The design is crossed, not a one-way sweep. Reading the step-9,000 checkpoint
(learned r = 1.196, the run's best evaluation) against the step-30,000 one
(learned r = 1.448, the drifted end) at the *same* overridden radii separates
two hypotheses that the training trajectory cannot:

  * If step 30,000 at r = 1.22 recovers step 9,000's geometry, and step 9,000
    pushed to r = 1.448 degrades to step-30,000-like numbers, then the radius
    is the whole story and freezing it is the next rung.
  * If step 30,000 stays near LID 24 whatever the angle, the trunk collapsed
    underneath and the radius is a symptom, not the cause; freezing it buys
    nothing and the rung needs a rank or critic change.

The `trunk_only` (x = u) and `tangent_only` (x = t) rows read the two
components on their own, so a collapse in the trunk's direction distribution
shows up directly rather than being inferred from the mixture.

Every set is measured with `ann_difficulty.compute` at the canonical NYTimes
conditions (20,000 rows, k 100, hub k 10, nlist 256, angular), the same call
the gate uses, plus effective rank and the median 1-NN distance.

Note on reproducing the committed v3 numbers: `src/sample/generate.py` seeds
the *global* torch RNG and drew v3's samples on CUDA, so a CPU run cannot
reproduce that latent draw bit-for-bit. The `learned_r` rows here are a fresh
draw from the same weights; they should agree with the committed canonical
numbers within draw noise, and that agreement is the end-to-end check on this
tool's pipeline. Exactness is only claimed for the paired comparison across
radii within one checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.ann_difficulty import compute, summary  # noqa: E402
from src.eval.eda.metrics import effective_rank  # noqa: E402
from src.models.generator import SphericalGenerator, build_generator  # noqa: E402

CANON = {"k": 100, "k_hub": 10, "nlist": 256, "max_rows": 20000, "seed": 42}

# The invariants the recombination relies on. Violating either means the row
# tables are measuring something other than "the same draw at a new angle".
UNIT_TOL = 1.0e-5
ORTHO_TOL = 1.0e-5


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
    p.add_argument(
        "--radii",
        default="1.10,1.20,1.25,1.30,1.40,1.448",
        help="shared angles to decode at, in radians; the learned r is always added",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def measure(x: np.ndarray) -> dict[str, float | int | None]:
    m = compute(x, metric="angular", **CANON)
    out = dict(summary(m))
    out["median_nn1_distance"] = float(np.median(m.nearest_distance))
    out["effective_rank"] = effective_rank(x)
    return out


def check_components(name: str, u: torch.Tensor, t: torch.Tensor) -> dict[str, float]:
    """Confirm u and t are unit and orthogonal before anything is built on them."""
    u_err = float((torch.linalg.vector_norm(u, dim=1) - 1.0).abs().max())
    t_err = float((torch.linalg.vector_norm(t, dim=1) - 1.0).abs().max())
    dot = float((u * t).sum(dim=1).abs().max())
    if max(u_err, t_err) > UNIT_TOL:
        raise SystemExit(f"{name}: components are not unit norm (u {u_err}, t {t_err})")
    if dot > ORTHO_TOL:
        raise SystemExit(f"{name}: components are not orthogonal (max |u.t| {dot})")
    return {"u_norm_err": u_err, "t_norm_err": t_err, "max_abs_dot": dot}


def decode(u: torch.Tensor, t: torch.Tensor, r: float) -> np.ndarray:
    """x = cos(r) u + sin(r) t. Unit norm follows from u, t unit and orthogonal,
    so this deliberately does not renormalise -- a norm that drifts is a bug in
    the components, and the caller checks for it."""
    x = math.cos(r) * u + math.sin(r) * t
    norm_err = float((torch.linalg.vector_norm(x, dim=1) - 1.0).abs().max())
    if norm_err > UNIT_TOL:
        raise SystemExit(f"recombination at r={r} left norms off by {norm_err}")
    return x.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = yaml.safe_load(Path(args.config).read_text())
    radii = [float(s) for s in args.radii.split(",")]
    latent_dim = int(cfg["model"]["latent_dim"])
    out_dim = int(cfg["data"]["descriptor_dim"])

    real = np.load(args.real_path, mmap_mode="r")
    result: dict = {
        "config": args.config,
        "real_path": args.real_path,
        "num_samples": args.num_samples,
        "sample_seed": args.seed,
        "conditions": {**CANON, "metric": "angular"},
        "radii": radii,
        "real": measure(np.asarray(real, dtype=np.float32)),
        "checkpoints": {},
    }
    print(f"real: {json.dumps(result['real'])}", flush=True)

    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint wants NAME=PATH, got {spec!r}")
        gen = build_generator(cfg["model"], output_dim=out_dim).to(device)
        if not isinstance(gen, SphericalGenerator):
            raise SystemExit(f"{name}: generator_type is not spherical")
        state = torch.load(path, map_location=device)
        gen.load_state_dict(state["generator_state_dict"])
        gen.eval()
        with torch.no_grad():
            learned_r = float(gen.radius)

        # One draw, reused at every angle, so the rows below are paired.
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z = torch.randn(args.num_samples, latent_dim, generator=g).to(device)
        with torch.no_grad():
            u, t = gen.components(z)
        u = u.float()
        t = t.float()
        invariants = check_components(name, u, t)

        rows: dict[str, dict] = {}
        # The learned angle first: this is the row that ties back to the
        # committed v3 numbers.
        rows["learned_r"] = measure(decode(u, t, learned_r))
        print(
            f"{name} learned_r {learned_r:.4f}: {json.dumps(rows['learned_r'])}",
            flush=True,
        )
        for r in radii:
            key = f"r_{r:g}"
            rows[key] = measure(decode(u, t, r))
            print(f"{name} {key}: {json.dumps(rows[key])}", flush=True)
        # The components alone. r = 0 and r = pi/2 exactly, which the band
        # forbids during training but which read the two parts on their own.
        rows["trunk_only"] = measure(u.cpu().numpy().astype(np.float32, copy=False))
        print(f"{name} trunk_only: {json.dumps(rows['trunk_only'])}", flush=True)
        rows["tangent_only"] = measure(t.cpu().numpy().astype(np.float32, copy=False))
        print(f"{name} tangent_only: {json.dumps(rows['tangent_only'])}", flush=True)

        result["checkpoints"][name] = {
            "path": path,
            "step": int(state.get("step", -1)),
            "learned_radius": learned_r,
            "radius_min": gen.radius_min,
            "radius_max": gen.radius_max,
            "direction_effective_rank": effective_rank(
                u.cpu().numpy().astype(np.float32, copy=False)
            ),
            "tangent_effective_rank": effective_rank(
                t.cpu().numpy().astype(np.float32, copy=False)
            ),
            "invariants": invariants,
            "rows": rows,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
