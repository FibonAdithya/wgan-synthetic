"""Generate the DEEP seed/alpha sweep configs from the committed ladder rungs.

The sweep exists to answer the two questions issues #20 and #21 raise about
the DEEP ladder:

  1. The three shipped rungs rest on one seed each, and the two draws that do
     exist disagree by more than the rungs differ. Three seeds per rung give
     a noise floor, without which every gate band in `gates/deep.yaml` would
     be fitted to noise.
  2. `spectrum_reg_alpha: 0.1` contributes ~1.2% of the generator gradient at
     its strongest, and v1 moved effective rank by 0.02 against a 1.96 gap to
     real. Two probes at 1.0 and 5.0 test whether the term binds *at all*
     before a full alpha grid is worth running.

Each generated cell is its own file because `train_wgan_gp` takes only
`--config` -- there is no `--seed` or `--output-dir` override -- and because
the runner checks out a pinned commit, so a cell that is not committed cannot
be run reproducibly.

Cells derive from the ladder rungs by changing only `seed`, `output_dir` and,
for the alpha probes, `training.spectrum_reg_alpha`. `tests/test_deep_sweep_configs.py`
machine-checks that invariant, which is what makes a difference between two
cells attributable to the thing being swept.

    python scripts/make_deep_sweep_configs.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "deep"
SWEEP_DIR = CONFIG_DIR / "sweep"

SEEDS = (42, 43, 44)
RUNGS = ("v0", "v1", "v2")
ALPHA_PROBES = (1.0, 5.0)
ALPHA_PROBE_SEED = 42


def _alpha_tag(alpha: float) -> str:
    """1.0 -> '1', 0.5 -> '0p5'. Keeps the filename shell-safe."""
    return f"{alpha:g}".replace(".", "p")


def cells() -> list[tuple[str, str, dict[str, Any]]]:
    """(name, base rung, overrides) for every cell in the sweep."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for rung in RUNGS:
        for seed in SEEDS:
            out.append((f"{rung}_s{seed}", rung, {"seed": seed}))
    for alpha in ALPHA_PROBES:
        name = f"v1_alpha{_alpha_tag(alpha)}_s{ALPHA_PROBE_SEED}"
        out.append(
            (
                name,
                "v1",
                {"seed": ALPHA_PROBE_SEED, "training.spectrum_reg_alpha": alpha},
            )
        )
    return out


def _apply(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"{dotted} is not a key in the base config")
    node[parts[-1]] = value


def build(name: str, rung: str, overrides: dict[str, Any]) -> dict[str, Any]:
    base = yaml.safe_load((CONFIG_DIR / f"{rung}.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    for dotted, value in overrides.items():
        _apply(config, dotted, value)
    config["output_dir"] = f"runs/deep/sweep/{name}"
    return config


def header(name: str, rung: str, overrides: dict[str, Any]) -> str:
    deltas = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    return (
        f"# DEEP sweep cell `{name}` -- GENERATED, do not edit by hand.\n"
        f"#\n"
        f"# Regenerate with: python scripts/make_deep_sweep_configs.py\n"
        f"# Base rung: configs/deep/{rung}.yaml\n"
        f"# Deltas from that rung: {deltas} (plus output_dir)\n"
        f"#\n"
        f"# Every other key is inherited from the base rung, which is what makes\n"
        f"# a difference between two cells attributable to the swept variable.\n"
    )


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rung, overrides in cells():
        config = build(name, rung, overrides)
        body = yaml.safe_dump(config, sort_keys=False)
        (SWEEP_DIR / f"{name}.yaml").write_text(
            header(name, rung, overrides) + "\n" + body, encoding="utf-8"
        )
        written.append(name)
    print(f"wrote {len(written)} configs to {SWEEP_DIR.relative_to(REPO_ROOT)}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
