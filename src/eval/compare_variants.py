"""Overlay every trained variant of one dataset family on its real data.

src.eval.eda_report can already overlay any number of synthetic sets; this
drives it across a family's named variants so the comparison does not have to
be retyped. Each variant is one config delta from the one before it, so a
difference visible in the report attributes to a single cause.

Which family is selected with `--dataset`; the ladders live in `LADDERS`
below. Everything else here is family-agnostic, including the inversion of a
whitened training space -- see `invert_samples`.

A variant whose checkpoint is not on this machine is skipped with a message
rather than aborting: checkpoints usually live on the training box, and a
partial comparison is still worth reading.

Example:
    python -m src.eval.compare_variants --dataset sift \
        --real-path data/sift_base.npy \
        --output-dir runs/eda_variants
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import PreprocessState, invert_preprocess
from src.eval import eda_report
from src.eval.evaluate_distribution import get_device, load_generator
from src.train.train_wgan_gp import sample_generator


@dataclass(frozen=True)
class Variant:
    """One named model variant and where its trained artifacts live."""

    name: str
    config_path: str
    run_dir: str


# Ordered so the report legend reads as a progression. Each entry is one
# config delta from the previous. Run directories point at the longest run of
# each variant that exists; v0 was never taken to 100k steps.
SIFT_VARIANTS: tuple[Variant, ...] = (
    Variant("v0", "configs/sift/v0.yaml", "runs/long_baseline"),
    Variant("v1", "configs/sift/v1.yaml", "runs/x100k_ema_only"),
    Variant("v1_5", "configs/sift/v1_5.yaml", "runs/x100k_improved"),
    Variant("v2", "configs/sift/v2.yaml", "runs/x100k_sparse_clamp4"),
)

# v2 trains in a PCA-whitened space, so its samples only mean anything after
# `invert_samples` maps them back -- which is why that step lives in
# `generate_samples` rather than in a deep-specific sampler.
DEEP_VARIANTS: tuple[Variant, ...] = (
    Variant("v0", "configs/deep/v0.yaml", "runs/deep/v0"),
    Variant("v1", "configs/deep/v1.yaml", "runs/deep/v1"),
    Variant("v2", "configs/deep/v2.yaml", "runs/deep/v2"),
)

LADDERS: dict[str, tuple[Variant, ...]] = {
    "sift": SIFT_VARIANTS,
    "deep": DEEP_VARIANTS,
}

# Retained under its historical name: `VARIANTS` is what the SIFT-era callers
# and docs refer to.
VARIANTS: tuple[Variant, ...] = SIFT_VARIANTS

CHECKPOINT_NAME = "best_generator.pt"
RUN_CONFIG_NAME = "run_config.yaml"
RUN_METADATA_NAME = "run_metadata.json"


def resolve_variants(
    variants: Sequence[Variant], root: Path
) -> tuple[list[Variant], list[tuple[Variant, str]]]:
    """Split variants into those whose artifacts are on disk and those not.

    The run config is required alongside the checkpoint because the generator
    architecture is rebuilt from it -- the checkpoint records which weights it
    holds ("live"/"ema") but not which generator produced them.

    A run whose config asks for centering or whitening is checked further, by
    `_inversion_blocker`: its samples are only meaningful once the fitted
    transform has been undone, and several things can make that impossible.
    Checked here rather than at sampling time so every such run is reported
    alongside the other skips, before any of the earlier variants in the loop
    have generated hundreds of thousands of vectors.
    """
    found: list[Variant] = []
    skipped: list[tuple[Variant, str]] = []
    for variant in variants:
        run_dir = root / variant.run_dir
        checkpoint = run_dir / CHECKPOINT_NAME
        run_config = run_dir / RUN_CONFIG_NAME
        if not run_dir.is_dir():
            skipped.append((variant, f"no run directory at {run_dir}"))
        elif not checkpoint.exists():
            skipped.append((variant, f"no {CHECKPOINT_NAME} in {run_dir}"))
        elif not run_config.exists():
            skipped.append((variant, f"no {RUN_CONFIG_NAME} in {run_dir}"))
        elif (blocker := _inversion_blocker(run_config, run_dir)) is not None:
            skipped.append((variant, blocker))
        else:
            found.append(variant)
    return found, skipped


def _needs_inversion(run_config: Path) -> bool:
    """True when a run's preprocessing has to be undone at sample time.

    Reads the run config rather than the variant's checked-in config: what
    matters is the transform the run was actually trained under.
    """
    config = yaml.safe_load(run_config.read_text(encoding="utf-8")) or {}
    preprocess = (config.get("data") or {}).get("preprocess") or {}
    return bool(preprocess.get("center")) or bool(preprocess.get("whiten"))


def _inversion_blocker(run_config: Path, run_dir: Path) -> str | None:
    """Why this run's samples could not be returned to original coordinates.

    None when there is nothing to undo, or when the fitted transform is on disk
    and invertible -- so a run that only L2-normalizes never reaches any of the
    checks below, and the whole SIFT ladder resolves as it always did.

    Deliberately loads the state rather than just stat-ing run_metadata.json.
    A file that exists but carries no `preprocess_state` key, or one truncated
    mid-write, both leave `load_preprocess_state` returning None -- which
    `invert_samples` reads as "nothing was fitted" and passes the samples
    straight through, writing them out in whitened coordinates with no error
    raised anywhere. Existence is not the property that matters; being able to
    reconstruct the transform is.

    The centering rejection is the same rule `invert_samples` enforces, applied
    early. It is asked of the fitted state rather than the config because the
    state is what inversion actually consumes.
    """
    if not _needs_inversion(run_config):
        return None

    metadata = Path(run_dir) / RUN_METADATA_NAME
    if not metadata.exists():
        return (
            f"no {RUN_METADATA_NAME} in {run_dir}, but its config centers or "
            "whitens -- the fitted transform is recorded there and samples "
            "cannot be returned to original coordinates without it"
        )
    try:
        state = load_preprocess_state(run_dir)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return (
            f"{RUN_METADATA_NAME} in {run_dir} could not be read back into a "
            f"transform ({exc}), but its config centers or whitens -- samples "
            "cannot be returned to original coordinates without it"
        )
    if state is None:
        return (
            f"{RUN_METADATA_NAME} in {run_dir} records no preprocess_state, "
            "but its config centers or whitens -- samples cannot be returned "
            "to original coordinates without the fitted transform"
        )
    if state.mean is not None and state.config.l2_normalize:
        return (
            f"this run was fitted with both centering and l2_normalize, which "
            f"{__name__}.invert_samples refuses: sample_generator "
            "L2-normalizes its raw output, and inverting a centered transform "
            "afterwards yields systematically wrong directions with nothing "
            "downstream to flag them"
        )
    return None


def load_preprocess_state(run_dir: Path) -> PreprocessState | None:
    """Read the transform `train` fitted, or None if this run recorded none.

    None is the ordinary case for the SIFT ladder and for any run predating
    run_metadata.json; those runs preprocess with L2 normalization alone,
    which is not invertible and does not need to be.

    Callers that require the transform must treat None as a failure rather than
    as "no transform needed" -- see `_inversion_blocker`, which is where that
    distinction is drawn.
    """
    path = Path(run_dir) / RUN_METADATA_NAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "preprocess_state" not in payload:
        return None
    return PreprocessState.from_serializable(payload["preprocess_state"])


def invert_samples(x: np.ndarray, state: PreprocessState | None) -> np.ndarray:
    """Map generator output back to the corpus's original coordinates.

    A no-op when the run fitted no centering or whitening, which keeps this
    safe to call unconditionally for every family.

    Refuses centering combined with L2 normalization: `sample_generator`
    L2-normalizes its raw output, and `invert_preprocess` only recovers
    directions exactly when no mean was subtracted (see its docstring). With
    both on, the result is wrong in a way nothing downstream would flag, so
    this raises instead of returning it.
    """
    if state is None:
        return x
    if state.mean is not None and state.config.l2_normalize:
        raise ValueError(
            "This run was fitted with both centering and l2_normalize. "
            "sample_generator L2-normalizes its raw output, and "
            "invert_preprocess only exactly recovers directions when there is "
            "no centering step: with a mean subtracted, its relative "
            "contribution varies per generated vector, so re-normalizing "
            "after inversion yields systematically wrong directions with no "
            "error otherwise. Retrain with `center: false`, or compare with a "
            "metric that does not depend on angular exactness."
        )
    if state.mean is None and state.whitening_matrix is None:
        return x
    return invert_preprocess(x, state)


def variant_seed(base_seed: int, name: str) -> int:
    """Derive a per-variant seed that does not depend on run order.

    A single seed set before the sampling loop would leave each variant's
    latents dependent on how many variants preceded it -- and variants get
    skipped whenever their checkpoint is not on this machine, so v2's samples
    would differ between a machine holding all four checkpoints and one
    holding only v2. Keying off the variant name instead makes a variant's
    samples reproducible from `--seed` alone.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def generate_samples(
    variant: Variant,
    root: Path,
    num_samples: int,
    batch_size: int,
    out_dir: Path,
    seed: int,
) -> Path:
    """Sample a variant's best checkpoint to an .npy file, and return its path.

    Samples land in the corpus's original coordinates: if the run trained in a
    whitened space, `invert_samples` maps them back before they are written.
    """
    run_dir = root / variant.run_dir
    config = yaml.safe_load((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))
    device = get_device(config["device"])
    generator = load_generator(config, run_dir / CHECKPOINT_NAME, device)
    torch.manual_seed(variant_seed(seed, variant.name))
    x = sample_generator(
        generator,
        num_samples=num_samples,
        latent_dim=int(config["model"]["latent_dim"]),
        batch_size=batch_size,
        device=device,
    )
    x = invert_samples(x, load_preprocess_state(run_dir))
    out_path = out_dir / f"{variant.name}.npy"
    np.save(out_path, x)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sift",
        choices=sorted(LADDERS),
        help="Which family's ladder to compare. Defaults to sift.",
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root that variant config and run paths resolve against.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Vectors to draw from each variant. Defaults to --max-vectors, since "
            "the report subsamples to that anyway and anything beyond it is "
            "generated and then discarded. Raise it only to keep a larger .npy "
            "under <output-dir>/samples for separate use."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-vectors", type=int, default=50000)
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument(
        "--ann-k",
        type=int,
        default=eda_report.ANN_K_DEFAULT,
        help="Neighbours per query for the LID and relative-contrast panels.",
    )
    parser.add_argument(
        "--ann-hub-k",
        type=int,
        default=eda_report.ANN_HUB_K_DEFAULT,
        help="Neighbour depth for the k-occurrence count behind the hubness panel.",
    )
    parser.add_argument(
        "--ann-max-rows",
        type=int,
        default=eda_report.ANN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for every ANN-difficulty metric. LID, "
            "contrast and hubness all drift with sample count, so every set "
            "must be cut to the same size."
        ),
    )
    parser.add_argument(
        "--knn-max-rows",
        type=int,
        default=eda_report.KNN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for the within-set k-NN distance panel, "
            "which is not an ANN-difficulty panel."
        ),
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=eda_report.IVF_NLIST_DEFAULT,
        help="Cluster count for the IVF cell-balance panel.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top-divergent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--glyph-samples", type=int, default=eda_report.GLYPH_SAMPLES_DEFAULT
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def build_report_args(args: argparse.Namespace, specs: list[str]) -> argparse.Namespace:
    """Build the Namespace `eda_report.run` expects from our own parsed args.

    Field-for-field parity with `eda_report.parse_args` is load-bearing: if
    `eda_report` gains a required argument and this Namespace is not updated
    to match, sampling hundreds of thousands of vectors will succeed before
    the mismatch surfaces as a runtime `AttributeError`. See
    `tests/test_compare_variants.py::test_report_args_match_eda_report_fields`.
    """
    return argparse.Namespace(
        real_path=args.real_path,
        real_format=args.real_format,
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=args.output_dir,
        preprocess="l2",
        max_vectors=args.max_vectors,
        num_pairs=args.num_pairs,
        knn=args.knn,
        ann_k=args.ann_k,
        ann_hub_k=args.ann_hub_k,
        ann_max_rows=args.ann_max_rows,
        knn_max_rows=args.knn_max_rows,
        ivf_nlist=args.ivf_nlist,
        bins=args.bins,
        top_divergent=args.top_divergent,
        seed=args.seed,
        glyph_samples=args.glyph_samples,
        no_png=args.no_png,
        plotlyjs=args.plotlyjs,
    )


def main() -> None:
    args = parse_args()
    num_samples = args.num_samples if args.num_samples is not None else args.max_vectors
    root = Path(args.root)
    out_dir = Path(args.output_dir)

    # Resolve before creating anything, so an aborted run leaves no empty tree.
    found, skipped = resolve_variants(LADDERS[args.dataset], root)
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        raise SystemExit(
            f"No {args.dataset} variant has both a checkpoint and a run config "
            "on this machine. Copy them from the training box, or pass --root "
            "at the tree holding them."
        )

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for variant in found:
        print(f"sampling {variant.name} from {variant.run_dir}")
        path = generate_samples(
            variant,
            root,
            num_samples,
            args.batch_size,
            samples_dir,
            seed=args.seed,
        )
        specs.append(f"{variant.name}={path}")

    report_args = build_report_args(args, specs)
    report_path = eda_report.run(report_args)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
