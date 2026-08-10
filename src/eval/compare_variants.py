"""Overlay every trained variant of one dataset family on its real data.

src.eval.eda.pipeline can already overlay any number of synthetic sets; this
drives it across a family's named variants so the comparison does not have to
be retyped. Each variant is one config delta from the one before it, so a
difference visible in the report attributes to a single cause.

Which family is selected with `--dataset`. Everything else here is
family-agnostic, including the inversion of a whitened training space -- see
`invert_samples`.

Which variants a family contains is a manifest, not a literal in this file:
`--dataset <name>` reads `configs/eval/<name>.yaml`, and `--variants-manifest`
overrides that with any path. `runs/` is gitignored, so the run directories a
manifest names exist only on the machine that trained them, and anyone else
has to be able to say where their own runs live without editing source.

A variant whose artifacts are missing aborts the run before any sampling
starts, naming the path and what would produce it. Pass `--allow-missing` to
fall back to the older behaviour of skipping it with a message and reporting
on whatever is present -- checkpoints usually live on the training box, and a
partial comparison is still worth reading once you have decided it is partial
on purpose.

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
from src.eval.ann_difficulty import METRICS
from src.eval.eda import config as eda_config
from src.eval.eda import pipeline
from src.eval.evaluate_distribution import get_device, load_generator
from src.train.train_wgan_gp import sample_generator


@dataclass(frozen=True)
class Variant:
    """One named model variant and where its trained artifacts live."""

    name: str
    config_path: str
    run_dir: str


CHECKPOINT_NAME = "best_generator.pt"
RUN_CONFIG_NAME = "run_config.yaml"
RUN_METADATA_NAME = "run_metadata.json"

# Repo root, so the default manifest is found regardless of the working
# directory the module is imported from. src/eval/compare_variants.py -> repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = "configs/eval"
DEFAULT_DATASET = "sift"
DEFAULT_MANIFEST = f"{MANIFEST_DIR}/{DEFAULT_DATASET}.yaml"


def manifest_for_dataset(dataset: str) -> Path:
    """Where a family's variant manifest lives.

    One manifest per family rather than one file listing every family: the
    ladders are numbered independently per family and are edited by whoever
    owns that family, so a shared file would make two agents contend for one
    set of lines.
    """
    return REPO_ROOT / MANIFEST_DIR / f"{dataset}.yaml"


def known_datasets() -> tuple[str, ...]:
    """Family names that ship a manifest, for `--dataset`'s choices.

    Read off disk rather than hard-coded so adding `configs/eval/<name>.yaml`
    is the whole of adding a family here -- the literal list this replaces is
    exactly what went stale when DEEP arrived.
    """
    directory = REPO_ROOT / MANIFEST_DIR
    if not directory.is_dir():
        return (DEFAULT_DATASET,)
    return tuple(sorted(p.stem for p in directory.glob("*.yaml")))


def load_variants(path: Path) -> tuple[Variant, ...]:
    """Read a variant manifest, rejecting anything a later stage would trip on.

    Validation is strict and up front because every failure mode here costs a
    caller the whole sampling pass otherwise: a duplicate name would have two
    variants overwrite each other's `<name>.npy`, and a missing key would
    surface as a `KeyError` several hundred thousand vectors in.
    """
    if not path.is_file():
        raise SystemExit(
            f"no variant manifest at {path}. Pass --variants-manifest, or "
            f"restore the default one at {DEFAULT_MANIFEST}."
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("variants"), list):
        raise SystemExit(
            f"{path} must be a YAML mapping with a 'variants' list; see "
            f"{DEFAULT_MANIFEST} for the shape."
        )
    entries = doc["variants"]
    if not entries:
        raise SystemExit(f"{path} lists no variants; there is nothing to compare.")

    variants: list[Variant] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: variant {index} is not a mapping.")
        missing = [key for key in ("name", "config", "run_dir") if not entry.get(key)]
        if missing:
            raise SystemExit(
                f"{path}: variant {index} is missing {', '.join(missing)}; "
                "every entry needs name, config and run_dir."
            )
        name = str(entry["name"])
        if name in seen:
            raise SystemExit(
                f"{path}: duplicate variant name {name!r}. Names label the "
                "report and name the sample file, so they must be unique."
            )
        seen.add(name)
        variants.append(Variant(name, str(entry["config"]), str(entry["run_dir"])))
    return tuple(variants)


# The default set, ordered so the report legend reads as a progression. Loaded
# at import so `plot_descriptor_grid` and the tests keep a module-level
# `VARIANTS` to reach for; `--variants-manifest` overrides it per invocation.
VARIANTS: tuple[Variant, ...] = load_variants(manifest_for_dataset(DEFAULT_DATASET))


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


def family_metric(variants: Sequence[Variant], root: Path) -> str:
    """The distance this family's corpus is searched under.

    Read from each variant's repo config, never from
    `run_dir/run_config.yaml`. Run configs predate the `data.metric` field, so
    a run trained before it existed would fall back to `l2` -- silently wrong
    for exactly the angular families this exists for. A run config is evidence
    of what ran, not a statement about what the corpus is.

    Every manifest entry is read, not only the ones whose checkpoints resolved
    on this box, so the geometry a report is measured under cannot depend on
    which runs happen to be present.

    A value outside `METRICS` is rejected here, before any sampling, rather
    than left to surface inside `ann_difficulty.compute` afterwards --
    `eda.cli`'s `--metric` flag is already guarded by `choices=list(METRICS)`,
    and a config-sourced value deserves the same guard.

    `variants` must be non-empty; `load_variants` already refuses an empty
    manifest, so this only guards against a caller that bypasses it.
    """
    if not variants:
        raise ValueError("family_metric requires at least one variant.")

    by_metric: dict[str, list[str]] = {}
    for variant in variants:
        path = root / variant.config_path
        if not path.is_file():
            raise SystemExit(
                f"no config at {path} for variant {variant.name!r}. The "
                "manifest names it, and its data.metric decides the distance "
                "the report measures under."
            )
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        metric = str((doc.get("data") or {}).get("metric", eda_config.METRIC_DEFAULT))
        if metric not in METRICS:
            raise SystemExit(
                f"{path} (variant {variant.name!r}) sets data.metric="
                f"{metric!r}, which is not one of {METRICS}. Fix the config "
                "before this family's corpus can be measured."
            )
        by_metric.setdefault(metric, []).append(variant.name)

    if len(by_metric) > 1:
        detail = "; ".join(
            f"{metric}: {', '.join(names)}"
            for metric, names in sorted(by_metric.items())
        )
        raise SystemExit(
            "variants disagree on data.metric, so there is no single distance "
            f"to measure this family under ({detail}). Variant numbers are "
            "per-family, so one ladder is one corpus and one metric; fix the "
            "configs, or compare only the variants that agree."
        )
    return next(iter(by_metric))


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


def describe_missing(
    skipped: Sequence[tuple[Variant, str]], manifest: Path, root: Path
) -> str:
    """Explain every unresolvable variant and what would produce its artifacts.

    Spelled out rather than summarised because the reader is as likely to be
    an agent on a fresh clone as a human who knows the history: `runs/` is
    gitignored, so the default manifest's directories are on the training box
    and nowhere else, and "no run directory" on its own gives no way forward.
    """
    lines = [
        f"{len(skipped)} variant(s) named by {manifest} have no usable "
        "artifacts under this root:",
        "",
    ]
    for variant, reason in skipped:
        lines.append(f"  {variant.name}: {reason}")
        lines.append(
            "      train it with: python -m src.train.train_wgan_gp "
            f"--config {variant.config_path}"
        )
    lines.extend(
        [
            "",
            "Training writes to the output_dir named inside each config, which is"
            " not the run_dir above -- the manifest points at historical runs."
            " So: copy those runs under the tree given by --root"
            f" ({root}), or edit the manifest (or pass --variants-manifest) to"
            " name runs you do have.",
        ]
    )
    return "\n".join(lines)


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
        default=DEFAULT_DATASET,
        choices=known_datasets(),
        help=(
            f"Which family's ladder to compare, read from {MANIFEST_DIR}/"
            f"<dataset>.yaml. Defaults to {DEFAULT_DATASET}. Ignored when "
            "--variants-manifest names a file directly."
        ),
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
        "--variants-manifest",
        type=str,
        default=None,
        help=(
            "YAML manifest listing the variants to compare (name, config, "
            "run_dir). Overrides --dataset; without it the manifest is "
            f"{MANIFEST_DIR}/<dataset>.yaml in the repo this module lives in."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Report on the variants that resolved instead of stopping when one "
            "has no checkpoint on this machine."
        ),
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
        default=eda_config.ANN_K_DEFAULT,
        help="Neighbours per query for the LID and relative-contrast panels.",
    )
    parser.add_argument(
        "--ann-hub-k",
        type=int,
        default=eda_config.ANN_HUB_K_DEFAULT,
        help="Neighbour depth for the k-occurrence count behind the hubness panel.",
    )
    parser.add_argument(
        "--ann-max-rows",
        type=int,
        default=eda_config.ANN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for every ANN-difficulty metric. LID, "
            "contrast and hubness all drift with sample count, so every set "
            "must be cut to the same size."
        ),
    )
    parser.add_argument(
        "--knn-max-rows",
        type=int,
        default=eda_config.KNN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for the within-set k-NN distance panel, "
            "which is not an ANN-difficulty panel."
        ),
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=eda_config.IVF_NLIST_DEFAULT,
        help="Cluster count for the IVF cell-balance panel.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top-divergent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--glyph-samples", type=int, default=eda_config.GLYPH_SAMPLES_DEFAULT
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def build_report_args(
    args: argparse.Namespace, specs: list[str], metric: str
) -> argparse.Namespace:
    """Build the Namespace `eda.pipeline.run` expects from our own parsed args.

    Field-for-field parity with `eda.cli.parse_args` is load-bearing: if
    `eda.cli` gains a required argument and this Namespace is not updated
    to match, sampling hundreds of thousands of vectors will succeed before
    the mismatch surfaces as a runtime `AttributeError`. See
    `tests/test_compare_variants.py::test_report_args_match_eda_report_fields`.

    `metric` is passed rather than read off `args` because it is a property of
    the corpus, recorded per family in config. A `--metric` flag would be a
    second place to state it, and so a place for it to go stale.
    """
    return argparse.Namespace(
        real_path=args.real_path,
        real_format=args.real_format,
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=args.output_dir,
        preprocess="l2",
        metric=metric,
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

    manifest = (
        Path(args.variants_manifest)
        if args.variants_manifest is not None
        else manifest_for_dataset(args.dataset)
    )
    variants = load_variants(manifest)

    # Resolve before creating anything, so an aborted run leaves no empty tree.
    found, skipped = resolve_variants(variants, root)
    if skipped and not args.allow_missing:
        raise SystemExit(
            describe_missing(skipped, manifest, root)
            + "\n\nPass --allow-missing to report on the variants that did"
            " resolve instead of stopping here."
        )
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        raise SystemExit(
            describe_missing(skipped, manifest, root)
            + "\n\nNo variant resolved, so there is nothing to report on even"
            " with --allow-missing."
        )

    # After the resolve checks, so a fresh clone still hears about missing
    # runs first; before sampling, so a config problem does not cost the
    # caller several hundred thousand vectors.
    metric = family_metric(variants, root)

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

    report_args = build_report_args(args, specs, metric)
    report_path = pipeline.run(report_args)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
