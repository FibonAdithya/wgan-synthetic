"""Run the GPU ANN benchmark over real SIFT and each trained variant.

Example:
    python -m src.eval.ann_benchmark \
        --real-path data/sift_1m.npy \
        --work-dir runs/ann_benchmark \
        --output-dir docs/results/ann-gpu-benchmark
"""

from __future__ import annotations

import argparse
import platform
from collections.abc import Sequence
from pathlib import Path

from src.eval.ann_benchmark import corpora as corpora_mod
from src.eval.ann_benchmark import indexes, report, runner
from src.eval.compare_variants import (
    DEFAULT_MANIFEST,
    describe_missing,
    load_variants,
    resolve_variants,
)

DEFAULT_NUM_VECTORS = 1_000_000
DEFAULT_NUM_QUERIES = 10_000
DEFAULT_K = 10
DEFAULT_REPEATS = 5
DEFAULT_TARGET_RECALL = 0.90


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, default="data/sift_1m.npy")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument(
        "--real-hdf5-path",
        type=str,
        default=None,
        help=(
            "Explicit ann-benchmarks HDF5 to read SIFT's query set from. "
            "Overrides selecting one out of --cache-dir by name match, which "
            "is required (and raises) when the cache holds more than one "
            "family's file -- e.g. deep-image, glove and sift side by side."
        ),
    )
    parser.add_argument("--variants-manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--work-dir", type=str, default="runs/ann_benchmark")
    parser.add_argument(
        "--output-dir", type=str, default="docs/results/ann-gpu-benchmark"
    )
    parser.add_argument("--num-vectors", type=int, default=DEFAULT_NUM_VECTORS)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=list(indexes.DEFAULT_INDEX_NAMES),
        choices=list(indexes.ADAPTER_NAMES),
        help=(
            "Which indexes to run. Defaults to the four cuVS indexes the "
            "published grid reports. The torch_flat* brute-force baselines "
            "are selectable but off by default: they measure whether cuVS "
            "brute force is near the hardware's limit, which is a separate "
            "question from how the variant ladder searches."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Proceed on a partial ladder instead of aborting. Checkpoints "
            "live on the training box, and a partial comparison is worth "
            "reading once you have decided it is partial on purpose."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)

    manifest = Path(args.variants_manifest)
    variants = load_variants(manifest)
    found, skipped = resolve_variants(variants, root)
    if skipped and not args.allow_missing:
        raise SystemExit(describe_missing(skipped, manifest, root))
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")

    adapters = indexes.build_adapters(args.indexes)

    # Fail here rather than forty cells in. Materializing seven corpora and
    # their ground truth is most of an hour's work, and discovering the
    # missing dependency afterwards wastes all of it.
    try:
        indexes.require_device_stack()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    built_corpora = [
        corpora_mod.materialize_real(
            real_path=Path(args.real_path),
            cache_dir=Path(args.cache_dir),
            work_dir=work_dir,
            num_vectors=args.num_vectors,
            num_queries=args.num_queries,
            k=args.k,
            hdf5_path=Path(args.real_hdf5_path) if args.real_hdf5_path else None,
        )
    ]
    for variant in found:
        built_corpora.append(
            corpora_mod.materialize_variant(
                variant,
                root=root,
                work_dir=work_dir,
                num_vectors=args.num_vectors,
                num_queries=args.num_queries,
                k=args.k,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
        print(f"materialized {variant.name}")

    builds, searches = run_and_report(
        built_corpora, adapters, args, work_dir, output_dir
    )
    print(
        f"{len(builds)} build records, {len(searches)} search records -> {output_dir}"
    )


def environment_block(args) -> dict[str, object]:
    """What the run was, recorded so the artifact can be read years later.

    `versions` is read off the interpreter that is running, not off
    requirements.txt: the two disagree on this box (torch is pinned at
    2.13.0, installed at 2.12.0), and a provenance field copied from the pin
    would record the disagreement as agreement. See `indexes.stack_versions`.
    """
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": indexes.stack_versions(),
        "num_vectors": args.num_vectors,
        "num_queries": args.num_queries,
        "k": args.k,
        "repeats": args.repeats,
        "target_recall": args.target_recall,
        "normalized": True,
    }


def run_and_report(built_corpora, adapters, args, work_dir, output_dir):
    builds, searches = runner.run_grid(
        built_corpora,
        adapters,
        k=args.k,
        repeats=args.repeats,
        records_path=work_dir / "records.json",
    )
    environment = environment_block(args)
    report.write_json(
        output_dir / "ann_benchmark.json",
        builds=builds,
        searches=searches,
        environment=environment,
    )
    rows = report.headline_rows(builds, searches, target_recall=args.target_recall)
    report.write_markdown(
        output_dir / "ann_benchmark.md", rows, target_recall=args.target_recall
    )
    report.write_html(
        output_dir / "report.html",
        builds,
        searches,
        target_recall=args.target_recall,
    )
    return builds, searches


if __name__ == "__main__":
    main()
