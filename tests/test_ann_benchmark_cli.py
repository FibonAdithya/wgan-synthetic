"""Tests for the ANN benchmark's argument surface and provenance block.

Runs CPU-only and cuVS-free, like the rest of the ann_benchmark tests.
"""

from src.eval.ann_benchmark import cli, indexes


def test_default_indexes_are_the_published_grid():
    """--indexes must not pick up every registered adapter.

    The torch brute-force baselines are registered so they can be selected,
    but they answer a different question from the ladder comparison. If the
    default were `ADAPTER_NAMES`, adding any future probe would silently
    change the shape of the artifact the project's conclusions cite.
    """
    args = cli.parse_args([])
    assert args.indexes == ["flat", "ivf_flat", "ivf_pq", "cagra"]


def test_torch_baselines_are_selectable():
    args = cli.parse_args(["--indexes", "flat", "torch_flat", "torch_flat_fp16"])
    assert args.indexes == ["flat", "torch_flat", "torch_flat_fp16"]


def test_stack_versions_reports_torch_and_tolerates_a_missing_cuvs():
    versions = indexes.stack_versions()
    assert set(versions) == {"torch", "cuvs", "cupy"}

    import torch

    assert versions["torch"] == torch.__version__

    # cuVS is absent on the `make check` box and present on the GPU box; both
    # are correct, and neither may raise.
    try:
        import cuvs
    except ImportError:
        assert versions["cuvs"] is None
    else:
        assert versions["cuvs"] == cuvs.__version__


def test_stack_versions_returns_none_rather_than_raising(monkeypatch):
    """A provenance field must not be able to fail the run it describes."""

    def explode(name):
        raise RuntimeError(f"broken import of {name}")

    monkeypatch.setattr(indexes.importlib, "import_module", explode)
    assert indexes.stack_versions() == {"torch": None, "cuvs": None, "cupy": None}


def test_environment_block_records_the_measured_versions():
    """The published grid recorded the *pinned* torch version, not the one it
    ran on -- 2.13.0 against the 2.12.0 that had been installed on the box
    since June. Reading it off the live interpreter is what stops a future
    write-up repeating that."""
    args = cli.parse_args([])
    block = cli.environment_block(args)

    import torch

    assert block["versions"]["torch"] == torch.__version__
    assert block["num_vectors"] == 1_000_000
    assert block["num_queries"] == 10_000
    assert block["k"] == 10
    assert block["normalized"] is True
