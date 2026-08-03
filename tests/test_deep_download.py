from pathlib import Path

import h5py
import numpy as np
import pytest

from src.deep.download import DEEP_URL, fetch, subset


@pytest.fixture
def fake_hdf5(tmp_path: Path) -> Path:
    """A miniature stand-in for deep-image-96-angular.hdf5."""
    path = tmp_path / "fake.hdf5"
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 96)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=train[:10])
    return path


def test_subset_writes_requested_shape_and_dtype(fake_hdf5: Path, tmp_path: Path):
    out = subset(fake_hdf5, tmp_path / "sub.npy", num_rows=100)
    arr = np.load(out)
    assert arr.shape == (100, 96)
    assert arr.dtype == np.float32


def test_subset_is_deterministic_under_the_same_seed(fake_hdf5: Path, tmp_path: Path):
    a = np.load(subset(fake_hdf5, tmp_path / "a.npy", num_rows=50, seed=7))
    b = np.load(subset(fake_hdf5, tmp_path / "b.npy", num_rows=50, seed=7))
    np.testing.assert_array_equal(a, b)


def test_subset_differs_under_a_different_seed(fake_hdf5: Path, tmp_path: Path):
    a = np.load(subset(fake_hdf5, tmp_path / "a.npy", num_rows=50, seed=7))
    b = np.load(subset(fake_hdf5, tmp_path / "b.npy", num_rows=50, seed=8))
    assert not np.array_equal(a, b)


def test_subset_takes_everything_when_num_rows_exceeds_the_file(
    fake_hdf5: Path, tmp_path: Path
):
    arr = np.load(subset(fake_hdf5, tmp_path / "all.npy", num_rows=10_000))
    assert arr.shape == (500, 96)


def test_fetch_leaves_no_partial_file_when_the_download_fails(
    tmp_path: Path, monkeypatch
):
    """A crashed fetch must not leave a truncated file a reader could load."""
    dest = tmp_path / "deep.hdf5"

    class Boom(Exception):
        pass

    def exploding_urlopen(*args, **kwargs):
        raise Boom("network down")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)
    with pytest.raises(Boom):
        fetch("http://example.invalid/x.hdf5", dest)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_fetch_skips_the_download_when_the_destination_already_exists(
    tmp_path: Path, monkeypatch
):
    dest = tmp_path / "deep.hdf5"
    dest.write_bytes(b"already here")

    def exploding_urlopen(*args, **kwargs):
        raise AssertionError("fetch must not download over an existing file")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)
    assert fetch("http://example.invalid/x.hdf5", dest) == dest
    assert dest.read_bytes() == b"already here"


def test_fetch_waits_for_a_concurrent_downloader_instead_of_duplicating_it(
    tmp_path: Path, monkeypatch
):
    """Another agent sharing the cache may already be pulling the 4GB file.

    A held .part file means a fetch is in flight; the second caller waits for
    the result rather than starting a second 4GB download.
    """
    dest = tmp_path / "deep.hdf5"
    lock = dest.with_suffix(dest.suffix + ".part")
    lock.write_bytes(b"")  # simulate the in-flight download

    def exploding_urlopen(*args, **kwargs):
        raise AssertionError("must not download while another fetch holds the lock")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)

    def finish_the_other_download(_seconds):
        dest.write_bytes(b"complete")
        lock.unlink()

    monkeypatch.setattr("src.deep.download.time.sleep", finish_the_other_download)
    assert fetch("http://example.invalid/x.hdf5", dest) == dest
    assert dest.read_bytes() == b"complete"


def test_fetch_gives_up_on_a_stalled_concurrent_downloader(tmp_path: Path, monkeypatch):
    dest = tmp_path / "deep.hdf5"
    dest.with_suffix(dest.suffix + ".part").write_bytes(b"")
    monkeypatch.setattr("src.deep.download.time.sleep", lambda _s: None)
    with pytest.raises(TimeoutError, match="in-flight download"):
        fetch("http://example.invalid/x.hdf5", dest, timeout_seconds=0.0)


def test_deep_url_points_at_the_angular_variant():
    assert DEEP_URL.endswith("deep-image-96-angular.hdf5")
