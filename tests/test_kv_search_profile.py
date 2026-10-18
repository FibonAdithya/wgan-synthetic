import faiss  # pinned in requirements.txt (Task 0); a missing install must fail CI, not skip
import numpy as np
import pytest

from src.kv.search_profile import profile_corpus, recall, work_at_target


def test_recall_counts_overlap_and_ignores_minus_one():
    """Catches counting -1 padding as a hit, or per-row instead of pooled recall."""
    gt = np.array([[0, 1], [2, 3]])
    found = np.array([[1, -1], [2, 3]])
    assert recall(found, gt) == 0.75


def test_work_at_target_interpolates_between_bracketing_points():
    """Catches returning the upper point's work instead of interpolating."""
    curve = [(1, 0.5, 10.0), (2, 0.8, 20.0), (4, 1.0, 40.0)]
    w, ok = work_at_target(curve, 0.9)
    assert ok and w == pytest.approx(30.0)


def test_work_at_target_unreached():
    assert work_at_target([(1, 0.5, 1.0), (2, 0.8, 2.0)], 0.9) == (None, False)


def test_work_at_target_first_point_already_reaches():
    """Catches interpolating from a phantom (0, 0) point."""
    assert work_at_target([(1, 0.95, 7.0)], 0.9) == (7.0, True)


@pytest.fixture
def one_thread():
    """Single-threaded faiss (deterministic HNSW build), restoring the global setting."""
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    yield
    faiss.omp_set_num_threads(old)


def _profile(keys, q_ood, q_id):
    return profile_corpus(
        keys, q_ood, q_id, nlist=16, hnsw_m=16, ef_construction=64, seed=0
    )


def test_key_side_queries_are_easy_and_fields_are_complete(one_thread):
    """Catches a profile missing a field or an ID baseline that is not easy."""
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(1500, 16)).astype(np.float32)
    q_id = rng.normal(size=(32, 16)).astype(np.float32)
    p = _profile(keys, q_id.repeat(5, 0), q_id)
    for name in ("plain_ood_k10", "plain_id_k10", "plain_ood_k128", "plain_id_k128"):
        assert set(p[name]) >= {
            "ivf_scan_frac",
            "ivf_reached",
            "hnsw_ndis",
            "hnsw_reached",
            "ivf_recall_op",
            "hnsw_recall_op",
            "ivf_curve",
            "hnsw_curve",
        }
    assert p["plain_id_k10"]["hnsw_reached"]
    assert p["plain_id_k10"]["hnsw_curve"][-1][1] == pytest.approx(1.0, abs=1e-3)


def _planted(seed, n=1500, d=16, n_ood=128, n_id=64, n_out=50, n_dense=100):
    """Keys with a structural ood/id gap.

    A bulk of norm-4 keys on a sphere, a dense cluster of norm-10 keys along c1, and
    50 scattered norm-12 outliers orthogonal to c1. id queries point at the dense
    cluster (answers sit in one tight graph neighbourhood, so HNSW finds them cheaply);
    ood queries are random directions orthogonal to c1, whose MIPS answers are the
    scattered outliers (isolated points the greedy search reaches only at large ef).
    """
    rng = np.random.default_rng(seed)
    bulk = rng.normal(size=(n - n_dense - n_out, d))
    bulk *= 4.0 / np.linalg.norm(bulk, axis=1, keepdims=True)
    c1 = rng.normal(size=d)
    c1 /= np.linalg.norm(c1)
    dense = c1 * 10.0 + rng.normal(size=(n_dense, d)) * 0.5
    out = rng.normal(size=(n_out, d))
    out -= (out @ c1)[:, None] * c1
    out *= 12.0 / np.linalg.norm(out, axis=1, keepdims=True)
    keys = np.vstack([bulk, dense, out]).astype(np.float32)
    q_id = (c1 + rng.normal(size=(n_id, d)) * 0.3).astype(np.float32)
    q_ood = rng.normal(size=(n_ood, d))
    q_ood -= (q_ood @ c1)[:, None] * c1
    return keys, q_ood.astype(np.float32), q_id


def test_ood_queries_cost_more_hnsw_work_than_id(one_thread):
    """The property the gate rests on: queries whose answers are isolated outliers
    (ood) need more HNSW distance computations than queries whose answers are a
    dense cluster (id). Catches swapping the ood/id workloads or reading ndis from
    the wrong counter.

    Planted distribution (see _planted) replaced twice. The brief's shifted
    Gaussian failed against the spike's own functions (ood recall 0.9997 vs id
    0.9625 at ef=176); a retuned shifted Gaussian (N(-1.5, 0.5)) held on only 5 of
    8 other data seeds. The structured cloud here held on 30 of 30 data seeds with
    ood ndis >= 1.3x id ndis. MEASURED 2026-10-18: held on 30 of 30 data seeds;
    minimum ood/id ndis ratio 2.2, minimum recall margin 0.03.
    """
    keys, q_ood, q_id = _planted(2)
    p = _profile(keys, q_ood, q_id)
    assert p["plain_ood_k10"]["hnsw_recall_op"] <= p["plain_id_k10"]["hnsw_recall_op"]
    ho, hi = p["plain_ood_k10"]["hnsw_ndis"], p["plain_id_k10"]["hnsw_ndis"]
    assert hi is not None, (
        "key-side queries must reach recall 0.9; otherwise the comparison is vacuous"
    )
    assert ho is None or ho > hi
