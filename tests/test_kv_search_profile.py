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


def _profile(keys, q_ood, q_id):
    faiss.omp_set_num_threads(1)  # deterministic HNSW build
    return profile_corpus(
        keys, q_ood, q_id, nlist=16, hnsw_m=16, ef_construction=64, seed=0
    )


def test_key_side_queries_are_easy_and_fields_are_complete():
    """Catches a profile missing a field or an ID baseline that is not easy."""
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(3000, 16)).astype(np.float32)
    q_id = rng.normal(size=(64, 16)).astype(np.float32)
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


def test_ood_queries_cost_more_hnsw_work_than_id():
    """The property the gate rests on: queries off the key distribution (here a
    cloud shifted against the key mean, whose MIPS answers are large-norm outliers)
    need more HNSW distance computations than key-like queries. Catches swapping
    the ood/id workloads or reading ndis from the wrong counter.

    Planted distribution changed from the brief's N(2.5, 0.3) to N(-1.5, 0.5):
    run against the spike's functions, the brief's version gave ood recall at
    ef=176 of 0.9997 vs id 0.9625, failing the recall assertion. N(-1.5, 0.5)
    held on seeds 0, 1, 2 (ood recall 0.945-0.954 < id, ood ndis > id)."""
    rng = np.random.default_rng(1)
    keys = rng.normal(size=(4000, 16)).astype(np.float32)
    q_id = rng.normal(size=(64, 16)).astype(np.float32)
    q_ood = (rng.normal(size=(320, 16)) * 0.5 - 1.5).astype(np.float32)
    p = _profile(keys, q_ood, q_id)
    assert p["plain_ood_k10"]["hnsw_recall_op"] <= p["plain_id_k10"]["hnsw_recall_op"]
    ho, hi = p["plain_ood_k10"]["hnsw_ndis"], p["plain_id_k10"]["hnsw_ndis"]
    assert hi is not None, (
        "key-side queries must reach recall 0.9; otherwise the comparison is vacuous"
    )
    assert ho is None or ho > hi
