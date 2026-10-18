"""IVF (inner product) and Xbox-HNSW search profiling for the scout_kv gate.

Ported from docs/results/kv-cache-profile/spike/profile_v2.py. The attention-error
estimators are out of scope. ``sweep_ivf`` / ``sweep_hnsw`` return only the curve.
"""

import faiss
import numpy as np

from src.kv.transforms import pad0, xbox

NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 176, 256, 384, 512, 768, 1024, 1536, 2048]
KS = (10, 128)
OP_NPROBE, OP_EF = 24, 176


def recall(found, gt):
    """Pooled recall sum|found & gt| / sum|gt|; ids within a row are distinct, -1 never matches."""
    return float((found[:, :, None] == gt[:, None, :]).any(2).sum() / gt.size)


def work_at_target(curve, target):
    """Work at the first recall >= target, linearly interpolated from the previous point.

    Returns (work, True), or (None, False) if the curve never reaches the target.
    """
    prev = (0.0, 0.0)
    for _, r, w in curve:
        if r >= target:
            r0, w0 = prev
            return (
                w
                if r == r0 or prev == (0.0, 0.0)
                else w0 + (w - w0) * (target - r0) / (r - r0)
            ), True
        prev = (r, w)
    return None, False


def sweep_ivf(ivf, q, gt, n_keys, k):
    """[(nprobe, recall, scanned fraction of the corpus)] over NPROBES."""
    out = []
    for nprobe in NPROBES:
        ivf.nprobe = nprobe
        faiss.cvar.indexIVF_stats.reset()
        _, found = ivf.search(q, k)
        out.append(
            (
                nprobe,
                recall(found, gt),
                faiss.cvar.indexIVF_stats.ndis / (len(q) * n_keys),
            )
        )
    return out


def sweep_hnsw(hnsw, q, gt, k):
    """[(efSearch, recall, distance computations per query)] over EFS with ef >= k."""
    out = []
    for ef in EFS:
        if ef < k:
            continue
        hnsw.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        _, found = hnsw.search(q, k)
        out.append((ef, recall(found, gt), faiss.cvar.hnsw_stats.ndis / len(q)))
    return out


def build(base, nlist, m, efc, seed):
    d = base.shape[1]
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.cp.seed = seed
    ivf.train(base)
    ivf.add(base)
    xb = xbox(base)
    hnsw = faiss.IndexHNSWFlat(xb.shape[1], m)
    hnsw.hnsw.efConstruction = efc
    hnsw.add(xb)
    return ivf, hnsw


def conditions(nlist, hnsw_m, ef_construction, target):
    return {
        "nlist": nlist,
        "hnsw_m": hnsw_m,
        "ef_construction": ef_construction,
        "target": target,
        "nprobes": list(NPROBES),
        "efs": list(EFS),
    }


def profile_corpus(
    keys, q_ood, q_id, *, nlist=256, hnsw_m=48, ef_construction=500, seed=0, target=0.9
):
    keys = np.ascontiguousarray(keys, np.float32)
    ivf, hnsw = build(keys, nlist, hnsw_m, ef_construction, seed)
    out = {}
    for qname, q in (("ood", q_ood), ("id", q_id)):
        q = np.ascontiguousarray(q, np.float32)
        sims = q @ keys.T
        qx = pad0(q)
        for k in KS:
            gt = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            ic = sweep_ivf(ivf, q, gt, len(keys), k)
            hc = sweep_hnsw(hnsw, qx, gt, k)
            iw, ir = work_at_target(ic, target)
            hw, hr = work_at_target(hc, target)
            out[f"plain_{qname}_k{k}"] = {
                "ivf_scan_frac": iw,
                "ivf_reached": ir,
                "hnsw_ndis": hw,
                "hnsw_reached": hr,
                "ivf_recall_op": dict((p, r) for p, r, _ in ic)[OP_NPROBE],
                "hnsw_recall_op": dict((p, r) for p, r, _ in hc)[OP_EF],
                "ivf_curve": ic,
                "hnsw_curve": hc,
            }
    return out
