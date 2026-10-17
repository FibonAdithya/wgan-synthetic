"""THROWAWAY SPIKE (KV-cache track, 2026-10-16): measure Q->K search difficulty per head.

For every (doc, global layer, kv head): keys are W_k h at positions [0, T - n_tail), OOD
queries are s(pos) W_q h for the head's n_rep query heads at the last n_tail positions,
in-distribution (ID) queries are the keys at those same tail positions (not in the index).
Search is maximum inner product, ground truth by brute force.

Work to reach recall@10 >= target:
  IVF  (IndexIVFFlat, IP, nlist lists): fraction of keys scanned  (indexIVF_stats.ndis)
  HNSW (IndexHNSWFlat, IP, M, efC):     distance computations/query (hnsw_stats.ndis)
Work is linearly interpolated between the two sweep points that bracket the target; a curve
that never reaches it is recorded as null with "reached": false.
"""

import argparse
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from safetensors.torch import load_file

NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
K = 10


def recall(found: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([len(set(f) & set(g)) / K for f, g in zip(found, gt)]))


def work_at_target(curve, target):
    """curve: list of (param, recall, work). Interpolate work at recall == target."""
    for (p0, r0, w0), (p1, r1, w1) in zip([(None, 0.0, 0.0)] + curve[:-1], curve):
        if r1 >= target:
            if p0 is None or r1 == r0:
                return w1, True
            return w0 + (w1 - w0) * (target - r0) / (r1 - r0), True
    return None, False


def ivf_curve(index, q, gt, n_keys):
    out = []
    for nprobe in NPROBES:
        if nprobe > index.nlist:
            break
        index.nprobe = nprobe
        faiss.cvar.indexIVF_stats.reset()
        _, found = index.search(q, K)
        out.append(
            (
                nprobe,
                recall(found, gt),
                faiss.cvar.indexIVF_stats.ndis / (len(q) * n_keys),
            )
        )
    return out


def hnsw_curve(index, q, gt):
    out = []
    for ef in EFS:
        index.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        _, found = index.search(q, K)
        out.append((ef, recall(found, gt), faiss.cvar.hnsw_stats.ndis / len(q)))
    return out


def diagnostics(keys_t, q_t, gt_q, head_dim):
    """Cheap side measurements: attention sink, hubness of q->k top-10, top-10 softmax mass."""
    counts = np.bincount(gt_q.ravel(), minlength=keys_t.shape[0])
    c = counts.astype(np.float64)
    skew = float(((c - c.mean()) ** 3).mean() / (c.std() ** 3 + 1e-12))
    logits = (q_t @ keys_t.T) / math.sqrt(head_dim)
    p = torch.softmax(logits, dim=1)
    top_mass = float(p.topk(K, dim=1).values.sum(1).mean())
    return {
        "top1_is_pos0": float(np.mean(gt_q[:, 0] == 0)),
        "pos0_in_top10": float(np.mean((gt_q == 0).any(1))),
        "hub_skew_k10": skew,
        "top10_softmax_mass": top_mass,
        "key_norm_median": float(keys_t.norm(dim=1).median()),
        "query_norm_median": float(q_t.norm(dim=1).median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-tail", type=int, default=1024)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=15)
    args = ap.parse_args()
    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)

    meta = json.loads((args.capture_dir / "meta.json").read_text())
    layers = meta["global_layers"]
    s_cfg = (meta["attn_scale"], meta["floor_scale"])
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["doc"], r["layer"], r["kv_head"]))

    for d in range(len(meta["docs"])):
        for L in layers:
            w = load_file(
                str(args.capture_dir / "weights" / f"layer{L:02d}.safetensors")
            )
            h = (
                load_file(
                    str(args.capture_dir / f"doc{d}" / f"h_L{L:02d}.safetensors")
                )["h"]
                .cuda()
                .float()
            )
            T = h.shape[0]
            pos = torch.arange(T, device="cuda", dtype=torch.float64)
            s = (
                1.0 + s_cfg[0] * torch.log1p(torch.floor((pos + 1.0) / s_cfg[1]))
            ).float()
            wk, wq = w["k_proj"].cuda().float(), w["q_proj"].cuda().float()
            n_kv, d_h = wk.shape[0] // 128, 128
            n_rep = wq.shape[0] // wk.shape[0]
            k_all = (h @ wk.T).view(T, n_kv, d_h)
            q_all = ((h @ wq.T) * s[:, None]).view(T, n_kv * n_rep, d_h)
            Tk = T - args.n_tail
            for g in range(n_kv):
                if (d, L, g) in done:
                    continue
                t0 = time.time()
                keys_t = k_all[:Tk, g].contiguous()
                q_ood_t = (
                    q_all[Tk:, g * n_rep : (g + 1) * n_rep]
                    .reshape(-1, d_h)
                    .contiguous()
                )
                q_id_t = k_all[Tk:, g].contiguous()
                gt_ood = (q_ood_t @ keys_t.T).topk(K, dim=1).indices.cpu().numpy()
                gt_id = (q_id_t @ keys_t.T).topk(K, dim=1).indices.cpu().numpy()
                keys, q_ood, q_id = (
                    x.cpu().numpy().astype(np.float32)
                    for x in (keys_t, q_ood_t, q_id_t)
                )

                quant = faiss.IndexFlatIP(d_h)
                ivf = faiss.IndexIVFFlat(
                    quant, d_h, args.nlist, faiss.METRIC_INNER_PRODUCT
                )
                ivf.cp.seed = 1234
                ivf.train(keys)
                ivf.add(keys)
                hnsw = faiss.IndexHNSWFlat(d_h, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                hnsw.hnsw.efConstruction = args.ef_construction
                hnsw.add(keys)

                rec = {
                    "doc": d,
                    "layer": L,
                    "kv_head": g,
                    "n_keys": Tk,
                    "n_q_ood": len(q_ood),
                    "n_q_id": len(q_id),
                }
                for name, q, gt in (("ood", q_ood, gt_ood), ("id", q_id, gt_id)):
                    ic, hc = ivf_curve(ivf, q, gt, Tk), hnsw_curve(hnsw, q, gt)
                    iw, ir = work_at_target(ic, args.target)
                    hw, hr = work_at_target(hc, args.target)
                    rec[name] = {
                        "ivf_scan_frac": iw,
                        "ivf_reached": ir,
                        "hnsw_ndis": hw,
                        "hnsw_reached": hr,
                        "ivf_curve": ic,
                        "hnsw_curve": hc,
                    }
                rec["diag_ood"] = diagnostics(keys_t, q_ood_t, gt_ood, d_h)
                rec["seconds"] = round(time.time() - t0, 1)
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    d,
                    L,
                    g,
                    rec["seconds"],
                    "s | ivf ood/id",
                    rec["ood"]["ivf_scan_frac"],
                    rec["id"]["ivf_scan_frac"],
                    "| hnsw ood/id",
                    rec["ood"]["hnsw_ndis"],
                    rec["id"]["hnsw_ndis"],
                    flush=True,
                )
            del h, k_all, q_all
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
