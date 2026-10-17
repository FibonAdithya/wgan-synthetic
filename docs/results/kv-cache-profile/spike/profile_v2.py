"""THROWAWAY SPIKE (KV-cache track, 2026-10-17): re-profile Q->K search per head.

Changes from profile_heads.py, following Zharkov et al. 2026 (arXiv 2610.10135):
  - the sink (position 0) is dropped from the index; the recent window is the last n_tail
    positions, which supply the queries and are not indexed either;
  - HNSW is built on the Xbox reduction (MIPS -> L2), M=48, efConstruction=500;
  - k in {10, 128};
  - two workloads: "plain" (q, k) and "lift" (SoftmaxLift: keys (k, -log u), queries
    (q / sqrt(d), 1), so the inner product is the log priority);
  - attention error over the indexed segment, k=128, for TopK on the plain workload and
    priority sampling on the lift workload, with exact, IVF and HNSW retrieval.

Operating points: IVF nprobe=24 of nlist=256 probes the same 9.4% of lists as the paper's
nprobe=96 of nlist=1024 (a scaled equivalent, not their setting); HNSW efSearch=176 is theirs.
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 176, 256, 384, 512, 768, 1024, 1536, 2048]
KS = (10, 128)
OP_NPROBE, OP_EF = 24, 176
K_ATTN = 128


def recall(found, gt):
    """Pooled recall sum|found & gt| / sum|gt|; ids within a row are distinct, -1 never matches."""
    return float((found[:, :, None] == gt[:, None, :]).any(2).sum() / gt.size)


def work_at_target(curve, target):
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


def xbox(base):
    """MIPS -> L2: append sqrt(M^2 - |x|^2); queries get a 0. Ranking is preserved exactly."""
    n2 = (base.astype(np.float64) ** 2).sum(1)
    extra = np.sqrt(np.maximum(n2.max() - n2, 0.0)).astype(np.float32)
    return np.hstack([base, extra[:, None]])


def pad0(q):
    return np.hstack([q, np.zeros((len(q), 1), np.float32)])


def sweep_ivf(ivf, q, gt, n_keys, k, keep_op=False):
    out, op_ids = [], None
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
        if keep_op and nprobe == OP_NPROBE:
            op_ids = found
    return out, op_ids


def sweep_hnsw(hnsw, q, gt, k, keep_op=False):
    out, op_ids = [], None
    for ef in EFS:
        if ef < k:
            continue
        hnsw.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        _, found = hnsw.search(q, k)
        out.append((ef, recall(found, gt), faiss.cvar.hnsw_stats.ndis / len(q)))
        if keep_op and ef == OP_EF:
            op_ids = found
    return out, op_ids


def pooled_rel_err(est, exact):
    return float(((est - exact) ** 2).sum().sqrt() / (exact**2).sum().sqrt())


def topk_estimate(logits, V, ids):
    ids = torch.as_tensor(ids, device=logits.device).clamp_min(0)
    sel = logits.gather(1, ids)
    return torch.einsum("qk,qkd->qd", torch.softmax(sel, 1), V[ids])


def priority_estimate(logits, neglogu, V, ids):
    """Priority sampling from a returned set of K+1 keys: tau = smallest returned priority."""
    ids = torch.as_tensor(ids, device=logits.device).clamp_min(0)
    lg = logits.gather(1, ids)
    lp = lg + neglogu[ids]
    order = lp.argsort(1, descending=True)
    lg, lp, ids = lg.gather(1, order), lp.gather(1, order), ids.gather(1, order)
    log_tau = lp[:, -1:]
    lw = torch.maximum(lg[:, :-1], log_tau)
    w = torch.softmax(lw, 1)
    return torch.einsum("qk,qkd->qd", w, V[ids[:, :-1]])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-tail", type=int, default=1024)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--hnsw-m", type=int, default=48)
    ap.add_argument("--ef-construction", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--max-corpora", type=int, default=0)
    args = ap.parse_args()
    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)

    meta = json.loads((args.capture_dir / "meta.json").read_text())
    a_scale, f_scale = meta["attn_scale"], meta["floor_scale"]
    wmap = json.loads((args.model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    done = set()
    if args.out.exists():
        done = {
            (r["doc"], r["layer"], r["kv_head"])
            for r in map(json.loads, args.out.read_text().splitlines())
        }
    n_run = 0

    for d in range(len(meta["docs"])):
        for L in meta["global_layers"]:
            w = load_file(
                str(args.capture_dir / "weights" / f"layer{L:02d}.safetensors")
            )
            vname = f"language_model.model.layers.{L}.self_attn.v_proj.weight"
            with safe_open(args.model_dir / wmap[vname], "pt") as f:
                wv = f.get_tensor(vname).cuda().float()
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
                1.0 + a_scale * torch.log1p(torch.floor((pos + 1.0) / f_scale))
            ).float()
            wk, wq = w["k_proj"].cuda().float(), w["q_proj"].cuda().float()
            d_h = 128
            n_kv = wk.shape[0] // d_h
            n_rep = wq.shape[0] // wk.shape[0]
            scaling = d_h**-0.5
            k_all = (h @ wk.T).view(T, n_kv, d_h)
            v_all = (h @ wv.T).view(T, n_kv, d_h)
            q_all = ((h @ wq.T) * s[:, None]).view(T, n_kv * n_rep, d_h)
            Tk = T - args.n_tail
            for g in range(n_kv):
                if (d, L, g) in done or (
                    args.max_corpora and n_run >= args.max_corpora
                ):
                    continue
                t0 = time.time()
                seed = 1000003 * d + 1009 * L + g
                keys_t = k_all[args.n_sink : Tk, g].contiguous()
                V = v_all[args.n_sink : Tk, g].contiguous()
                n_keys = keys_t.shape[0]
                q_ood_t = (
                    q_all[Tk:, g * n_rep : (g + 1) * n_rep]
                    .reshape(-1, d_h)
                    .contiguous()
                )
                q_id_t = k_all[Tk:, g].contiguous()
                gen = torch.Generator(device="cuda").manual_seed(seed)
                u = 1.0 - torch.rand(n_keys, device="cuda", generator=gen)  # (0, 1]
                neglogu = -torch.log(u)
                logits = (
                    q_ood_t * scaling
                ) @ keys_t.T  # attention logits over the indexed segment
                exact = torch.softmax(logits, 1) @ V

                lift_keys_t = torch.cat([keys_t, neglogu[:, None]], 1)
                lift_q_t = torch.cat(
                    [q_ood_t * scaling, torch.ones(len(q_ood_t), 1, device="cuda")], 1
                )
                workloads = {
                    "plain": (keys_t, {"ood": q_ood_t, "id": q_id_t}),
                    "lift": (lift_keys_t, {"ood": lift_q_t}),
                }
                rec = {
                    "doc": d,
                    "layer": L,
                    "kv_head": g,
                    "n_keys": n_keys,
                    "n_q_ood": len(q_ood_t),
                    "n_q_id": len(q_id_t),
                    "n_sink": args.n_sink,
                    "n_tail": args.n_tail,
                }
                op_sets = {}
                for wname, (base_t, qsets) in workloads.items():
                    base = base_t.cpu().numpy().astype(np.float32)
                    ivf, hnsw = build(
                        base, args.nlist, args.hnsw_m, args.ef_construction, seed
                    )
                    for qname, q_t in qsets.items():
                        sims = q_t @ base_t.T
                        q = q_t.cpu().numpy().astype(np.float32)
                        qx = pad0(q)
                        for k in KS:
                            gt = sims.topk(k, 1).indices.cpu().numpy()
                            keep = qname == "ood" and k == K_ATTN
                            ic, iv_ids = sweep_ivf(ivf, q, gt, n_keys, k, keep)
                            hc, hn_ids = sweep_hnsw(hnsw, qx, gt, k, keep)
                            iw, ir = work_at_target(ic, args.target)
                            hw, hr = work_at_target(hc, args.target)
                            rec[f"{wname}_{qname}_k{k}"] = {
                                "ivf_scan_frac": iw,
                                "ivf_reached": ir,
                                "hnsw_ndis": hw,
                                "hnsw_reached": hr,
                                "ivf_recall_op": dict((p, r) for p, r, _ in ic)[
                                    OP_NPROBE
                                ],
                                "hnsw_recall_op": dict((p, r) for p, r, _ in hc)[OP_EF],
                                "ivf_curve": ic,
                                "hnsw_curve": hc,
                            }
                            if keep:
                                op_sets[wname] = {
                                    "flat": gt,
                                    "ivf": iv_ids,
                                    "hnsw": hn_ids,
                                }

                errs = {}
                for oracle in ("flat", "ivf", "hnsw"):
                    errs[f"topk_{oracle}"] = pooled_rel_err(
                        topk_estimate(logits, V, op_sets["plain"][oracle]), exact
                    )
                    errs[f"lift_{oracle}"] = pooled_rel_err(
                        priority_estimate(logits, neglogu, V, op_sets["lift"][oracle]),
                        exact,
                    )
                rec["attn_rel_err_k128"] = errs

                gt10 = logits.topk(10, 1).indices.cpu().numpy()
                c = np.bincount(gt10.ravel(), minlength=n_keys).astype(np.float64)
                p = torch.softmax(logits, 1)
                rec["diag_ood"] = {
                    "hub_skew_k10": float(
                        ((c - c.mean()) ** 3).mean() / (c.std() ** 3 + 1e-12)
                    ),
                    "distinct_top10_keys": int((c > 0).sum()),
                    "top10_softmax_mass": float(p.topk(10, 1).values.sum(1).mean()),
                    "top128_softmax_mass": float(p.topk(128, 1).values.sum(1).mean()),
                    "key_norm_median": float(keys_t.norm(dim=1).median()),
                    "query_norm_median": float(q_ood_t.norm(dim=1).median()),
                }
                rec["seconds"] = round(time.time() - t0, 1)
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_run += 1
                pk = rec["plain_ood_k128"]
                print(
                    d,
                    L,
                    g,
                    rec["seconds"],
                    "s | plain ood k128 ivf",
                    pk["ivf_scan_frac"],
                    "hnsw",
                    pk["hnsw_ndis"],
                    "| attn err topk/lift flat",
                    round(errs["topk_flat"], 3),
                    round(errs["lift_flat"], 3),
                    flush=True,
                )
            del h, k_all, q_all, v_all
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
