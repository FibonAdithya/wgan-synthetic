"""THROWAWAY SPIKE (KV-cache track, 2026-10-18): is a corpus mixed from 4 books as hard as one book?

A generator trained on all books pooled emits i.i.d. h, so each synthetic corpus resembles a
mixture of books. This builds real mixed corpora with the same sizes as the single-book ones in
profile_v2.py and runs the same plain-workload IVF / Xbox-HNSW sweeps, so the two are comparable.

Per (layer, kv_head):
  keys    n_per_doc = (Tk - n_sink) // n_docs indexed positions from each book (Tk = T - n_tail)
  ood     n_tail // n_docs random tail positions per book x n_rep query heads
  id      the keys at those same tail positions
Records the same fields as profile_v2.py's plain_* entries, plus the share of each query's
exact top-10 that comes from its own book.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_v2 import KS, OP_EF, OP_NPROBE, build, pad0, sweep_hnsw, sweep_ivf, work_at_target  # noqa: E402

import faiss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-tail", type=int, default=1024)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--hnsw-m", type=int, default=48)
    ap.add_argument("--ef-construction", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--mix-seed", type=int, default=0)
    ap.add_argument("--max-corpora", type=int, default=0)
    args = ap.parse_args()
    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)

    meta = json.loads((args.capture_dir / "meta.json").read_text())
    a_scale, f_scale = meta["attn_scale"], meta["floor_scale"]
    n_docs = len(meta["docs"])
    done = set()
    if args.out.exists():
        done = {(r["layer"], r["kv_head"]) for r in map(json.loads, args.out.read_text().splitlines())}
    n_run = 0

    for L in meta["global_layers"]:
        w = load_file(str(args.capture_dir / "weights" / f"layer{L:02d}.safetensors"))
        wk, wq = w["k_proj"].cuda().float(), w["q_proj"].cuda().float()
        d_h = 128
        n_kv = wk.shape[0] // d_h
        n_rep = wq.shape[0] // wk.shape[0]
        ks, qs = [], []
        T = None
        for d in range(n_docs):
            h = load_file(str(args.capture_dir / f"doc{d}" / f"h_L{L:02d}.safetensors"))["h"].cuda().float()
            if T is None:
                T = h.shape[0]
            assert h.shape[0] == T, f"doc{d} has {h.shape[0]} tokens, doc0 has {T}"
            pos = torch.arange(T, device="cuda", dtype=torch.float64)
            s = (1.0 + a_scale * torch.log1p(torch.floor((pos + 1.0) / f_scale))).float()
            ks.append((h @ wk.T).view(T, n_kv, d_h))
            qs.append(((h @ wq.T) * s[:, None]).view(T, n_kv * n_rep, d_h))
            del h
        Tk = T - args.n_tail
        n_per_doc = (Tk - args.n_sink) // n_docs
        n_tail_per_doc = args.n_tail // n_docs

        for g in range(n_kv):
            if (L, g) in done or (args.max_corpora and n_run >= args.max_corpora):
                continue
            t0 = time.time()
            seed = 7919 * args.mix_seed + 1009 * L + g
            rng = np.random.default_rng(seed)
            keys, key_doc, q_ood, q_doc, q_id = [], [], [], [], []
            for d in range(n_docs):
                kpos = np.sort(rng.choice(np.arange(args.n_sink, Tk), n_per_doc, replace=False))
                tpos = np.sort(rng.choice(np.arange(Tk, T), n_tail_per_doc, replace=False))
                kpos_t, tpos_t = torch.as_tensor(kpos, device="cuda"), torch.as_tensor(tpos, device="cuda")
                keys.append(ks[d][kpos_t, g])
                key_doc.append(np.full(n_per_doc, d))
                qo = qs[d][tpos_t, g * n_rep : (g + 1) * n_rep].reshape(-1, d_h)
                q_ood.append(qo)
                q_doc.append(np.full(len(qo), d))
                q_id.append(ks[d][tpos_t, g])
            keys_t = torch.cat(keys).contiguous()
            key_doc = np.concatenate(key_doc)
            q_doc = np.concatenate(q_doc)
            qsets = {"ood": torch.cat(q_ood).contiguous(), "id": torch.cat(q_id).contiguous()}
            n_keys = keys_t.shape[0]

            base = keys_t.cpu().numpy().astype(np.float32)
            ivf, hnsw = build(base, args.nlist, args.hnsw_m, args.ef_construction, seed)
            rec = {
                "layer": L,
                "kv_head": g,
                "mix_seed": args.mix_seed,
                "n_docs": n_docs,
                "n_keys": n_keys,
                "n_q_ood": len(qsets["ood"]),
                "n_q_id": len(qsets["id"]),
            }
            for qname, q_t in qsets.items():
                sims = q_t @ keys_t.T
                q = q_t.cpu().numpy().astype(np.float32)
                qx = pad0(q)
                for k in KS:
                    gt = sims.topk(k, 1).indices.cpu().numpy()
                    ic, _ = sweep_ivf(ivf, q, gt, n_keys, k)
                    hc, _ = sweep_hnsw(hnsw, qx, gt, k)
                    iw, ir = work_at_target(ic, args.target)
                    hw, hr = work_at_target(hc, args.target)
                    rec[f"plain_{qname}_k{k}"] = {
                        "ivf_scan_frac": iw,
                        "ivf_reached": ir,
                        "hnsw_ndis": hw,
                        "hnsw_reached": hr,
                        "ivf_recall_op": dict((p, r) for p, r, _ in ic)[OP_NPROBE],
                        "hnsw_recall_op": dict((p, r) for p, r, _ in hc)[OP_EF],
                        "ivf_curve": ic,
                        "hnsw_curve": hc,
                    }
                    if qname == "ood" and k == 10:
                        rec["own_doc_share_top10"] = float((key_doc[gt] == q_doc[:, None]).mean())
            rec["seconds"] = round(time.time() - t0, 1)
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n_run += 1
            p10 = rec["plain_ood_k10"]
            print(L, g, rec["seconds"], "s | ood k10 hnsw@176", round(p10["hnsw_recall_op"], 3),
                  "own-doc", round(rec["own_doc_share_top10"], 3), flush=True)
        del ks, qs
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
