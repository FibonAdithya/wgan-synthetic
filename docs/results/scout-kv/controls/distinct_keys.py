"""Distinct keys among real-style queries' exact top-10, for three corpora (seed 0, head 0).

Run on tig-gpu-alt from /workspace/repo with PYTHONPATH=. ; output committed as distinct_keys.txt.
"""

import sys, numpy as np, torch
from src.eval import kv_gate
from src.kv.corpus import mixed_real_corpus, corpus_seed
from src.kv.transforms import load_layer_weights
cap="/workspace/kv/capture"
for layer, g in ((3,0),(19,0),(43,0)):
    hb=[h.cuda().float() for h in kv_gate.load_books(cap, layer, list(range(12,20)))]
    w=load_layer_weights(cap, layer); wk,wq=w["k_proj"].cuda().float(), w["q_proj"].cuda().float()
    c=mixed_real_corpus(hb, wk, wq, g, n_sink=1, n_tail=1024, seed=corpus_seed(layer,g,0))
    sims=np.ascontiguousarray(c.q_ood,np.float32)@np.ascontiguousarray(c.keys,np.float32).T
    gt=np.argpartition(-sims,9,axis=1)[:,:10]
    cnt=np.bincount(gt.ravel(),minlength=len(c.keys)); used=np.flatnonzero(cnt)
    kn=np.linalg.norm(c.keys,axis=1); order=np.argsort(-cnt)
    print(layer,g,"distinct_keys_in_top10",len(used),"of",len(c.keys),
          "| top1%share",cnt[order[:len(c.keys)//100]].sum()/gt.size,
          "| keynorm pct of used keys (median)",float(np.median((kn[used][:,None]>kn[None,:]).mean(1))),
          "| corr(norm,count) over all keys",float(np.corrcoef(kn,cnt)[0,1]), flush=True)
    del hb
