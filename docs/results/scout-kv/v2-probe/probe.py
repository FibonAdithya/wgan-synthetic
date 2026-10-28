"""scout_kv v2 probe (throwaway spike): generate whole books, token by token or
from a drifting latent, and compare.

Real KV caches are sequence-local (../locality/): from L15 on, consecutive h
are alike, and at L23-L35 a key's nearest keys are mostly nearby tokens of its
own book. v1 draws every h independently, so it cannot produce these local
runs. This probe trains three small generators that write books position by
position, with everything else held equal, and scores the books they write
the same way real books are scored.

Arms (the token decoder is v1's trunk, 1024-2048-4096-5120, in all three):

  A  token by token:  h_t = dec(z_t, s_{t-1}, pos_t),  s_t = GRU(enc(h_t), s_{t-1})
  B  drifting latent: h_t = dec(z_t, u_book(t), pos_t), u_book a random smooth
                      signal over position (64 channels, length scales 2 to
                      16,384 tokens, random Fourier features, new per book)
  C  control:         h_t = dec(z_t, pos_t), tokens independent given position

pos_t is sinusoidal features of t/T plus a tail bit (t >= T - 1024), so every
arm knows position precisely. The critic is shared: it sees windows of 8
tokens at stride 1, 4, 16, 64, 256 or 1024 ending at position t_end, scores
each token (with its position) and the window as a whole. Real windows are
drawn at the same (t_end, stride) from a random training book, so the critic
compares like with like. t_end is uniform over the book and the stride
uniform over those that fit, for every arm. A trains by streaming: 64
synthetic books are written 8 tokens per step, with gradient through the new
8 tokens and earlier tokens read back detached (truncated backpropagation).

Data: books 0-11 of one layer, centred with their mean (as v1). Writes 8
books named 12-19 into a capture-format directory, so the unchanged gate
(`kv_gate measure-real`), locality.py and hub_position.py score them.

    python docs/results/scout-kv/v2-probe/probe.py train --arm B --layer 27 --seed 42 \
        --capture-dir /workspace/kv/capture --out /workspace/kv/v2probe/L27/B_s42
    python docs/results/scout-kv/v2-probe/probe.py sample --run /workspace/kv/v2probe/L27/B_s42 \
        --capture-dir /workspace/kv/capture
    python docs/results/scout-kv/v2-probe/probe.py intrinsic --run /workspace/kv/v2probe/L27/B_s42

Ablation switches (2026-10-26). Every arm lost LID against v1, so each change
the probe made to v1 can be undone on its own:

  --critic token     v1's per-token critic (x + position -> 2048-1024-512-1)
                     instead of the window critic; tokens still come in windows
  --pos flag         position input is the tail bit only, as v1's flag
  --tail-share 0.25  a quarter of the windows lie inside the tail, as v1's
                     tail_batch_share; the rest end in the body (arms B, C)

The ablation found the window critic costs LID and v1's per-token critic
restores it. --critic both trains the two critics side by side, each with its
own optimizer and gradient penalty, on the same batches. The generator loss is
-(token score) - window_weight * (window score). At log steps the per-token
gradient each term puts on the fake tokens is recorded (grad_tok, grad_win).

Latent size (2026-10-27): --cond sets arm B's latent channels (default 64)
and --max-len its longest length scale (default 16,384 tokens).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.kv_capture import h_path, read_safetensor, write_bf16_safetensor

TRAIN_BOOKS = list(range(12))
OUT_BOOKS = list(range(12, 20))
N_TAIL = 1024
W = 8
STRIDES = (1, 4, 16, 64, 256, 1024)
N_WIN = 64
LATENT = 128
COND = 64
N_FREQ_POS = 8
POS_DIM = 2 * N_FREQ_POS + 2
CRITIC_HIDDEN = (2048, 1024, 512)  # v1's critic
ENC = 256
CHUNK = 8
RING = 8192
N_RFF = 8
SLOPE = 0.2


def mlp(dims):
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(SLOPE)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def pos_dim(mode: str) -> int:
    return {"full": POS_DIM, "flag": 1}[mode]


def pos_features(t: torch.Tensor, T: int, mode: str = "full") -> torch.Tensor:
    tail = (t >= T - N_TAIL).float()[..., None]
    if mode == "flag":
        return tail
    assert mode == "full", mode
    x = t.float() / T
    f = 2 * math.pi * x[..., None] * (2.0 ** torch.arange(N_FREQ_POS, device=t.device))
    return torch.cat([f.sin(), f.cos(), x[..., None], tail], -1)


class DriftLatent(nn.Module):
    """u_c(t) = sum_m (a_cm cos(w_cm t) + b_cm sin(w_cm t)) / sqrt(M): a smooth
    random signal per channel with length scale l_c; a, b ~ N(0,1) per book."""

    def __init__(self, seed: int, cond: int = COND, max_len: float = 16384.0):
        super().__init__()
        self.cond = cond
        g = torch.Generator().manual_seed(seed)
        ell = torch.logspace(math.log10(2.0), math.log10(max_len), cond)
        self.register_buffer("omega", torch.randn(cond, N_RFF, generator=g) / ell[:, None])

    def sample_books(self, n, dev):
        return torch.randn(n, 2, self.cond, N_RFF, device=dev)

    def forward(self, coef, t):  # coef (N,2,C,M), t (N,L) -> (N,L,C)
        ph = t.float()[..., None, None] * self.omega
        a, b = coef[:, None, 0], coef[:, None, 1]
        return (a * ph.cos() + b * ph.sin()).sum(-1) / math.sqrt(N_RFF)


class ARState(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.enc = mlp([D, ENC, ENC])
        self.cell = nn.GRUCell(ENC, COND)
        self.init = nn.Linear(COND, COND)

    def s0(self, n, dev):
        return torch.tanh(self.init(torch.randn(n, COND, device=dev)))

    def step(self, h, s):
        return self.cell(self.enc(h), s)


class Gen(nn.Module):
    def __init__(self, arm, D, seed, pos="full", cond=COND, max_len=16384.0):
        super().__init__()
        self.arm = arm
        assert arm == "B" or (cond == COND and max_len == 16384.0), "--cond/--max-len are for arm B"
        self.dec = mlp([LATENT + (0 if arm == "C" else cond) + pos_dim(pos), 1024, 2048, 4096, D])
        if arm == "A":
            self.ar = ARState(D)
        if arm == "B":
            self.lat = DriftLatent(seed, cond, max_len)

    def decode(self, z, c, pf):
        return self.dec(torch.cat([z, pf] if c is None else [z, c, pf], -1))


class TokenCritic(nn.Module):
    """v1's critic: one score per token from (x, position). Called on windows
    of length 1, so the gradient penalty is per token."""

    def __init__(self, D, pos="full"):
        super().__init__()
        self.net = mlp([D + pos_dim(pos), *CRITIC_HIDDEN, 1])

    def forward(self, x, pf, s_idx):  # x (N,1,D)
        assert x.shape[1] == 1, x.shape
        return self.net(torch.cat([x[:, 0], pf[:, 0]], -1)).squeeze(-1)


class WindowCritic(nn.Module):
    def __init__(self, D, pos="full"):
        super().__init__()
        self.tok = mlp([D + pos_dim(pos), 1024, 512])
        self.tok_head = nn.Linear(512, 1)
        self.win = mlp([W * 512 + len(STRIDES), 1024, 512, 1])

    def forward(self, x, pf, s_idx):  # x (N,W,D), pf (N,W,P), s_idx (N,)
        e = F.leaky_relu(self.tok(torch.cat([x, pf], -1)), SLOPE)
        t = self.tok_head(e).mean(1)
        oh = F.one_hot(s_idx, len(STRIDES)).to(x.dtype)
        w = self.win(torch.cat([e.flatten(1), oh], -1))
        return (t + w).squeeze(-1)


STRIDES_T = torch.tensor(STRIDES)


def window_positions(t_end, s):  # oldest -> newest, (N,W)
    return t_end[:, None] - s[:, None] * torch.arange(W - 1, -1, -1, device=t_end.device)


def sample_stride(t_end, lo, g=None):
    """Uniform over the strides whose window fits in [lo, t_end]."""
    st = STRIDES_T.to(t_end.device)
    ok = (t_end[:, None] - (W - 1) * st[None]) >= lo[:, None]
    n_ok = ok.sum(1)
    assert int(n_ok.min()) >= 1, "no stride fits; t_end too early"
    r = (torch.rand(len(t_end), device=t_end.device) * n_ok).long()
    return r  # strides are increasing, so the r-th fitting stride is index r


class RealBooks:
    def __init__(self, capture_dir, layer, dev):
        hs = []
        for b in TRAIN_BOOKS:
            hs.append(torch.from_numpy(read_safetensor(h_path(capture_dir, b, layer), "h")).to(torch.bfloat16))
        Ts = {h.shape[0] for h in hs}
        assert len(Ts) == 1, Ts
        self.T = Ts.pop()
        self.h = torch.stack(hs).to(dev)
        self.D = self.h.shape[-1]
        self.mean = sum(self.h[i, 1:].float().mean(0) for i in range(len(hs))) / len(hs)

    def windows(self, pos):  # pos (N,W) -> (N,W,D) centred
        b = torch.randint(len(TRAIN_BOOKS), (pos.shape[0],), device=pos.device)
        return self.h[b[:, None], pos].float() - self.mean


class Streams:
    """64 synthetic books written CHUNK tokens per step (arm A)."""

    def __init__(self, gen, n, T, D, dev):
        self.n, self.T, self.dev = n, T, dev
        self.pos = torch.randint(1, T - CHUNK, (n,), device=dev)  # staggered first life
        self.born = self.pos.clone()
        with torch.no_grad():
            self.s = gen.ar.s0(n, dev)
        self.buf = torch.zeros(n, RING, D, dtype=torch.bfloat16, device=dev)

    def advance(self, gen):
        """Write CHUNK new tokens; returns (chunk (n,CHUNK,D), chunk_start (n,))."""
        reset = self.pos + CHUNK - 1 > self.T - 1
        if reset.any():
            with torch.no_grad():
                s0 = gen.ar.s0(int(reset.sum()), self.dev)
            self.s = self.s.clone()
            self.s[reset] = s0
            self.pos[reset] = 1
            self.born[reset] = 1
        start = self.pos.clone()
        s, outs = self.s, []
        for j in range(CHUNK):
            t = start + j
            h = gen.decode(torch.randn(self.n, LATENT, device=self.dev), s, pos_features(t, self.T))
            s = gen.ar.step(h, s)
            outs.append(h)
        chunk = torch.stack(outs, 1)
        idx = (start[:, None] + torch.arange(CHUNK, device=self.dev)) % RING
        self.buf[torch.arange(self.n, device=self.dev)[:, None], idx] = chunk.detach().to(torch.bfloat16)
        self.s = s.detach()
        self.pos = start + CHUNK
        return chunk, start

    def windows(self, chunk, start):
        t_end = start + CHUNK - 1
        s_idx = sample_stride(t_end, self.born)
        pos = window_positions(t_end, STRIDES_T.to(self.dev)[s_idx])
        rows = torch.arange(self.n, device=self.dev)[:, None]
        x = self.buf[rows, pos % RING].float()
        in_chunk = pos >= start[:, None]
        ci = (pos - start[:, None]).clamp(0, CHUNK - 1)
        x = torch.where(in_chunk[..., None], chunk[rows, ci], x)
        return x, pos, s_idx


def window_ends(n, T, dev, tail_share=None):
    """(t_end, lo): uniform over the book, or with tail_share of the windows
    wholly inside the tail and the rest ending in the body."""
    if tail_share is None:
        t_end = torch.randint(W, T, (n,), device=dev)
        return t_end, torch.ones_like(t_end)
    in_tail = torch.rand(n, device=dev) < tail_share
    t_body = torch.randint(W, T - N_TAIL, (n,), device=dev)
    t_tail = torch.randint(T - N_TAIL + W - 1, T, (n,), device=dev)
    t_end = torch.where(in_tail, t_tail, t_body)
    lo = torch.where(in_tail, torch.full_like(t_end, T - N_TAIL), torch.ones_like(t_end))
    return t_end, lo


def fake_windows(gen, n, T, dev, tail_share=None, pos_mode="full"):
    """Arms B and C: n windows, each from its own fresh book."""
    t_end, lo = window_ends(n, T, dev, tail_share)
    s_idx = sample_stride(t_end, lo)
    pos = window_positions(t_end, STRIDES_T.to(dev)[s_idx])
    z = torch.randn(n, W, LATENT, device=dev)
    c = gen.lat(gen.lat.sample_books(n, dev), pos) if gen.arm == "B" else None
    return gen.decode(z, c, pos_features(pos, T, pos_mode)), pos, s_idx


def gp(critic, real, fake, pf, s_idx):
    a = torch.rand(real.shape[0], 1, 1, device=real.device)
    xi = (a * real + (1 - a) * fake).requires_grad_(True)
    out = critic(xi, pf, s_idx)
    g = torch.autograd.grad(out.sum(), xi, create_graph=True)[0]
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


def train(a):
    from src.train.gpu_lock import claim_gpu

    out = Path(a.out)
    if (out / "final.pt").exists():
        raise SystemExit(f"{out} already trained")
    if a.arm == "A" and (a.tail_share is not None or a.pos != "full"):
        raise SystemExit("--tail-share and --pos flag are for arms B and C only")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    with claim_gpu(dev, run_dir=out, timeout_s=a.lock_timeout):
        data = RealBooks(a.capture_dir, a.layer, dev)
        T, D = data.T, data.D
        gen = Gen(a.arm, D, a.seed, a.pos, a.cond, a.max_len).to(dev)
        # (name, critic, optimizer, scores single tokens, weight in the generator loss)
        kinds = {"token": [("tok", True, 1.0)], "window": [("win", False, 1.0)],
                 "both": [("tok", True, 1.0), ("win", False, a.window_weight)]}[a.critic]
        critics = []
        for name, tokens, weight in kinds:
            c = (TokenCritic if tokens else WindowCritic)(D, a.pos).to(dev)
            critics.append((name, c, torch.optim.Adam(c.parameters(), lr=1e-4, betas=(0.0, 0.9)), tokens, weight))
        ema = copy.deepcopy(gen).requires_grad_(False)
        og = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
        streams = Streams(gen, N_WIN, T, D, dev) if a.arm == "A" else None
        cfg = {k: v for k, v in vars(a).items() if k != "func"}
        cfg |= {"T": T, "D": D, "W": W, "strides": STRIDES, "n_win": N_WIN, "chunk": CHUNK,
                "cond": a.cond, "max_len": a.max_len, "latent": LATENT, "lambda_gp": a.lambda_gp, "n_critic": 3,
                "ema_decay": 0.999, "train_books": TRAIN_BOOKS}
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")
        np.save(out / "mean.npy", data.mean.cpu().numpy())

        def fake_batch(grad):
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                if streams is not None:
                    chunk, start = streams.advance(gen)
                    return streams.windows(chunk, start)
                return fake_windows(gen, N_WIN, T, dev, a.tail_share, a.pos)

        def as_tokens(x, pf, s_idx, tokens):  # token critic: each token is its own length-1 window
            if not tokens:
                return x, pf, s_idx
            return x.reshape(-1, 1, x.shape[-1]), pf.reshape(-1, 1, pf.shape[-1]), s_idx.repeat_interleave(W)

        log = open(out / "train_log.jsonl", "a")
        t0 = time.time()
        for step in range(1, a.steps + 1):
            stats = {}
            for _ in range(3):
                fake_w, pos, s_idx_w = fake_batch(False)
                real_w = data.windows(pos)
                pf_w = pos_features(pos, T, a.pos)
                for name, critic, oc, tokens, _ in critics:
                    real, _, _ = as_tokens(real_w, pf_w, s_idx_w, tokens)
                    fake, pf, s_idx = as_tokens(fake_w, pf_w, s_idx_w, tokens)
                    d_real, d_fake = critic(real, pf, s_idx), critic(fake, pf, s_idx)
                    pen = gp(critic, real, fake, pf, s_idx)
                    loss_c = d_fake.mean() - d_real.mean() + a.lambda_gp * pen
                    oc.zero_grad(set_to_none=True)
                    loss_c.backward()
                    oc.step()
                    stats[name] = (d_real.mean() - d_fake.mean(), pen)
            fake_w, pos, s_idx_w = fake_batch(True)
            pf_w = pos_features(pos, T, a.pos)
            terms = {name: -weight * critic(*as_tokens(fake_w, pf_w, s_idx_w, tokens)).mean()
                     for name, critic, _, tokens, weight in critics}
            loss_g = sum(terms.values())
            logging = step % 250 == 0 or step == 1
            grads = {}
            if logging and len(terms) > 1:  # per-token gradient norm each term puts on the fakes
                for name, term in terms.items():
                    g = torch.autograd.grad(term, fake_w, retain_graph=True)[0]
                    grads[f"grad_{name}"] = float(g.norm(dim=-1).mean())
            og.zero_grad(set_to_none=True)
            loss_g.backward()
            og.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), gen.parameters()):
                    pe.lerp_(p, 1 - 0.999)
            if logging:
                (wd, pen), rec = stats[critics[0][0]], {"step": step}
                rec |= {"w_dist": float(wd), "gp": float(pen)}
                for name, (wd2, pen2) in list(stats.items())[1:]:
                    rec |= {f"w_dist_{name}": float(wd2), f"gp_{name}": float(pen2)}
                rec |= grads | {"loss_g": float(loss_g), "seconds": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step % 5000 == 0 or step == a.steps:
                ck = {"step": step, "gen": gen.state_dict(), "ema": ema.state_dict(),
                      "critic": critics[0][1].state_dict()}
                ck |= {f"critic_{c[0]}": c[1].state_dict() for c in critics[1:]}
                torch.save(ck, out / "last.pt")
        (out / "last.pt").rename(out / "final.pt")
        print("TRAINED", out, round(time.time() - t0, 1), "s", flush=True)


@torch.no_grad()
def sample(a):
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, D = cfg["T"], cfg["D"]
    pm = cfg.get("pos", "full")
    gen = Gen(cfg["arm"], D, cfg["seed"], pm, cfg.get("cond", COND), cfg.get("max_len", 16384.0)).to(dev)
    gen.load_state_dict(torch.load(run / "final.pt", map_location=dev)["ema"])
    gen.eval()
    mean = torch.from_numpy(np.load(run / "mean.npy")).to(dev)
    cap = run / "capture"
    if cap.exists():
        raise SystemExit(f"{cap} exists")
    (cap / "weights").parent.mkdir(parents=True)
    (cap / "weights").symlink_to(Path(a.capture_dir).resolve() / "weights")
    torch.manual_seed(a.sample_seed)
    n = len(OUT_BOOKS)
    h = torch.zeros(n, T, D, device=dev)  # position 0 (sink) left at the mean
    t0 = time.time()
    if cfg["arm"] == "A":
        s = gen.ar.s0(n, dev)
        for t in range(1, T):
            tt = torch.full((n,), t, device=dev)
            x = gen.decode(torch.randn(n, LATENT, device=dev), s, pos_features(tt, T, pm))
            s = gen.ar.step(x, s)
            h[:, t] = x
    else:
        coef = gen.lat.sample_books(n, dev) if cfg["arm"] == "B" else None
        for s0 in range(1, T, 2048):
            t = torch.arange(s0, min(s0 + 2048, T), device=dev)[None].expand(n, -1)
            c = gen.lat(coef, t) if coef is not None else None
            h[:, t[0]] = gen.decode(torch.randn(n, t.shape[1], LATENT, device=dev), c, pos_features(t, T, pm))
    h += mean
    for i, b in enumerate(OUT_BOOKS):
        p = h_path(cap, b, cfg["layer"])
        p.parent.mkdir(parents=True, exist_ok=True)
        write_bf16_safetensor(p, "h", h[i].cpu().numpy())
    print("SAMPLED", cap, round(time.time() - t0, 1), "s", flush=True)


def intrinsic(a):
    """LID, contrast, hubness per flag, with v1's intrinsic.py settings."""
    from src.data.kv_capture import load_kv_capture
    from src.eval.ann_difficulty import compute, summary
    from src.models.conditional import BODY, TAIL

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    out = run / "intrinsic.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists")
    x, flags = load_kv_capture(run / "capture", cfg["layer"], OUT_BOOKS, n_sink=1, n_tail=N_TAIL)
    for name, f in (("body", BODY), ("tail", TAIL)):
        t = time.time()
        rec = {"layer": cfg["layer"], "source": run.name, "flag": name,
               "n_rows": int((flags == f).sum())}
        rec |= summary(compute(x[flags == f], k=100, k_hub=10, nlist=256, max_rows=20000, seed=42, metric="l2"))
        rec["seconds"] = round(time.time() - t, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=["A", "B", "C"], required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock-timeout", type=float, default=0.0)
    p.add_argument("--critic", choices=["window", "token", "both"], default="window")
    p.add_argument("--window-weight", type=float, default=0.1)
    p.add_argument("--cond", type=int, default=COND)
    p.add_argument("--max-len", type=float, default=16384.0)
    p.add_argument("--pos", choices=["full", "flag"], default="full")
    p.add_argument("--tail-share", type=float, default=None)
    p.set_defaults(func=train)
    p = sub.add_parser("sample")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--sample-seed", type=int, default=0)
    p.set_defaults(func=sample)
    p = sub.add_parser("intrinsic")
    p.add_argument("--run", type=Path, required=True)
    p.set_defaults(func=intrinsic)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
