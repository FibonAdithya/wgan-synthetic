"""Capture Llama 4 Scout attention inputs for the scout_kv family.

Ported from docs/results/kv-cache-profile/spike/capture_scout.py (2026-10-16).

Records h -- the attention input after input_layernorm -- at the no-RoPE global layers,
plus the real W_q / W_k / W_v for those layers, and checks that

    k = W_k h            q = s(pos) * W_q h,   s(pos) = 1 + attn_scale * log1p(floor((pos+1)/floor_scale))

reproduces the q/k the model actually hands to attention. The design assumes this; if the
check fails, nothing downstream is valid and the script exits non-zero before the long run.

Books are numbered globally (PG-19 rows with >= ctx tokens, in row order) and written to
``doc{book}``. A run refuses to overwrite an existing book dir, and never rewrites an
existing top-level meta.json; it checks that the new run is compatible with it instead.

The model runs unquantized in bf16, held in CPU RAM and streamed to the GPU per module by
accelerate. Heavy imports (torch, transformers, ...) happen in main() so the pure helpers
can be imported and tested without them.
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

TOKEN_CHUNK = 4096
REL_ERR_LIMIT = 2e-2  # bf16 rounding is ~4e-3; a wrong formula or layer is off by >5e-2
META_COMPAT_KEYS = ("global_layers", "attn_scale", "floor_scale", "gqa")


def select_books(
    token_counts: Sequence[int], ctx: int, first_book: int, n_books: int
) -> list[int]:
    """PG-19 row indices of books first_book .. first_book+n_books-1.

    Only rows with >= ctx tokens are books; short rows are skipped without taking a number.
    """
    qualifying = [i for i, n in enumerate(token_counts) if n >= ctx]
    need = first_book + n_books
    if len(qualifying) < need:
        raise ValueError(
            f"only {len(qualifying)} rows have >= {ctx} tokens; need {need} "
            f"(books {first_book}..{need - 1})"
        )
    return qualifying[first_book:need]


def check_meta_compatible(existing: dict, new: dict) -> None:
    """Raise ValueError naming the first key that differs between two meta dicts."""
    for key in META_COMPAT_KEYS:
        if existing.get(key) != new.get(key):
            raise ValueError(
                f"meta.json incompatible on {key!r}: "
                f"existing {existing.get(key)!r} vs new {new.get(key)!r}"
            )


def planned_dirs(out_dir: Path, books: Sequence[int]) -> list[Path]:
    """Output dirs doc{b} for each book; FileExistsError if any already exists."""
    dirs = [out_dir / f"doc{b}" for b in books]
    for d in dirs:
        if d.exists():
            raise FileExistsError(
                f"{d} already exists; refusing to overwrite book data"
            )
    return dirs


GLOBAL_LAYERS: list[int] = []
CAP = {"on": False, "q": {}, "k": {}, "h": {}}


def install_patches():
    """Experts chunking patch and the SDPA capture registration (need transformers)."""
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.llama4 import modeling_llama4 as M

    def chunked_experts_forward(self, hidden_states):
        """Llama4TextExperts.forward, token-chunked. Experts are per-token, so this is exact."""
        n_exp = self.gate_up_proj.shape[0]
        hs = hidden_states.view(n_exp, -1, self.hidden_size)
        out = torch.empty_like(hs)
        for s in range(0, hs.shape[1], TOKEN_CHUNK):
            gate, up = torch.bmm(hs[:, s : s + TOKEN_CHUNK], self.gate_up_proj).chunk(
                2, dim=-1
            )
            out[:, s : s + TOKEN_CHUNK] = torch.bmm(
                up * self.act_fn(gate), self.down_proj
            )
        return out.view(-1, self.hidden_size)

    M.Llama4TextExperts.forward = (
        chunked_experts_forward  # before load, so accelerate wraps it
    )

    def capture_sdpa(module, query, key, value, attention_mask, **kw):
        if CAP["on"] and module.layer_idx in GLOBAL_LAYERS:
            CAP["q"][module.layer_idx] = (
                query[0].transpose(0, 1).contiguous()
            )  # (T, n_q, d)
            CAP["k"][module.layer_idx] = (
                key[0].transpose(0, 1).contiguous()
            )  # (T, n_kv, d)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register("capture_sdpa", capture_sdpa)


def h_hook(layer_idx):
    def hook(_module, _args, kwargs):
        if CAP["on"]:
            CAP["h"][layer_idx] = kwargs["hidden_states"][0].detach()

    return hook


def read_tensor(model_dir: Path, weight_map: dict, name: str):
    from safetensors import safe_open

    with safe_open(model_dir / weight_map[name], "pt") as f:
        return f.get_tensor(name)


def verify(cfg, weights, h, q, k, positions):
    """Recompute q/k from h and the checkpoint weights; return relative errors per layer."""
    import torch

    out = {}
    pos = positions.double()
    s = 1.0 + cfg.attn_scale * torch.log1p(torch.floor((pos + 1.0) / cfg.floor_scale))
    late = (
        positions >= cfg.floor_scale - 1
    )  # where s != 1; the scale check means nothing elsewhere
    for L in GLOBAL_LAYERS:
        hl = h[L].double()
        wq, wk = (weights[L][n].cuda().double() for n in ("q_proj", "k_proj"))
        k_hat = (hl @ wk.T).view(k[L].shape)
        q_raw = (hl @ wq.T).view(q[L].shape)
        q_hat = q_raw * s.cuda().view(-1, 1, 1)
        rel = lambda a, b: float((a - b.double()).norm() / b.double().norm())  # noqa: E731
        r = {"k_rel_err": rel(k_hat, k[L]), "q_rel_err": rel(q_hat, q[L])}
        if late.any():
            m = late.cuda()
            r["q_rel_err_late_with_scale"] = rel(q_hat[m], q[L][m])
            r["q_rel_err_late_without_scale"] = rel(
                q_raw[m], q[L][m]
            )  # negative control
        out[L] = r
    return out


def write_meta_if_absent(path: Path, meta: dict) -> None:
    """Write the top-level meta.json only when it does not exist yet."""
    if not path.exists():
        path.write_text(json.dumps(meta, indent=1))


def main():
    import pyarrow.parquet as pq
    import torch
    import transformers
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Llama4Config, Llama4ForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--pg19-parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--first-book", type=int, default=0)
    ap.add_argument("--n-books", type=int, default=4)
    ap.add_argument("--smoke-ctx", type=int, default=2048)
    ap.add_argument("--gpu-weights", default="30GiB")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    cfg = Llama4Config.from_pretrained(args.model_dir).text_config
    GLOBAL_LAYERS.extend(i for i, r in enumerate(cfg.no_rope_layers) if r == 0)
    assert all(cfg.layer_types[i] == "full_attention" for i in GLOBAL_LAYERS), (
        cfg.layer_types
    )
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    meta |= {
        "global_layers": GLOBAL_LAYERS,
        "gqa": f"query head j reads kv head j // {n_rep}",
        "attn_scale": cfg.attn_scale,
        "floor_scale": cfg.floor_scale,
    }
    print("global layers", GLOBAL_LAYERS, flush=True)

    meta_path = args.out_dir / "meta.json"
    if meta_path.exists():
        check_meta_compatible(json.loads(meta_path.read_text()), meta)

    # Tokenize rows lazily; select_books decides when enough qualifying rows are counted.
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    table = pq.read_table(args.pg19_parquet).to_pylist()
    counts: list[int] = []
    kept: dict[int, dict] = {}
    books = None
    for i, row in enumerate(table):
        ids = tok(row["text"], return_tensors="pt").input_ids[0]
        counts.append(len(ids))
        if len(ids) >= args.ctx:
            kept[i] = {
                "row": i,
                "title": row["short_book_title"],
                "n_tokens_full": len(ids),
                "ids": ids[: args.ctx],
            }
        try:
            books = select_books(counts, args.ctx, args.first_book, args.n_books)
            break
        except ValueError:
            continue
    if books is None:
        books = select_books(counts, args.ctx, args.first_book, args.n_books)  # raises
    docs = [kept[r] for r in books]
    book_dirs = planned_dirs(
        args.out_dir, range(args.first_book, args.first_book + len(books))
    )
    print(
        "books",
        dict(zip(range(args.first_book, args.first_book + len(books)), books)),
        flush=True,
    )

    weight_map = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    weights = {}
    wdir = args.out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    for L in GLOBAL_LAYERS:
        p = f"language_model.model.layers.{L}.self_attn."
        weights[L] = {
            n: read_tensor(args.model_dir, weight_map, p + n + ".weight")
            for n in ("q_proj", "k_proj", "v_proj")
        }
        wpath = wdir / f"layer{L:02d}.safetensors"
        has_v = False
        if wpath.exists():
            with safe_open(wpath, "pt") as f:
                has_v = "v_proj" in f.keys()  # noqa: SIM118
        if not has_v:
            save_file(weights[L], str(wpath))

    install_patches()
    t0 = time.time()
    model, info = Llama4ForCausalLM.from_pretrained(
        args.model_dir,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_weights, "cpu": "600GiB"},
        attn_implementation="capture_sdpa",
        output_loading_info=True,
    )
    meta["load_seconds"] = round(time.time() - t0, 1)
    unexpected_text = [
        k
        for k in info["unexpected_keys"]
        if "vision" not in k and "multi_modal" not in k
    ]
    print(
        "load",
        meta["load_seconds"],
        "s; missing",
        info["missing_keys"][:5],
        "unexpected(text)",
        unexpected_text[:5],
        flush=True,
    )
    if info["missing_keys"] or unexpected_text or info.get("mismatched_keys"):
        sys.exit(
            "weights did not load cleanly -- refusing to capture from a partly random model"
        )
    model.eval()
    for L in GLOBAL_LAYERS:
        model.model.layers[L].self_attn.register_forward_pre_hook(
            h_hook(L), with_kwargs=True
        )

    def run(ids, d_out):
        CAP.update(on=True, q={}, k={}, h={})
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.inference_mode():
            model.model(input_ids=ids[None].to("cuda:0"), use_cache=False)
        CAP["on"] = False
        res = {
            "forward_seconds": round(time.time() - t, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
        res["verify"] = verify(
            cfg, weights, CAP["h"], CAP["q"], CAP["k"], torch.arange(len(ids))
        )
        res["worst_rel_err"] = max(
            max(v["k_rel_err"], v["q_rel_err"]) for v in res["verify"].values()
        )
        if d_out is not None:
            d_out.mkdir(exist_ok=True)
            for L in GLOBAL_LAYERS:
                save_file(
                    {"h": CAP["h"][L].cpu().contiguous()},
                    str(d_out / f"h_L{L:02d}.safetensors"),
                )
        CAP.update(q={}, k={}, h={})
        torch.cuda.empty_cache()
        return res

    # Smoke: short prefix of the first selected book, then a prefix past floor_scale so
    # the temperature term is exercised.
    for n in (args.smoke_ctx, min(args.ctx, cfg.floor_scale + 2048)):
        r = run(docs[0]["ids"][:n], None)
        meta.setdefault("smoke", {})[n] = r
        print("smoke", n, json.dumps(r), flush=True)
        if not math.isfinite(r["worst_rel_err"]) or r["worst_rel_err"] > REL_ERR_LIMIT:
            write_meta_if_absent(meta_path, meta)
            sys.exit(
                f"q/k reconstruction failed at n={n}: worst rel err {r['worst_rel_err']:.3g}"
            )

    meta["docs"] = []
    write_meta_if_absent(meta_path, meta)
    for b, doc, d_out in zip(
        range(args.first_book, args.first_book + len(books)), docs, book_dirs
    ):
        r = run(doc["ids"], d_out)
        entry = (
            {k: v for k, v in doc.items() if k != "ids"}
            | {"n_tokens": len(doc["ids"])}
            | r
        )
        (d_out / "book.json").write_text(json.dumps(entry, indent=1))
        print("book", b, doc["title"], json.dumps(r), flush=True)
        if r["worst_rel_err"] > REL_ERR_LIMIT:
            sys.exit(f"q/k reconstruction failed on book {b}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
