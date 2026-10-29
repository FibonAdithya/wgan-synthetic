"""Write synthetic books from a scout_kv v2 (book_latent) run in the real capture format.

    python -m src.sample.sample_books --run-dir R --capture-dir CAP --out-dir R/capture \
        [--seed 0] [--checkpoint final.pt] [--book-ids 12-19]

The books land at `doc{id}/h_L{layer}.safetensors` with `weights` linked to the
capture's, so `kv_gate measure-real --capture-dir R/capture --source v2` scores them
as it scores real books. Layer, n_sink and n_tail come from the run config and T from
the checkpoint; the training books are never read.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from torch import Tensor

from src.data.kv_capture import h_path, write_bf16_safetensor
from src.eval.kv_gate import parse_books
from src.models.book_latent import (
    BookLatentGenerator,
    book_latent_settings,
    build_book_latent,
)
from src.train.book_windows import tail_flags
from src.train.train_wgan_gp import load_ema_into_model


def load_book_latent_run(
    run_dir: Path, checkpoint_name: str = "final.pt", device="cpu"
) -> tuple[BookLatentGenerator, Tensor, dict]:
    """The generator with bias-corrected EMA weights, the data mean and the run config."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    if config["model"].get("generator_type") != "book_latent":
        raise ValueError(
            f"{run_dir} is not a book_latent run (generator_type "
            f"{config['model'].get('generator_type')!r}); use the v0/v1 sampler"
        )
    ckpt = torch.load(
        run_dir / checkpoint_name, map_location=device, weights_only=False
    )
    if ckpt.get("generator_type") != "book_latent":
        raise ValueError(f"{checkpoint_name} is not a book_latent checkpoint")
    if ckpt["book_latent"] != book_latent_settings(config):
        raise ValueError(
            f"{checkpoint_name} book_latent settings differ from {run_dir}/run_config.yaml"
        )
    gen, _, _ = build_book_latent(config)
    gen.load_state_dict(ckpt["generator_state_dict"])
    load_ema_into_model(
        ckpt["ema_params"],
        gen,
        decay=ckpt["ema_decay"],
        ema_step=ckpt["ema_step"],
    )
    gen.to(device).eval()
    ckpt["data_mean"] = ckpt["data_mean"].to(device)
    config["_T"] = int(ckpt["T"])
    return gen, ckpt["data_mean"], config


@torch.no_grad()
def sample_books(
    gen: BookLatentGenerator,
    mean: Tensor,
    *,
    n_books: int,
    T: int,
    n_sink: int,
    n_tail: int,
    seed: int,
    chunk: int = 2048,
    device="cpu",
) -> Tensor:
    """`(n_books, T, D)` float32: one latent draw per book, positions n_sink.. in chunks.

    Seeds the global torch RNG with `seed` (the committed v2 samples were drawn this
    way), so a caller's own torch random state is reset.
    """
    torch.manual_seed(seed)
    mean = mean.to(device)
    h = mean.expand(n_books, T, -1).clone()
    coef = gen.latent.sample_coefficients(n_books, device)
    for start in range(n_sink, T, chunk):
        pos = torch.arange(start, min(start + chunk, T), device=device)
        pos = pos[None].expand(n_books, -1)
        z = torch.randn(n_books, pos.shape[1], gen.latent_dim, device=device)
        u = gen.latent(coef, pos)
        h[:, start : start + pos.shape[1]] = (
            gen(z, u, tail_flags(pos, T, n_tail)) + mean
        )
    return h


def write_books(
    h: Tensor, out_dir: Path, layer: int, book_ids: Sequence[int], weights_from: Path
) -> None:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite")
    if len(book_ids) != h.shape[0]:
        raise ValueError(f"{len(book_ids)} book ids for {h.shape[0]} books")
    for i, b in enumerate(book_ids):
        p = h_path(out_dir, b, layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_bf16_safetensor(p, "h", h[i].cpu().numpy())
    (out_dir / "weights").symlink_to(Path(weights_from).resolve())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--book-ids", type=parse_books, default=parse_books("12-19"))
    a = ap.parse_args(argv)
    if a.out_dir.exists():
        raise FileExistsError(f"{a.out_dir} exists; refusing to overwrite")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen, mean, cfg = load_book_latent_run(a.run_dir, a.checkpoint, device)
    data = cfg["data"]
    h = sample_books(
        gen,
        mean,
        n_books=len(a.book_ids),
        T=cfg["_T"],
        n_sink=int(data["kv_n_sink"]),
        n_tail=int(data["kv_n_tail"]),
        seed=a.seed,
        device=device,
    )
    write_books(
        h, a.out_dir, int(data["kv_layer"]), a.book_ids, a.capture_dir / "weights"
    )
    print(f"wrote {len(a.book_ids)} books to {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
