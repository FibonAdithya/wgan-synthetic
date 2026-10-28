"""WGAN-GP trainer for scout_kv v2 (`model.generator_type: book_latent`).

The data unit is a whole book. Each batch is `n_windows` windows of W tokens:
the first ones end in the body, `round(tail_window_share * n_windows)` lie
wholly in the tail. Every fake window is a fresh synthetic book with its own
drifting latent; its real counterpart is read at the same positions from a
uniformly drawn training book. Two critics are trained side by side on each
batch, each with its own optimizer and gradient penalty: v1's per-token critic
on the n*W tokens and a window critic on the windows. The generator loss is
`-mean(token score) - window_weight * mean(window score)`.

    python -m src.train.train_book_latent --config PATH [--resume PATH]

`last.pt` is overwritten every `save_every` steps and renamed to `final.pt`
after the last step. There is no holdout and no checkpoint selector; v2 is
judged on the final EMA weights (`src.sample.sample_books`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch import Tensor

from src.data.kv_capture import load_kv_books
from src.device import resolve_device
from src.models.book_latent import (
    BookLatentGenerator,
    WindowCritic,
    book_latent_settings,
    build_book_latent,
)
from src.train.book_windows import (
    RealBooks,
    sample_stride,
    tail_flags,
    window_ends,
    window_positions,
)
from src.train.gpu_lock import claim_gpu
from src.train.train_wgan_gp import (
    ema_update,
    gpu_preflight,
    gradient_penalty,
    init_ema_params,
    set_seed,
)

GENERATOR_TYPE = "book_latent"
PREPROCESS = {"center": True, "whiten": False, "l2_normalize": False}

# Every key the trainer reads; none has a default.
REQUIRED = (
    ("seed",),
    ("device",),
    ("output_dir",),
    ("data", "real_path"),
    ("data", "format"),
    ("data", "kv_layer"),
    ("data", "kv_books"),
    ("data", "kv_n_sink"),
    ("data", "kv_n_tail"),
    ("data", "descriptor_dim"),
    ("data", "preprocess"),
    ("model", "generator_type"),
    ("model", "latent_dim"),
    ("model", "generator_hidden_dims"),
    ("model", "critic_hidden_dims"),
    ("model", "negative_slope"),
    ("model", "output_normalize"),
    ("model", "n_flags"),
    ("model", "book_latent", "channels"),
    ("model", "book_latent", "n_features"),
    ("model", "book_latent", "min_length"),
    ("model", "book_latent", "max_length"),
    ("model", "window_critic", "window"),
    ("model", "window_critic", "strides"),
    ("model", "window_critic", "token_hidden_dims"),
    ("model", "window_critic", "window_hidden_dims"),
    ("training", "n_windows"),
    ("training", "tail_window_share"),
    ("training", "num_gen_steps"),
    ("training", "n_critic"),
    ("training", "lr_g"),
    ("training", "lr_d"),
    ("training", "betas"),
    ("training", "lambda_gp"),
    ("training", "window_weight"),
    ("training", "ema_decay"),
    ("training", "log_every"),
    ("training", "save_every"),
)


def _n_tail_windows(training: dict) -> int:
    return round(float(training["tail_window_share"]) * int(training["n_windows"]))


def validate_config(config: dict) -> None:
    """Raise KeyError for a missing v2 key and ValueError for a setting v2 cannot train."""
    for path in REQUIRED:
        node = config
        for i, key in enumerate(path):
            if key not in node:
                raise KeyError(f"config has no {'.'.join(path[: i + 1])}")
            node = node[key]
    model, data, training = config["model"], config["data"], config["training"]
    if model["generator_type"] != GENERATOR_TYPE:
        raise ValueError(
            f"model.generator_type must be {GENERATOR_TYPE!r}, got {model['generator_type']!r}"
        )
    if model["n_flags"] != 2:
        raise ValueError(f"model.n_flags must be 2, got {model['n_flags']!r}")
    if model["output_normalize"] is not False:
        raise ValueError(
            f"model.output_normalize must be false, got {model['output_normalize']!r}"
        )
    if data["format"] != "kv_capture":
        raise ValueError(f"data.format must be 'kv_capture', got {data['format']!r}")
    if dict(data["preprocess"]) != PREPROCESS:
        raise ValueError(
            f"data.preprocess must be {PREPROCESS} (centring only), got {data['preprocess']}"
        )
    for key in ("kv_n_tail", "kv_n_sink"):
        v = data[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"data.{key} must be an int >= 1, got {v!r}")
    strides = list(model["window_critic"]["strides"])
    if any(b <= a for a, b in zip(strides, strides[1:], strict=False)):
        raise ValueError(
            f"model.window_critic.strides must be strictly increasing, got {strides}"
        )
    n_windows, n_tail_windows = int(training["n_windows"]), _n_tail_windows(training)
    if not 1 <= n_tail_windows <= n_windows - 1:
        raise ValueError(
            f"training.tail_window_share={training['tail_window_share']} gives "
            f"{n_tail_windows} tail windows of {n_windows}; both classes need at least one"
        )


def window_gradient_penalty(
    critic: WindowCritic,
    real: Tensor,
    fake: Tensor,
    flags: Tensor,
    stride_idx: Tensor,
) -> Tensor:
    """WGAN-GP over whole windows: one interpolation weight per window, the norm of
    the summed score's gradient over all W x D entries of each window."""
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
    xi = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    out = critic(xi, flags, stride_idx)
    g = torch.autograd.grad(out.sum(), xi, create_graph=True)[0]
    return ((g.flatten(1).norm(dim=1) - 1.0) ** 2).mean()


def fake_windows(
    gen: BookLatentGenerator, pos: Tensor, T: int, n_tail: int
) -> tuple[Tensor, Tensor]:
    """One fresh synthetic book per window, read at `pos (n, W)`: `(x (n, W, D), flags (n, W))`."""
    n, w = pos.shape
    flags = tail_flags(pos, T, n_tail)
    u = gen.latent(gen.latent.sample_coefficients(n, pos.device), pos)
    z = torch.randn(n, w, gen.latent_dim, device=pos.device)
    return gen(z, u, flags), flags


def generator_loss(
    token_score: Tensor, window_score: Tensor, window_weight: float
) -> Tensor:
    return -token_score.mean() - window_weight * window_score.mean()


def _first_difference(saved, wanted, prefix: str = "") -> str | None:
    """Dotted key of the first entry where two settings dicts differ, in `wanted`'s order."""
    for key in list(wanted) + [k for k in saved if k not in wanted]:
        name = f"{prefix}{key}"
        a, b = saved.get(key), wanted.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            inner = _first_difference(a, b, f"{name}.")
            if inner is not None:
                return inner
        elif a != b:
            return name
    return None


def _check_resume_record(
    ckpt: dict, config: dict, gen: BookLatentGenerator, resume
) -> None:
    """The refusals that need no data: checkpoint type, settings, omega."""
    if ckpt.get("generator_type") != GENERATOR_TYPE:
        raise ValueError(
            f"{resume} has generator_type {ckpt.get('generator_type')!r}, not "
            f"{GENERATOR_TYPE!r}; only a book_latent checkpoint can be resumed here"
        )
    settings = book_latent_settings(config)
    if ckpt["book_latent"] != settings:
        key = _first_difference(ckpt["book_latent"], settings)
        raise ValueError(
            f"{resume} was trained with different settings: {key} differs from the config"
        )
    saved = ckpt["generator_state_dict"]["latent.omega"].cpu()
    if not torch.equal(saved, gen.latent.omega.cpu()):
        raise ValueError(
            f"{resume} has a latent omega other than the one built from seed "
            f"{config['seed']}; a resume cannot change the seed"
        )


def _checkpoint(
    step, T, config, gen, token, window, optims, ema, ema_step, mean
) -> dict:
    optim_g, optim_token, optim_window = optims
    return {
        "step": step,
        "generator_type": GENERATOR_TYPE,
        "book_latent": book_latent_settings(config),
        "T": T,
        "generator_state_dict": gen.state_dict(),
        "ema_params": ema,
        "ema_step": ema_step,
        "ema_decay": float(config["training"]["ema_decay"]),
        "token_critic_state_dict": token.state_dict(),
        "window_critic_state_dict": window.state_dict(),
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_token_state_dict": optim_token.state_dict(),
        "optim_window_state_dict": optim_window.state_dict(),
        "data_mean": mean.detach().float().cpu(),
    }


def _critic_step(optim, d_real, d_fake, gp, lambda_gp) -> tuple[Tensor, Tensor]:
    """One critic update; returns the detached (W-distance, penalty) for the log."""
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return (d_real.mean() - d_fake.mean()).detach(), gp.detach()


def train(config: dict, resume: str | None = None) -> Path:
    """Train a book_latent run; returns the path of its `final.pt`."""
    validate_config(config)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config["device"], strict=True)
    gpu_meta = gpu_preflight(device)
    out_dir = Path(config["output_dir"])
    final = out_dir / "final.pt"
    if final.exists():
        raise FileExistsError(f"{final} exists; this run is already trained")
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = float(config["training"].get("gpu_lock_timeout_s", 1800.0))
    with claim_gpu(device, run_dir=out_dir, timeout_s=timeout_s):
        _train(config, resume, device, gpu_meta, out_dir)
    return final


def _train(config, resume, device, gpu_meta, out_dir: Path) -> None:
    data, model, training = config["data"], config["model"], config["training"]
    t0 = time.time()
    gen, token_critic, window_critic = build_book_latent(config)

    ckpt = None
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        _check_resume_record(ckpt, config, gen, resume)

    h = load_kv_books(Path(data["real_path"]), int(data["kv_layer"]), data["kv_books"])
    books = RealBooks(h.to(device), int(data["kv_n_sink"]))
    del h
    T, D = books.T, books.D
    num_gen_steps = int(training["num_gen_steps"])
    start_step = 0
    if ckpt is not None:
        if not torch.allclose(ckpt["data_mean"].to(books.mean), books.mean, atol=1e-6):
            raise ValueError(
                f"{resume}: the data mean changed (the training books differ from "
                "the ones the checkpoint was trained on)"
            )
        start_step = int(ckpt["step"])
        if start_step >= num_gen_steps:
            raise ValueError(
                f"{resume} is at step {start_step}, already at or past "
                f"num_gen_steps={num_gen_steps}; raise the budget to continue"
            )

    gen.to(device)
    token_critic.to(device)
    window_critic.to(device)
    betas = tuple(float(b) for b in training["betas"])
    lr_g, lr_d = float(training["lr_g"]), float(training["lr_d"])
    optim_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=betas)
    optim_token = torch.optim.Adam(token_critic.parameters(), lr=lr_d, betas=betas)
    optim_window = torch.optim.Adam(window_critic.parameters(), lr=lr_d, betas=betas)
    optims = (optim_g, optim_token, optim_window)
    ema_decay = float(training["ema_decay"])
    ema_params = init_ema_params(gen)
    ema_step = 0
    if ckpt is not None:
        gen.load_state_dict(ckpt["generator_state_dict"])
        token_critic.load_state_dict(ckpt["token_critic_state_dict"])
        window_critic.load_state_dict(ckpt["window_critic_state_dict"])
        optim_g.load_state_dict(ckpt["optim_g_state_dict"])
        optim_token.load_state_dict(ckpt["optim_token_state_dict"])
        optim_window.load_state_dict(ckpt["optim_window_state_dict"])
        ema_params = {k: v.to(device) for k, v in ckpt["ema_params"].items()}
        ema_step = int(ckpt["ema_step"])
        del ckpt

    with (out_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    n_sink, n_tail = int(data["kv_n_sink"]), int(data["kv_n_tail"])
    wc = model["window_critic"]
    W, strides = int(wc["window"]), [int(s) for s in wc["strides"]]
    stride_t = torch.tensor(strides, device=device)
    n_windows, n_tail_windows = int(training["n_windows"]), _n_tail_windows(training)
    n_critic = int(training["n_critic"])
    lambda_gp = float(training["lambda_gp"])
    window_weight = float(training["window_weight"])
    log_every, save_every = int(training["log_every"]), int(training["save_every"])

    def draw_positions() -> tuple[Tensor, Tensor]:
        t_end, lo = window_ends(n_windows, n_tail_windows, T, n_sink, n_tail, W, device)
        s_idx = sample_stride(t_end, lo, strides, W)
        return window_positions(t_end, stride_t[s_idx], W), s_idx

    last = out_dir / "last.pt"
    # A fresh run starts a new log (a crashed earlier run in this dir would otherwise leave
    # its steps in front of ours); a resume continues the log it is resuming.
    log_mode = "w" if resume is None else "a"
    with (out_dir / "train_log.jsonl").open(log_mode, encoding="utf-8") as log:
        for step in range(start_step + 1, num_gen_steps + 1):
            for _ in range(n_critic):
                pos, s_idx = draw_positions()
                with torch.no_grad():
                    fake, flags = fake_windows(gen, pos, T, n_tail)
                real = books.windows(pos)
                rt, ft, ftok = (
                    real.reshape(-1, D),
                    fake.reshape(-1, D),
                    flags.reshape(-1),
                )
                token_stats = _critic_step(
                    optim_token,
                    token_critic(rt, ftok),
                    token_critic(ft, ftok),
                    gradient_penalty(token_critic, rt, ft, device, flags=ftok),
                    lambda_gp,
                )
                window_stats = _critic_step(
                    optim_window,
                    window_critic(real, flags, s_idx),
                    window_critic(fake, flags, s_idx),
                    window_gradient_penalty(window_critic, real, fake, flags, s_idx),
                    lambda_gp,
                )

            pos, s_idx = draw_positions()
            fake, flags = fake_windows(gen, pos, T, n_tail)
            token_score = token_critic(fake.reshape(-1, D), flags.reshape(-1))
            window_score = window_critic(fake, flags, s_idx)
            loss_g = generator_loss(token_score, window_score, window_weight)
            logging = step == 1 or step % log_every == 0
            if logging:
                # Mean per-token gradient norm each term puts on the fake windows.
                terms = {
                    "grad_token": -token_score.mean(),
                    "grad_window": -window_weight * window_score.mean(),
                }
                grads = {
                    name: float(
                        torch.autograd.grad(term, fake, retain_graph=True)[0]
                        .norm(dim=-1)
                        .mean()
                    )
                    for name, term in terms.items()
                }
            optim_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optim_g.step()
            ema_update(ema_params, gen, ema_decay)
            ema_step += 1

            if logging:
                rec = {
                    "step": step,
                    "w_dist_token": float(token_stats[0]),
                    "gp_token": float(token_stats[1]),
                    "w_dist_window": float(window_stats[0]),
                    "gp_window": float(window_stats[1]),
                    "loss_g": float(loss_g.detach()),
                    **grads,
                    "seconds": round(time.time() - t0, 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % save_every == 0 or step == num_gen_steps:
                state = _checkpoint(
                    step,
                    T,
                    config,
                    gen,
                    token_critic,
                    window_critic,
                    optims,
                    ema_params,
                    ema_step,
                    books.mean,
                )
                tmp = out_dir / "last.pt.tmp"
                torch.save(state, tmp)
                tmp.replace(last)
    last.rename(out_dir / "final.pt")

    meta = {
        "seed": int(config["seed"]),
        "device": str(device),
        "gpu": gpu_meta,
        "T": T,
        "D": D,
        "n_books": books.n_books,
        "seconds": round(time.time() - t0, 1),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scout_kv v2 book_latent run.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to continue from. num_gen_steps is the target total.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    final = train(config, resume=args.resume)
    print(f"Training complete. Final checkpoint: {final}")


if __name__ == "__main__":
    main()
