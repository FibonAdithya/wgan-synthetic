from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import (
    NumpyTensorDataset,
    PreprocessConfig,
    build_training_data,
)
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.spectrum import spectrum_distance


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def gradient_penalty(critic: Critic, real: Tensor, fake: Tensor, device: torch.device) -> Tensor:
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)
    grad_outputs = torch.ones_like(critic_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    norms = gradients.norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean()


def tensor_stats(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    # zero_fraction_gap, negative_fraction, per_dim_zero_rate_l1, and
    # nnz_std_gap exist because raw SIFT descriptors carry heavy mass at
    # exactly zero, which a dense MLP generator cannot reproduce and the
    # critic does not reliably penalize -- these measure that support gap.
    real_mean = real.mean(axis=0)
    fake_mean = fake.mean(axis=0)
    real_var = real.var(axis=0)
    fake_var = fake.var(axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)
    real_zero = real == 0.0
    fake_zero = fake == 0.0
    real_nnz = (~real_zero).sum(axis=1)
    fake_nnz = (~fake_zero).sum(axis=1)
    return {
        "mean_l2": float(np.linalg.norm(real_mean - fake_mean)),
        "var_l2": float(np.linalg.norm(real_var - fake_var)),
        "cov_fro": float(np.linalg.norm(cov_real - cov_fake, ord="fro")),
        "zero_fraction_gap": float(abs(fake_zero.mean() - real_zero.mean())),
        "negative_fraction": float((fake < 0).mean()),
        "per_dim_zero_rate_l1": float(
            np.abs(fake_zero.mean(axis=0) - real_zero.mean(axis=0)).mean()
        ),
        "nnz_std_gap": float(abs(fake_nnz.std() - real_nnz.std())),
    }


def normalize_l2(x: Tensor, eps: float = 1.0e-8) -> Tensor:
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def batch_pairwise_distance_mean(x: Tensor, max_points: int = 128) -> Tensor:
    n = x.shape[0]
    if n < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if max_points > 1 and n > max_points:
        idx = torch.randperm(n, device=x.device)[:max_points]
        x = x[idx]
    pd = torch.pdist(x, p=2)
    return pd.mean() if pd.numel() > 0 else torch.zeros((), device=x.device, dtype=x.dtype)


def collapse_stats(fake: np.ndarray, max_points: int = 512) -> Dict[str, float]:
    """Lightweight mode-collapse monitor.

    A healthy generator spreads samples across the descriptor space; a
    collapsed one packs them into a few points. We report:
      - fake_std: mean per-dimension std (collapse -> near-zero)
      - fake_min_pdist / fake_mean_pdist: min / mean pairwise L2 among a
        subsample of generated vectors (collapse -> both -> 0)

    Note: the ``np.random.default_rng(0)`` reseed is deliberate, not a bug.
    Re-seeding per call makes the subsample pick the *same* row indices at
    every eval, so the numbers are directly comparable across steps instead of
    also moving with the subsampling noise. All returned values are plain
    Python floats because they are JSON-serialised into run_metadata.json.
    """
    n = fake.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        fake = fake[idx]
    std = float(fake.std(axis=0).mean())
    if fake.shape[0] >= 2:
        d = torch.pdist(torch.from_numpy(np.ascontiguousarray(fake)), p=2)
        min_pd = float(d.min())
        mean_pd = float(d.mean())
    else:
        min_pd = mean_pd = 0.0
    return {
        "fake_std": std,
        "fake_min_pdist": min_pd,
        "fake_mean_pdist": mean_pd,
    }


EMA_BIAS_CORRECTION_EPS = 1.0e-8


def init_ema_params(model: torch.nn.Module) -> Dict[str, Tensor]:
    """Zero-initialised EMA accumulator for the trainable params of `model`.

    Zero (rather than "a copy of the initial weights") is what makes the
    Adam-style bias correction in `load_ema_into_model` exact: the raw
    accumulator after t updates is (1 - decay**t) times a proper weighted
    average of the *observed* parameter values, with no contribution from the
    random initialisation.
    """
    return {
        name: torch.zeros_like(p.data)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def ema_update(ema_params: Dict[str, Tensor], model: torch.nn.Module, decay: float) -> None:
    """In-place exponential moving average of model parameters.

    The accumulator is left *uncorrected*; bias correction is applied only
    where the weights are read out (`load_ema_into_model`).
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in ema_params:
                ema_params[name].mul_(decay).add_(p.data, alpha=1.0 - decay)


def ema_bias_correction(decay: float, ema_step: int) -> float:
    """Multiplier undoing the zero-init bias of an EMA after `ema_step` updates.

    Returns 1.0 (no correction) when there is nothing to correct or when the
    denominator (1 - decay**ema_step) is too close to zero to divide by safely.
    """
    if ema_step <= 0 or not 0.0 < decay < 1.0:
        return 1.0
    denom = 1.0 - decay**ema_step
    if denom < EMA_BIAS_CORRECTION_EPS:
        return 1.0
    return 1.0 / denom


def load_ema_into_model(
    ema_params: Dict[str, Tensor],
    model: torch.nn.Module,
    decay: float = 0.0,
    ema_step: int = 0,
) -> None:
    """Copy bias-corrected EMA weights into the model (for saving / sampling).

    Without the correction a short run evaluates weights that are still mostly
    the zero-initialised accumulator (e.g. decay=0.999 after 200 steps only
    accumulates a factor 1 - 0.999**200 ~= 0.18).
    """
    correction = ema_bias_correction(decay, ema_step)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema_params:
                p.data.copy_(ema_params[name])
                if correction != 1.0:
                    p.data.mul_(correction)


@contextlib.contextmanager
def ema_weights(
    model: torch.nn.Module,
    ema_params: Dict[str, Tensor],
    decay: float = 0.0,
    ema_step: int = 0,
) -> Iterator[None]:
    """Temporarily swap bias-corrected EMA weights into `model`.

    Live weights are snapshotted on entry and restored in a `finally`, so an
    exception raised inside the block can never strand the model on EMA
    weights. A no-op when `ema_params` is empty (EMA disabled).
    """
    live_backup = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if name in ema_params
    }
    try:
        load_ema_into_model(ema_params, model, decay=decay, ema_step=ema_step)
        yield
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in live_backup:
                    p.data.copy_(live_backup[name])


def save_checkpoint(
    generator: nn.Module,
    critic: Critic,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    out_dir: Path,
    step: int,
    best: bool = False,
    generator_weights: str = "live",
) -> None:
    """Write a checkpoint.

    `generator_weights` records which parameters `generator_state_dict` holds:
    "live" (the currently optimised parameters) or "ema" (the bias-corrected
    EMA swapped in for evaluation). Everything else in the file -- critic and
    both optimiser states -- is always live; the EMA is a read-only shadow of
    the generator and has no optimiser moments of its own.
    """
    if generator_weights not in ("live", "ema"):
        raise ValueError(f"generator_weights must be 'live' or 'ema', got {generator_weights!r}")
    ckpt = {
        "step": step,
        "generator_weights": generator_weights,
        "generator_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_d_state_dict": optim_d.state_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / f"checkpoint_step_{step}.pt")
    if best:
        torch.save(ckpt, out_dir / "best_generator.pt")


def sample_generator(
    generator: nn.Module,
    num_samples: int,
    latent_dim: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    # Restore the mode we found rather than forcing train(): callers outside
    # the training loop (evaluate_distribution, compare_variants) hand us an
    # eval()-mode generator and would otherwise get it back in train mode.
    was_training = generator.training
    generator.eval()
    out = []
    generated = 0
    with torch.no_grad():
        n_batches = math.ceil(num_samples / batch_size)
        for _ in range(n_batches):
            cur = min(batch_size, num_samples - generated)
            if cur <= 0:
                break
            z = torch.randn(cur, latent_dim, device=device)
            x = normalize_l2(generator(z)).detach().cpu().numpy()
            out.append(x)
            generated += cur
    generator.train(was_training)
    return np.concatenate(out, axis=0)[:num_samples]


def train(config: Dict) -> Tuple[Path, Dict]:
    seed = int(config["seed"])
    set_seed(seed)
    device = get_device(config["device"])
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    preprocess_cfg = PreprocessConfig(
        **data_cfg["preprocess"], metric=data_cfg.get("metric", "l2")
    )
    x_train, x_holdout, preprocess_state = build_training_data(
        descriptor_path=data_cfg["real_path"],
        file_format=data_cfg["format"],
        descriptor_dim=int(data_cfg["descriptor_dim"]),
        holdout_fraction=float(data_cfg["holdout_fraction"]),
        preprocess_cfg=preprocess_cfg,
        seed=seed,
        synthetic_if_missing=bool(data_cfg["synthetic_if_missing"]),
        synthetic_num_vectors=int(data_cfg["synthetic_num_vectors"]),
    )

    model_cfg = config["model"]
    train_cfg = config["training"]
    latent_dim = int(model_cfg["latent_dim"])
    descriptor_dim = int(data_cfg["descriptor_dim"])

    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
    critic = Critic(
        input_dim=descriptor_dim,
        hidden_dims=model_cfg["critic_hidden_dims"],
        negative_slope=float(model_cfg["negative_slope"]),
    ).to(device)

    lr_g = float(train_cfg["lr_g"])
    lr_d = float(train_cfg["lr_d"])
    betas = tuple(float(x) for x in train_cfg["betas"])
    optim_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    optim_d = torch.optim.Adam(critic.parameters(), lr=lr_d, betas=betas)

    dataset = NumpyTensorDataset(x_train)
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        # Keep workers alive across epochs: the loop re-creates the iterator on
        # StopIteration, which would otherwise respawn the pool every epoch.
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(loader)

    # Generator EMA (stabilises samples / checkpoint). Disabled when decay <= 0.
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    use_ema = ema_decay > 0.0
    # decay >= 1 never accumulates: the zero-initialised shadow would stay at
    # zero and bias correction cannot rescue it, so eval and best_generator.pt
    # would silently come from an all-zero generator. Reject it up front.
    if use_ema and ema_decay >= 1.0:
        raise ValueError(f"ema_decay must be < 1.0 to accumulate, got {ema_decay}")
    ema_params: Dict[str, Tensor] = init_ema_params(generator) if use_ema else {}
    ema_step = 0

    amp = bool(train_cfg["amp"] and device.type == "cuda")
    scaler_d = GradScaler(device="cuda", enabled=amp)
    scaler_g = GradScaler(device="cuda", enabled=amp)

    lambda_gp = float(train_cfg["lambda_gp"])
    n_critic = int(train_cfg["n_critic"])
    num_gen_steps = int(train_cfg["num_gen_steps"])
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])
    save_every = int(train_cfg["save_every"])
    distance_reg_alpha = float(train_cfg.get("distance_reg_alpha", 0.0))
    distance_reg_max_points = int(train_cfg.get("distance_reg_max_points", 128))
    spectrum_reg_alpha = float(train_cfg.get("spectrum_reg_alpha", 0.0))

    run_meta = {
        "seed": seed,
        "device": str(device),
        "data": {
            "num_train": int(x_train.shape[0]),
            "num_holdout": int(x_holdout.shape[0]),
            "descriptor_dim": descriptor_dim,
        },
        "preprocess_state": preprocess_state.to_serializable(),
        "metrics": [],
    }

    best_cov = float("inf")

    for step in range(1, num_gen_steps + 1):
        d_loss_val = 0.0
        gp_val = 0.0
        wasserstein_val = 0.0

        for _ in range(n_critic):
            try:
                real_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_batch = next(data_iter)

            real = real_batch.to(device)
            batch_size = real.shape[0]
            z = torch.randn(batch_size, latent_dim, device=device)

            optim_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp):
                fake = normalize_l2(generator(z).detach())
                d_real = critic(real)
                d_fake = critic(fake)
                gp = gradient_penalty(critic, real, fake, device=device)
                d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp

            scaler_d.scale(d_loss).backward()
            scaler_d.step(optim_d)
            scaler_d.update()

            d_loss_val += float(d_loss.item())
            gp_val += float(gp.item())
            wasserstein_val += float((d_real.mean() - d_fake.mean()).item())

        d_loss_val /= n_critic
        gp_val /= n_critic
        wasserstein_val /= n_critic

        try:
            real_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_batch = next(data_iter)
        batch_size = real_batch.shape[0]

        optim_g.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = normalize_l2(generator(z))
            adv_loss = -critic(fake).mean()
            g_loss = adv_loss
            if distance_reg_alpha > 0.0:
                real_for_reg = real_batch.to(device)
                dist_real = batch_pairwise_distance_mean(
                    real_for_reg, max_points=distance_reg_max_points
                )
                dist_fake = batch_pairwise_distance_mean(
                    fake, max_points=distance_reg_max_points
                )
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = g_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
            if spectrum_reg_alpha > 0.0:
                spectrum_reg = spectrum_distance(real_batch.to(device), fake)
                g_loss = g_loss + spectrum_reg_alpha * spectrum_reg
            else:
                spectrum_reg = torch.zeros((), device=device, dtype=fake.dtype)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(optim_g)
        scaler_g.update()

        if use_ema:
            ema_update(ema_params, generator, ema_decay)
            ema_step += 1

        if step % log_every == 0 or step == 1:
            msg = {
                "step": step,
                "g_loss": float(g_loss.item()),
                "d_loss": d_loss_val,
                "gp": gp_val,
                "wasserstein": wasserstein_val,
                "adv_loss": float(adv_loss.item()),
                "distance_reg": float(distance_reg.item()),
                "spectrum_reg": float(spectrum_reg.item()),
            }
            run_meta["metrics"].append(msg)
            print(json.dumps(msg))

        if step % eval_every == 0 or step == num_gen_steps:
            # Evaluate / sample from the (bias-corrected) EMA weights for a
            # stabilised view; the context manager restores the live weights on
            # the way out, including if the eval body raises. No-op when EMA is
            # disabled.
            with ema_weights(generator, ema_params, decay=ema_decay, ema_step=ema_step):
                fake_holdout = sample_generator(
                    generator=generator,
                    num_samples=x_holdout.shape[0],
                    latent_dim=latent_dim,
                    batch_size=int(train_cfg["batch_size"]),
                    device=device,
                )
                stats = tensor_stats(x_holdout, fake_holdout)
                stats.update(collapse_stats(fake_holdout))
                stats["step"] = step
                run_meta.setdefault("eval", []).append(stats)
                print(json.dumps({"eval": stats}))
                if stats["cov_fro"] < best_cov:
                    best_cov = stats["cov_fro"]
                    # Saved from inside the EMA swap: generator_state_dict holds
                    # EMA weights, but optim_g_state_dict (and the critic) are
                    # the LIVE training state -- the optimiser moments belong to
                    # the live parameters, not to the EMA shadow. Resuming from
                    # this file therefore mixes EMA weights with live moments;
                    # the "generator_weights" key records which is which.
                    save_checkpoint(
                        generator,
                        critic,
                        optim_g,
                        optim_d,
                        out_dir,
                        step,
                        best=True,
                        generator_weights="ema" if use_ema else "live",
                    )

        if step % save_every == 0:
            # Outside the EMA swap: this one holds live weights throughout.
            save_checkpoint(
                generator,
                critic,
                optim_g,
                optim_d,
                out_dir,
                step,
                best=False,
                generator_weights="live",
            )

    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with (out_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return out_dir / "best_generator.pt", run_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WGAN-GP for SIFT1M-like descriptors.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    best_ckpt, _ = train(config)
    print(f"Training complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
