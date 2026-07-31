from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.sift1m_dataset import (
    NumpyTensorDataset,
    PreprocessConfig,
    build_training_data,
)
from src.models.critic import Critic
from src.models.generator import build_generator


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


def save_checkpoint(
    generator: nn.Module,
    critic: Critic,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    out_dir: Path,
    step: int,
    best: bool = False,
) -> None:
    ckpt = {
        "step": step,
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
    generator.train()
    return np.concatenate(out, axis=0)[:num_samples]


def train(config: Dict) -> Tuple[Path, Dict]:
    seed = int(config["seed"])
    set_seed(seed)
    device = get_device(config["device"])
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    preprocess_cfg = PreprocessConfig(**data_cfg["preprocess"])
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
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    data_iter = iter(loader)

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
            if distance_reg_alpha > 0.0:
                real_for_reg = real_batch.to(device)
                dist_real = batch_pairwise_distance_mean(
                    real_for_reg, max_points=distance_reg_max_points
                )
                dist_fake = batch_pairwise_distance_mean(
                    fake, max_points=distance_reg_max_points
                )
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = adv_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
                g_loss = adv_loss
        scaler_g.scale(g_loss).backward()
        scaler_g.step(optim_g)
        scaler_g.update()

        if step % log_every == 0 or step == 1:
            msg = {
                "step": step,
                "g_loss": float(g_loss.item()),
                "d_loss": d_loss_val,
                "gp": gp_val,
                "wasserstein": wasserstein_val,
                "adv_loss": float(adv_loss.item()),
                "distance_reg": float(distance_reg.item()),
            }
            run_meta["metrics"].append(msg)
            print(json.dumps(msg))

        if step % eval_every == 0 or step == num_gen_steps:
            fake_holdout = sample_generator(
                generator=generator,
                num_samples=x_holdout.shape[0],
                latent_dim=latent_dim,
                batch_size=int(train_cfg["batch_size"]),
                device=device,
            )
            stats = tensor_stats(x_holdout, fake_holdout)
            stats["step"] = step
            run_meta.setdefault("eval", []).append(stats)
            print(json.dumps({"eval": stats}))
            if stats["cov_fro"] < best_cov:
                best_cov = stats["cov_fro"]
                save_checkpoint(generator, critic, optim_g, optim_d, out_dir, step, best=True)

        if step % save_every == 0:
            save_checkpoint(generator, critic, optim_g, optim_d, out_dir, step, best=False)

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
