import torch

from src.models.critic import Critic
from src.models.generator import Generator
from src.train.train_wgan_gp import save_checkpoint


def _tiny_setup():
    generator = Generator(latent_dim=4, output_dim=6, hidden_dims=[8])
    critic = Critic(input_dim=6, hidden_dims=[8])
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)
    return generator, critic, optim_g, optim_d


def test_checkpoint_carries_the_ema_shadow_and_best_cov(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    ema = {name: p.detach().clone() for name, p in generator.named_parameters()}

    save_checkpoint(
        generator, critic, optim_g, optim_d, tmp_path, step=500,
        ema_params=ema, ema_step=500, best_cov=0.25,
    )

    ckpt = torch.load(tmp_path / "checkpoint_step_500.pt", weights_only=False)
    assert ckpt["ema_step"] == 500
    assert ckpt["best_cov"] == 0.25
    assert set(ckpt["ema_params"]) == set(ema)
    for name, tensor in ema.items():
        assert torch.allclose(ckpt["ema_params"][name], tensor)


def test_checkpoint_without_ema_records_an_empty_shadow(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    save_checkpoint(generator, critic, optim_g, optim_d, tmp_path, step=10)
    ckpt = torch.load(tmp_path / "checkpoint_step_10.pt", weights_only=False)
    assert ckpt["ema_params"] == {}
    assert ckpt["ema_step"] == 0
