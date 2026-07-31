import pytest
import torch

from src.models.generator import Generator, SparseGenerator

LATENT = 16
OUTPUT = 128
HIDDEN = [32, 32]


def build(**overrides):
    kwargs = dict(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    kwargs.update(overrides)
    return SparseGenerator(**kwargs)


@pytest.fixture
def gen():
    torch.manual_seed(0)
    return build()


@pytest.fixture
def out(gen):
    torch.manual_seed(1)
    return gen(torch.randn(64, LATENT))


def test_output_shape(out):
    assert out.shape == (64, OUTPUT)


def test_non_negative(out):
    assert (out >= 0).all()


def test_unit_norm(out):
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_zeros_are_exact(out):
    assert (out == 0.0).any()


def test_no_all_zero_row(out):
    assert ((out > 0).sum(dim=1) > 0).all()


def test_single_output_cannot_become_all_zero():
    generator = SparseGenerator(latent_dim=2, output_dim=1, hidden_dims=[2])
    with torch.no_grad():
        generator.gate_head.weight.zero_()
        generator.gate_head.bias.fill_(-100.0)
    out = generator(torch.randn(32, 2))
    assert (out > 0).all()
    assert torch.equal(out, torch.ones_like(out))


def test_gate_head_receives_gradient(gen):
    torch.manual_seed(2)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.gate_head.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_magnitude_head_receives_gradient(gen):
    torch.manual_seed(3)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.magnitude_head.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_gate_noise_is_kept_at_sample_time(gen):
    gen.eval()
    z = torch.randn(64, LATENT)
    with torch.no_grad():
        a, b = gen(z), gen(z)
    assert not torch.equal(a, b)


def test_existing_generator_is_unchanged():
    torch.manual_seed(4)
    generator = Generator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    x = generator(torch.randn(8, LATENT))
    assert x.shape == (8, OUTPUT)
    assert (x < 0).any()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hidden_dims": []}, "hidden dimension"),
        ({"gate_temperature": 0}, "gate_temperature"),
        ({"logit_clamp": 0}, "logit_clamp"),
        ({"eps": 0}, "eps"),
        ({"output_dim": 0}, "dimensions"),
    ],
)
def test_invalid_sparse_configuration_fails_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(**kwargs)
