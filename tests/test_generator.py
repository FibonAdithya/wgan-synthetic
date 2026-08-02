import pytest
import torch
import torch.nn.functional as F

from src.models.generator import Generator, GatedGenerator

LATENT = 16
OUTPUT = 128
HIDDEN = [32, 32]


def build(**overrides):
    kwargs = dict(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    kwargs.update(overrides)
    return GatedGenerator(**kwargs)


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
    generator = GatedGenerator(latent_dim=2, output_dim=1, hidden_dims=[2])
    with torch.no_grad():
        generator.gate_head.weight.zero_()
        generator.gate_head.bias.fill_(-100.0)
    out = generator(torch.randn(32, 2))
    assert (out > 0).all()
    assert torch.equal(out, torch.ones_like(out))


def test_saturated_magnitude_still_yields_unit_norm():
    # softplus underflows to exactly 0.0 below about -90 in float32. The gate
    # fallback alone does not rescue this: an open gate over a zero magnitude
    # still normalizes to the zero vector. Only the magnitude floor does.
    generator = build()
    with torch.no_grad():
        generator.magnitude_head.weight.zero_()
        generator.magnitude_head.bias.fill_(-1000.0)
    torch.manual_seed(9)
    out = generator(torch.randn(32, LATENT))
    assert (F.softplus(torch.tensor(-1000.0)) == 0.0).item(), "premise: softplus underflows"
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert ((out > 0).sum(dim=1) > 0).all()


def test_magnitude_floor_leaves_gate_zeros_exact(out):
    # The floor must not leak a nonzero value through a closed gate.
    assert (out == 0.0).any()


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
        ({"negative_slope": -0.1}, "negative_slope"),
    ],
)
def test_invalid_gated_configuration_fails_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(**kwargs)


LOW_PRECISION = [
    pytest.param(torch.bfloat16, 3e-2, id="bfloat16"),
    pytest.param(torch.float16, 5e-3, id="float16"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_gate_preserves_input_dtype(gen, dtype):
    torch.manual_seed(5)
    logits = torch.randn(64, OUTPUT, dtype=dtype)
    gate = gen._sample_gate(logits)
    assert gate.dtype == dtype
    assert torch.isfinite(gate).all()
    assert (gate.sum(dim=1) > 0).all()


@pytest.mark.parametrize("dtype, atol", LOW_PRECISION)
def test_low_precision_forward_preserves_dtype(dtype, atol):
    torch.manual_seed(6)
    generator = build().to(dtype)
    out = generator(torch.randn(64, LATENT, dtype=dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    norms = out.float().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=atol)


def test_float32_output_is_unchanged_by_dtype_handling(gen):
    torch.manual_seed(7)
    z = torch.randn(16, LATENT)
    torch.manual_seed(8)
    a = gen(z)
    torch.manual_seed(8)
    b = gen(z)
    assert a.dtype == torch.float32
    assert torch.equal(a, b)
