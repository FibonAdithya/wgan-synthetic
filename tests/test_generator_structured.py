import pytest
import torch
import torch.nn.functional as F

from src.models.generator import GatedGenerator, StructuredGateGenerator

LATENT = 16
OUTPUT = 128
HIDDEN = [32, 32]


def build(**overrides):
    kwargs = dict(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    kwargs.update(overrides)
    return StructuredGateGenerator(**kwargs)


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


def test_all_gates_closed_still_yields_a_unit_vector():
    # The fallback path: if every stochastic gate in a row closes, the row would
    # normalize to the zero vector. One coordinate is forced open instead.
    generator = build(output_dim=8, layout=(1, 1, 8))
    with torch.no_grad():
        generator.gate_head.weight.zero_()
        generator.gate_head.bias.fill_(-100.0)
        generator.sparsity_head.weight.zero_()
        generator.sparsity_head.bias.fill_(-100.0)
    torch.manual_seed(12)
    out = generator(torch.randn(32, LATENT))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Deliberately `>= 1`, not `== 1`: at a clamped logit of -10 a coordinate
    # still opens with probability about 4.5e-5, so an exact count would be
    # flaky across seeds. The contract under test is that no row is empty.
    assert ((out > 0).sum(dim=1) >= 1).all()


def test_saturated_magnitude_still_yields_unit_norm():
    # softplus underflows to exactly 0.0 below about -90 in float32, and an
    # open gate over a zero magnitude still normalizes to the zero vector.
    # Only the magnitude floor rescues this.
    generator = build()
    with torch.no_grad():
        generator.magnitude_head.weight.zero_()
        generator.magnitude_head.bias.fill_(-1000.0)
    torch.manual_seed(9)
    out = generator(torch.randn(32, LATENT))
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert ((out > 0).sum(dim=1) > 0).all()


def test_gate_noise_is_kept_at_sample_time(gen):
    gen.eval()
    z = torch.randn(64, LATENT)
    with torch.no_grad():
        a, b = gen(z), gen(z)
    assert not torch.equal(a, b)


def test_seeded_forward_is_deterministic(gen):
    z = torch.randn(16, LATENT)
    torch.manual_seed(8)
    a = gen(z)
    torch.manual_seed(8)
    b = gen(z)
    assert torch.equal(a, b)


@pytest.mark.parametrize("head", ["gate_head", "magnitude_head", "sparsity_head"])
def test_every_head_receives_gradient(gen, head):
    torch.manual_seed(2)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = getattr(gen, head).weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hidden_dims": []}, "hidden dimension"),
        ({"gate_temperature": 0}, "gate_temperature"),
        ({"logit_clamp": 0}, "logit_clamp"),
        ({"eps": 0}, "eps"),
        ({"output_dim": 0}, "dimensions"),
        ({"negative_slope": -0.1}, "negative_slope"),
        ({"layout": (4, 4, 4)}, "layout"),
        ({"layout": (4, 32)}, "layout"),
        ({"gate_kernel": 2}, "gate_kernel"),
        ({"noise_kernel_sigma": 0.0}, "noise_kernel_sigma"),
    ],
)
def test_invalid_configuration_fails_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(**kwargs)


def _nnz_stats(generator, n=4096, seed=11):
    torch.manual_seed(seed)
    with torch.no_grad():
        x = generator(torch.randn(n, LATENT))
    nnz = (x > 0).sum(dim=1).float()
    p = nnz.mean().item() / OUTPUT
    binomial = (OUTPUT * p * (1.0 - p)) ** 0.5
    return nnz.std().item(), binomial


def test_sparsity_level_makes_support_size_over_dispersed():
    # The whole reason this class exists. Independent per-coordinate gates give
    # nnz ~ Binomial(128, p), std = sqrt(128 p (1-p)), *by construction*. Real
    # SIFT is 3x that (14.45 vs 4.76). A per-vector logit shift turns nnz into
    # a mixture of binomials, whose variance can reach the measured value.
    generator = build()
    with torch.no_grad():
        # Amplify the head so the spread is unambiguous at random init.
        generator.sparsity_head.weight.mul_(8.0)
    observed, binomial = _nnz_stats(generator)
    assert observed > 1.5 * binomial


def test_v2_gate_stays_near_the_binomial_baseline():
    # The contrast that makes the previous test meaningful: v2 has no per-vector
    # level, so no amount of training moves it off the binomial.
    torch.manual_seed(0)
    v2 = GatedGenerator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    observed, binomial = _nnz_stats(v2)
    assert observed < 1.5 * binomial


@pytest.mark.parametrize("dtype, atol", [
    pytest.param(torch.bfloat16, 3e-2, id="bfloat16"),
    pytest.param(torch.float16, 5e-3, id="float16"),
])
def test_low_precision_forward_preserves_dtype(dtype, atol):
    torch.manual_seed(6)
    generator = build().to(dtype)
    out = generator(torch.randn(64, LATENT, dtype=dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    norms = out.float().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_gate_preserves_input_dtype(gen, dtype):
    torch.manual_seed(5)
    logits = torch.randn(64, OUTPUT, dtype=dtype)
    gate = gen._sample_gate(logits)
    assert gate.dtype == dtype
    assert torch.isfinite(gate).all()
    assert (gate.sum(dim=1) > 0).all()
