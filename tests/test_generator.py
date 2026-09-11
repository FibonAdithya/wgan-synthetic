import pytest
import torch
import torch.nn.functional as F

from src.models.generator import GatedGenerator, Generator, LinearSkipGenerator

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
    assert (F.softplus(torch.tensor(-1000.0)) == 0.0).item(), (
        "premise: softplus underflows"
    )
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


# --- linear_skip -----------------------------------------------------------


def _cov_rank(x: torch.Tensor, floor: float = 1e-6) -> int:
    x = x - x.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh(x.T @ x / (x.shape[0] - 1))
    return int((eig > floor).sum())


def test_linear_skip_output_shape():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(
        latent_dim=48, output_dim=32, hidden_dims=[16, 16], skip_dim=32
    )
    out = gen(torch.randn(64, 48))
    assert out.shape == (64, 32)
    assert gen.trunk_latent_dim == 16 and gen.skip_dim == 32


def test_linear_skip_with_trunk_zeroed_is_the_skip_map():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(
        latent_dim=48, output_dim=32, hidden_dims=[16, 16], skip_dim=32
    )
    last = gen.trunk.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    z = torch.randn(8, 48)
    expected = z[:, 16:] @ gen.skip.weight.T
    assert torch.allclose(gen(z), expected, atol=1e-6)


def test_linear_skip_output_is_full_rank_where_mlp_is_not():
    # The discriminating claim of the design: the skip path gives the output
    # distribution full rank by construction, where an MLP of the same width
    # collapses. 4096 latents, covariance rank measured against a floor.
    #
    # The MLP's bound is the width of its last hidden layer (16), not the
    # latent (8): the output is Linear(16 -> 64) applied to a 16-vector, and a
    # piecewise-linear map of an 8-d latent fills all 16 of those directions.
    # Measured at rank 16 on three seeds during the plan audit, so `<= 8`
    # would fail; `<= 16 < 64` is the bound that actually holds.
    torch.manual_seed(0)
    n, out_dim = 4096, 64
    mlp = Generator(latent_dim=8, output_dim=out_dim, hidden_dims=[16, 16])
    skip = LinearSkipGenerator(
        latent_dim=8 + out_dim,
        output_dim=out_dim,
        hidden_dims=[16, 16],
        skip_dim=out_dim,
    )
    with torch.no_grad():
        rank_mlp = _cov_rank(mlp(torch.randn(n, 8)))
        rank_skip = _cov_rank(skip(torch.randn(n, 8 + out_dim)))
    assert rank_mlp <= 16 < out_dim  # bounded by the last hidden width
    assert rank_skip == out_dim  # the skip map supplies all 64


def test_linear_skip_identity_init_is_the_identity():
    gen = LinearSkipGenerator(
        latent_dim=40,
        output_dim=32,
        hidden_dims=[16],
        skip_dim=32,
        skip_init="identity",
    )
    assert torch.equal(gen.skip.weight, torch.eye(32))


def test_linear_skip_orthogonal_init_has_orthonormal_columns():
    torch.manual_seed(0)
    gen = LinearSkipGenerator(
        latent_dim=40, output_dim=32, hidden_dims=[16], skip_dim=32, skip_init_gain=2.0
    )
    w = gen.skip.weight
    assert torch.allclose(w.T @ w, 4.0 * torch.eye(32), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(latent_dim=32, output_dim=32, hidden_dims=[16], skip_dim=32), "skip_dim"),
        (
            dict(
                latent_dim=48,
                output_dim=32,
                hidden_dims=[16],
                skip_dim=16,
                skip_init="identity",
            ),
            "identity",
        ),
        (
            dict(
                latent_dim=48,
                output_dim=32,
                hidden_dims=[16],
                skip_dim=32,
                skip_init="nope",
            ),
            "skip_init",
        ),
    ],
)
def test_linear_skip_rejects_bad_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinearSkipGenerator(**kwargs)
