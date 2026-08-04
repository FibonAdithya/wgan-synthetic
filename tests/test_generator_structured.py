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


def test_coupling_is_identity_at_initialisation():
    # v3 must start where v2 starts; a random conv would scramble the logits at
    # step 0 for reasons unrelated to the mechanism under test.
    generator = build()
    torch.manual_seed(3)
    logits = torch.randn(8, OUTPUT)
    assert torch.allclose(generator._couple(logits), logits, atol=1e-6)


def test_orientation_axis_wraps_circularly():
    # A gradient direction falling between bins 7 and 0 deposits in both, so
    # the last orientation bin must neighbour the first.
    generator = build()
    rows, cols, orient = generator.layout
    with torch.no_grad():
        generator.gate_coupling.weight.zero_()
        # Pick up only the neighbour one step *back* along orientation.
        generator.gate_coupling.weight[0, 0, 1, 1, 0] = 1.0
    logits = torch.zeros(1, OUTPUT)
    logits[0, orient - 1] = 5.0  # last orientation bin of the first cell
    coupled = generator._couple(logits).reshape(rows, cols, orient)
    assert coupled[0, 0, 0].item() == pytest.approx(5.0)


def test_spatial_edge_replicates_rather_than_wrapping():
    # The 4x4 grid is not periodic: cell (0,0) and cell (3,3) are opposite
    # corners of the patch, not neighbours.
    generator = build()
    rows, cols, orient = generator.layout
    with torch.no_grad():
        generator.gate_coupling.weight.zero_()
        generator.gate_coupling.weight[0, 0, 0, 1, 1] = 1.0  # one step back in rows
    logits = torch.zeros(1, OUTPUT)
    logits[0, ((rows - 1) * cols + 0) * orient + 0] = 7.0  # last row, first cell
    coupled = generator._couple(logits).reshape(rows, cols, orient)
    # Row 0 pulls from replicated row 0, not from wrapped row 3.
    assert coupled[0, 0, 0].item() == pytest.approx(0.0)


def test_coupling_receives_gradient(gen):
    torch.manual_seed(4)
    gen(torch.randn(32, LATENT)).sum().backward()
    grad = gen.gate_coupling.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_coupling_preserves_shape_and_dtype(gen):
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(4, OUTPUT, dtype=dtype)
        coupled = gen.to(dtype)._couple(logits)
        assert coupled.shape == (4, OUTPUT)
        assert coupled.dtype == dtype
    gen.to(torch.float32)


def test_noise_kernel_is_variance_preserving():
    # Smoothing must not change the noise scale, or it silently shifts the
    # gate's effective temperature.
    generator = build()
    assert generator.noise_kernel.pow(2).sum().item() == pytest.approx(1.0, abs=1e-6)


def test_noise_kernel_is_not_a_learnable_parameter():
    # A learned noise kernel could be driven to zero, killing gate
    # stochasticity and collapsing the support distribution.
    generator = build()
    assert "noise_kernel" in dict(generator.named_buffers())
    assert "noise_kernel" not in dict(generator.named_parameters())


def _adjacent_vs_distant_gate_correlation(generator, n=8192, seed=21):
    """Correlation between neighbouring vs far-apart gates, logits held flat.

    Takes any generator exposing `_sample_gate`, so v2 and v3 can be compared
    on identical terms.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        gate = generator._sample_gate(torch.zeros(n, OUTPUT))
    g = gate - gate.mean(dim=0, keepdim=True)
    sd = g.std(dim=0, keepdim=True).clamp(min=1e-8)
    g = g / sd
    corr = (g.T @ g) / n
    idx = torch.arange(OUTPUT)
    sep = (idx[:, None] - idx[None, :]).abs()
    off_diagonal = ~torch.eye(OUTPUT, dtype=torch.bool)
    adjacent = corr[(sep == 1) & off_diagonal].mean().item()
    distant = corr[(sep >= 16) & off_diagonal].mean().item()
    return adjacent, distant


def test_gate_noise_is_spatially_correlated():
    # With logits flat at zero the gate is pure noise, so any correlation
    # between neighbouring coordinates comes from the smoothing.
    adjacent, distant = _adjacent_vs_distant_gate_correlation(build())
    assert adjacent > 0.05
    assert adjacent > distant + 0.04


def test_v2_gate_noise_is_uncorrelated():
    # The contrast: v2 samples every coordinate independently, so neighbours
    # are no more alike than distant coordinates.
    torch.manual_seed(0)
    v2 = GatedGenerator(latent_dim=LATENT, output_dim=OUTPUT, hidden_dims=HIDDEN)
    adjacent, distant = _adjacent_vs_distant_gate_correlation(v2)
    assert abs(adjacent - distant) < 0.03
