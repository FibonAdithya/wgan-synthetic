from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
    ):
        super().__init__()
        dims: List[int] = [latent_dim, *list(hidden_dims), output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GatedGenerator(nn.Module):
    """Generate non-negative unit vectors with a learnable point mass at zero.

    The gate is sampled in `eval()` mode too: `eval()` does not switch to a
    deterministic threshold, so two forward passes on the same `z` give
    different supports. That is intentional -- the gate noise is part of the
    generative distribution, not a training-time regularizer like dropout, and
    thresholding at inference would sample from a different distribution than
    the critic was trained against. Callers needing reproducible output should
    seed the global RNG (`torch.manual_seed`) rather than rely on `eval()`.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        gate_temperature: float = 0.5,
        logit_clamp: float = 10.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        if latent_dim <= 0 or output_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("model dimensions must be greater than zero")
        if not hidden_dims:
            raise ValueError("GatedGenerator requires at least one hidden dimension")
        if negative_slope < 0:
            raise ValueError("negative_slope must not be negative")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be greater than zero")
        if logit_clamp <= 0:
            raise ValueError("logit_clamp must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        dims: List[int] = [latent_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.magnitude_head = nn.Linear(dims[-1], output_dim)
        self.gate_head = nn.Linear(dims[-1], output_dim)
        self.gate_temperature = float(gate_temperature)
        self.logit_clamp = float(logit_clamp)
        self.eps = float(eps)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a hard binary-concrete gate, returned in ``logits.dtype``.

        Draw in float32 under AMP: eps=1e-8 rounds to zero in float16, which
        would leave log(0) capable of poisoning gate gradients. The upcast is
        needed only for the noise sampling itself, so the gate is cast back
        before returning; the module therefore preserves the dtype implied by
        the enclosing autocast region instead of silently promoting it.
        """
        sample_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        u = torch.rand_like(sample_logits).clamp(self.eps, 1.0 - self.eps)
        logistic = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((sample_logits + logistic) / self.gate_temperature)
        hard = (soft > 0.5).to(soft.dtype)
        # Preserve the unit-norm contract even if every stochastic gate in a
        # row is off. This is vanishingly rare at 128 dimensions but otherwise
        # silently produces an invalid zero vector.
        empty = hard.sum(dim=1, keepdim=True) == 0
        fallback = F.one_hot(
            sample_logits.argmax(dim=1), sample_logits.shape[1]
        ).to(hard.dtype)
        # Note: on a rescued row the straight-through gradient still comes from
        # `soft`, i.e. from the gate the forward pass did *not* take. This is
        # the one place forward and backward genuinely disagree. It is harmless
        # in practice -- the rescued coordinate is the argmax logit, so `soft`
        # is its largest entry anyway, and all-off rows are vanishingly rare at
        # d=128 -- so the behaviour is left as is.
        hard = torch.where(empty, fallback, hard)
        return (hard + soft - soft.detach()).to(logits.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        # The gate fallback rescues a row whose gates all close, but not one
        # whose magnitudes all underflow: softplus saturates to exactly 0.0
        # below about -90 in float32, and a rescued gate over a zero magnitude
        # still normalizes to the zero vector. Floor the magnitude above `eps`
        # so the unit-norm contract holds under divergence too -- a floor at or
        # below `eps` would not, since the norm would stay inside the clamp
        # below. Exact zeros are unaffected: those come from the gate.
        magnitude = F.softplus(self.magnitude_head(h)).clamp(min=self.eps * 100.0)
        raw_logits = self.gate_head(h)
        logits = self.logit_clamp * torch.tanh(raw_logits / self.logit_clamp)
        x = self._sample_gate(logits) * magnitude
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=self.eps)


class StructuredGateGenerator(nn.Module):
    """Gated generator whose support statistics match SIFT's measured shape.

    `GatedGenerator` samples every coordinate's gate independently at a rate
    the trunk controls, so its non-zero count is Binomial(d, p) with standard
    deviation sqrt(d p (1-p)) -- about 4.76 at d=128, p=0.77. Real SIFT
    measures 14.45. That is an expressiveness ceiling, not a tuning problem:
    no parameter setting of independent gates reaches it.

    Three additions lift it, each targeting a measured property of the real
    descriptors (see tools/probes/):

    1. A per-vector scalar added to every gate logit, making the non-zero
       count a mixture of binomials whose variance the trunk can learn.
    2. A convolution over the logits reshaped to the (4,4,8) descriptor grid,
       producing the measured local correlation (Task 2).
    3. Smoothing of the gate *noise* with a fixed kernel, so sampling is
       correlated and not merely the logits (Task 3).

    Deliberately not a subclass of GatedGenerator: v2 is the frozen baseline
    this variant is measured against, and inheritance would let a change to
    it silently move the comparison.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        negative_slope: float = 0.2,
        gate_temperature: float = 0.5,
        logit_clamp: float = 10.0,
        layout: Sequence[int] = (4, 4, 8),
        gate_kernel: int = 3,
        noise_kernel_sigma: float = 0.65,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        layout = tuple(int(v) for v in layout)
        if latent_dim <= 0 or output_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("model dimensions must be greater than zero")
        if not hidden_dims:
            raise ValueError("StructuredGateGenerator requires at least one hidden dimension")
        if negative_slope < 0:
            raise ValueError("negative_slope must not be negative")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be greater than zero")
        if logit_clamp <= 0:
            raise ValueError("logit_clamp must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")
        if len(layout) != 3 or any(v <= 0 for v in layout):
            raise ValueError(f"layout must be three positive dimensions, got {layout}")
        if layout[0] * layout[1] * layout[2] != output_dim:
            raise ValueError(
                f"layout {layout} has {layout[0] * layout[1] * layout[2]} cells "
                f"but output_dim is {output_dim}"
            )
        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError(f"gate_kernel must be a positive odd number, got {gate_kernel}")
        if noise_kernel_sigma <= 0:
            raise ValueError("noise_kernel_sigma must be greater than zero")

        dims: List[int] = [latent_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.magnitude_head = nn.Linear(dims[-1], output_dim)
        self.gate_head = nn.Linear(dims[-1], output_dim)
        # One scalar per vector, broadcast across all coordinates. This is the
        # entire over-dispersion mechanism.
        self.sparsity_head = nn.Linear(dims[-1], 1)
        # Local coupling over the (row, col, orientation) grid. Identity-init
        # so the module starts as an uncoupled gate and learns structure.
        self.gate_coupling = nn.Conv3d(1, 1, kernel_size=gate_kernel, bias=False)
        with torch.no_grad():
            self.gate_coupling.weight.zero_()
            centre = gate_kernel // 2
            self.gate_coupling.weight[0, 0, centre, centre, centre] = 1.0
        self.layout = layout
        self.gate_kernel = int(gate_kernel)
        self.noise_kernel_sigma = float(noise_kernel_sigma)
        self.gate_temperature = float(gate_temperature)
        self.logit_clamp = float(logit_clamp)
        self.eps = float(eps)
        # Fixed, not learned: a trainable noise kernel could be driven toward
        # zero, removing gate stochasticity and collapsing the support
        # distribution -- the failure this class exists to prevent.
        #
        # The flip side is that the default sigma has to be right on its own,
        # since training cannot correct it. 0.65 was picked by sweeping sigma
        # and measuring the resulting gate correlation against the real
        # profile: it lands +0.329 at separation 1 and +0.271 at offset 8,
        # against the measured +0.317 and +0.275.
        #
        # Non-persistent: the kernel is a pure function of `gate_kernel` and
        # `noise_kernel_sigma`, both of which come from the run config. Were it
        # in the state dict, loading a checkpoint trained at one sigma into a
        # model configured with another would silently reinstate the old kernel
        # while `noise_kernel_sigma` still reported the configured value.
        self.register_buffer(
            "noise_kernel",
            self._gaussian_kernel(gate_kernel, noise_kernel_sigma),
            persistent=False,
        )
        self.register_buffer(
            "noise_scale", self._position_noise_scale(output_dim), persistent=False
        )

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Separable 3-D Gaussian, L2-normalised.

        Scaling by the L2 norm rather than the sum is deliberate: convolving
        i.i.d. unit-variance noise with weights w gives variance sum(w^2), so
        an L2-normalised kernel leaves the noise scale unchanged *where the
        convolution sees independent inputs*. It does not at the grid edges:
        replicate padding feeds one draw into several taps, whose coefficients
        add before squaring, so a corner cell would carry about 64% more noise
        std than an interior one. `_position_noise_scale` corrects that
        exactly; this kernel alone does not.
        """
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        line = torch.exp(-(coords**2) / (2.0 * sigma**2))
        kernel = line[:, None, None] * line[None, :, None] * line[None, None, :]
        kernel = kernel / torch.linalg.vector_norm(kernel)
        return kernel.reshape(1, 1, size, size, size)

    def _position_noise_scale(self, output_dim: int) -> torch.Tensor:
        """Per-position reciprocal std of the smoothing map, computed exactly.

        Smoothing is linear, so pushing the `output_dim` one-hot basis vectors
        through it recovers its matrix M, where column j lists the coefficients
        every input draw contributes to output position j. On i.i.d.
        unit-variance noise the output std at j is therefore the L2 norm of
        column j -- padding-induced coefficient sharing included -- and the
        reciprocal restores unit std at every position, not just the interior.
        """
        with torch.no_grad():
            basis = torch.eye(output_dim)
            smoothed = F.conv3d(self._pad_grid(basis), self.noise_kernel)
            column_norms = torch.linalg.vector_norm(
                smoothed.reshape(output_dim, -1), dim=0
            )
        return 1.0 / column_norms

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a hard binary-concrete gate, returned in ``logits.dtype``.

        Draw in float32 under AMP: eps=1e-8 rounds to zero in float16, which
        would leave log(0) capable of poisoning gate gradients.
        """
        sample_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        u = torch.rand_like(sample_logits).clamp(self.eps, 1.0 - self.eps)
        logistic = self._smooth_noise(torch.log(u) - torch.log1p(-u))
        soft = torch.sigmoid((sample_logits + logistic) / self.gate_temperature)
        hard = (soft > 0.5).to(soft.dtype)
        # Preserve the unit-norm contract even if every gate in a row closes.
        empty = hard.sum(dim=1, keepdim=True) == 0
        fallback = F.one_hot(
            sample_logits.argmax(dim=1), sample_logits.shape[1]
        ).to(hard.dtype)
        hard = torch.where(empty, fallback, hard)
        return (hard + soft - soft.detach()).to(logits.dtype)

    def _pad_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to descriptor grid and apply padding for convolutions.

        Orientation is padded **circularly** -- bin 7 neighbours bin 0, since a
        gradient direction between two bins deposits in both. The 4x4 spatial
        grid is not periodic, so its edges replicate: opposite corners of the
        patch are not neighbours.
        """
        batch = x.shape[0]
        rows, cols, orient = self.layout
        pad = self.gate_kernel // 2
        grid = x.reshape(batch, 1, rows, cols, orient)
        # F.pad's tuple runs last-dim-first: (W, W, H, H, D, D).
        grid = F.pad(grid, (pad, pad, 0, 0, 0, 0), mode="circular")
        grid = F.pad(grid, (0, 0, pad, pad, pad, pad), mode="replicate")
        return grid

    def _couple(self, logits: torch.Tensor) -> torch.Tensor:
        """Mix each gate logit with its neighbours on the descriptor grid."""
        grid = self._pad_grid(logits)
        return self.gate_coupling(grid).reshape(logits.shape[0], -1)

    def _smooth_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Correlate i.i.d. noise over the descriptor grid.

        Correlated logits with independent noise still sample near
        independently: the correlation has to be in the draw, not only in the
        mean.

        The per-position rescaling makes the output std exactly 1 at every
        coordinate for i.i.d. unit-variance input, so smoothing leaves the
        gate's effective temperature unchanged across the whole grid rather
        than only in its interior. It is a per-position gain, so the
        neighbour correlation the smoothing exists to produce survives it.
        """
        batch = noise.shape[0]
        grid = self._pad_grid(noise)
        kernel = self.noise_kernel.to(grid.dtype)
        smoothed = F.conv3d(grid, kernel).reshape(batch, -1)
        return smoothed * self.noise_scale.to(smoothed.dtype)

    def _gate_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self._couple(self.gate_head(h)) + self.sparsity_head(h)
        return self.logit_clamp * torch.tanh(logits / self.logit_clamp)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        # Floor the magnitude above `eps` so the unit-norm contract holds under
        # divergence: softplus saturates to exactly 0.0 below about -90 in
        # float32, and an open gate over a zero magnitude still normalizes to
        # the zero vector. Exact zeros come from the gate, not from here.
        magnitude = F.softplus(self.magnitude_head(h)).clamp(min=self.eps * 100.0)
        x = self._sample_gate(self._gate_logits(h)) * magnitude
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / torch.clamp(norm, min=self.eps)


def build_generator(model_cfg: Mapping[str, Any], output_dim: int) -> nn.Module:
    """Build the configured generator, defaulting to the legacy MLP."""
    kind = model_cfg.get("generator_type", "mlp")
    common = {
        "latent_dim": int(model_cfg["latent_dim"]),
        "output_dim": output_dim,
        "hidden_dims": model_cfg["generator_hidden_dims"],
        "negative_slope": float(model_cfg["negative_slope"]),
    }
    if kind == "mlp":
        return Generator(**common)
    if kind == "gated":
        return GatedGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
        )
    if kind == "structured_gated":
        return StructuredGateGenerator(
            **common,
            gate_temperature=float(model_cfg.get("gate_temperature", 0.5)),
            logit_clamp=float(model_cfg.get("logit_clamp", 10.0)),
            layout=tuple(model_cfg.get("layout", (4, 4, 8))),
            gate_kernel=int(model_cfg.get("gate_kernel", 3)),
            noise_kernel_sigma=float(model_cfg.get("noise_kernel_sigma", 0.65)),
        )
    raise ValueError(f"Unknown generator_type: {kind}")
