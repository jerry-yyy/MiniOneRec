"""Characteristic-function geometry regularization for MiniOneRec.

The batch centering, single differentiable RMS, normalized quadrature and
mean reductions are this project's ``cf_mean`` adaptation conventions. They
are not asserted to reproduce unpublished GeoSID implementation details.
"""

import math

import torch
from torch import nn


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_residual(residual):
    if residual.ndim != 2 or residual.shape[0] < 2 or residual.shape[1] < 1:
        raise ValueError("GeoReg expects a B x d tensor with B >= 2 and d >= 1")
    if not residual.is_floating_point():
        raise ValueError("GeoReg expects floating-point residuals")
    if not torch.isfinite(residual).all():
        raise ValueError("GeoReg residuals must be finite")


def _work_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def frequency_grid(num_frequencies=17, tmax=3.0, *, dtype=torch.float32, device=None):
    """Return frequencies, the Gaussian target, and normalized trapezoid weights."""
    _positive_int("num_frequencies", num_frequencies, minimum=2)
    if not math.isfinite(tmax) or tmax <= 0:
        raise ValueError("tmax must be finite and positive")
    t = torch.linspace(0.0, tmax, num_frequencies, dtype=dtype, device=device)
    phi = torch.exp(-0.5 * t.square())
    q = torch.full_like(t, tmax / (num_frequencies - 1))
    q[0] *= 0.5
    q[-1] *= 0.5
    weights = q * phi
    return t, phi, weights / weights.sum()


def normalize_residual(residual, eps=1e-6):
    """Center each coordinate and divide by one differentiable overall RMS.

    Returns ``(normalized, scale)``. Half/bfloat16 inputs are evaluated in
    float32; double inputs retain double precision for numerical tests.
    """
    _validate_residual(residual)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    with torch.autocast(device_type=residual.device.type, enabled=False):
        x = residual.to(dtype=_work_dtype(residual))
        centered = x - x.mean(dim=0, keepdim=True)
        scale = (centered.square().mean() + eps * eps).sqrt()
        return centered / scale, scale


def _validate_directions(directions, dimension):
    if directions.ndim != 2 or directions.shape[0] != dimension or directions.shape[1] < 1:
        raise ValueError("directions must have shape d x M with M >= 1")
    if not directions.is_floating_point() or not torch.isfinite(directions).all():
        raise ValueError("directions must be finite floating-point values")
    norms = torch.linalg.vector_norm(directions, dim=0)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError("directions must be unit vectors in columns")


def _cf_discrepancy(residual, directions, t, phi, weights, projection_chunk):
    # Chunk only directions: each empirical CF still uses the complete batch.
    # Summing before division handles a final, shorter projection chunk.
    total = residual.new_zeros(())
    for start in range(0, directions.shape[1], projection_chunk):
        projected = residual @ directions[:, start:start + projection_chunk]
        angles = projected.unsqueeze(-1) * t
        real = angles.cos().mean(dim=0)
        imag = angles.sin().mean(dim=0)
        total = total + (((real - phi).square() + imag.square()) * weights).sum()
    return total / directions.shape[1]


def cf_discrepancy(residual, directions, *, num_frequencies=17, tmax=3.0,
                   projection_chunk=64):
    """Pure, *uncentered* CF kernel with fixed unit projection directions.

    This function permits numerical gradcheck. Centering and scaling belong
    to :func:`normalize_residual`, not to this kernel. No batch-size factor
    is applied.
    """
    _validate_residual(residual)
    _positive_int("projection_chunk", projection_chunk)
    with torch.autocast(device_type=residual.device.type, enabled=False):
        x = residual.to(dtype=_work_dtype(residual))
        directions = directions.to(device=x.device, dtype=x.dtype)
        _validate_directions(directions, x.shape[1])
        grid = frequency_grid(num_frequencies, tmax, dtype=x.dtype, device=x.device)
        return _cf_discrepancy(x, directions, *grid, projection_chunk)


class GeoReg(nn.Module):
    """Normalized ``cf_mean`` loss, without trainable or persistent tensors.

    Sampling uses a dedicated CPU generator, independently of model/device
    RNG streams. Save its state in the *outer training checkpoint*, using
    ``get_rng_state`` and ``set_rng_state``. Supply fixed directions for
    diagnostics so results do not depend on the number of training calls.
    """

    def __init__(self, nproj=256, num_frequencies=17, tmax=3.0, eps=1e-6,
                 projection_chunk=64, seed=2024):
        super().__init__()
        _positive_int("nproj", nproj)
        _positive_int("projection_chunk", projection_chunk)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.nproj = nproj
        self.num_frequencies = num_frequencies
        self.tmax = tmax
        self.eps = eps
        self.projection_chunk = projection_chunk
        self.seed = seed
        # Double construction also preserves the reference grid under gradcheck.
        for name, value in zip(("frequencies", "target_cf", "frequency_weights"),
                               frequency_grid(num_frequencies, tmax, dtype=torch.float64)):
            self.register_buffer(name, value, persistent=False)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def get_rng_state(self):
        return self._generator.get_state().clone()

    def set_rng_state(self, state):
        self._generator.set_state(state.detach().cpu())

    def sample_directions(self, dimension, *, dtype=torch.float32, device=None):
        _positive_int("dimension", dimension)
        # Sampling on CPU makes device transfers and global RNG immaterial.
        directions = torch.randn(dimension, self.nproj, generator=self._generator,
                                 dtype=dtype, device="cpu")
        directions = directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True)
        return directions.to(device=device)

    def forward(self, residual, directions=None, *, return_stats=False):
        with torch.autocast(device_type=residual.device.type, enabled=False):
            normalized, scale = normalize_residual(residual, self.eps)
            if directions is None:
                directions = self.sample_directions(normalized.shape[1],
                                                    dtype=normalized.dtype,
                                                    device=normalized.device)
            else:
                directions = directions.to(device=normalized.device, dtype=normalized.dtype)
                _validate_directions(directions, normalized.shape[1])
            grid = [value.to(device=normalized.device, dtype=normalized.dtype) for value in
                    (self.frequencies, self.target_cf, self.frequency_weights)]
            loss = _cf_discrepancy(normalized, directions, *grid, self.projection_chunk)
            if not return_stats:
                return loss
            # Statistics are logs only; the loss retains differentiable scale/mean.
            rms = residual.to(dtype=normalized.dtype).square().mean().sqrt()
            return loss, {"scale": scale.detach(), "residual_rms": rms.detach()}
