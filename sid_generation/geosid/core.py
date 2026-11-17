"""Backend-independent characteristic-function geometry (``cf_mean``).

Centering, differentiable scalar RMS, normalized quadrature and mean reductions
are this project's adaptation choices, not unpublished GeoSID implementation
details. Only CPU/CUDA and float16/bfloat16/float32/float64 inputs are supported.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch
from torch import nn


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive_scalar(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _device(device):
    device = torch.device('cpu' if device is None else device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError(f"GeoReg supports CPU/CUDA only, got {device.type}")
    return device


def _validate_residual(residual):
    if not isinstance(residual, torch.Tensor):
        raise ValueError("GeoReg expects a tensor")
    _device(residual.device)
    if residual.ndim != 2 or residual.shape[0] < 2 or residual.shape[1] < 1:
        raise ValueError("GeoReg expects a B x d tensor with B >= 2 and d >= 1")
    if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("GeoReg expects float16, bfloat16, float32 or float64 residuals")
    if not torch.isfinite(residual).all():
        raise ValueError("GeoReg residuals must be finite")


def _work_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


@dataclass(frozen=True)
class GeoRegConfig:
    nproj: int = 256
    num_frequencies: int = 17
    tmax: float = 3.0
    eps: float = 1e-6
    projection_chunk: int = 64
    seed: int = 2024

    def __post_init__(self):
        _positive_int('nproj', self.nproj)
        _positive_int('num_frequencies', self.num_frequencies, minimum=2)
        _positive_int('projection_chunk', self.projection_chunk)
        _positive_scalar('tmax', self.tmax)
        _positive_scalar('eps', self.eps)
        if (isinstance(self.seed, bool) or not isinstance(self.seed, int) or
                not -(2**63) <= self.seed <= 2**64 - 1):
            raise ValueError('seed must be an integer in the torch.Generator seed range')

    def to_dict(self):
        return asdict(self)


def frequency_grid(num_frequencies=17, tmax=3.0, *, dtype=torch.float32, device=None):
    """Frequencies, Gaussian CF target and normalized Gaussian/trapezoid weights."""
    _positive_int('num_frequencies', num_frequencies, minimum=2)
    _positive_scalar('tmax', tmax)
    device = _device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('CF grid requires float32 or float64 working precision')
    if tmax > torch.finfo(dtype).max:
        raise ValueError('tmax exceeds working precision')
    t = torch.linspace(0.0, tmax, num_frequencies, dtype=dtype, device=device)
    phi = torch.exp(-0.5 * t.square())
    q = torch.full_like(t, tmax / (num_frequencies - 1))
    q[0] *= 0.5
    q[-1] *= 0.5
    weights = q * phi
    normalizer = weights.sum()
    if not torch.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('CF quadrature is not representable in working precision')
    return t, phi, weights / normalizer


def normalize_residual(residual, eps=1e-6):
    """Center by coordinate and divide by one differentiable overall RMS."""
    _validate_residual(residual)
    _positive_scalar('eps', eps)
    with torch.autocast(device_type=residual.device.type, enabled=False):
        x = residual.to(dtype=_work_dtype(residual))
        centered = x - x.mean(dim=0, keepdim=True)
        scale = (centered.square().mean() + eps * eps).sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError('Residual normalization overflow/underflow in working precision')
        normalized = centered / scale
        if not torch.isfinite(normalized).all():
            raise ValueError('Normalized residuals must be finite')
        return normalized, scale


def _validate_directions(directions, dimension):
    if not isinstance(directions, torch.Tensor):
        raise ValueError('directions must be a tensor')
    _device(directions.device)
    if directions.ndim != 2 or directions.shape[0] != dimension or directions.shape[1] < 1:
        raise ValueError('directions must have shape d x M with M >= 1')
    if not directions.is_floating_point() or not torch.isfinite(directions).all():
        raise ValueError('directions must be finite floating-point values')
    norms = torch.linalg.vector_norm(directions, dim=0)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError('directions must be unit vectors in columns')


def _directions_for(directions, residual):
    if not isinstance(directions, torch.Tensor):
        raise ValueError('directions must be a tensor')
    _device(directions.device)
    if directions.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError('directions must be finite floating-point values')
    result = directions.to(device=residual.device, dtype=_work_dtype(residual))
    _validate_directions(result, residual.shape[1])
    return result


def _cf_discrepancy(residual, directions, t, phi, weights, projection_chunk):
    # Chunk projections only: each empirical CF still uses the entire batch.
    total = residual.new_zeros(())
    for start in range(0, directions.shape[1], projection_chunk):
        projected = residual @ directions[:, start:start + projection_chunk]
        angles = projected.unsqueeze(-1) * t
        if not torch.isfinite(angles).all():
            raise ValueError('Projected CF arguments overflowed in working precision')
        real = angles.cos().mean(dim=0)
        imag = angles.sin().mean(dim=0)
        total = total + (((real - phi).square() + imag.square()) * weights).sum()
    return total / directions.shape[1]


def cf_discrepancy(residual, directions, *, num_frequencies=17, tmax=3.0,
                   projection_chunk=64):
    """Pure uncentered fixed-projection CF kernel, suitable for gradcheck."""
    _validate_residual(residual)
    _positive_int('projection_chunk', projection_chunk)
    with torch.autocast(device_type=residual.device.type, enabled=False):
        x = residual.to(dtype=_work_dtype(residual))
        directions = _directions_for(directions, x)
        grid = frequency_grid(num_frequencies, tmax, dtype=x.dtype, device=x.device)
        return _cf_discrepancy(x, directions, *grid, projection_chunk)


class GeoReg(nn.Module):
    """Normalized ``cf_mean`` with no trainable or persistent model tensors.

    Save the independent CPU generator with get_rng_state/set_rng_state in the
    backend's own checkpoint. Fixed diagnostic directions never advance it.
    """

    def __init__(self, nproj=256, num_frequencies=17, tmax=3.0, eps=1e-6,
                 projection_chunk=64, seed=2024):
        super().__init__()
        self.config = GeoRegConfig(nproj, num_frequencies, tmax, eps, projection_chunk, seed)
        for name, value in self.config.to_dict().items():
            setattr(self, name, value)
        # These immutable CPU constants are deliberately not Module buffers:
        # .half()/.bfloat16() must not permanently round the CF target/weights.
        self._grid = frequency_grid(num_frequencies, tmax, dtype=torch.float64)
        self._generator = torch.Generator(device='cpu').manual_seed(seed)

    def get_rng_state(self):
        return self._generator.get_state().clone()

    def set_rng_state(self, state):
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError('GeoReg RNG state must be a one-dimensional byte tensor')
        self._generator.set_state(state.detach().cpu())

    def sample_directions(self, dimension, *, dtype=torch.float32, device=None):
        _positive_int('dimension', dimension)
        device = _device(device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Projection sampling requires float32 or float64')
        directions = torch.randn(dimension, self.nproj, generator=self._generator,
                                 dtype=dtype, device='cpu')
        directions = directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True)
        if not torch.isfinite(directions).all():
            raise ValueError('Sampled projection normalization was nonfinite')
        return directions.to(device=device)

    def forward(self, residual, directions=None, *, return_stats=False):
        _validate_residual(residual)
        with torch.autocast(device_type=residual.device.type, enabled=False):
            normalized, scale = normalize_residual(residual, self.eps)
            if directions is None:
                directions = self.sample_directions(normalized.shape[1], dtype=normalized.dtype,
                                                    device=normalized.device)
            else:
                directions = _directions_for(directions, normalized)
            grid = [value.to(device=normalized.device, dtype=normalized.dtype) for value in self._grid]
            loss = _cf_discrepancy(normalized, directions, *grid, self.projection_chunk)
            if not return_stats:
                return loss
            rms = residual.to(dtype=normalized.dtype).square().mean().sqrt()
            if not torch.isfinite(rms):
                raise ValueError('Residual RMS overflowed in working precision')
            return loss, {'scale': scale.detach(), 'residual_rms': rms.detach()}


def geometry_blocks_loss(blocks, georeg, directions=None, *, return_stats=False):
    """Mean geometry across named blocks, independent of their training backend.

    ``blocks`` is a nonempty insertion-ordered mapping of names to B x d tensors.
    Blocks must share d, device and working precision; each has B >= 2. Fixed
    directions are either one shared d x M tensor or a complete name->tensor
    mapping. Batch counts may differ. No hidden weighting by batch or block size.
    """
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError('Geometry blocks must be a nonempty named mapping')
    if not isinstance(georeg, GeoReg):
        raise ValueError('georeg must be a GeoReg module')
    names = tuple(blocks)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('Geometry block names must be nonempty strings')
    signature = None
    for residual in blocks.values():
        _validate_residual(residual)
        current = (residual.shape[1], residual.device, _work_dtype(residual))
        if signature is not None and current != signature:
            raise ValueError('Geometry blocks must share dimension, device and working precision')
        signature = current
    if isinstance(directions, Mapping):
        if set(directions) != set(names):
            raise ValueError('Fixed direction names must match all geometry blocks')
        fixed = {name: _directions_for(directions[name], blocks[name]) for name in names}
    elif directions is not None:
        shared = _directions_for(directions, next(iter(blocks.values())))
        fixed = {name: shared for name in names}
    else:
        fixed = {name: None for name in names}
    losses, scales, rms = [], [], []
    for name, residual in blocks.items():
        loss, stats = georeg(residual, fixed[name], return_stats=True)
        losses.append(loss)
        scales.append(stats['scale'])
        rms.append(stats['residual_rms'])
    level_losses = torch.stack(losses)
    loss = level_losses.mean()
    if not return_stats:
        return loss
    return loss, {'names': names, 'level_losses': level_losses,
                  'scales': torch.stack(scales), 'residual_rms': torch.stack(rms)}
