"""Independent GeoReg adapter; the original MiniOneRec RQVAE is unchanged."""
from __future__ import annotations

import math
from numbers import Real

import torch

from rq.models.rqvae import RQVAE
from ..core import GeoReg, geometry_blocks_loss


class RQVAEAdapter(RQVAE):
    """Original RQVAE with explicit optional auxiliary geometry.

    The ordinary forward always returns the original triple. Request return_aux
    for a fourth dictionary; disabled/zero-weight calls return empty zero stats
    without entering geometry or consuming projection RNG. Model weights remain
    strictly compatible with the original model, including state-dict keys.
    """

    def __init__(self, *args, geo_mode='none', geo_weight=0.0, geo_nproj=256,
                 geo_num_frequencies=17, geo_tmax=3.0, geo_eps=1e-6,
                 geo_projection_chunk=64, geo_seed=2024, **kwargs):
        if geo_mode not in ('none', 'global', 'residual'):
            raise ValueError('geo_mode must be none, global or residual')
        if (isinstance(geo_weight, bool) or not isinstance(geo_weight, Real) or
                not math.isfinite(geo_weight) or geo_weight < 0):
            raise ValueError('geo_weight must be finite and nonnegative')
        super().__init__(*args, **kwargs)
        self.geo_mode = geo_mode
        self.geo_weight = float(geo_weight)
        self.georeg = GeoReg(nproj=geo_nproj, num_frequencies=geo_num_frequencies,
                             tmax=geo_tmax, eps=geo_eps,
                             projection_chunk=geo_projection_chunk, seed=geo_seed)

    @property
    def geo_enabled(self):
        return self.geo_mode != 'none' and self.geo_weight > 0

    @property
    def geo_config(self):
        config = self.georeg.config.to_dict()
        return {'geo_mode': self.geo_mode, 'geo_weight': self.geo_weight,
                **{'geo_' + key: value for key, value in config.items()}}

    def forward(self, x, use_sk=True, return_aux=False, geo_directions=None):
        if not return_aux or not self.geo_enabled:
            original = super().forward(x, use_sk=use_sk)
            if not return_aux:
                return original
            reference = original[0]
            return (*original, {'geo_loss': reference.new_zeros(()),
                                'geo_names': (), 'geo_level_losses': reference.new_empty(0),
                                'geo_residual_rms': reference.new_empty(0),
                                'geo_scales': reference.new_empty(0)})

        # The one quantization forward supplies both the unchanged base graph
        # and its actual selected indices. No hook or graph-bearing cache lives
        # on the module, and no second nearest-neighbor/Sinkhorn pass is used.
        if x.device.type not in ('cpu', 'cuda'):
            raise ValueError(f'GeoReg supports CPU/CUDA only, got {x.device.type}')
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError('GeoReg auxiliary forward requires a B x d batch with B >= 2')
        z = self.encoder(x)
        x_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(x_q)
        with torch.autocast(device_type=z.device.type, enabled=False):
            residual = z if z.dtype == torch.float64 else z.float()
            nlevels = 1 if self.geo_mode == 'global' else len(self.rq.vq_layers)
            blocks = {}
            for level in range(nlevels):
                blocks[f'level_{level}'] = residual
                if level + 1 < nlevels:
                    selected = self.rq.vq_layers[level].get_codebook_entry(indices[..., level])
                    residual = residual - selected.detach().to(dtype=residual.dtype)
            loss, stats = geometry_blocks_loss(blocks, self.georeg, geo_directions,
                                               return_stats=True)
        return out, rq_loss, indices, {'geo_loss': loss, 'geo_names': stats['names'],
                                       'geo_level_losses': stats['level_losses'],
                                       'geo_residual_rms': stats['residual_rms'],
                                       'geo_scales': stats['scales']}
