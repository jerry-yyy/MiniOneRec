import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from .georeg import GeoReg


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 geo_mode="none",
                 geo_weight=0.0,
                 geo_nproj=256,
                 geo_num_frequencies=17,
                 geo_tmax=3.0,
                 geo_eps=1e-6,
                 geo_projection_chunk=64,
                 geo_seed=2024,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        if geo_mode not in ("none", "global", "residual"):
            raise ValueError("geo_mode must be none, global, or residual")
        if not math.isfinite(geo_weight) or geo_weight < 0:
            raise ValueError("geo_weight must be finite and nonnegative")
        self.geo_mode = geo_mode
        self.geo_weight = float(geo_weight)
        self.georeg = GeoReg(nproj=geo_nproj, num_frequencies=geo_num_frequencies,
                             tmax=geo_tmax, eps=geo_eps,
                             projection_chunk=geo_projection_chunk, seed=geo_seed)

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    @property
    def geo_enabled(self):
        return self.geo_mode != "none" and self.geo_weight > 0

    @property
    def geo_config(self):
        return {"geo_mode": self.geo_mode, "geo_weight": self.geo_weight,
                "geo_nproj": self.georeg.nproj,
                "geo_num_frequencies": self.georeg.num_frequencies,
                "geo_tmax": self.georeg.tmax, "geo_eps": self.georeg.eps,
                "geo_projection_chunk": self.georeg.projection_chunk,
                "geo_seed": self.georeg.seed}

    def forward(self, x, use_sk=True, return_aux=False, geo_directions=None):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        if not return_aux:
            return out, rq_loss, indices

        # Never reconstruct residuals or consume projection RNG when disabled.
        aux = {"geo_loss": x.new_zeros(()), "geo_level_losses": x.new_empty(0),
               "geo_residual_rms": x.new_empty(0), "geo_scales": x.new_empty(0)}
        if self.geo_enabled:
            losses, scales, rms = [], [], []
            nlevels = 1 if self.geo_mode == "global" else len(self.rq.vq_layers)
            # The original RQ path and all its STE/commitment gradients stay
            # intact. This auxiliary surrogate sends each level's gradient to
            # the encoder while adding no codebook gradient.
            with torch.autocast(device_type=x.device.type, enabled=False):
                residual = x if x.dtype == torch.float64 else x.float()
                for level in range(nlevels):
                    loss, stats = self.georeg(residual, directions=geo_directions,
                                             return_stats=True)
                    losses.append(loss)
                    scales.append(stats["scale"])
                    rms.append(stats["residual_rms"])
                    if level + 1 < nlevels:
                        selected = self.rq.vq_layers[level].get_codebook_entry(indices[..., level])
                        residual = residual - selected.detach().to(dtype=residual.dtype)
            aux = {"geo_loss": torch.stack(losses).mean(),
                   "geo_level_losses": torch.stack(losses),
                   "geo_residual_rms": torch.stack(rms), "geo_scales": torch.stack(scales)}
        return out, rq_loss, indices, aux

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
