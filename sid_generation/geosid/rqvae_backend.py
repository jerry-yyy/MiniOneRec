"""A small opt-in RQ-VAE training driver; upstream model/trainer are untouched.

This driver deliberately has no scheduler, checkpoint-selection or resume
framework. It uses the original reconstruction/quantization loss, AdamW and
the original unit gradient clipping. Saved artifacts are inference models.
"""

import math
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from sid_generation.sid_common import canonical_item_ids, load_npz, save_npz, vectors
from .adapters.rqvae import RQVAEAdapter
from .core import GeoReg, geometry_blocks_loss


@contextmanager
def seeded(seed, device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            # torch.manual_seed also changes every CUDA/MPS/XPU RNG. Only seed
            # generators whose state this context actually saves and restores.
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def checked_options(defaults, supplied, label):
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError(f"unknown {label} options; allowed: {sorted(defaults)}")
    return dict(defaults, **supplied)


def positive(value, name, integer=False):
    if isinstance(value, bool) or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class RQVAEBackend:
    def __init__(self, model=None, geometry=None, training=None, seed=2024):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.model_options = checked_options({
            "num_emb_list": [256, 256, 256], "e_dim": 64, "layers": [512, 256, 128],
            "dropout_prob": 0.0, "bn": False, "loss_type": "mse",
            "quant_loss_weight": 1.0, "beta": 0.25, "kmeans_init": False,
            "kmeans_iters": 100, "sk_epsilons": None, "sk_iters": 100,
        }, model, "RQ-VAE model")
        for key in ("num_emb_list", "layers"):
            values = self.model_options[key]
            if not isinstance(values, (list, tuple)) or (key == "num_emb_list" and not values):
                raise ValueError(f"{key} must be a list of positive integers")
            for value in values:
                positive(value, key, integer=True)
        for key in ("e_dim", "kmeans_iters", "sk_iters"):
            positive(self.model_options[key], key, integer=True)
        for key in ("bn", "kmeans_init"):
            if not isinstance(self.model_options[key], bool):
                raise ValueError(f"{key} must be a boolean")
        for key in ("beta", "quant_loss_weight", "dropout_prob"):
            value = self.model_options[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.model_options["dropout_prob"] >= 1 or self.model_options["loss_type"] not in ("mse", "l1"):
            raise ValueError("dropout_prob must be below 1; loss_type must be mse or l1")
        epsilons = self.model_options["sk_epsilons"]
        if epsilons is not None:
            if not isinstance(epsilons, (list, tuple)) or len(epsilons) != len(self.capacities):
                raise ValueError("sk_epsilons must match num_emb_list length")
            for value in epsilons:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("sk_epsilons must be finite and nonnegative")
        self.geometry = checked_options({
            "mode": "none", "weight": 0.0, "nproj": 256, "num_frequencies": 17,
            "tmax": 3.0, "eps": 1e-6, "projection_chunk": 64, "seed": 952024,
        }, geometry, "geometry")
        if self.geometry["mode"] not in ("none", "global", "residual"):
            raise ValueError("RQ-VAE geometry mode must be none, global or residual")
        weight = self.geometry["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("geometry weight must be finite and nonnegative")
        GeoReg(**{key: value for key, value in self.geometry.items() if key not in ("mode", "weight")})
        self.training = checked_options({
            "epochs": 1, "batch_size": 256, "lr": 1e-4,
            "weight_decay": 0.0, "device": "cpu",
        }, training, "RQ-VAE training")
        for key in ("epochs", "batch_size"):
            positive(self.training[key], key, integer=True)
        positive(self.training["lr"], "lr")
        decay = self.training["weight_decay"]
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.device = torch.device(self.training["device"])
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("RQ-VAE GeoSID supports CPU and CUDA only")
        self.model = None
        self.history = []

    @property
    def capacities(self):
        return self.model_options["num_emb_list"]

    def configuration(self):
        return {"backend": "rqvae", "model": self.model_options,
                "geometry": self.geometry, "training": self.training, "seed": self.seed}

    def _construct(self, dimension):
        options = dict(self.model_options)
        if options["sk_epsilons"] is None:
            options["sk_epsilons"] = [0.0] * len(self.capacities)
        geometry = {"geo_" + key: value for key, value in self.geometry.items()}
        return RQVAEAdapter(in_dim=dimension, **options, **geometry).to(self.device)

    def fit(self, data, item_ids):
        data = vectors(data)
        canonical_item_ids(item_ids, len(data))
        batch = self.training["batch_size"]
        active = self.geometry["mode"] != "none" and self.geometry["weight"] != 0
        if (active or self.model_options["bn"]) and (min(batch, len(data)) < 2 or len(data) % batch == 1):
            raise ValueError("geometry/BatchNorm requires every batch >= 2; change batch_size (no rows are dropped)")
        with seeded(self.seed, self.device):
            self.model = self._construct(data.shape[1])
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training["lr"],
                                          weight_decay=self.training["weight_decay"])
            loader = DataLoader(torch.from_numpy(data), batch_size=batch, shuffle=True,
                                num_workers=0, generator=torch.Generator().manual_seed(self.seed))
            self.history = []
            for epoch in range(self.training["epochs"]):
                self.model.train()
                totals = np.zeros(4, dtype=np.float64)
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    out, quantization, _, aux = self.model(inputs, return_aux=True)
                    base, reconstruction = self.model.compute_loss(out, quantization, xs=inputs)
                    loss = base + self.geometry["weight"] * aux["geo_loss"]
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    totals += np.array([loss.item(), reconstruction.item(), quantization.item(),
                                        aux["geo_loss"].item()]) * len(inputs)
                self.history.append(dict(epoch=epoch + 1, **dict(zip(
                    ("total_loss", "reconstruction", "quantization", "geometry"), (totals / len(data)).tolist()))))
            self.model.eval()
        return self

    def _inputs(self, data, item_ids):
        if self.model is None:
            raise ValueError("fit or load an RQ-VAE GeoSID model first")
        data = vectors(data, self.model.in_dim)
        canonical_item_ids(item_ids, len(data))
        return data

    def _quantize(self, x):
        z = self.model.encoder(x)
        if not torch.isfinite(z).all():
            raise ValueError("nonfinite encoder representation")
        # The original squared-distance identity may become inf+inf-inf even
        # when nearby huge vectors have a finite commitment loss. Bound every
        # possible residual/code distance before invoking that unchanged path.
        magnitude = z.detach().abs().max().item()
        magnitude += sum(q.embedding.weight.detach().abs().max().item() for q in self.model.rq.vq_layers)
        limit = math.sqrt(torch.finfo(z.dtype).max / (16 * z.shape[1]))
        if not math.isfinite(magnitude) or magnitude > limit:
            raise ValueError("nonfinite-risk quantizer distances; rescale inputs/weights")
        q, loss, codes = self.model.rq(z, use_sk=False)
        if not torch.isfinite(q).all() or not torch.isfinite(loss).all():
            raise ValueError("nonfinite quantization arithmetic; rescale inputs/weights")
        return z, q, codes

    @torch.no_grad()
    def encode(self, data, item_ids):
        data = self._inputs(data, item_ids)
        self.model.eval()
        batch = self.training["batch_size"]
        return np.concatenate([self._quantize(torch.from_numpy(data[i:i + batch]).to(self.device))[2].cpu().numpy()
                               for i in range(0, len(data), batch)])

    @torch.no_grad()
    def diagnose(self, data, item_ids, diagnostic_seed=314159):
        data = self._inputs(data, item_ids)
        if len(data) < 2:
            raise ValueError("geometry diagnostics require at least two items")
        self.model.eval()
        latents, indices, recon_error = [], [], 0.0
        batch = self.training["batch_size"]
        for start in range(0, len(data), batch):
            x = torch.from_numpy(data[start:start + batch]).to(self.device)
            z, q, codes = self._quantize(x)
            out = self.model.decoder(q)
            if not torch.isfinite(out).all():
                raise ValueError("nonfinite decoded reconstruction")
            recon_error += (out.double() - x.double()).square().sum().item()
            latents.append(z)
            indices.append(codes)
        residual = torch.cat(latents)
        codes = torch.cat(indices)
        blocks = {}
        for level, quantizer in enumerate(self.model.rq.vq_layers):
            blocks[f"residual_{level}"] = residual
            if level + 1 < len(self.capacities):
                residual = residual - quantizer.embedding(codes[:, level]).detach()
        config = {k: v for k, v in self.geometry.items() if k not in ("mode", "weight")}
        config["seed"] = diagnostic_seed
        regularizer = GeoReg(**config)
        directions = regularizer.sample_directions(residual.shape[1], device=residual.device)
        loss, stats = geometry_blocks_loss(blocks, regularizer, directions=directions, return_stats=True)
        return {"geometry_mean": loss.item(), "geometry_per_block": stats["level_losses"].cpu().tolist(),
                "block_names": list(blocks), "reconstruction_mse": recon_error / data.size,
                "diagnostic_seed": diagnostic_seed, "scope": "all supplied items; quantizer inputs before each RQ level"}

    def save(self, path, overwrite=False):
        if self.model is None:
            raise ValueError("cannot save an unfitted backend")
        arrays = {"weight_" + key: value.detach().cpu().numpy() for key, value in self.model.state_dict().items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("cannot save nonfinite model weights")
        metadata = {"algorithm": "geosid_rqvae", "configuration": self.configuration(),
                    "dimension": self.model.in_dim, "history": self.history,
                    "purpose": "inference only; no training resume contract"}
        save_npz(path, metadata, arrays, overwrite)

    @classmethod
    def load(cls, path):
        metadata, arrays = load_npz(path, "geosid_rqvae")
        config = dict(metadata["configuration"])
        if config.pop("backend") != "rqvae":
            raise ValueError("wrong backend in RQ-VAE artifact")
        # Portable inference loading does not allocate CUDA without opt-in.
        config["training"] = dict(config["training"], device="cpu")
        result = cls(**config)
        dimension = positive(metadata["dimension"], "dimension", integer=True)
        with seeded(result.seed, result.device):
            result.model = result._construct(dimension)
        expected = result.model.state_dict()
        if set(arrays) != {"weight_" + key for key in expected}:
            raise ValueError("artifact weight keys do not match the model")
        state = {}
        for key, reference in expected.items():
            value = arrays["weight_" + key]
            target_dtype = reference.cpu().numpy().dtype
            if value.shape != tuple(reference.shape) or value.dtype != target_dtype or not np.isfinite(value).all():
                raise ValueError(f"invalid weight {key}")
            state[key] = torch.from_numpy(value)
        result.model.load_state_dict(state, strict=True)
        for quantizer in result.model.rq.vq_layers:
            quantizer.initted = True
        result.model.eval()
        result.history = metadata.get("history", [])
        return result
