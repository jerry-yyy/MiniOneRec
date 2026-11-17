"""Experimental catalog-bound RQ + shaped OPQ residual + refitted PQ.

This is a new transductive hybrid adaptation, not the paper's original OPQ
algorithm. RQ assignments/codebooks and the fitted rotation stay frozen.
Free catalog rows Z are optimized, then only a new PQ is fitted on Z. The
additional PQ refit requires its own explicit refit-only experimental control.
Ordinary disabled/zero-weight execution delegates to the original RQOPQ.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np

from ...rq_opq import (RQOPQ, _check_faiss_magnitude, _rotate_stably,
                       faiss_threads, require_faiss, unpack_pq_codes)
from ...sid_common import (build_indexes, canonical_item_ids, load_npz,
                           positive_int, save_npz, vectors)


_GEOMETRY = dict(mode="none", weight=0.0, nproj=256, num_frequencies=17,
                 tmax=3.0, eps=1e-6, projection_chunk=64, seed=2024)
_TRAINING = dict(steps=100, lr=1e-3, semantic_weight=1.0,
                 cosine_weight=1.0, device="cpu", pq_refit_only=False)
_ALGORITHM = "geosid_rq_opq_transductive"
_ARRAY_NAMES = {"catalog_ids", "row_hashes", "prefix_codes", "learned_z",
                "final_codes", "refit_pq_index"}


def _options(defaults, supplied, name):
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError(f"{name} must be a mapping")
    supplied = dict(supplied or {})
    if set(supplied) - set(defaults):
        raise ValueError(f"unknown {name} options: {sorted(set(supplied)-set(defaults))}")
    return dict(defaults, **supplied)


def _scalar(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _row_hashes(data):
    # Canonical computational identity: finite little-endian FP32, including
    # signed-zero bytes. Higher precision differences rounded away by vectors()
    # are intentionally outside the original backend's computational identity.
    rows = np.ascontiguousarray(data, dtype="<f4")
    return np.stack([np.frombuffer(hashlib.sha256(row.tobytes()).digest(), dtype=np.uint8)
                     for row in rows])


def _catalog_hash(ids, row_hashes):
    digest = hashlib.sha256(b"geosid-rq-opq-catalog-v1")
    for item_id, row_hash in zip(ids, row_hashes):
        encoded = item_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _base_payload(model):
    # Reuse original serialization and validation rather than cloning a second
    # implementation of the upstream RQ/rotation/PQ persistence contract.
    with tempfile.TemporaryDirectory(prefix="geosid-opq-base-") as temporary:
        path = Path(temporary) / "base.npz"
        model.save(path)
        metadata, arrays = load_npz(path, "rq_opq")
    return metadata, arrays


def _base_from_payload(metadata, arrays):
    with tempfile.TemporaryDirectory(prefix="geosid-opq-base-") as temporary:
        path = Path(temporary) / "base.npz"
        save_npz(path, metadata, arrays)
        return RQOPQ.load(path)


def _clone_base(model):
    return _base_from_payload(*_base_payload(model))


def _rotation(model):
    faiss = require_faiss()
    transform = faiss.downcast_VectorTransform(model.opq_index.chain.at(0))
    return np.ascontiguousarray(faiss.vector_to_array(transform.A).reshape(model.dimension, model.dimension))


def _prefix_sum(model, codes):
    # Explicit selected-codebook sum; FP32 RQ subtraction and inverse rotation
    # can have small rounding residuals at initialization. Never assert bitwise
    # equality of reconstructed full embeddings with their original input.
    result = np.zeros((len(codes), model.dimension), dtype=np.float64)
    for level, codebook in enumerate(model.rq_codebooks):
        result += codebook[codes[:, level]].astype(np.float64)
    return vectors(result, model.dimension)


def _validate_pq(pq, model):
    faiss = require_faiss()
    if (not isinstance(pq, faiss.IndexPQ) or not pq.is_trained or pq.ntotal != 0
            or pq.metric_type != faiss.METRIC_L2 or pq.d != model.dimension
            or pq.pq.M != model.opq_subspaces or pq.pq.nbits != model.nbits):
        raise ValueError("refitted PQ structure does not match the frozen baseline")
    centroids = faiss.vector_to_array(pq.pq.centroids)
    if centroids.size != model.dimension * (1 << model.nbits):
        raise ValueError("refitted PQ centroid count does not match baseline")
    _check_faiss_magnitude(centroids, model.dimension, "refitted PQ")


def _pq_codes(pq, z, model):
    faiss = require_faiss()
    _check_faiss_magnitude(z, model.dimension, "shaped PQ encoding")
    codes = []
    with faiss_threads(faiss, model.threads):
        for start in range(0, len(z), model.batch_size):
            packed = pq.sa_encode(np.ascontiguousarray(z[start:start + model.batch_size]))
            codes.append(unpack_pq_codes(packed, model.opq_subspaces, model.nbits))
    return np.concatenate(codes)


def _pack_codes(codes, nbits):
    packed = np.zeros((len(codes), (codes.shape[1] * nbits + 7) // 8), dtype=np.uint8)
    for column in range(codes.shape[1]):
        byte, shift = divmod(column * nbits, 8)
        value = codes[:, column].astype(np.uint16)
        packed[:, byte] |= (value << shift).astype(np.uint8)
        if shift + nbits > 8:
            packed[:, byte + 1] |= (value >> (8 - shift)).astype(np.uint8)
    return packed


class RQOPQBackend:
    """Catalog-bound shaping; optional unfitted/fitted original RQOPQ input.

    ``model`` is an RQOPQ configuration mapping or an RQOPQ instance. A fitted
    instance is cloned and reused without refitting, allowing baseline,
    refit-only and geometry variants to branch from exactly one fitted model.
    Enabled/refit-only encode requires known unique IDs and matching canonical
    FP32 row hashes, allowing catalog reorder/subsets but no unseen embeddings.
    Saved models are final inference artifacts, not optimizer-resume files.
    """
    encoding_revision = 1

    def __init__(self, model=None, geometry=None, training=None, seed=2024):
        self.seed = positive_int(seed, "seed", minimum=0)
        if self.seed > 2**31 - 1:
            raise ValueError("seed must fit a signed 32-bit FAISS seed")
        self.geometry = _options(dict(_GEOMETRY, seed=self.seed), geometry, "geometry")
        self.training = _options(_TRAINING, training, "training")
        if self.geometry["mode"] not in ("none", "block"):
            raise ValueError("RQ-OPQ geometry mode must be none or block")
        for key in ("nproj", "projection_chunk"):
            positive_int(self.geometry[key], key)
        positive_int(self.geometry["num_frequencies"], "num_frequencies", minimum=2)
        positive_int(self.geometry["seed"], "geometry seed", minimum=0)
        if self.geometry["seed"] > 2**63 - 1:
            raise ValueError("geometry seed is outside the supported range")
        for key in ("tmax", "eps"):
            _scalar(self.geometry[key], key, positive=True)
        _scalar(self.geometry["weight"], "geometry weight")
        positive_int(self.training["steps"], "steps")
        for key in ("lr", "semantic_weight"):
            _scalar(self.training[key], key, positive=True)
        _scalar(self.training["cosine_weight"], "cosine_weight")
        if self.training["device"] != "cpu":
            raise ValueError("the experimental FAISS RQ-OPQ adapter currently supports CPU only")
        if not isinstance(self.training["pq_refit_only"], bool):
            raise ValueError("pq_refit_only must be boolean")
        if self.training["pq_refit_only"] and self.geo_enabled:
            raise ValueError("refit-only control cannot also enable geometry")
        self._reuse_fitted_baseline = isinstance(model, RQOPQ) and model.opq_index is not None
        if isinstance(model, RQOPQ):
            self.model = _clone_base(model) if model.opq_index is not None else RQOPQ(**model.configuration())
        elif model is None or isinstance(model, Mapping):
            config = dict(model or {})
            config.setdefault("seed", self.seed)
            self.model = RQOPQ(**config)
        else:
            raise ValueError("model must be an RQOPQ configuration or instance")
        self.history = []
        self.catalog_ids = []
        self.row_hashes = None
        self.prefix_codes = None
        self.learned_z = None
        self.final_codes = None
        self._refitted_pq = None
        self._rows = {}
        self._frozen_hashes = None

    @property
    def geo_enabled(self):
        return self.geometry["mode"] == "block" and self.geometry["weight"] > 0

    @property
    def catalog_bound(self):
        return self.geo_enabled or self.training["pq_refit_only"]

    @property
    def capacities(self):
        return self.model.capacities

    def configuration(self):
        return {"model": self.model.configuration(), "geometry": dict(self.geometry),
                "training": dict(self.training), "seed": self.seed}

    def _freeze_fingerprints(self):
        arrays = {f"rq_codebook_{j}": cb for j, cb in enumerate(self.model.rq_codebooks)}
        arrays["opq_index"] = require_faiss().serialize_index(self.model.opq_index)
        self._frozen_hashes = {key: _array_hash(value) for key, value in arrays.items()}

    def _check_frozen(self):
        if self._frozen_hashes is None:
            raise ValueError("fit the catalog-bound adapter first")
        arrays = {f"rq_codebook_{j}": cb for j, cb in enumerate(self.model.rq_codebooks)}
        arrays["opq_index"] = require_faiss().serialize_index(self.model.opq_index)
        if self._frozen_hashes != {key: _array_hash(value) for key, value in arrays.items()}:
            raise ValueError("the frozen RQ prefix/OPQ baseline was modified")

    def fit(self, data, item_ids):
        data = vectors(data)
        ids = canonical_item_ids(item_ids, len(data))
        if not self._reuse_fitted_baseline or self.model.opq_index is None:
            self.model.fit(data)
        elif self.model.dimension != data.shape[1]:
            raise ValueError("fitted baseline and catalog dimensions differ")
        self.history = []
        if not self.catalog_bound:
            # No torch import, geometry module, random projection or PQ refit.
            return self
        if len(data) < (1 << self.model.nbits):
            raise ValueError("bound catalog must contain at least one row per PQ centroid")
        self.learned_z = self.final_codes = self._refitted_pq = None
        prefix, residuals = self.model.rq_residuals(data)
        matrix = _rotation(self.model)
        z = _rotate_stably(residuals, matrix)
        self.catalog_ids = ids
        self._rows = {value: index for index, value in enumerate(ids)}
        self.row_hashes = _row_hashes(data)
        self.prefix_codes = prefix.copy()
        self._freeze_fingerprints()
        if self.geo_enabled:
            z = self._shape(data, z, _prefix_sum(self.model, prefix), matrix)
        self.learned_z = np.ascontiguousarray(z, dtype=np.float32)
        self.refit_pq()
        self._check_frozen()
        return self

    def _shape(self, data, initial_z, prefix, matrix):
        import torch
        from torch.nn import functional as F
        from ..core import GeoReg, geometry_blocks_loss

        _check_faiss_magnitude(data, self.model.dimension, "full-embedding FP32 semantic objective", len(data))
        _check_faiss_magnitude(prefix, self.model.dimension, "fixed-prefix FP32 semantic objective", len(data))
        config = {key: self.geometry[key] for key in _GEOMETRY if key not in ("mode", "weight")}
        georeg = GeoReg(**config)
        z = torch.nn.Parameter(torch.from_numpy(initial_z.copy()))
        target = torch.from_numpy(data.copy())
        fixed_prefix = torch.from_numpy(prefix.copy())
        rotation = torch.from_numpy(matrix.copy())
        optimizer = torch.optim.Adam([z], lr=self.training["lr"], weight_decay=0)
        width = self.model.dimension // self.model.opq_subspaces
        for step in range(1, self.training["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", enabled=False):
                reconstructed = fixed_prefix + z @ rotation
                mse = (reconstructed - target).square().mean()
                cosine = (1 - F.cosine_similarity(reconstructed, target, dim=1, eps=1e-8)).mean()
                blocks = {f"opq_subspace_{j}": z[:, j * width:(j + 1) * width]
                          for j in range(self.model.opq_subspaces)}
                cf, stats = geometry_blocks_loss(blocks, georeg, return_stats=True)
                loss = (self.training["semantic_weight"] * mse
                        + self.training["cosine_weight"] * cosine + self.geometry["weight"] * cf)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite full-embedding shaping objective")
            loss.backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise ValueError("nonfinite free-catalog gradient")
            row = {"step": step, "loss": loss.item(), "semantic_mse": mse.item(),
                   "semantic_cosine_loss": cosine.item(), "geometry": cf.item(),
                   "weighted_geometry": self.geometry["weight"] * cf.item(),
                   "gradient_norm": z.grad.norm().item(),
                   "block_geometry": stats["level_losses"].detach().tolist()}
            optimizer.step()
            if not torch.isfinite(z).all():
                raise ValueError("nonfinite shaped catalog rows")
            self.history.append(row)
        return z.detach().numpy().copy()

    def refit_pq(self):
        """Fit only a fresh PQ on bound Z, never RQ or the OPQ rotation.

        Explicit ``training.pq_refit_only=True`` produces the matched control
        without optimizing Z or sampling geometry. This helper only operates
        after bound rows were prepared; it never turns a disabled fit into an
        implicit refit.
        """
        if not self.catalog_bound or self.learned_z is None:
            raise ValueError("PQ refit requires an explicitly bound geometry/refit-only fit")
        self._check_frozen()
        faiss = require_faiss()
        _check_faiss_magnitude(self.learned_z, self.model.dimension, "PQ refit training", len(self.learned_z))
        child = faiss.IndexPQ(self.model.dimension, self.model.opq_subspaces, self.model.nbits)
        child.pq.cp.seed = self.model.seed
        child.pq.cp.niter = self.model.pq_niter
        child.pq.cp.max_points_per_centroid = max(256, len(self.learned_z))
        with faiss_threads(faiss, self.model.threads):
            child.train(self.learned_z)
            child = faiss.deserialize_index(faiss.serialize_index(child))
        _validate_pq(child, self.model)
        # Attribute has a distinct name from this public method.
        self._refitted_pq = child
        suffix = _pq_codes(child, self.learned_z, self.model)
        self.final_codes = np.column_stack((self.prefix_codes, suffix)).astype(np.int32)
        self._check_frozen()
        return self

    def _matching_rows(self, data, item_ids):
        if self.learned_z is None or self.final_codes is None or self._refitted_pq is None:
            raise ValueError("complete a catalog-bound fit before encoding")
        self._check_frozen()
        data = vectors(data, self.model.dimension)
        ids = canonical_item_ids(item_ids, len(data))
        if any(value not in self._rows for value in ids):
            raise ValueError("transductive RQ-OPQ cannot encode unknown catalog IDs")
        rows = np.array([self._rows[value] for value in ids], dtype=np.int64)
        if not np.array_equal(_row_hashes(data), self.row_hashes[rows]):
            raise ValueError("catalog ID embedding differs from its fitted FP32 row fingerprint")
        actual_prefix, _ = self.model.rq_residuals(data)
        if not np.array_equal(actual_prefix, self.prefix_codes[rows]):
            raise ValueError("saved prefix does not agree with frozen RQ on the authenticated rows")
        return data, ids, rows

    def encode(self, data, item_ids):
        if not self.catalog_bound:
            canonical_item_ids(item_ids, len(data))
            return self.model.encode(data)
        _, _, rows = self._matching_rows(data, item_ids)
        return self.final_codes[rows].copy()

    def diagnose(self, data, item_ids, diagnostic_seed=314159):
        import torch
        from torch.nn import functional as F
        from ..core import GeoReg, geometry_blocks_loss

        data = vectors(data, self.model.dimension)
        ids = canonical_item_ids(item_ids, len(data))
        if len(data) < 2:
            raise ValueError("geometry diagnostics require at least two catalog rows")
        if self.catalog_bound:
            _, _, rows = self._matching_rows(data, ids)
            z, prefix, codes = self.learned_z[rows], self.prefix_codes[rows], self.final_codes[rows]
            pq = self._refitted_pq
        else:
            prefix, residual = self.model.rq_residuals(data)
            z = _rotate_stably(residual, _rotation(self.model))
            codes = self.model.encode(data)
            pq = require_faiss().downcast_index(self.model.opq_index.index)
        matrix = _rotation(self.model)
        prefix_sum = _prefix_sum(self.model, prefix)
        _, original_residual = self.model.rq_residuals(data)
        original_z = _rotate_stably(original_residual, matrix)
        config = {key: self.geometry[key] for key in _GEOMETRY if key not in ("mode", "weight", "seed")}
        georeg = GeoReg(**config, seed=positive_int(diagnostic_seed, "diagnostic_seed", minimum=0))
        width = self.model.dimension // self.model.opq_subspaces
        directions = georeg.sample_directions(width)
        blocks = {f"opq_subspace_{j}": torch.from_numpy(z[:, j * width:(j + 1) * width].copy())
                  for j in range(self.model.opq_subspaces)}
        with torch.no_grad():
            cf, stats = geometry_blocks_loss(blocks, georeg, directions=directions, return_stats=True)
            shaped_full = torch.from_numpy(prefix_sum).double() + torch.from_numpy(z.copy()).double() @ torch.from_numpy(matrix).double()
            target = torch.from_numpy(data).double()
            semantic_mse = (shaped_full - target).square().mean().item()
            cosine_rows = 1 - F.cosine_similarity(shaped_full, target, dim=1, eps=1e-8)
            semantic_cos = cosine_rows.mean().item()
            nonzero = torch.linalg.vector_norm(target, dim=1) > 1e-8
            nonzero_cos = cosine_rows[nonzero].mean().item() if nonzero.any() else None
        with faiss_threads(require_faiss(), self.model.threads):
            # Decode the very assignments returned by encode(), without a
            # different query batching causing a fresh assignment decision.
            packed = _pack_codes(codes[:, self.model.rq_levels:], self.model.nbits)
            decoded = pq.sa_decode(packed)
        reconstructed = prefix_sum.astype(np.float64) + _rotate_stably(decoded, matrix.T).astype(np.float64)
        _, _, statistics = build_indexes(ids, codes, self.capacities, "allow")
        return {"backend": _ALGORITHM, "catalog_bound": self.catalog_bound, "items": len(data),
                "geometry_mean": cf.item(), "mean_block_cf": cf.item(), "block_names": list(stats["names"]),
                "block_cf": stats["level_losses"].tolist(), "block_rms": stats["residual_rms"].tolist(),
                "semantic_mse": semantic_mse, "semantic_cosine_loss": semantic_cos,
                "nonzero_semantic_cosine_loss": nonzero_cos,
                "zero_target_count": int((target == 0).all(dim=1).sum()),
                "near_zero_target_count": int((~nonzero).sum()), "nonzero_target_norm_threshold": 1e-8,
                "tail_drift_mse": float(np.mean((z.astype(np.float64) - original_z.astype(np.float64)) ** 2)),
                "shaped_quantization_mse": float(np.mean((z.astype(np.float64) - decoded.astype(np.float64)) ** 2)),
                "reconstruction_mse": float(np.mean((reconstructed - data.astype(np.float64)) ** 2)),
                "quantized_reconstruction_mse": float(np.mean((reconstructed - data.astype(np.float64)) ** 2)),
                "diagnostic_seed": diagnostic_seed, "code_statistics": statistics,
                "adaptation": "frozen RQ prefix/rotation; optional free catalog Z; extra final PQ refit",
                "cosine_convention": "full reconstructed embedding; cosine_similarity eps=1e-8; zero-target cosine=0"}

    def save(self, path, overwrite=False):
        if self.model.opq_index is None:
            raise ValueError("cannot save an unfitted backend")
        base_metadata, base_arrays = _base_payload(self.model)
        arrays = {"base_" + key: value for key, value in base_arrays.items()}
        count = 0
        catalog_hash = None
        if self.catalog_bound:
            if self.learned_z is None or self.final_codes is None or self._refitted_pq is None:
                raise ValueError("cannot save an incomplete catalog-bound fit")
            self._check_frozen()
            count = len(self.catalog_ids)
            catalog_hash = _catalog_hash(self.catalog_ids, self.row_hashes)
            arrays.update(catalog_ids=np.array(self.catalog_ids, dtype=np.str_),
                          row_hashes=self.row_hashes, prefix_codes=self.prefix_codes,
                          learned_z=self.learned_z, final_codes=self.final_codes,
                          refit_pq_index=require_faiss().serialize_index(self._refitted_pq))
        metadata = {"algorithm": _ALGORITHM, "encoding_revision": self.encoding_revision,
                    "configuration": self.configuration(), "base_metadata": base_metadata,
                    "catalog_bound": self.catalog_bound, "catalog_count": count,
                    "catalog_sha256": catalog_hash, "history": self.history,
                    "array_sha256": {key: _array_hash(value) for key, value in arrays.items()},
                    "persistence_scope": "final inference artifact; no optimizer-resume state"}
        save_npz(path, metadata, arrays, overwrite)

    @classmethod
    def load(cls, path):
        metadata, arrays = load_npz(path, _ALGORITHM)
        expected = {"algorithm", "format_version", "encoding_revision", "configuration", "base_metadata",
                    "catalog_bound", "catalog_count", "catalog_sha256", "history", "array_sha256", "persistence_scope"}
        if set(metadata) != expected or metadata["encoding_revision"] != cls.encoding_revision:
            raise ValueError("unsupported or inconsistent adapter metadata")
        if metadata["persistence_scope"] != "final inference artifact; no optimizer-resume state":
            raise ValueError("unsupported persistence scope")
        config = metadata["configuration"]
        if not isinstance(config, dict) or set(config) != {"model", "geometry", "training", "seed"}:
            raise ValueError("invalid adapter configuration")
        backend = cls(**config)
        if metadata["catalog_bound"] is not backend.catalog_bound:
            raise ValueError("catalog binding metadata does not match configuration")
        baseline_names = {f"base_rq_codebook_{i}" for i in range(backend.model.rq_levels)} | {"base_opq_index"}
        expected_arrays = baseline_names | (_ARRAY_NAMES if backend.catalog_bound else set())
        if set(arrays) != expected_arrays or set(metadata["array_sha256"]) != expected_arrays:
            raise ValueError("unexpected or missing adapter arrays")
        if any(_array_hash(value) != metadata["array_sha256"][key] for key, value in arrays.items()):
            raise ValueError("adapter array checksum mismatch")
        base_arrays = {key.removeprefix("base_"): arrays[key] for key in baseline_names}
        backend.model = _base_from_payload(metadata["base_metadata"], base_arrays)
        if backend.model.configuration() != config["model"]:
            raise ValueError("frozen baseline configuration mismatch")
        history = metadata["history"]
        if not isinstance(history, list) or len(history) != (backend.training["steps"] if backend.geo_enabled else 0):
            raise ValueError("shaping history length does not match configuration")
        expected_history = {"step", "loss", "semantic_mse", "semantic_cosine_loss", "geometry", "weighted_geometry", "gradient_norm", "block_geometry"}
        for step, record in enumerate(history, 1):
            if not isinstance(record, dict) or set(record) != expected_history or record["step"] != step:
                raise ValueError("invalid shaping history record")
            if not isinstance(record["block_geometry"], list):
                raise ValueError("block geometry history must be a list")
            values = [v for k, v in record.items() if k != "block_geometry"] + record["block_geometry"]
            if (len(record["block_geometry"]) != backend.model.opq_subspaces
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                raise ValueError("nonfinite or malformed shaping history")
        backend.history = history
        if not backend.catalog_bound:
            if metadata["catalog_count"] != 0 or metadata["catalog_sha256"] is not None:
                raise ValueError("disabled adapter must not contain catalog binding")
            return backend
        count = positive_int(metadata["catalog_count"], "catalog_count")
        if count < (1 << backend.model.nbits):
            raise ValueError("bound catalog is smaller than its PQ centroid count")
        if arrays["catalog_ids"].ndim != 1 or arrays["catalog_ids"].dtype.kind != "U":
            raise ValueError("catalog IDs must be a one-dimensional unicode array")
        ids = canonical_item_ids(arrays["catalog_ids"].tolist(), count)
        hashes = arrays["row_hashes"]
        if hashes.dtype != np.uint8 or hashes.shape != (count, 32):
            raise ValueError("invalid catalog row fingerprints")
        if _catalog_hash(ids, hashes) != metadata["catalog_sha256"]:
            raise ValueError("catalog identity checksum mismatch")
        z = arrays["learned_z"]
        if z.dtype != np.float32 or z.shape != (count, backend.model.dimension) or not np.isfinite(z).all():
            raise ValueError("invalid shaped catalog rows")
        prefix, codes = arrays["prefix_codes"], arrays["final_codes"]
        if (prefix.dtype != np.int32 or prefix.shape != (count, backend.model.rq_levels)
                or codes.dtype != np.int32 or codes.shape != (count, len(backend.capacities))):
            raise ValueError("invalid frozen prefix/final code shape")
        for level, capacity in enumerate(backend.capacities):
            if np.any(codes[:, level] < 0) or np.any(codes[:, level] >= capacity):
                raise ValueError("catalog codes outside their codebook capacities")
        if not np.array_equal(prefix, codes[:, :backend.model.rq_levels]):
            raise ValueError("final codes changed the frozen RQ prefix")
        blob = arrays["refit_pq_index"]
        if blob.dtype != np.uint8 or blob.ndim != 1:
            raise ValueError("invalid refitted PQ byte stream")
        try:
            pq = require_faiss().deserialize_index(blob)
        except RuntimeError as exc:
            raise ValueError("invalid refitted PQ serialization") from exc
        _validate_pq(pq, backend.model)
        if not np.array_equal(_pq_codes(pq, z, backend.model), codes[:, backend.model.rq_levels:]):
            raise ValueError("stored PQ codes do not reproduce from persisted Z and PQ")
        backend.catalog_ids, backend.row_hashes = ids, hashes.copy()
        backend.prefix_codes, backend.learned_z, backend.final_codes = prefix.copy(), z.copy(), codes.copy()
        backend._rows = {value: index for index, value in enumerate(ids)}
        backend._refitted_pq = pq
        backend._freeze_fingerprints()
        return backend
