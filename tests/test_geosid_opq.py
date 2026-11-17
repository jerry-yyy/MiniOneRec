"""CPU contracts for the explicitly transductive, experimental hybrid adapter."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sid_generation.rq_opq import RQOPQ, _rotate_stably, require_faiss, unpack_pq_codes
from sid_generation.sid_common import load_npz, save_npz
from sid_generation.geosid.adapters.rq_opq import (
    RQOPQBackend, _ALGORITHM, _array_hash, _pack_codes, _rotation,
)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "optional CPU FAISS is required")
class OPQBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.data = np.random.default_rng(19).normal(size=(256, 8)).astype(np.float32)
        cls.ids = [f"catalog-{i}" for i in range(len(cls.data))]
        cls.model_config = dict(rq_levels=2, codebook_size=4, opq_subspaces=2, nbits=2,
                                rq_niter=3, opq_niter=2, pq_niter=3,
                                batch_size=37, seed=2024, threads=1)
        cls.baseline = RQOPQ(**cls.model_config).fit(cls.data)
        cls.base_codes = cls.baseline.encode(cls.data)
        cls.geometry = dict(mode="block", weight=.5, nproj=8, num_frequencies=5,
                            projection_chunk=3, seed=123)
        cls.training = dict(steps=3, lr=.01, semantic_weight=2., cosine_weight=.3)
        cls.shaped = RQOPQBackend(cls.baseline, cls.geometry, cls.training).fit(cls.data, cls.ids)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_disabled_fresh_fit_matches_original_and_skips_geometry_and_refit(self):
        before = torch.get_rng_state().clone()
        for geometry in (dict(mode="none", weight=7), dict(mode="block", weight=0)):
            with self.subTest(geometry=geometry), patch(
                    "sid_generation.geosid.core.GeoReg", side_effect=AssertionError("geometry constructed")), patch.object(
                    RQOPQBackend, "refit_pq", side_effect=AssertionError("refit called")):
                backend = RQOPQBackend(self.model_config, geometry).fit(self.data, self.ids)
                np.testing.assert_array_equal(backend.encode(self.data, self.ids), self.base_codes)
                self.assertEqual(backend.history, [])
                self.assertFalse(backend.catalog_bound)
                np.testing.assert_array_equal(backend.encode(self.data[:3], ["new-a", "new-b", "new-c"]), self.baseline.encode(self.data[:3]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_fitted_baseline_is_shared_by_value_without_refitting_or_mutation(self):
        original_blob = require_faiss().serialize_index(self.baseline.opq_index).copy()
        with patch.object(RQOPQ, "fit", side_effect=AssertionError("baseline refitted")):
            control = RQOPQBackend(self.baseline, training=dict(pq_refit_only=True)).fit(self.data, self.ids)
        np.testing.assert_array_equal(original_blob, require_faiss().serialize_index(self.baseline.opq_index))
        np.testing.assert_array_equal(_rotation(control.model), _rotation(self.shaped.model))
        for a, b in zip(control.model.rq_codebooks, self.shaped.model.rq_codebooks):
            np.testing.assert_array_equal(a, b)
        self.assertIsNot(control.model, self.baseline)

    def test_ordinary_repeated_fit_retrains_like_original(self):
        backend = RQOPQBackend(self.model_config).fit(self.data, self.ids)
        changed = self.data * .7 + np.arange(8, dtype=np.float32)
        expected = RQOPQ(**self.model_config).fit(changed).encode(changed)
        backend.fit(changed, self.ids)
        np.testing.assert_array_equal(backend.encode(changed, self.ids), expected)

    def test_shaping_changes_only_free_rows_then_refits_pq(self):
        _, residual = self.baseline.rq_residuals(self.data)
        initial = _rotate_stably(residual, _rotation(self.baseline))
        self.assertFalse(np.array_equal(initial, self.shaped.learned_z))
        np.testing.assert_array_equal(self.shaped.prefix_codes, self.base_codes[:, :2])
        np.testing.assert_array_equal(self.shaped.encode(self.data, self.ids)[:, :2], self.base_codes[:, :2])
        np.testing.assert_array_equal(_rotation(self.shaped.model), _rotation(self.baseline))
        self.assertEqual(len(self.shaped.history), 3)
        self.assertTrue(all(row["gradient_norm"] > 0 and np.isfinite(row["loss"]) for row in self.shaped.history))

    def test_refit_only_explicit_control_preserves_z_and_skips_geometry(self):
        with patch.object(RQOPQBackend, "_shape", side_effect=AssertionError("shaping called")), patch(
                "sid_generation.geosid.core.GeoReg", side_effect=AssertionError("geometry constructed")):
            backend = RQOPQBackend(self.baseline, training=dict(pq_refit_only=True)).fit(self.data, self.ids)
        _, residual = self.baseline.rq_residuals(self.data)
        np.testing.assert_array_equal(backend.learned_z, _rotate_stably(residual, _rotation(self.baseline)))
        self.assertEqual(backend.history, [])
        self.assertEqual(backend.model.seed, self.shaped.model.seed)
        self.assertEqual(backend.model.pq_niter, self.shaped.model.pq_niter)
        with self.assertRaises(ValueError):
            RQOPQBackend(self.baseline, self.geometry, dict(pq_refit_only=True))

    def test_catalog_reorder_subset_and_unknown_or_changed_rows(self):
        order = np.array([71, 9, 200, 2])
        np.testing.assert_array_equal(self.shaped.encode(self.data[order], [self.ids[i] for i in order]), self.shaped.final_codes[order])
        altered = self.data[order].copy()
        altered[0, 0] = np.nextafter(altered[0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.shaped.encode(altered, [self.ids[i] for i in order])
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.shaped.encode(self.data[:1], ["unseen"])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.shaped.encode(self.data[:2], [self.ids[0], self.ids[0]])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.shaped.encode(self.data[:2], self.ids[1::-1])

    def test_enabled_encoding_does_not_call_original_encode(self):
        with patch.object(RQOPQ, "encode", side_effect=AssertionError("silent out-of-sample original path")):
            np.testing.assert_array_equal(self.shaped.encode(self.data[:3], self.ids[:3]), self.shaped.final_codes[:3])

    def test_inverse_rotation_and_full_embedding_semantic_loss(self):
        diagnostic = self.shaped.diagnose(self.data, self.ids)
        prefix = np.zeros_like(self.data, dtype=np.float64)
        for level, codebook in enumerate(self.shaped.model.rq_codebooks):
            prefix += codebook[self.shaped.prefix_codes[:, level]]
        # Diagnostics accumulate semantic errors in FP64, independently of
        # the FP32 training objective; retain the declared prefix FP32 rounding.
        reconstructed = torch.from_numpy(prefix.astype(np.float32)).double() + torch.from_numpy(self.shaped.learned_z).double() @ torch.from_numpy(_rotation(self.shaped.model)).double()
        target = torch.from_numpy(self.data).double()
        self.assertAlmostEqual(diagnostic["semantic_mse"], (reconstructed - target).square().mean().item(), places=10)
        expected_cos = (1 - torch.nn.functional.cosine_similarity(reconstructed, target, dim=1, eps=1e-8)).mean().item()
        self.assertAlmostEqual(diagnostic["semantic_cosine_loss"], expected_cos, places=10)
        self.assertGreaterEqual(diagnostic["shaped_quantization_mse"], 0)
        self.assertGreaterEqual(diagnostic["reconstruction_mse"], 0)

    def test_diagnostics_fixed_seed_and_baseline_comparable(self):
        tiny = self.data[:4].copy()
        tiny[0] = 0
        tiny[1] = 1e-10
        tiny_stats = RQOPQBackend(self.baseline).diagnose(tiny, self.ids[:4])
        self.assertEqual(tiny_stats["zero_target_count"], 1)
        self.assertEqual(tiny_stats["near_zero_target_count"], 2)
        self.assertEqual(tiny_stats["nonzero_target_norm_threshold"], 1e-8)
        before = torch.get_rng_state().clone()
        a = self.shaped.diagnose(self.data, self.ids, 314159)
        b = self.shaped.diagnose(self.data, self.ids, 314159)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertAlmostEqual(a["geometry_mean"], sum(a["block_cf"]) / 2, places=7)
        baseline = RQOPQBackend(self.baseline)
        diagnostic = baseline.diagnose(self.data, self.ids)
        self.assertFalse(diagnostic["catalog_bound"])
        self.assertEqual(a.keys(), diagnostic.keys())

    def test_save_load_preserves_codes_z_config_history_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.npz"
            self.shaped.save(path)
            restored = RQOPQBackend.load(path)
            self.assertEqual(restored.configuration(), self.shaped.configuration())
            self.assertEqual(restored.history, self.shaped.history)
            np.testing.assert_array_equal(restored.learned_z, self.shaped.learned_z)
            np.testing.assert_array_equal(restored.encode(self.data, self.ids), self.shaped.final_codes)
            self.assertEqual(restored.diagnose(self.data, self.ids), self.shaped.diagnose(self.data, self.ids))
            with self.assertRaises(FileExistsError):
                self.shaped.save(path)
            self.shaped.save(path, overwrite=True)

    def test_disabled_and_refit_only_roundtrips(self):
        for training in ({}, {"pq_refit_only": True}):
            with self.subTest(training=training), tempfile.TemporaryDirectory() as temp:
                backend = RQOPQBackend(self.baseline, training=training).fit(self.data, self.ids)
                path = Path(temp) / "model.npz"
                backend.save(path)
                restored = RQOPQBackend.load(path)
                np.testing.assert_array_equal(backend.encode(self.data, self.ids), restored.encode(self.data, self.ids))

    def test_load_rejects_shapes_nonfinite_checksums_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            source, broken = Path(temp) / "source.npz", Path(temp) / "broken.npz"
            self.shaped.save(source)
            original_meta, original_arrays = load_npz(source, _ALGORITHM)
            for name in ("nan_z", "bad_shape", "prefix_mismatch", "bad_history", "bad_revision", "unknown_array", "base_config", "pq_codes"):
                with self.subTest(case=name):
                    meta, arrays = copy.deepcopy(original_meta), copy.deepcopy(original_arrays)
                    if name == "nan_z": arrays["learned_z"][0, 0] = np.nan
                    elif name == "bad_shape": arrays["learned_z"] = arrays["learned_z"][:-1]
                    elif name == "prefix_mismatch": arrays["prefix_codes"][0, 0] = (arrays["prefix_codes"][0, 0] + 1) % 4
                    elif name == "bad_history": meta["history"][0]["loss"] = "bad"
                    elif name == "bad_revision": meta["encoding_revision"] = 999
                    elif name == "unknown_array": arrays["extra"] = np.zeros(1)
                    elif name == "base_config": meta["base_metadata"]["configuration"]["threads"] = 2
                    elif name == "pq_codes": arrays["final_codes"][0, -1] = (arrays["final_codes"][0, -1] + 1) % 4
                    meta["array_sha256"] = {k: _array_hash(v) for k, v in arrays.items()}
                    save_npz(broken, meta, arrays, overwrite=True)
                    with self.assertRaises(ValueError): RQOPQBackend.load(broken)
            arrays = copy.deepcopy(original_arrays)
            arrays["learned_z"][0, 0] += .1
            save_npz(broken, original_meta, arrays, overwrite=True)
            with self.assertRaisesRegex(ValueError, "checksum"): RQOPQBackend.load(broken)

    def test_mutated_frozen_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.npz"
            self.shaped.save(path)
            restored = RQOPQBackend.load(path)
            restored.model.rq_codebooks[0][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "modified"):
                restored.encode(self.data[:2], self.ids[:2])

    def test_bit_packing_roundtrip_across_byte_boundaries(self):
        for bits in range(1, 9):
            codes = np.random.default_rng(bits).integers(0, 1 << bits, size=(11, 5), dtype=np.int32)
            np.testing.assert_array_equal(unpack_pq_codes(_pack_codes(codes, bits), 5, bits), codes)

    def test_invalid_options_and_unfitted_models_fail_closed(self):
        for kw in (dict(geometry={"mode": "residual"}), dict(geometry={"weight": float("nan")}),
                   dict(training={"semantic_weight": 0}), dict(training={"device": "cuda"}),
                   dict(training={"steps": True}), dict(training={"unknown": 1})):
            with self.subTest(options=kw), self.assertRaises(ValueError): RQOPQBackend(**kw)
        unfitted = RQOPQBackend(self.model_config, self.geometry)
        with self.assertRaises(ValueError): unfitted.encode(self.data[:1], self.ids[:1])


if __name__ == "__main__":
    unittest.main()
