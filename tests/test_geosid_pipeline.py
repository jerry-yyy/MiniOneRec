"""Real CPU training, inference-artifact and catalog/CLI contract tests."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from rq.models.rqvae import RQVAE
from sid_generation.geosid.__main__ import parser, run
from sid_generation.geosid.rqvae_backend import RQVAEBackend, seeded
from sid_generation.sid_common import build_indexes, load_npz, save_npz


MODEL = {"e_dim": 4, "layers": [12], "num_emb_list": [4, 4, 4], "dropout_prob": 0.2}
TRAINING = {"epochs": 2, "batch_size": 8, "lr": 0.001}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.x = np.random.default_rng(12).normal(size=(24, 6)).astype("float32")
        self.x[-1] = self.x[0]
        self.ids = [f"item-{k}" for k in range(len(self.x))]

    def backend(self, weight=0.0):
        return RQVAEBackend(model=MODEL, training=TRAINING,
                            geometry={"mode": "residual", "weight": weight}, seed=71)

    def test_disabled_training_matches_original_model_and_optimizer(self):
        actual = self.backend().fit(self.x, self.ids)
        with seeded(71, torch.device("cpu")):
            options = dict(actual.model_options, sk_epsilons=[0.0] * 3)
            original = RQVAE(in_dim=6, **options)
            optimizer = torch.optim.AdamW(original.parameters(), lr=0.001, weight_decay=0)
            loader = DataLoader(torch.from_numpy(self.x), batch_size=8, shuffle=True,
                                generator=torch.Generator().manual_seed(71), num_workers=0)
            for _ in range(2):
                original.train()
                for x in loader:
                    optimizer.zero_grad(set_to_none=True)
                    out, q, _ = original(x)
                    base, _ = original.compute_loss(out, q, x)
                    base.backward()
                    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, actual.model.state_dict()[key]), key)

    def test_active_fit_save_load_reorder_diagnose(self):
        model = self.backend(0.03).fit(self.x, self.ids)
        codes = model.encode(self.x, self.ids)
        np.testing.assert_array_equal(codes[0], codes[-1])
        path = self.root / "model.npz"
        model.save(path)
        loaded = RQVAEBackend.load(path)
        np.testing.assert_array_equal(codes[::-1], loaded.encode(self.x[::-1], self.ids[::-1]))
        before = model.model.georeg.get_rng_state()
        diagnostic = model.diagnose(self.x, self.ids)
        self.assertEqual(diagnostic, loaded.diagnose(self.x, self.ids))
        self.assertTrue(torch.equal(before, model.model.georeg.get_rng_state()))
        self.assertEqual(len(diagnostic["geometry_per_block"]), 3)
        self.assertTrue(np.isfinite(diagnostic["reconstruction_mse"]))
        self.assertEqual(len(model.history), 2)

    def test_kmeans_artifact_is_inference_not_reinitialization(self):
        model = RQVAEBackend(model=dict(MODEL, kmeans_init=True, kmeans_iters=2),
                             training=TRAINING).fit(self.x, self.ids)
        path = self.root / "model.npz"
        model.save(path)
        loaded = RQVAEBackend.load(path)
        self.assertTrue(all(q.initted for q in loaded.model.rq.vq_layers))
        np.testing.assert_array_equal(model.encode(self.x, self.ids), loaded.encode(self.x, self.ids))

    def test_collisions_suffixes_and_permutation_use_shared_upstream_rules(self):
        codes = np.zeros((4, 3), dtype=np.int32)
        codes[-1] = 1
        ids = ["z", "a", "k", "single"]
        raw, final, stats = build_indexes(ids, codes, [4, 4, 4])
        self.assertEqual(final["a"][-1], "<d_0>")
        self.assertEqual(final["single"][-1], "<d_0>")
        self.assertEqual(stats["raw"]["excess_collision_count"], 2)
        self.assertEqual(stats["final"]["unique_sids"], 4)
        self.assertEqual(final, build_indexes(ids[::-1], codes[::-1], [4, 4, 4])[1])
        self.assertEqual(len(raw), len(final))

    def test_bad_ids_and_nonfinite_input_rejected(self):
        for data, ids in ((self.x, ["same"] * 24), (self.x, self.ids[:-1]),
                          (self.x * np.nan, self.ids), (self.x.astype(complex), self.ids)):
            with self.subTest(ids=ids[:1]), self.assertRaises(ValueError):
                self.backend().fit(data, ids)

    def test_singleton_tail_rejected_without_silent_drop(self):
        with self.assertRaisesRegex(ValueError, "every batch"):
            self.backend(0.03).fit(self.x[:17], self.ids[:17])

    def test_checkpoint_weight_tampering_rejected(self):
        model = self.backend().fit(self.x, self.ids)
        original, bad = self.root / "model.npz", self.root / "bad.npz"
        model.save(original)
        metadata, arrays = load_npz(original, "geosid_rqvae")
        first = next(iter(arrays))
        arrays[first].flat[0] = np.nan
        save_npz(bad, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "invalid weight"):
            RQVAEBackend.load(bad)
        with self.assertRaises(FileExistsError):
            model.save(original)

    def test_options_device_and_unknown_backend_rejected(self):
        for options in ({"training": {"device": "meta"}}, {"model": {"unknown": 1}},
                        {"training": {"epochs": True}}, {"training": {"lr": float("nan")}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RQVAEBackend(**options)
        from sid_generation.geosid import backend_class
        with self.assertRaises(ValueError):
            backend_class("flash")

    def test_cpu_seed_context_never_seeds_cuda_and_restores_cpu(self):
        before = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA RNG touched")):
            self.backend().fit(self.x, self.ids)
            path = self.root / "model.npz"
            self.backend().fit(self.x, self.ids).save(path)
            RQVAEBackend.load(path)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_architecture_rejected_before_silent_zip_truncation(self):
        for options in ({"sk_epsilons": [0.0]}, {"sk_epsilons": []},
                        {"num_emb_list": []}, {"num_emb_list": [True, 4]},
                        {"e_dim": 0}, {"layers": [3.5]}, {"beta": float("nan")},
                        {"sk_epsilons": [0, float("inf"), 0]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RQVAEBackend(model=dict(MODEL, **options))

    def test_nonfinite_inference_arithmetic_and_save_fail_closed(self):
        model = self.backend().fit(self.x, self.ids)
        with torch.no_grad():
            first = next(model.model.encoder.parameters())
            first.fill_(torch.finfo(first.dtype).max)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model.encode(self.x, self.ids)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model.diagnose(self.x, self.ids)
        with torch.no_grad():
            first.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model.save(self.root / "invalid.npz")

    def test_finite_nearby_huge_vectors_cannot_hide_distance_overflow(self):
        model = RQVAEBackend(model=dict(MODEL, num_emb_list=[2]), training=TRAINING).fit(self.x, self.ids)
        z = torch.full((24, 4), 1e20)
        with torch.no_grad():
            model.model.rq.vq_layers[0].embedding.weight[0].fill_(1e20 + 1e13)
            model.model.rq.vq_layers[0].embedding.weight[1].fill_(1e20)
        with patch.object(model.model.encoder, "forward", return_value=z):
            with self.assertRaisesRegex(ValueError, "distances"):
                model._quantize(torch.from_numpy(self.x))

    def cli(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(parser().parse_args(arguments))

    def test_cli_train_export_diagnose_full_catalog(self):
        np.save(self.root / "x.npy", self.x)
        (self.root / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({"model": MODEL, "training": TRAINING,
            "geometry": {"mode": "residual", "weight": 0.03}}), encoding="utf-8")
        common = ["--backend", "rqvae", "--data-path", str(self.root / "x.npy"),
                  "--item-ids", str(self.root / "ids.json")]
        self.cli(["train", *common, "--config", str(self.root / "config.json"),
                  "--output-dir", str(self.root / "train"), "--dataset-name", "toy"])
        model = self.root / "train/toy.tokenizer.npz"
        self.cli(["export", *common, "--model-path", str(model),
                  "--output-dir", str(self.root / "export"), "--dataset-name", "toy"])
        for name in ("index.json", "raw.index.json", "codes.npy", "item_ids.json"):
            self.assertEqual((self.root / f"train/toy.{name}").read_bytes(),
                             (self.root / f"export/toy.{name}").read_bytes())
        index = json.loads((self.root / "train/toy.index.json").read_text())
        self.assertEqual(set(index), set(self.ids))
        self.assertEqual(len({tuple(v) for v in index.values()}), len(self.ids))
        result = self.cli(["diagnose", *common, "--model-path", str(model),
                           "--report", str(self.root / "diagnostics.json")])
        self.assertEqual(result["sid_statistics"]["final"]["excess_collision_count"], 0)
        with self.assertRaises(FileExistsError):
            self.cli(["export", *common, "--model-path", str(model),
                      "--output-dir", str(self.root / "export"), "--dataset-name", "toy"])


@unittest.skipUnless(os.environ.get("GEOSID_TEST_CUDA") == "1", "CUDA tests require budgeted opt-in")
class CudaPipelineTests(unittest.TestCase):
    def test_cuda_training_inference_roundtrip_and_rng_isolation(self):
        self.assertTrue(torch.cuda.is_available())
        torch.set_num_threads(1)
        data = np.random.default_rng(5).normal(size=(16, 6)).astype("float32")
        ids = list(range(16))
        device = torch.device("cuda", torch.cuda.current_device())
        before_cpu = torch.get_rng_state().clone()
        before_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model = RQVAEBackend(model=MODEL, training=dict(TRAINING, device=str(device), epochs=1),
                             geometry={"mode": "residual", "weight": 0.03}).fit(data, ids)
        codes = model.encode(data, ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.npz"
            model.save(path)
            loaded = RQVAEBackend.load(path)
            loaded.device = device
            loaded.model.to(device)
            np.testing.assert_array_equal(codes, loaded.encode(data, ids))
        self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
        for before, after in zip(before_cuda, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
