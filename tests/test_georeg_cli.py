"""Small CPU integration test for the public train/export/diagnose entry point."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sid_generation.geosid import main


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = Path(__file__).resolve().parents[1]

    def invoke(self, args):
        # Test real command implementations while keeping tqdm/model dumps out
        # of the test log. No subprocess or downloaded model is required.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def test_subcommand_help_is_dispatched(self):
        for command, expected in (("train", "--geo_mode"), ("export", "--checkpoint"),
                                  ("diagnose", "--diagnostic-seed")):
            with self.subTest(command=command):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exited:
                    main([command, "--help"])
                self.assertEqual(exited.exception.code, 0)
                self.assertIn(expected, output.getvalue())

    def test_synthetic_train_export_diagnose_and_reordered_item_alignment(self):
        data = np.random.default_rng(85).normal(size=(48, 12)).astype(np.float32)
        data[1] = data[0]  # Force a raw collision and exercise local-rank suffixes.
        embedding_path = self.root / "embeddings.npy"
        ids_path = self.root / "ids.json"
        ids = [f"item-{i + 10}" for i in range(len(data))]
        np.save(embedding_path, data)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        self.invoke(["train", "--config", str(self.repo / "configs/geosid/smoke.json"),
                     "--data_path", str(embedding_path), "--ckpt_dir", str(self.root / "checkpoints"),
                     "--geo_weight", "0.03"])
        checkpoints = list((self.root / "checkpoints").rglob("last_epoch.pth"))
        self.assertEqual(len(checkpoints), 1)
        checkpoint = checkpoints[0]
        state = torch.load(checkpoint, weights_only=False)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["global_step"], 6)
        self.assertEqual(state["geo_config"]["geo_mode"], "residual")
        self.assertEqual(state["geo_config"]["geo_weight"], .03)
        self.assertTrue(all(state["vq_initted"]))
        self.assertEqual(state["loader_config"]["num_workers"], 0)
        for action in ("export", "diagnose"):
            self.invoke([action, "--checkpoint", str(checkpoint), "--embeddings", str(embedding_path),
                         "--item-ids", str(ids_path), "--output-dir", str(self.root / action)])
        export = self.root / "export"
        raw = json.loads((export / "raw.index.json").read_text())
        final = json.loads((export / "index.json").read_text())
        metrics = json.loads((export / "metrics.json").read_text())
        self.assertEqual(set(final), set(ids))
        self.assertTrue(all(len(tokens) == 3 for tokens in raw.values()))
        self.assertTrue(all(len(tokens) == 4 for tokens in final.values()))
        self.assertEqual(len({tuple(tokens) for tokens in final.values()}), len(ids))
        self.assertEqual(raw[ids[0]], raw[ids[1]])
        self.assertNotEqual(final[ids[0]][-1], final[ids[1]][-1])
        self.assertEqual(metrics["final_collision"]["duplicate_excess"], 0)
        self.assertGreater(metrics["raw_collision"]["duplicate_excess"], 0)
        self.assertEqual(len(metrics["geometry"]), 3)
        self.assertTrue(np.isfinite(metrics["reconstruction_mse"]))
        self.assertEqual(metrics, json.loads((self.root / "diagnose/metrics.json").read_text()))
        self.assertEqual(np.load(export / "raw_codes.npy").shape, (len(ids), 3))

        permutation = np.random.default_rng(4).permutation(len(ids))
        np.save(embedding_path, data[permutation])
        ids_path.write_text(json.dumps([ids[i] for i in permutation]), encoding="utf-8")
        self.invoke(["export", "--checkpoint", str(checkpoint), "--embeddings", str(embedding_path),
                     "--item-ids", str(ids_path), "--output-dir", str(self.root / "reordered")])
        self.assertEqual(final, json.loads((self.root / "reordered/index.json").read_text()))


if __name__ == "__main__":
    unittest.main()
