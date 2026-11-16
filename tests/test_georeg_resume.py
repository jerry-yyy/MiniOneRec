"""Actual Trainer integration and epoch-boundary restart contracts."""

import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from rq.models.rqvae import RQVAE
from rq.trainer import Trainer
from tests.test_georeg_model import assert_rng_equal, model_args, reset_seed, rng_snapshot


class RecordingDataset(Dataset):
    """Num_workers=0 also exercises Python/NumPy restoration through the data path."""

    def __init__(self):
        self.x = torch.randn(27, 12, generator=torch.Generator().manual_seed(61))
        self.accesses = []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        self.accesses.append(index)
        return self.x[index] + 0.001 * (random.random() + np.random.random())


def trainer_args(directory, **changes):
    args = dict(lr=.001, learner="AdamW", lr_scheduler_type="linear", weight_decay=0.,
                epochs=4, warmup_epochs=1, save_limit=2, eval_step=1, device="cpu",
                ckpt_dir=str(directory), save_last=True)
    args.update(changes)
    return SimpleNamespace(**args)


def assert_nested_equal(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            assert_nested_equal(test, left, right)
    else:
        test.assertEqual(a, b)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        reset_seed()

    def test_epoch_restart_restores_order_outputs_updates_and_all_rng(self):
        kwargs = model_args(dropout_prob=.2, kmeans_init=True, kmeans_iters=4,
                            geo_mode="residual", geo_weight=.3, geo_seed=173)
        original = RQVAE(**kwargs)
        data = RecordingDataset()
        loader = DataLoader(data, batch_size=9, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(79))
        trainer = Trainer(trainer_args(self.root / "original"), original, len(loader))
        trainer.train_data = loader
        trainer._train_epoch(loader, 0)
        # Match fit(): its catalog validation also advances the shuffled loader.
        trainer._valid_epoch(loader)
        self.assertTrue(all(q.initted for q in original.rq.vq_layers))
        trainer.best_loss, trainer.best_collision_rate = .7, .2
        path = trainer._save_checkpoint(0, ckpt_file="boundary.pth")
        saved_global_rng = rng_snapshot()
        saved_geo_rng = original.georeg.get_rng_state()
        saved_loader_rng = loader.generator.get_state()
        outputs = []
        handle = original.register_forward_hook(
            lambda module, inputs, out: outputs.append(tuple(x.detach().clone() for x in out[:3])))
        data.accesses.clear()
        expected_losses = trainer._train_epoch(loader, 1)
        handle.remove()
        expected_order = list(data.accesses)
        expected_metrics = copy.deepcopy(trainer.last_epoch_metrics)
        expected_model = copy.deepcopy(original.state_dict())
        expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
        expected_rng = rng_snapshot()
        expected_geo_rng = original.georeg.get_rng_state()
        expected_loader_rng = loader.generator.get_state()

        # Recreate every object with different seeds to prove restoration rather
        # than accidental reuse of the uninterrupted process's mutable state.
        reset_seed(999)
        resumed_model = RQVAE(**kwargs)
        resumed_data = RecordingDataset()
        resumed_loader = DataLoader(resumed_data, batch_size=9, shuffle=True, num_workers=0,
                                    generator=torch.Generator().manual_seed(999))
        resumed = Trainer(trainer_args(self.root / "resumed"), resumed_model, len(resumed_loader))
        summary = resumed.load_checkpoint(path, resumed_loader)
        self.assertTrue(summary["exact_epoch_resume"])
        self.assertEqual(summary["start_epoch"], 1)
        self.assertEqual(summary["global_step"], 3)
        self.assertEqual(resumed.best_loss, .7)
        self.assertEqual(resumed.best_collision_rate, .2)
        assert_rng_equal(self, saved_global_rng, rng_snapshot())
        self.assertTrue(torch.equal(saved_geo_rng, resumed_model.georeg.get_rng_state()))
        self.assertTrue(torch.equal(saved_loader_rng, resumed_loader.generator.get_state()))
        resumed_outputs = []
        handle = resumed_model.register_forward_hook(
            lambda module, inputs, out: resumed_outputs.append(tuple(x.detach().clone() for x in out[:3])))
        with patch("rq.models.vq.VectorQuantizer.init_emb", side_effect=AssertionError("KMeans restarted")):
            actual_losses = resumed._train_epoch(resumed_loader, resumed.start_epoch)
        handle.remove()
        self.assertEqual(expected_order, resumed_data.accesses)
        self.assertEqual(expected_losses, actual_losses)
        self.assertEqual(expected_metrics, resumed.last_epoch_metrics)
        assert_nested_equal(self, outputs, resumed_outputs)
        assert_nested_equal(self, expected_model, resumed_model.state_dict())
        assert_nested_equal(self, expected_optimizer, resumed.optimizer.state_dict())
        assert_nested_equal(self, expected_scheduler, resumed.scheduler.state_dict())
        assert_rng_equal(self, expected_rng, rng_snapshot())
        self.assertTrue(torch.equal(expected_geo_rng, resumed_model.georeg.get_rng_state()))
        self.assertTrue(torch.equal(expected_loader_rng, resumed_loader.generator.get_state()))
        self.assertEqual(resumed.global_step, 6)

    def test_old_checkpoint_disables_geometry_and_marks_continuation_nonexact(self):
        kwargs = model_args(kmeans_init=False)
        model = RQVAE(**kwargs)
        trainer = Trainer(trainer_args(self.root / "legacy"), model, 1)
        legacy = {"state_dict": copy.deepcopy(model.state_dict()), "optimizer": trainer.optimizer.state_dict(),
                  "epoch": 7, "args": trainer.args}
        path = self.root / "legacy.pth"
        torch.save(legacy, path)
        loaded_model = RQVAE(**model_args(kmeans_init=True, geo_mode="residual", geo_weight=.5))
        loaded = Trainer(trainer_args(self.root / "loaded"), loaded_model, 1)
        with self.assertLogs(level="WARNING"):
            result = loaded.load_checkpoint(path)
        self.assertFalse(result["exact_epoch_resume"])
        self.assertEqual(result["start_epoch"], 8)
        self.assertFalse(loaded_model.geo_enabled)
        self.assertTrue(all(q.initted for q in loaded_model.rq.vq_layers))
        x = torch.randn(19, 12)
        model.eval()
        loaded_model.eval()
        with patch("rq.models.vq.VectorQuantizer.init_emb", side_effect=AssertionError("KMeans restarted")):
            assert_nested_equal(self, model(x), loaded_model(x))
        loaded_model.train()
        with patch("rq.models.vq.VectorQuantizer.init_emb", side_effect=AssertionError("KMeans restarted")):
            loaded_model(x)

    def test_config_mismatch_and_missing_required_generator_fail(self):
        model = RQVAE(**model_args(geo_mode="residual", geo_weight=.2))
        trainer = Trainer(trainer_args(self.root / "saved"), model, 3)
        loader = DataLoader(RecordingDataset(), batch_size=9, shuffle=True,
                            generator=torch.Generator().manual_seed(41))
        trainer.train_data = loader
        path = trainer._save_checkpoint(-1, ckpt_file="initial.pth")
        mismatched = Trainer(trainer_args(self.root / "mismatch"),
                             RQVAE(**model_args(geo_mode="global", geo_weight=.2)), 3)
        before = copy.deepcopy(mismatched.model.state_dict())
        with self.assertRaisesRegex(ValueError, "configuration"):
            mismatched.load_checkpoint(path, loader)
        assert_nested_equal(self, before, mismatched.model.state_dict())
        model2 = RQVAE(**model_args(geo_mode="residual", geo_weight=.2))
        resumed = Trainer(trainer_args(self.root / "generator"), model2, 3)
        with self.assertRaisesRegex(ValueError, "generator"):
            resumed.load_checkpoint(path, DataLoader(RecordingDataset(), batch_size=9, shuffle=True))

    def test_incomplete_epoch_checkpoint_is_not_silently_called_exact(self):
        model = RQVAE(**model_args(geo_mode="residual", geo_weight=.2))
        trainer = Trainer(trainer_args(self.root / "saved"), model, 3)
        loader = DataLoader(RecordingDataset(), batch_size=9, shuffle=True,
                            generator=torch.Generator().manual_seed(41))
        trainer.train_data = loader
        source = trainer._save_checkpoint(-1, ckpt_file="initial.pth")
        state = torch.load(source, weights_only=False)
        for key in ("scheduler", "global_step", "vq_initted", "geo_config", "geo_rng",
                    "loader_generator", "sampler_generator", "rng", "loader_config",
                    "loader_sampler_generator_shared"):
            with self.subTest(missing=key):
                incomplete = dict(state)
                del incomplete[key]
                path = self.root / f"missing_{key}.pth"
                torch.save(incomplete, path)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    trainer.load_checkpoint(path, loader)

    def test_loader_generator_topology_and_batch_layout_cannot_silently_change(self):
        kwargs = model_args(geo_mode="residual", geo_weight=.2)
        trainer = Trainer(trainer_args(self.root / "saved"), RQVAE(**kwargs), 3)
        data = RecordingDataset()
        loader = DataLoader(data, batch_size=9, shuffle=True,
                            generator=torch.Generator().manual_seed(41))
        trainer.train_data = loader
        path = trainer._save_checkpoint(-1, ckpt_file="initial.pth")
        separate = DataLoader(data, batch_size=9,
                              sampler=RandomSampler(data, generator=torch.Generator().manual_seed(41)),
                              generator=torch.Generator().manual_seed(41))
        with self.assertRaisesRegex(ValueError, "sharing"):
            trainer.load_checkpoint(path, separate)
        different_batch = DataLoader(data, batch_size=13, shuffle=True,
                                     generator=torch.Generator().manual_seed(41))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            trainer.load_checkpoint(path, different_batch)

    def test_global_default_loader_rng_is_restored_and_cannot_change_to_dedicated(self):
        kwargs = model_args(dropout_prob=.2, geo_mode="residual", geo_weight=.2)
        model = RQVAE(**kwargs)
        trainer = Trainer(trainer_args(self.root / "global"), model, 3)
        dataset = RecordingDataset()
        loader = DataLoader(dataset, batch_size=9, shuffle=True)
        trainer.train_data = loader
        trainer._train_epoch(loader, 0)
        path = trainer._save_checkpoint(0, ckpt_file="boundary.pth")
        dataset.accesses.clear()
        trainer._train_epoch(loader, 1)
        expected_order = list(dataset.accesses)
        expected = copy.deepcopy(model.state_dict())
        resumed_data = RecordingDataset()
        resumed_loader = DataLoader(resumed_data, batch_size=9, shuffle=True)
        resumed = Trainer(trainer_args(self.root / "resumed"), RQVAE(**kwargs), 3)
        result = resumed.load_checkpoint(path, resumed_loader)
        self.assertTrue(result["exact_epoch_resume"])
        resumed._train_epoch(resumed_loader, 1)
        self.assertEqual(expected_order, resumed_data.accesses)
        assert_nested_equal(self, expected, resumed.model.state_dict())
        with self.assertRaisesRegex(ValueError, "generator"):
            resumed.load_checkpoint(path, DataLoader(RecordingDataset(), batch_size=9, shuffle=True,
                                                     generator=torch.Generator().manual_seed(3)))

    def test_actual_trainer_applies_geo_weight_outside_quant_weight(self):
        kwargs = model_args(geo_mode="residual", geo_weight=.31, quant_loss_weight=.17)
        model = RQVAE(**kwargs)
        direct_model = copy.deepcopy(model)
        trainer = Trainer(trainer_args(self.root / "trainer", warmup_epochs=0), model, 1)
        direct = Trainer(trainer_args(self.root / "direct", warmup_epochs=0), direct_model, 1)
        x = torch.randn(19, 12)
        # No shuffle/dropout here, isolating Trainer's loss aggregation and step.
        out, quant, _, aux = direct_model(x, return_aux=True)
        base, recon = direct_model.compute_loss(out, quant, x)
        expected = recon + .17 * quant + .31 * aux["geo_loss"]
        expected.backward()
        torch.nn.utils.clip_grad_norm_(direct_model.parameters(), 1.0)
        direct.optimizer.step()
        direct.scheduler.step()
        trainer._train_epoch([x], 0)
        metrics = trainer.last_epoch_metrics[0]
        self.assertAlmostEqual(metrics["loss"], expected.item(), places=7)
        self.assertAlmostEqual(metrics["base"], base.item(), places=7)
        self.assertAlmostEqual(metrics["weighted_geo"], .31 * aux["geo_loss"].item(), places=7)
        assert_nested_equal(self, direct_model.state_dict(), model.state_dict())


if __name__ == "__main__":
    unittest.main()
