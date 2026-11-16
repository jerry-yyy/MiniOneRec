"""Opt-in actual-device epoch recovery test; run only through the GPU budget."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from rq.models.rqvae import RQVAE
from rq.trainer import Trainer
from tests.test_georeg_model import assert_rng_equal, model_args, reset_seed, rng_snapshot
from tests.test_georeg_resume import RecordingDataset, assert_nested_equal, trainer_args


def cuda_rng_snapshot():
    return [state.clone() for state in torch.cuda.get_rng_state_all()]


@unittest.skipUnless(os.environ.get("GEOSID_TEST_CUDA") == "1", "CUDA tests require budgeted opt-in")
class CudaEpochResumeTests(unittest.TestCase):
    def test_saved_epoch_reproduces_cuda_training_and_rng(self):
        self.assertTrue(torch.cuda.is_available(), "Explicit CUDA verification requires an actual device")
        device = "cuda:0"
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        reset_seed(2024)
        kwargs = model_args(dropout_prob=.2, kmeans_init=True, kmeans_iters=4,
                            geo_mode="residual", geo_weight=.3, geo_seed=173)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            original = RQVAE(**kwargs)
            data = RecordingDataset()
            loader = DataLoader(data, batch_size=9, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(79))
            trainer = Trainer(trainer_args(root / "original", device=device), original, len(loader))
            trainer.train_data = loader
            self.assertFalse(any(quantizer.initted for quantizer in original.rq.vq_layers))
            trainer._train_epoch(loader, 0)
            # The real fit() path also consumes the loader during catalog diagnostics.
            trainer._valid_epoch(loader)
            torch.cuda.synchronize(device)
            self.assertTrue(all(quantizer.initted for quantizer in original.rq.vq_layers))
            path = trainer._save_checkpoint(0, ckpt_file="cuda_boundary.pth")
            saved_rng = rng_snapshot()
            saved_cuda_rng = cuda_rng_snapshot()
            saved_geo_rng = original.georeg.get_rng_state()
            saved_loader_rng = loader.generator.get_state().clone()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert_nested_equal(self, saved_cuda_rng, checkpoint["rng"]["cuda"])
            self.assertEqual(checkpoint["resume_boundary"], "epoch")

            expected_batches, expected_outputs = [], []
            input_hook = original.register_forward_pre_hook(
                lambda module, inputs: expected_batches.append(inputs[0].detach().clone()))
            output_hook = original.register_forward_hook(
                lambda module, inputs, output: expected_outputs.append(
                    tuple(value.detach().clone() for value in output[:3])))
            data.accesses.clear()
            try:
                expected_losses = trainer._train_epoch(loader, 1)
            finally:
                input_hook.remove()
                output_hook.remove()
            torch.cuda.synchronize(device)
            expected_order = list(data.accesses)
            expected_metrics = copy.deepcopy(trainer.last_epoch_metrics)
            expected_model = copy.deepcopy(original.state_dict())
            expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            expected_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
            expected_rng = rng_snapshot()
            expected_cuda_rng = cuda_rng_snapshot()
            expected_geo_rng = original.georeg.get_rng_state()
            expected_loader_rng = loader.generator.get_state().clone()

            # Use a genuinely fresh set of objects and perturb CUDA independently
            # before loading. A pass cannot depend on the live uninterrupted RNG.
            reset_seed(999)
            resumed_model = RQVAE(**kwargs)
            resumed_data = RecordingDataset()
            resumed_loader = DataLoader(resumed_data, batch_size=9, shuffle=True, num_workers=0,
                                        generator=torch.Generator().manual_seed(999))
            resumed = Trainer(trainer_args(root / "resumed", device=device),
                              resumed_model, len(resumed_loader))
            torch.rand(17, device=device)
            self.assertFalse(torch.equal(saved_cuda_rng[0], cuda_rng_snapshot()[0]))
            summary = resumed.load_checkpoint(path, resumed_loader)
            self.assertTrue(summary["exact_epoch_resume"])
            self.assertEqual(summary["start_epoch"], 1)
            self.assertEqual(summary["global_step"], 3)
            assert_rng_equal(self, saved_rng, rng_snapshot())
            assert_nested_equal(self, saved_cuda_rng, cuda_rng_snapshot())
            self.assertTrue(torch.equal(saved_geo_rng, resumed_model.georeg.get_rng_state()))
            self.assertTrue(torch.equal(saved_loader_rng, resumed_loader.generator.get_state()))
            self.assertTrue(all(quantizer.initted for quantizer in resumed_model.rq.vq_layers))
            for parameter, optimizer_state in resumed.optimizer.state.items():
                self.assertEqual(optimizer_state["step"].device.type, "cpu")
                self.assertEqual(optimizer_state["exp_avg"].device, parameter.device)
                self.assertEqual(optimizer_state["exp_avg_sq"].device, parameter.device)

            actual_batches, actual_outputs = [], []
            input_hook = resumed_model.register_forward_pre_hook(
                lambda module, inputs: actual_batches.append(inputs[0].detach().clone()))
            output_hook = resumed_model.register_forward_hook(
                lambda module, inputs, output: actual_outputs.append(
                    tuple(value.detach().clone() for value in output[:3])))
            try:
                with patch("rq.models.vq.VectorQuantizer.init_emb",
                           side_effect=AssertionError("Loaded CUDA codebooks were reinitialized")):
                    actual_losses = resumed._train_epoch(resumed_loader, resumed.start_epoch)
            finally:
                input_hook.remove()
                output_hook.remove()
            torch.cuda.synchronize(device)
            self.assertEqual(expected_order, resumed_data.accesses)
            self.assertEqual(expected_losses, actual_losses)
            self.assertEqual(expected_metrics, resumed.last_epoch_metrics)
            assert_nested_equal(self, expected_batches, actual_batches)
            assert_nested_equal(self, expected_outputs, actual_outputs)
            assert_nested_equal(self, expected_model, resumed_model.state_dict())
            assert_nested_equal(self, expected_optimizer, resumed.optimizer.state_dict())
            assert_nested_equal(self, expected_scheduler, resumed.scheduler.state_dict())
            assert_rng_equal(self, expected_rng, rng_snapshot())
            assert_nested_equal(self, expected_cuda_rng, cuda_rng_snapshot())
            self.assertTrue(torch.equal(expected_geo_rng, resumed_model.georeg.get_rng_state()))
            self.assertTrue(torch.equal(expected_loader_rng, resumed_loader.generator.get_state()))
            self.assertEqual(resumed.global_step, 6)


if __name__ == "__main__":
    unittest.main()
