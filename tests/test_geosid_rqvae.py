"""Surrogate-gradient and pinned-upstream compatibility contracts."""

import copy
import os
import random
import unittest

import numpy as np
import torch

from rq.models.rqvae import RQVAE as OriginalRQVAE
from sid_generation.geosid.adapters.rqvae import RQVAEAdapter as RQVAE


def directions(d=4, m=11, seed=7, dtype=torch.float64):
    u = torch.randn(d, m, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return u / torch.linalg.vector_norm(u, dim=0, keepdim=True)


UPSTREAM_SHA = "0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1"
# Declared before running comparisons; the upstream STE has float32
# cancellation roundoff whereas the auxiliary path subtracts code vectors.
RESIDUAL_ATOL = 2e-6
RESIDUAL_RTOL = 2e-5


def model_args(**changes):
    kwargs = dict(in_dim=12, num_emb_list=[8, 8, 8], e_dim=4, layers=[24, 16],
                  dropout_prob=0.0, kmeans_init=False, sk_epsilons=[0., 0., 0.],
                  sk_iters=10, quant_loss_weight=0.7)
    kwargs.update(changes)
    return kwargs


def original_rqvae_class():
    return OriginalRQVAE


def reset_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())


def assert_rng_equal(test, first, second):
    test.assertEqual(first[0], second[0])
    test.assertEqual(first[1][0], second[1][0])
    np.testing.assert_array_equal(first[1][1], second[1][1])
    test.assertEqual(first[1][2:], second[1][2:])
    test.assertTrue(torch.equal(first[2], second[2]))


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        reset_seed()
        self.x = torch.randn(31, 12)

    def test_geometry_matches_actual_quantizer_inputs_and_identity_jacobian(self):
        model = RQVAE(**model_args(geo_mode="residual", geo_weight=.2))
        actual, geometric, encoded = [], [], []
        hooks = [q.register_forward_pre_hook(lambda module, args: actual.append(args[0]))
                 for q in model.rq.vq_layers]
        hooks.append(model.georeg.register_forward_pre_hook(lambda module, args: geometric.append(args[0])))
        hooks.append(model.encoder.register_forward_hook(lambda module, args, output: encoded.append(output)))
        try:
            _, _, indices, aux = model(self.x, return_aux=True, geo_directions=directions(dtype=torch.float32))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(actual), 3)
        self.assertEqual(len(geometric), 3)
        self.assertEqual(indices.shape, (31, 3))
        self.assertEqual(aux["geo_level_losses"].shape, (3,))
        torch.testing.assert_close(aux["geo_loss"], aux["geo_level_losses"].mean())
        for a, g in zip(actual, geometric):
            torch.testing.assert_close(a, g, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL)
            jacobian = torch.autograd.grad(g.sum(), encoded[0], retain_graph=True)[0]
            torch.testing.assert_close(jacobian, torch.ones_like(jacobian), atol=0, rtol=0)
        # Demonstrate the upstream STE cancellation which motivates the separate
        # path; do not numerical-gradcheck the intentionally surrogate bridge.
        upstream_jacobian = torch.autograd.grad(actual[1].sum(), encoded[0], retain_graph=True)[0]
        self.assertTrue(torch.equal(upstream_jacobian, torch.zeros_like(upstream_jacobian)))

    def test_each_later_level_reaches_encoder_but_not_codebook(self):
        for level in (1, 2):
            with self.subTest(level=level):
                model = RQVAE(**model_args(geo_mode="residual", geo_weight=.3))
                _, _, _, aux = model(self.x, return_aux=True, geo_directions=directions(dtype=torch.float32))
                aux["geo_level_losses"][level].backward()
                gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.square().sum().item() for g in gradients), 0.)
                self.assertTrue(all(p.grad is None for p in model.rq.parameters()))
                self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))

    def test_enabled_aux_does_not_change_base_gradient_or_update(self):
        plain = RQVAE(**model_args())
        enabled = RQVAE(**model_args(geo_mode="residual", geo_weight=.3))
        enabled.load_state_dict(plain.state_dict(), strict=True)
        opt_plain = torch.optim.AdamW(plain.parameters(), lr=1e-3, weight_decay=0)
        opt_enabled = torch.optim.AdamW(enabled.parameters(), lr=1e-3, weight_decay=0)
        out, quant, _ = plain(self.x)
        base, recon = plain.compute_loss(out, quant, self.x)
        out2, quant2, _, aux = enabled(self.x, return_aux=True)
        base2, recon2 = enabled.compute_loss(out2, quant2, self.x)
        self.assertTrue(torch.equal(base, base2))
        self.assertTrue(torch.equal(recon, recon2))
        base.backward()
        base2.backward()
        for p, p2 in zip(plain.parameters(), enabled.parameters()):
            self.assertTrue(torch.equal(p.grad, p2.grad))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in plain.rq.parameters()))
        opt_plain.step()
        opt_enabled.step()
        for p, p2 in zip(plain.parameters(), enabled.parameters()):
            self.assertTrue(torch.equal(p, p2))
        # GeoReg remains a separate term, unaffected by quant_loss_weight.
        total = base2 + enabled.geo_weight * aux["geo_loss"]
        torch.testing.assert_close(total - base2, enabled.geo_weight * aux["geo_loss"])

    def test_global_uses_only_encoder_latent_and_default_api_skips_geometry(self):
        model = RQVAE(**model_args(geo_mode="global", geo_weight=.3))
        state = model.georeg.get_rng_state()
        self.assertEqual(len(model(self.x)), 3)
        model.get_indices(self.x)
        self.assertTrue(torch.equal(state, model.georeg.get_rng_state()))
        _, _, _, aux = model(self.x, return_aux=True)
        self.assertEqual(len(aux["geo_level_losses"]), 1)
        self.assertFalse(torch.equal(state, model.georeg.get_rng_state()))
        self.assertFalse(aux["geo_residual_rms"].requires_grad)
        self.assertFalse(aux["geo_scales"].requires_grad)

    def test_enabled_aux_has_one_base_forward_and_no_graph_cache(self):
        model = RQVAE(**model_args(geo_mode='residual', geo_weight=.2))
        counts = {'encoder': 0, 'rq': 0, 'decoder': 0}
        handles = []
        for name in counts:
            def count(module, args, output, name=name):
                counts[name] += 1
            handles.append(getattr(model, name).register_forward_hook(count))
        try:
            _, _, _, aux = model(self.x, return_aux=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, {'encoder': 1, 'rq': 1, 'decoder': 1})
        self.assertEqual(aux['geo_names'], ('level_0', 'level_1', 'level_2'))
        self.assertFalse(any(torch.is_tensor(value) and value.grad_fn is not None
                             for value in vars(model).values()))

    def test_cpu_autocast_keeps_auxiliary_geometry_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = RQVAE(**model_args(geo_mode='residual', geo_weight=.2))
                with torch.autocast('cpu', dtype=dtype):
                    out, rq_loss, _, aux = model(self.x, return_aux=True)
                    base, _ = model.compute_loss(out, rq_loss, self.x)
                    loss = base + model.geo_weight * aux['geo_loss']
                self.assertEqual(aux['geo_loss'].dtype, torch.float32)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_auxiliary_shape_device_rejection_precedes_original_forward(self):
        from unittest.mock import patch
        model = RQVAE(**model_args(geo_mode='residual', geo_weight=.2))
        with patch.object(model.encoder, 'forward', side_effect=AssertionError('encoder called')):
            with self.assertRaisesRegex(ValueError, 'B >= 2'):
                model(torch.ones(1, 12), return_aux=True)
            with self.assertRaisesRegex(ValueError, 'CPU/CUDA'):
                model(torch.empty(2, 12, device='meta'), return_aux=True)


class UpstreamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original_class = original_rqvae_class()

    def test_disabled_matches_pinned_upstream_through_optimizer_step(self):
        # Include dropout and Sinkhorn to exercise the original random and
        # quantization paths, not merely a deterministic encoder in eval mode.
        kwargs = model_args(dropout_prob=.2, sk_epsilons=[0., .05, .05])
        x = torch.randn(23, 12, generator=torch.Generator().manual_seed(17))
        for mode, weight in (("none", 0.), ("none", .5), ("global", 0.), ("residual", 0.)):
            with self.subTest(mode=mode, weight=weight):
                reset_seed()
                original = self.original_class(**kwargs)
                original_init_rng = rng_snapshot()
                reset_seed()
                current = RQVAE(**kwargs, geo_mode=mode, geo_weight=weight)
                assert_rng_equal(self, original_init_rng, rng_snapshot())
                self.assertEqual(list(original.state_dict()), list(current.state_dict()))
                self.assertEqual(sum(p.numel() for p in original.parameters()),
                                 sum(p.numel() for p in current.parameters()))
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, current.state_dict()[key]))
                # Explicit legacy strict loading is a distinct inference contract.
                current.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
                old_optimizer = torch.optim.AdamW(original.parameters(), lr=.001, weight_decay=0)
                new_optimizer = torch.optim.AdamW(current.parameters(), lr=.001, weight_decay=0)
                before_geo = current.georeg.get_rng_state()
                reset_seed(36)
                old_outputs = original(x)
                old_loss, _ = original.compute_loss(old_outputs[0], old_outputs[1], x)
                old_loss.backward()
                old_optimizer.step()
                old_rng = rng_snapshot()
                reset_seed(36)
                new_outputs = current(x, return_aux=True)
                new_loss, _ = current.compute_loss(new_outputs[0], new_outputs[1], x)
                new_loss = new_loss + current.geo_weight * new_outputs[3]["geo_loss"]
                new_loss.backward()
                new_optimizer.step()
                assert_rng_equal(self, old_rng, rng_snapshot())
                self.assertTrue(torch.equal(before_geo, current.georeg.get_rng_state()))
                self.assertEqual(new_outputs[3]["geo_level_losses"].numel(), 0)
                self.assertTrue(torch.equal(old_loss, new_loss))
                for old, new in zip(old_outputs, new_outputs[:3]):
                    self.assertTrue(torch.equal(old, new))
                for old, new in zip(original.parameters(), current.parameters()):
                    self.assertTrue(torch.equal(old.grad, new.grad))
                    self.assertTrue(torch.equal(old, new))
                original.eval()
                current.eval()
                self.assertTrue(torch.equal(original.get_indices(x), current.get_indices(x)))

    def test_disabled_really_does_not_enter_geometry(self):
        from unittest.mock import patch
        for mode, weight in (("none", 1.), ("residual", 0.)):
            model = RQVAE(**model_args(geo_mode=mode, geo_weight=weight))
            with patch.object(model.georeg, "forward", side_effect=AssertionError("geometry called")), \
                 patch.object(model.georeg, "sample_directions", side_effect=AssertionError("RNG used")):
                self.assertEqual(len(model(torch.ones(1, 12), return_aux=True)), 4)

    def test_invalid_mode_and_weight(self):
        for kwargs in ({"geo_mode": "tail"}, {"geo_weight": -1}, {"geo_weight": float("nan")},
                       {'geo_weight': True}, {'geo_weight': '1'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RQVAE(**model_args(**kwargs))


@unittest.skipUnless(os.environ.get("GEOSID_TEST_CUDA") == "1", "CUDA tests require budgeted opt-in")
class CudaModelTests(unittest.TestCase):
    def test_fp32_residual_values_and_later_level_encoder_gradients(self):
        self.assertTrue(torch.cuda.is_available())
        x = torch.randn(67, 12, device="cuda")
        u = directions(dtype=torch.float32).cuda()
        for level in (1, 2):
            model = RQVAE(**model_args(geo_mode="residual", geo_weight=.3)).cuda()
            actual, geometric = [], []
            hooks = [q.register_forward_pre_hook(lambda module, args: actual.append(args[0]))
                     for q in model.rq.vq_layers]
            hooks.append(model.georeg.register_forward_pre_hook(lambda module, args: geometric.append(args[0])))
            try:
                _, _, _, aux = model(x, return_aux=True, geo_directions=u)
            finally:
                for hook in hooks:
                    hook.remove()
            for expected, value in zip(actual, geometric):
                torch.testing.assert_close(expected, value, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL)
            aux["geo_level_losses"][level].backward()
            gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.square().sum().item() for g in gradients), 0.)
            self.assertTrue(all(p.grad is None for p in model.rq.parameters()))


if __name__ == "__main__":
    unittest.main()
