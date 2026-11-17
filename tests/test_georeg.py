"""Mathematical contracts for the explicitly documented cf_mean adaptation."""

import os
import unittest

import numpy as np
import torch

from rq.models.georeg import GeoReg, cf_discrepancy, frequency_grid, normalize_residual


def directions(d=4, m=11, seed=7, dtype=torch.float64):
    u = torch.randn(d, m, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return u / torch.linalg.vector_norm(u, dim=0, keepdim=True)


def numpy_reference(x, u, normalize=True, eps=1e-6):
    # Deliberately independent scalar-loop implementation of the specification.
    if normalize:
        c = x - np.mean(x, axis=0, keepdims=True)
        x = c / np.sqrt(np.mean(c * c) + eps * eps)
    t = np.linspace(0, 3, 17)
    phi = np.exp(-t * t / 2)
    q = np.full(17, 3 / 16)
    q[[0, -1]] /= 2
    w = q * phi / np.sum(q * phi)
    result = 0.0
    for column in range(u.shape[1]):
        p = x @ u[:, column]
        for frequency in range(len(t)):
            a = np.mean(np.cos(t[frequency] * p))
            b = np.mean(np.sin(t[frequency] * p))
            result += w[frequency] * ((a - phi[frequency]) ** 2 + b ** 2)
    return result / u.shape[1]


class MathematicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numpy_and_uneven_projection_chunks_match(self):
        x = torch.tensor(np.random.default_rng(2).normal(size=(37, 4)), dtype=torch.float64)
        u = directions()
        expected = numpy_reference(x.numpy(), u.numpy())
        for chunk in (1, 4, 11, 64):
            with self.subTest(chunk=chunk):
                actual = GeoReg(nproj=11, projection_chunk=chunk)(x, u)
                self.assertAlmostEqual(actual.item(), expected, places=13)
                raw = cf_discrepancy(x, u, projection_chunk=chunk)
                self.assertAlmostEqual(raw.item(), numpy_reference(x.numpy(), u.numpy(), False), places=13)

    def test_frequency_weights_and_no_sample_or_projection_multipliers(self):
        t, phi, w = frequency_grid(dtype=torch.float64)
        self.assertEqual(len(t), 17)
        self.assertEqual(t[0], 0)
        self.assertEqual(t[-1], 3)
        self.assertAlmostEqual(w.sum().item(), 1, places=15)
        self.assertAlmostEqual((w[0] / phi[0] / (w[1] / phi[1])).item(), .5, places=14)
        x = torch.randn(19, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        u = directions()
        g = GeoReg()
        # Exact replication leaves empirical means invariant. A hidden B or M
        # multiplier would violate these equalities.
        torch.testing.assert_close(g(x, u), g(x.repeat(2, 1), u), rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(g(x, u), g(x, u.repeat(1, 3)), rtol=1e-12, atol=1e-14)

    def test_pure_cf_and_normalized_loss_double_gradcheck(self):
        x = torch.randn(7, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(8),
                        requires_grad=True)
        u = directions(3, 5)
        self.assertTrue(torch.autograd.gradcheck(lambda y: cf_discrepancy(y, u, projection_chunk=2),
                                               (x,), eps=1e-6, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.autograd.gradcheck(lambda y: GeoReg(projection_chunk=2)(y, u),
                                               (x,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_normalization_is_one_differentiable_scalar_and_translation_invariant(self):
        x = torch.randn(13, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(9),
                        requires_grad=True)
        x = x * torch.tensor([1., 2., 4., 8.], dtype=torch.float64)
        y, scale = normalize_residual(x)
        self.assertEqual(scale.ndim, 0)
        torch.testing.assert_close(y.mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)
        # Different coordinate variances survive; this is not whitening.
        self.assertGreater(y[:, 3].square().mean(), y[:, 0].square().mean() * 10)
        self.assertTrue(scale.requires_grad)
        u = directions()
        torch.testing.assert_close(GeoReg()(x, u), GeoReg()(x + 17, u), rtol=1e-12, atol=1e-13)

    def test_zero_and_constant_batches_have_finite_loss_and_gradient(self):
        for constant in (0., 7.):
            x = torch.full((9, 4), constant, requires_grad=True)
            loss = GeoReg()(x, directions(dtype=torch.float32))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_inputs_are_rejected(self):
        g = GeoReg()
        for x in (torch.ones(4), torch.ones(1, 4), torch.ones(2, 0), torch.ones(2, 3, 4),
                  torch.ones(2, 3, dtype=torch.long), torch.full((2, 3), float("nan")),
                  torch.full((2, 3), float("inf"))):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                g(x)
        for u in (torch.ones(4, 2), torch.zeros(4, 2), torch.eye(3), torch.zeros(4, 0)):
            with self.subTest(direction_shape=u.shape), self.assertRaises(ValueError):
                g(torch.ones(5, 4), u)

    def test_invalid_configuration(self):
        for kwargs in ({"nproj": 0}, {"nproj": True}, {"num_frequencies": 1},
                       {"projection_chunk": 0}, {"eps": 0}, {"eps": float("nan")},
                       {"tmax": float("inf")}, {"tmax": 0}, {"seed": 1.2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeoReg(**kwargs)

    def test_gaussian_vs_anisotropic_and_uncentered_shift(self):
        x = torch.randn(8192, 8, generator=torch.Generator().manual_seed(31))
        u = directions(8, 64, dtype=torch.float32)
        anisotropic = x * torch.tensor([8., 1., 1., 1., 1., 1., 1., 1.])
        g = GeoReg()
        self.assertLess(g(x, u).item(), g(anisotropic, u).item())
        self.assertLess(cf_discrepancy(x, u).item(), cf_discrepancy(x + 2, u).item())

    def test_rng_is_independent_restorable_and_fixed_directions_do_not_advance_it(self):
        x = torch.ones(5, 4)
        before_global = torch.random.get_rng_state().clone()
        g = GeoReg(seed=18)
        initial = g.get_rng_state()
        first = g.sample_directions(4)
        self.assertFalse(torch.equal(g.get_rng_state(), initial))
        g.set_rng_state(initial)
        self.assertTrue(torch.equal(first, g.sample_directions(4)))
        before_fixed = g.get_rng_state()
        g(x, directions(dtype=torch.float32))
        self.assertTrue(torch.equal(before_fixed, g.get_rng_state()))
        self.assertTrue(torch.equal(before_global, torch.random.get_rng_state()))
        self.assertEqual(list(g.parameters()), [])
        self.assertEqual(list(g.state_dict()), [])

    def test_cpu_autocast_is_disabled_and_half_inputs_use_float32(self):
        x = torch.randn(29, 4, generator=torch.Generator().manual_seed(42))
        u = directions(dtype=torch.float32)
        g = GeoReg()
        expected = g(x, u)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = g(x, u)
            half_actual = g(x.to(torch.bfloat16), u)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(half_actual.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(g(x.double(), u.double()).dtype, torch.float64)

    def test_module_dtype_conversion_preserves_geometry_constants(self):
        x = torch.randn(29, 4, generator=torch.Generator().manual_seed(42))
        for dtype in (torch.float32, torch.float64):
            inputs = x.to(dtype)
            u = directions(dtype=dtype)
            expected = GeoReg()(inputs, u)
            for converted in (GeoReg().half(), GeoReg().bfloat16(),
                              GeoReg().float(), GeoReg().half().double()):
                with self.subTest(input_dtype=dtype, module=converted):
                    self.assertTrue(torch.equal(expected, converted(inputs, u)))
                    self.assertEqual(converted.state_dict(), {})


@unittest.skipUnless(os.environ.get("GEOSID_TEST_CUDA") == "1", "CUDA tests require budgeted opt-in")
class CudaTests(unittest.TestCase):
    def test_cuda_autocast_and_gradients(self):
        self.assertTrue(torch.cuda.is_available())
        x = torch.randn(67, 8, device="cuda", requires_grad=True)
        u = directions(8, 37, dtype=torch.float32).cuda()
        g = GeoReg().cuda()
        expected = g(x, u)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = g(x, u)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, actual))
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
