"""Manufactured qualification; no research data or evaluator annotations."""

import unittest
import numpy as np
from scipy.optimize import least_squares
from sfm_nonrigidity import geometry as G


def fixture(noise=0.0, seed=923):
    random = np.random.default_rng(seed)
    points = np.column_stack(
        [
            random.uniform(-1, 1, 120),
            random.uniform(-0.6, 0.6, 120),
            random.uniform(4, 7, 120),
        ]
    )
    poses, obs, ids = {}, {}, {}
    camera = np.array([450.0, 452.0, 320.0, 240.0])
    for iid in range(1, 7):
        center = np.array([0.15 * iid, 0.05 * np.sin(iid), 0.0])
        poses[iid] = (np.eye(3), center, f"{iid:04d}.png")
        pixel = (points - center)[:, :2] / (points - center)[:, 2, None] * camera[
            :2
        ] + camera[2:]
        pixel += random.normal(0, noise, pixel.shape)
        obs[iid] = np.column_stack([pixel, np.arange(len(points))])
        ids[iid] = 1
    tracks = [tuple((iid, i) for iid in poses) for i in range(len(points))]
    base = (poses, obs, ids, {1: ("PINHOLE", camera)}, points, tracks)
    original = (*base[:4], points + random.normal(0, 0.03, points.shape), tracks)
    return base, original


class ConditioningTests(unittest.TestCase):
    def test_roundoff_polish_requires_bounded_cost_and_better_stationarity(self):
        cost = np.array([0.19552683246728966] * 3)
        new_cost = cost + np.array([5e-14, 1e-6, 5e-14])
        accept, neutral = G.accept_step(
            cost,
            new_cost,
            np.array([1.18228e-7] * 3),
            np.array([9.45e-12, 9.45e-12, 2e-7]),
            np.array([10] * 3),
            640.0,
        )
        np.testing.assert_array_equal(accept, [True, False, False])
        np.testing.assert_array_equal(neutral, [True, False, False])

    def test_exact_points_and_all_track_identities(self):
        base, original = fixture()
        model, diagnostics = G.conditioned_model(base, original)
        self.assertTrue(diagnostics["converged"].all())
        # A 1e-7 pixel-gradient stop has a condition-dependent coordinate error.
        # Require sub-microunit coordinate accuracy and an independent residual.
        np.testing.assert_allclose(model[4], base[4], atol=1e-7, rtol=0)
        self.assertLess(float(diagnostics["final_sse"].max()), 1e-12)
        self.assertIs(model[5], original[5])
        self.assertIs(model[1], original[1])
        self.assertTrue(np.all(diagnostics["final_sse"] <= diagnostics["initial_sse"]))

    def test_noisy_points_against_independent_scipy_optimizer(self):
        base, original = fixture(0.2)
        model, diagnostics = G.conditioned_model(base, original)
        self.assertTrue(diagnostics["converged"].all())
        for point in range(8):

            def residual(x):
                result = []
                for iid, key in original[5][point]:
                    rotation, center, _ = base[0][iid]
                    cam = rotation @ (x - center)
                    k = base[3][base[2][iid]][1]
                    result.extend(
                        cam[:2] / cam[2] * k[:2] + k[2:] - original[1][iid][key, :2]
                    )
                return np.asarray(result)

            fit = least_squares(
                residual,
                original[4][point],
                method="lm",
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
            )
            np.testing.assert_allclose(model[4][point], fit.x, rtol=0, atol=2e-6)
            self.assertAlmostEqual(
                float(np.sum(residual(model[4][point]) ** 2)),
                float(np.sum(fit.fun**2)),
                places=10,
            )

    def test_degenerate_camera_gauge_rejected(self):
        base, original = fixture()
        poses = {i: (r, np.zeros(3), n) for i, (r, _, n) in original[0].items()}
        collapsed = (poses, *original[1:])
        with self.assertRaisesRegex(ValueError, "Degenerate camera baseline"):
            G.conditioned_model((poses, *base[1:]), collapsed)

    def test_transfer_inherited_identity_and_unknown_bounds(self):
        base, original = fixture(0.2)
        model, _ = G.conditioned_model(base, original)
        pairs = G.intersections(base[5], model[5])
        value, details = G.transfer_bounds(base, model, pairs, True)
        self.assertEqual(value, [0.5, 0.5])
        self.assertTrue(details["exact_inherited_equal_errors"])
        lo, hi = G.TV.point_bounds(
            np.array([0.0, 1.0]),
            np.array([0.0, 9.0]),
            np.array([True, False]),
            np.array([True, True]),
        )
        np.testing.assert_array_equal(lo, [0.5, 0.0])
        np.testing.assert_array_equal(hi, [0.5, 1.0])

    def test_gradient_matches_finite_difference(self):
        base, original = fixture(0.2)
        cost, grad, _, valid, _ = G.image_residual_terms(original, original[4])
        self.assertTrue(valid.all())
        for axis in range(3):
            plus, minus = original[4].copy(), original[4].copy()
            plus[0, axis] += 1e-6
            minus[0, axis] -= 1e-6
            cp = G.image_residual_terms(original, plus)[0][0]
            cm = G.image_residual_terms(original, minus)[0][0]
            self.assertAlmostEqual((cp - cm) / 2e-6, 2 * grad[0, axis], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
