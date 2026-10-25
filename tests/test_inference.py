import tempfile, unittest
from pathlib import Path
import numpy as np
from sfm_nonrigidity import rdsc_scores, compose, global_scores, global_projection
from sfm_nonrigidity.sequence import endpoint
from test_geometry import fixture


class InferenceTests(unittest.TestCase):
    def test_rdsc_similarity_invariance_and_determinism(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(200, 3))
        d = 0.003 * rng.normal(size=x.shape)
        d[:, 2] += 0.01 * np.sin(x[:, 0] * 4)
        a = rdsc_scores(x, d)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        b = rdsc_scores(7 * x @ q + 2, 7 * d @ q)
        self.assertEqual(a, rdsc_scores(x, d))
        for key in a:
            if key != "spatial_scale":
                self.assertAlmostEqual(a[key], b[key], places=10)

    def test_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):
            rdsc_scores(np.ones((3, 3)), np.full((3, 3), np.nan))

    def test_strict_positive_threshold_and_unknown_provenance(self):
        evidence = {"rdsc": dict(score=[1.0, 1.0], cutoff=[1.0, 1.0])}
        self.assertIs(compose("independent", evidence)["decision"], False)
        self.assertIsNone(compose("unknown", evidence)["decision"])

    def test_independent_branch_priority_and_fallback(self):
        evidence = {
            "rdsc": dict(score=[2.0, 2.0], cutoff=[1.0, 1.0]),
            "transfer_share": dict(score=[0.4, 0.4], cutoff=[0.5, 0.5]),
        }
        self.assertIs(compose("independent", evidence)["decision"], False)
        del evidence["transfer_share"]
        self.assertEqual(compose("independent", evidence)["selected_expert"], "rdsc")

    def test_inherited_requires_two_geometry_components(self):
        e = {
            "japd_conditioned": dict(score=[2.0, 2.0], cutoff=[1.0, 1.0]),
            "rdsc": dict(score=[0.0, 0.0], cutoff=[1.0, 1.0]),
        }
        self.assertEqual(compose("inherit_global", e)["selected_expert"], "rdsc")
        e["japd_quadratic_q95"] = dict(score=[0.0, 0.0], cutoff=[1.0, 1.0])
        self.assertIs(compose("inherit_global", e)["decision"], True)

    def test_persistence_missing_anchors_and_coverage(self):
        rows = [dict(index=i, scores={"transfer": 1.0, "rdsc": 1.0}) for i in range(9)]
        self.assertEqual(endpoint(rows, "transfer")["runs"], 7)
        self.assertEqual(endpoint(rows, "rdsc")["runs"], 8)
        for i in [2, 5]:
            rows[i]["scores"] = {}
        self.assertIsNone(endpoint(rows, "rdsc")["score"])
        with self.assertRaises(ValueError):
            endpoint(rows + [rows[0]], "rdsc")

    def test_streamed_energy_against_independent_dense_projection(self):
        rng = np.random.default_rng(55)
        n = 6
        rows = 40
        nc = 18
        acc = global_projection.Accumulator(nc)
        aa = []
        bb = []
        hh = []
        for i in range(n):
            a = rng.normal(size=(rows, 3))
            b = rng.normal(size=(rows, nc))
            h = rng.normal(size=(rows, 61))
            acc.add(a, b, h, np.arange(nc))
            a0 = np.zeros((rows, 3 * n))
            a0[:, 3 * i : 3 * i + 3] = a
            aa.append(a0)
            bb.append(b)
            hh.append(h)
        nuisance = np.column_stack((np.vstack(aa), np.vstack(bb)))
        h = np.vstack(hh)
        projected = h - nuisance @ np.linalg.lstsq(nuisance, h, rcond=None)[0]
        np.testing.assert_allclose(
            acc.close()["gram"], projected.T @ projected, rtol=1e-10, atol=1e-10
        )
        affine = projected[:, :24]
        datum = projected[:, -1]
        quad = projected[:, 24:60]
        null = datum - affine @ np.linalg.lstsq(affine, datum, rcond=None)[0]
        quad = quad - affine @ np.linalg.lstsq(affine, quad, rcond=None)[0]
        fitted = quad @ np.linalg.lstsq(quad, null, rcond=None)[0]
        result = global_projection.score(acc.close(), float(h[:, -1] @ h[:, -1]))
        self.assertAlmostEqual(result["gain"], float(fitted @ fitted), places=9)

    def test_global_view_gate_and_full_fixture(self):
        base, _ = fixture(0.2)
        self.assertEqual(global_scores(base)["reason"], "insufficient_registered_views")
        result = global_scores(base, requested_views=6)
        self.assertEqual(result["status"], "SCORED")
        self.assertGreaterEqual(result["scores"]["mode_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
