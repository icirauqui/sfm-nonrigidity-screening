import json, unittest, sys
from pathlib import Path
import numpy as np
from sfm_nonrigidity import epc_scores


class PrimaryPairTests(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "linux",
        "Strict retained scalar replay requires the qualified Linux numerical backend; see docs/reproduction.md",
    )
    def test_retained_primary_scalar_profile_and_controls(self):
        root = Path(__file__).resolve().parents[1] / "reproduction/reference"
        expected = json.loads((root / "primary_pair_expected.json").read_text())
        with np.load(root / "primary_pair.npz", allow_pickle=False) as z:
            actual = epc_scores(
                z["positions"],
                z["residuals"],
                window_index=expected["window_index"],
                normalization_extent=float(z["normalization_extent"]),
            )
        self.assertEqual(set(actual), set(expected["scores"]))
        for k, v in actual.items():
            self.assertTrue(
                np.isclose(v, expected["scores"][k], rtol=1e-8, atol=1e-10),
                (k, v, expected["scores"][k]),
            )
        a = actual["coherent_affine_q95"]
        b = a / actual["coherence_excess"]
        self.assertAlmostEqual(
            actual["coherent_squared_minus_shuffle_squared"], a * a - b * b, places=14
        )
        self.assertLess(actual["coherent_squared_minus_shuffle_squared"], 0.0)


if __name__ == "__main__":
    unittest.main()
