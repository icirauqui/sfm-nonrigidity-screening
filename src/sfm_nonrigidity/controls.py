"""Explicit additive controls, not a replacement for original RDSC."""

import numpy as np
from scipy.spatial import cKDTree


def score(x, y, index, detector):
    length = float(np.quantile(np.linalg.norm(x - x.mean(0), axis=1), 0.9))
    if length <= 1e-12:
        raise ValueError("degenerate_extent")
    residual, _ = detector.robust_affine_residuals(x, y - x)
    graph = cKDTree(x).query(x, k=min(9, len(x)), workers=1)[1]

    def amp(v):
        return float(
            np.quantile(np.linalg.norm(v[graph].mean(1), axis=1), 0.95) / length
        )

    a = amp(residual)
    rng = np.random.default_rng(20260904 + index)
    b = float(np.median([amp(residual[rng.permutation(len(x))]) for _ in range(20)]))
    return dict(
        coherent_minus_shuffle=a - b,
        coherent_squared_minus_shuffle_squared=a * a - b * b,
    ), dict(amplitude=a, shuffle_reference=b)
