"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import transfer_views as E


def transfer(x, z, rotations, centers, intrinsics):
    """Batched endpoint pixels to middle pixels; abstain on any invalid point."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    R, C, K = map(lambda v: np.asarray(v, float), (rotations, centers, intrinsics))
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or x.shape != z.shape
        or not len(x)
        or R.shape != (3, 3, 3)
        or C.shape != (3, 3)
        or K.shape != (3, 3, 3)
    ):
        raise ValueError("invalid_transfer_shape")
    if not all(np.isfinite(v).all() for v in (x, z, R, C, K)):
        raise ValueError("nonfinite_transfer")
    scale = np.linalg.norm(C[2] - C[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("zero_endpoint_baseline")
    c = (C - C[0]) / scale
    rays = []
    for pixels, i in ((x, 0), (z, 2)):
        ray = (
            np.linalg.solve(K[i], np.column_stack((pixels, np.ones(len(x)))).T).T @ R[i]
        )
        ray /= np.linalg.norm(ray, axis=1)[:, None]
        rays.append(ray)
    a, b = rays
    A = np.stack((a, -b), axis=2)
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    if np.any(s[:, 1] <= 64 * np.finfo(float).eps * s[:, 0]):
        raise ValueError("unidentifiable_endpoint_rays")
    coeff = np.einsum("nji,nj->ni", vh, np.einsum("nji,j->ni", u, c[2]) / s)
    point = 0.5 * (coeff[:, 0, None] * a + c[2] + coeff[:, 1, None] * b)
    for i in range(3):
        q = (point - c[i]) @ R[i].T
        if np.any(q[:, 2] <= 0) or not np.isfinite(q).all():
            raise ValueError("nonpositive_transfer_depth")
    q = (point - c[1]) @ R[1].T @ K[1].T
    prediction = q[:, :2] / q[:, 2, None]
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite_transfer_prediction")
    return prediction
