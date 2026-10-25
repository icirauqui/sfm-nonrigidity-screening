"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import models as G


def jacobian(points, kind, p):
    if np.any(points[:, 2] <= 0):
        raise ValueError("nonpositive_camera_depth")
    z = points[:, 2]
    uv = points[:, :2] / z[:, None]
    n = len(z)
    J = np.zeros((n, 2, 3))
    J[:, 0, 0] = 1 / z
    J[:, 1, 1] = 1 / z
    J[:, :, 2] = -uv / z[:, None]
    D = np.tile(np.eye(2), (n, 1, 1))
    if kind == "OPENCV_FISHEYE":
        r = np.linalg.norm(uv, axis=1)
        theta = np.arctan(r)
        F = theta + sum(p[4 + k] * theta ** (2 * k + 3) for k in range(4))
        Fp = 1 + sum((2 * k + 3) * p[4 + k] * theta ** (2 * k + 2) for k in range(4))
        a = np.divide(F, r, out=np.ones_like(r), where=r > 1e-8)
        b = np.divide(
            Fp * r / (1 + r * r) - F,
            r**3,
            out=np.full_like(r, 2 * (p[4] - 1 / 3)),
            where=r > 1e-8,
        )
        D = a[:, None, None] * D + b[:, None, None] * uv[:, :, None] * uv[:, None, :]
    elif kind != "PINHOLE":
        raise ValueError("unsupported_camera_model")
    D = D * p[:2][None, :, None]
    return D @ J


def scatter_gram(A, idx, n):
    # A is an observation-indexed 3 by 2 influence matrix.
    gram = A @ np.swapaxes(A, 1, 2)
    out = np.empty((n, 3, 3))
    for i in range(3):
        for j in range(3):
            out[:, i, j] = np.bincount(idx, weights=gram[:, i, j], minlength=n)
    return out


def inverse_psd(H, relative_floor=1e-12):
    values, V = np.linalg.eigh((H + H.swapaxes(-1, -2)) / 2)
    top = values[:, -1]
    if np.any(top <= 0):
        raise ValueError("zero_information_matrix")
    floor = relative_floor * top
    regularized = values < floor[:, None]
    inv = (V / np.maximum(values, floor[:, None])[:, None, :]) @ V.swapaxes(1, 2)
    return inv, int(np.any(regularized, axis=1).sum())


def standardized(d, cov):
    # Small isotropic floor is numerical regularization, explicitly not measured noise.
    extent = float(np.quantile(np.linalg.norm(d, axis=1), 0.9))
    if not np.any(d):
        return np.zeros(len(d)), 0
    total = cov + np.eye(3)[None, :, :] * (
        np.finfo(float).eps * max(extent**2, np.finfo(float).tiny)
    )
    inv, reg = inverse_psd(total)
    value = np.einsum("ni,nij,nj->n", d, inv, d)
    return np.sqrt(np.maximum(value, 0)), reg
