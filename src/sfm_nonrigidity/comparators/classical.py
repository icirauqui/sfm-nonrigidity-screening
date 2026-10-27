"""Same-input elementary geometric controls. SPDX-License-Identifier: GPL-3.0-or-later."""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from ..methods.rdsc.rdsc import detection_scores, robust_affine_residuals


def proper_fit(source, target, *, scale):
    """Batched proper Procrustes fit; final two axes are points and XYZ."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    xc, yc = x - x.mean(axis=-2, keepdims=True), y - y.mean(axis=-2, keepdims=True)
    cov = np.swapaxes(xc, -1, -2) @ yc
    u, singular, vt = np.linalg.svd(cov)
    signs = np.ones_like(singular)
    signs[..., -1] = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    rotation = (u * signs[..., None, :]) @ vt
    factor = np.ones(x.shape[:-2])
    if scale:
        den = np.sum(xc * xc, axis=(-2, -1))
        if np.any(den <= 0):
            raise ValueError("Degenerate similarity source")
        factor = np.sum(singular * signs, axis=-1) / den
    return factor[..., None, None] * (xc @ rotation) + y.mean(axis=-2, keepdims=True)


def extra_scores(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 9:
        raise ValueError("Need matching Nx3 arrays, N >= 9")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite points")
    length = np.quantile(np.linalg.norm(x - x.mean(0), axis=1), 0.9)
    if length <= 1e-12:
        raise ValueError("Degenerate extent")
    idx = cKDTree(x).query(x, k=9, workers=1)[1]
    src = np.repeat(np.arange(len(x)), 9)
    dst = idx.ravel()
    edges = np.unique(np.sort(np.column_stack((src, dst))[src != dst], axis=1), axis=0)
    a, b = edges.T
    changes = (
        np.abs(
            np.linalg.norm(y[b] - y[a], axis=1) - np.linalg.norm(x[b] - x[a], axis=1)
        )
        / length
    )
    fitted = proper_fit(y, x, scale=True)
    local = proper_fit(y[idx], x[idx], scale=False)
    local_rms = np.sqrt(np.mean(np.sum((local - x[idx]) ** 2, axis=2), axis=1)) / length
    residual, _ = robust_affine_residuals(x, y - x)
    z = residual - residual.mean(0)
    denominator = float(np.sum(z * z))
    moran = (
        float(len(x) * np.sum(z[a] * z[b]) / (len(edges) * denominator))
        if denominator > np.finfo(float).eps ** 2 * length**2 * len(x)
        else 0.0
    )
    return dict(
        edge_length_change_q95=float(np.quantile(changes, 0.95)),
        similarity_q95=float(
            np.quantile(np.linalg.norm(fitted - x, axis=1) / length, 0.95)
        ),
        local_rigid_q95=float(np.quantile(local_rms, 0.95)),
        vector_moran=moran,
    )


def score(x, y, index, sensitivity=False):
    result = detection_scores(
        x, y - x, neighbors=8, shuffle_count=20, seed=20260904 + index
    )
    result.update(extra_scores(x, y))
    variants = {}
    if sensitivity:
        for k in (4, 8, 16):
            for offset in range(5):
                value = (
                    result["rdsc"]
                    if k == 8 and offset == 0
                    else detection_scores(
                        x,
                        y - x,
                        neighbors=k,
                        shuffle_count=20,
                        seed=20260904 + index + 1000 * offset,
                    )["rdsc"]
                )
                variants[f"k{k}_seed{offset}"] = value
    return result, variants
