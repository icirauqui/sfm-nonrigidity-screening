"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def neighbors(x, k=8, keys=None):
    # Retained native IDs resolve coincident/equidistant geometry deterministically.
    if len(x) <= k or not np.isfinite(x).all():
        raise ValueError("invalid_metric_support")
    keys = np.arange(len(x)) if keys is None else np.asarray(keys)
    if len(np.unique(keys)) != len(x):
        raise ValueError("nonunique_graph_keys")
    tree = cKDTree(x)
    dd, jj = tree.query(x, k=min(k + 2, len(x)))
    result = []
    for i, (dist, idx) in enumerate(zip(dd, jj)):
        keep = idx != i
        dist, idx = dist[keep], idx[keep]
        if len(dist) > k and np.isclose(dist[k - 1], dist[k], rtol=1e-12, atol=0):
            threshold = dist[k - 1]
            idx = np.array(tree.query_ball_point(x[i], threshold * (1 + 1e-12)), int)
            idx = idx[idx != i]
            dist = np.linalg.norm(x[idx] - x[i], axis=1)
            inside = dist < threshold * (1 - 1e-12)
            first = idx[inside]
            boundary = idx[~inside]
            boundary = boundary[np.argsort(keys[boundary])]
            result.append(np.r_[first, boundary][:k])
        else:
            order = np.lexsort((keys[idx], dist))
            result.append(idx[order[:k]])
    return np.asarray(result)


def moments(m, d, zm, zd, sigma):
    z = np.concatenate([zm, zd], axis=2)
    n, k = z.shape[:2]
    flat = z.reshape(n, 6 * k, z.shape[-1])
    joint = (
        (sigma * (flat @ flat.swapaxes(1, 2)) / z.shape[-1])
        .reshape(n, k, 6, k, 6)
        .transpose(0, 1, 3, 2, 4)
    )
    diagonal = joint[:, np.arange(k), np.arange(k)]
    bias = 2 * np.trace(diagonal[:, :, :3, 3:], axis1=-2, axis2=-1)
    raw = 2 * np.sum(m * d, axis=2)
    gradient = 2 * np.concatenate([d, m], axis=2)
    linear = np.einsum("nei,nefij,nfj->nef", gradient, joint, gradient, optimize=True)
    second = 4 * (
        np.einsum("nefij,nefij->nef", joint[:, :, :, :3, :3], joint[:, :, :, 3:, 3:])
        + np.einsum("nefij,nefij->nef", joint[:, :, :, :3, 3:], joint[:, :, :, 3:, :3])
    )
    return raw, bias, linear, second, joint


def standardized(a, cov):
    c = 0.5 * (cov + cov.swapaxes(1, 2))
    floor0 = (64 * np.finfo(float).eps) ** 2 + np.finfo(float).eps * np.mean(
        a * a, axis=1
    )
    c = c + floor0[:, None, None] * np.eye(a.shape[1])[None]
    eigen, v = np.linalg.eigh(c)
    top = eigen[:, -1]
    if np.any(eigen[:, 0] < -1e-8 * np.maximum(top, np.finfo(float).tiny)):
        raise ValueError("indefinite_quadratic_covariance")
    floor = np.maximum(1e-12 * top, np.finfo(float).tiny)
    projected = np.einsum("nij,ni->nj", v, a)
    value = np.sqrt(
        np.mean(projected * projected / np.maximum(eigen, floor[:, None]), axis=1)
    )
    return value, np.sum(eigen < floor[:, None], axis=1)


def scores(x, y, zx, zy, sigma, keys=None):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid_pixel_variance")
    extent = float(np.quantile(np.linalg.norm(x - x.mean(0), axis=1), 0.9))
    if extent <= 1e-12:
        raise ValueError("degenerate_reference_extent")
    graph = neighbors(x, keys=keys)
    x = (x - x.mean(0)) / extent
    y = (y - y.mean(0)) / extent
    zx = zx / extent
    zy = zy / extent
    selected = np.unique(np.linspace(0, len(x) - 1, min(16, len(x)), dtype=int))
    parts = []
    diagnostics = []
    regs = []
    zero_edges = 0
    for start in range(0, len(x), 128):
        stop = min(start + 128, len(x))
        g = graph[start:stop]
        ex = x[g] - x[start:stop, None]
        ey = y[g] - y[start:stop, None]
        d = ey - ex
        m = 0.5 * (ey + ex)
        exz = zx[g] - zx[start:stop, None]
        eyz = zy[g] - zy[start:stop, None]
        zd = eyz - exz
        zm = 0.5 * (eyz + exz)
        raw, bias, linear, second, joint = moments(m, d, zm, zd, sigma)
        cov = linear + second
        a = raw - bias
        value, reg = standardized(a, cov)
        uncentered, _ = standardized(raw, cov)
        first, _ = standardized(a, linear)
        parts.append(
            np.column_stack(
                [value, uncentered, first, np.sqrt(np.mean(raw * raw, axis=1))]
            )
        )
        regs.append(reg)
        zero_edges += int(
            np.sum((np.sum(ex * ex, axis=2) == 0) | (np.sum(ey * ey, axis=2) == 0))
        )
        at = selected[(selected >= start) & (selected < stop)] - start
        if len(at):
            diagnostics.append(
                dict(
                    indices=at + start,
                    raw_coefficients=raw[at],
                    bias=bias[at],
                    coefficients=a[at],
                    linear_covariance=linear[at],
                    second_covariance=second[at],
                    covariance=cov[at],
                    joint_covariance=joint[at],
                    midpoints=m[at],
                    differences=d[at],
                    point_scores=value[at],
                )
            )
    p = np.concatenate(parts)
    reg = np.concatenate(regs)
    values = dict(
        japd_quadratic_q95=float(np.quantile(p[:, 0], 0.95)),
        japd_quadratic_median=float(np.median(p[:, 0])),
        japd_quadratic_uncentered_q95=float(np.quantile(p[:, 1], 0.95)),
        japd_quadratic_linear_q95=float(np.quantile(p[:, 2], 0.95)),
        raw_quadratic_q95=float(np.quantile(p[:, 3], 0.95)),
        raw_quadratic_median=float(np.median(p[:, 3])),
    )
    diag = {k: np.concatenate([r[k] for r in diagnostics]) for k in diagnostics[0]}
    diag.update(
        sigma=np.array(sigma),
        extent=np.array(extent),
        samples=np.array(zx.shape[-1]),
        point_count=np.array(len(x)),
    )
    info = dict(
        points=len(x),
        directed_edges=int(graph.size),
        neighbors=8,
        coincident_edges=zero_edges,
        regularized_points=int(np.sum(reg > 0)),
        regularized_modes=int(reg.sum()),
        regularization_fraction=float(np.mean(reg > 0)),
    )
    return values, info, diag
