"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import transfer as T

ARMS = ("transfer_share", "transfer_global", "transfer_local")
CAP = 4.0


def transfer_one(x, z, R, C, K):
    try:
        return T.transfer(
            np.asarray(x, float)[None], np.asarray(z, float)[None], R, C, K
        )[0], True
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.full(2, np.nan), False


def point_bounds(g, l, g_ok, l_ok):
    g = np.asarray(g, float)
    l = np.asarray(l, float)
    if g.shape != l.shape or g.ndim != 1:
        raise ValueError("invalid_point_errors")
    if np.any(g_ok & (~np.isfinite(g) | (g < 0))) or np.any(
        l_ok & (~np.isfinite(l) | (l < 0))
    ):
        raise ValueError("invalid_known_errors")
    scale = np.maximum(np.where(g_ok, g, 0), np.where(l_ok, l, 0))
    gs = np.divide(g, scale, out=np.zeros_like(g), where=g_ok & (scale > 0))
    ls = np.divide(l, scale, out=np.zeros_like(l), where=l_ok & (scale > 0))
    exact = np.divide(
        gs, gs + ls, out=np.full_like(g, 0.5), where=g_ok & l_ok & (gs + ls > 0)
    )
    both = g_ok & l_ok
    lo = np.where(both, exact, 0.0)
    hi = np.where(both, exact, 1.0)
    return lo, hi


def aggregate(g, l, g_ok, l_ok):
    scores = {}
    bounds = {}
    for j, name in enumerate(ARMS):
        if name == "transfer_share":
            lo, hi = point_bounds(g, l, g_ok, l_ok)
        elif name == "transfer_global":
            lo = np.where(g_ok, np.minimum(g, CAP), 0.0)
            hi = np.where(g_ok, np.minimum(g, CAP), CAP)
        else:
            lo = np.where(l_ok, np.minimum(l, CAP), 0.0)
            hi = np.where(l_ok, np.minimum(l, CAP), CAP)
        bounds[name] = [float(np.median(lo)), float(np.median(hi))]
        scores[name] = bounds[name]
    return scores, bounds


def maximum(values):
    values = [v for v in values if v is not None]
    return [max(v[j] for v in values) for j in range(2)] if values else None


def decide(value, cutoff):
    if value is None or cutoff is None:
        return None
    if value[0] > cutoff[1]:
        return True
    if value[1] <= cutoff[0]:
        return False
    return None
