"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np


def camera(kind, p):
    if kind != "PINHOLE":
        raise ValueError("unsupported_epipolar_camera:" + kind)
    if len(p) != 4 or not np.isfinite(p).all() or min(p[:2]) <= 0:
        raise ValueError("invalid_intrinsics")
    fx, fy, cx, cy = p
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def grouped_views(base, local, ib, il):
    shared = sorted(set(base[5][ib]) & set(local[5][il]))
    if len(shared) < 3:
        raise ValueError("insufficient_shared_observations")
    groups = {}
    for iid, key in shared:
        groups.setdefault(iid, []).append(key)
    if len(groups) < 2:
        raise ValueError("fewer_than_two_shared_views")
    views = []
    for iid in sorted(groups, key=lambda i: (base[0][i][2], i)):
        keys = groups[iid]
        if base[0][iid][2] != local[0][iid][2]:
            raise ValueError("image_identity_mismatch")
        x = base[1][iid][keys, :2]
        y = local[1][iid][keys, :2]
        if not np.allclose(x, y, rtol=0, atol=1e-8):
            raise ValueError("observation_coordinates_differ")
        views.append((iid, x[0] + np.mean(x - x[0], axis=0), len(keys)))
    return views
