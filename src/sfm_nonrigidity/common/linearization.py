"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import linear as L


def linearize(model):
    poses, observations, camera_ids, cameras, points, tracks = model
    ids = sorted(poses)
    lookup = {v: k for k, v in enumerate(ids)}
    center = points.mean(0)
    scale = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.9))
    if scale <= 1e-12:
        raise ValueError("degenerate_bundle")
    tr = [sorted(set(t)) for t in tracks]
    offset = np.r_[0, np.cumsum([len(t) for t in tr])]
    pi = np.repeat(np.arange(len(points)), np.diff(offset))
    obs = np.asarray([o for t in tr for o in t], int)
    if not len(obs) or len(set(map(tuple, obs))) != len(obs):
        raise ValueError("empty_or_duplicate_measurements")
    ci = np.array([lookup[i] for i in obs[:, 0]])
    jp = np.empty((len(obs), 2, 3))
    jc = np.empty((len(obs), 2, 6))
    for i in ids:
        at = np.flatnonzero(obs[:, 0] == i)
        r, c, _ = poses[i]
        cam = (points[pi[at]] - c) / scale @ r.T
        j, _, _ = L.projection_linearization(cam, *cameras[camera_ids[i]])
        jp[at] = j @ r
        jc[at] = j @ np.concatenate(
            [-L.skew(cam), np.tile(np.eye(3), (len(at), 1, 1))], axis=2
        )
    return dict(
        ids=ids, lookup=lookup, scale=scale, obs=obs, offset=offset, ci=ci, jp=jp, jc=jc
    )
