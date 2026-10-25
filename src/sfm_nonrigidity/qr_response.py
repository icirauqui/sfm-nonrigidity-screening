"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from . import linear as L


def eliminate(A, B, E):
    """Eliminate three point coordinates without subtracting large Gram matrices."""
    _, sv, vt = np.linalg.svd(A, full_matrices=False)
    if len(sv) != 3 or sv[0] <= 0:
        raise ValueError("zero_information_point")
    floor = 1e-12 * sv[0] ** 2
    deficit = np.maximum(floor - sv**2, 0)
    regularized = bool(np.any(deficit > 0))
    if regularized:
        A = np.concatenate([A, np.sqrt(deficit)[:, None] * vt])
        B = np.concatenate([B, np.zeros((3, B.shape[1]))])
        E = np.concatenate([E, np.zeros((3, E.shape[1]))])
    Q, R = np.linalg.qr(A, mode="complete")
    return (
        R[:3, :3],
        Q[:, :3].T @ B,
        Q[:, :3].T @ E,
        Q[:, 3:].T @ B,
        Q[:, 3:].T @ E,
        regularized,
    )


def response(model, samples=64):
    poses, observations, camera_ids, cameras, points, tracks = model
    ids = sorted(poses)
    camera_index = {iid: i for i, iid in enumerate(ids)}
    nc = len(ids)
    npnt = len(points)
    center = points.mean(0)
    scale = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.9))
    if scale <= 1e-12 or nc < 2:
        raise ValueError("degenerate_bundle")
    x = (points - center) / scale
    centers = {iid: (poses[iid][1] - center) / scale for iid in ids}
    tr = [sorted(set(t)) for t in tracks]
    offset = np.r_[0, np.cumsum([len(t) for t in tr])]
    idx = np.repeat(np.arange(npnt), np.diff(offset))
    obs = np.array([o for t in tr for o in t], int)
    if any(i not in poses for i in set(obs[:, 0])):
        raise ValueError("unregistered_track_camera")
    ci = np.array([camera_index[i] for i in obs[:, 0]], int)
    Jp = np.empty((len(idx), 2, 3))
    Jc = np.empty((len(idx), 2, 6))
    squared_error = 0.0
    negative_depths = 0
    for iid in ids:
        at = np.where(obs[:, 0] == iid)[0]
        rot = poses[iid][0]
        cam = (x[idx[at]] - centers[iid]) @ rot.T
        kind, p = cameras[camera_ids[iid]]
        J, pred, negative = L.projection_linearization(cam, kind, p)
        negative_depths += negative
        Jp[at] = J @ rot
        Jc[at] = J @ np.concatenate(
            [-L.skew(cam), np.tile(np.eye(3), (len(at), 1, 1))], axis=2
        )
        squared_error += float(np.sum((pred - observations[iid][obs[at, 1], :2]) ** 2))
    S = np.zeros((6 * nc, 6 * nc))
    bc = np.zeros((6 * nc, samples))
    bp = np.empty((npnt, 3, samples))
    triangular = np.empty((npnt, 3, 3))
    wr = []
    wc = []
    wv = []
    point_regularized = 0
    for i in range(npnt):
        a, b = offset[i : i + 2]
        count = b - a
        unique, inverse = np.unique(ci[a:b], return_inverse=True)
        cols = (6 * unique[:, None] + np.arange(6)).ravel()
        A = Jp[a:b].reshape(2 * count, 3)
        B = np.zeros((2 * count, len(cols)))
        B[
            np.arange(2 * count)[:, None],
            np.repeat(6 * inverse, 2)[:, None] + np.arange(6),
        ] = Jc[a:b].reshape(2 * count, 6)
        E = L.probes(obs[a:b, 0], obs[a:b, 1], samples).reshape(2 * count, samples)
        R, topB, topE, reducedB, reducedE, reg = eliminate(A, B, E)
        point_regularized += reg
        triangular[i] = R
        bp[i] = topE
        S[np.ix_(cols, cols)] += reducedB.T @ reducedB
        bc[cols] += reducedB.T @ reducedE
        wr.append(np.repeat(np.arange(3 * i, 3 * i + 3), len(cols)))
        wc.append(np.tile(cols, 3))
        wv.append(topB.ravel())
    W = sparse.coo_matrix(
        (np.concatenate(wv), (np.concatenate(wr), np.concatenate(wc))),
        shape=(3 * npnt, 6 * nc),
    ).tocsr()
    second = ids[1]
    direction = poses[second][0] @ (centers[ids[0]] - centers[second])
    axis = int(np.argmax(abs(direction)))
    if abs(direction[axis]) <= 1e-12:
        raise ValueError("coincident_gauge_cameras")
    fixed = list(range(6)) + [6 + 3 + axis]
    free = np.array([i for i in range(6 * nc) if i not in fixed])
    reduced = S[np.ix_(free, free)]
    reduced = (reduced + reduced.T) / 2
    values, V = np.linalg.eigh(reduced)
    if values[-1] <= 0 or values[0] < -1e-7 * values[-1]:
        raise ValueError("indefinite_square_root_system")
    floor = 1e-10 * values[-1]
    reg_pose = int(np.sum(values < floor))
    values = np.maximum(values, floor)
    dc = np.zeros((6 * nc, samples))
    dc[free] = V @ ((V.T @ bc[free]) / values[:, None])
    rhs = bp - (W @ dc).reshape(npnt, 3, samples)
    dp = np.linalg.solve(triangular, rhs) * scale
    info = dict(
        regularized_points=point_regularized,
        regularized_pose_directions=reg_pose,
        pose_condition=float(values[-1] / values[0]),
        points=npnt,
        cameras=nc,
        observations=len(idx),
        samples=samples,
        scale=scale,
        gauge_camera_ids=ids[:2],
        gauge_translation_axis=axis,
        squared_reprojection_error=squared_error,
        linear_dof=2 * len(idx) - 3 * npnt - 6 * nc + 7,
        negative_depth_observations=negative_depths,
        solver="square_root_point_elimination",
    )
    return dict(
        point=dp, pose=dc.reshape(nc, 6, samples), ids=ids, scale=scale, info=info
    )
