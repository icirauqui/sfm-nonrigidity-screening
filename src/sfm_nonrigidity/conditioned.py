"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import linear as L
from .qr_response import eliminate


def signed_projection(cam, kind, p):
    if kind not in ("PINHOLE", "OPENCV_FISHEYE"):
        raise ValueError("unsupported_camera_model")
    if np.any(cam[:, 2] == 0):
        raise ValueError("zero_camera_depth")
    sign = np.where(cam[:, 2] < 0, -1.0, 1.0)
    safe = cam * sign[:, None]
    return (
        L.P.jacobian(safe, kind, p) * sign[:, None, None],
        L.P.G.project(safe, kind, p),
        int(np.sum(sign < 0)),
    )


def response(base, local, rb, samples=128):
    poses, observations, camera_ids, cameras, points, tracks = local
    ids = sorted(poses)
    index = {i: k for k, i in enumerate(ids)}
    global_index = {i: k for k, i in enumerate(rb["ids"])}
    if not set(ids) <= set(base[0]):
        raise ValueError("conditioned_camera_not_in_global")
    s, R, t = L.P.G.alignment(base[0], poses)
    extent = float(np.quantile(np.linalg.norm(points - points.mean(0), axis=1), 0.9))
    center = points.mean(0)
    if extent <= 1e-12:
        raise ValueError("degenerate_bundle")
    rotation_error = max(
        float(np.max(abs(poses[i][0] - base[0][i][0] @ R))) for i in ids
    )
    position_error = max(
        float(np.linalg.norm(s * R @ poses[i][1] + t - base[0][i][1])) for i in ids
    ) / (s * extent)
    if rotation_error > 1e-5 or position_error > 1e-5:
        raise ValueError("not_global_conditioned_camera_poses")
    dc = np.zeros((len(ids), 6, samples))
    for k, i in enumerate(ids):
        pose = rb["pose"][global_index[i]]
        deltaC = -rb["scale"] * base[0][i][0].T @ pose[3:]
        localC = R.T @ deltaC / s
        dc[k, :3] = pose[:3]
        dc[k, 3:] = -poses[i][0] @ localC / extent
    tr = [sorted(set(t)) for t in tracks]
    offset = np.r_[0, np.cumsum([len(t) for t in tr])]
    pi = np.repeat(np.arange(len(points)), np.diff(offset))
    obs = np.array([o for t in tr for o in t], int)
    jp = np.empty((len(pi), 2, 3))
    jc = np.empty((len(pi), 2, 6))
    x = (points - center) / extent
    ci = np.array([index[i] for i in obs[:, 0]])
    negative = 0
    for i in ids:
        at = np.flatnonzero(obs[:, 0] == i)
        rot = poses[i][0]
        cam = (x[pi[at]] - (poses[i][1] - center) / extent) @ rot.T
        J, _, ng = signed_projection(cam, *cameras[camera_ids[i]])
        negative += ng
        jp[at] = J @ rot
        jc[at] = J @ np.concatenate(
            [-L.skew(cam), np.tile(np.eye(3), (len(at), 1, 1))], axis=2
        )
    dp = np.empty((len(points), 3, samples))
    regularized = 0
    for k in range(len(points)):
        a, b = offset[k : k + 2]
        A = jp[a:b].reshape(-1, 3)
        E = L.probes(obs[a:b, 0], obs[a:b, 1], samples) - np.einsum(
            "nai,nis->nas", jc[a:b], dc[ci[a:b]]
        )
        E = E.reshape(-1, samples)
        tri, _, rhs, _, _, reg = eliminate(A, np.empty((len(A), 0)), E)
        dp[k] = np.linalg.solve(tri, rhs) * extent
        regularized += reg
    return dict(
        point=dp,
        pose=dc,
        ids=ids,
        scale=extent,
        info=dict(
            solver="global_camera_conditioned_point_QR",
            points=len(points),
            cameras=len(ids),
            observations=len(obs),
            regularized_points=regularized,
            regularized_pose_directions=0,
            negative_depth_observations=negative,
            rotation_error=rotation_error,
            relative_camera_position_error=position_error,
            samples=samples,
        ),
    )
