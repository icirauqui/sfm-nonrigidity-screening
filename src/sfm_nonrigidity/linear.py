"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from . import precision as P


def skew(x):
    out = np.zeros((len(x), 3, 3))
    out[:, 0, 1] = -x[:, 2]
    out[:, 0, 2] = x[:, 1]
    out[:, 1, 0] = x[:, 2]
    out[:, 1, 2] = -x[:, 0]
    out[:, 2, 0] = -x[:, 1]
    out[:, 2, 1] = x[:, 0]
    return out


def probes(image, keypoint, samples, seed=20260924):
    # Counter-derived Rademacher probes; the same image/keypoint receives the same
    # perturbation in global and local models, regardless of file ordering.
    image = np.asarray(image, dtype=np.uint64)
    keypoint = np.asarray(keypoint, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = (
            image[:, None, None] * np.uint64(0x9E3779B97F4A7C15)
            + keypoint[:, None, None] * np.uint64(0xD1B54A32D192ED03)
            + np.arange(2, dtype=np.uint64)[None, :, None]
            * np.uint64(0x94D049BB133111EB)
            + np.arange(samples, dtype=np.uint64)[None, None, :]
            * np.uint64(0xBF58476D1CE4E5B9)
            + np.uint64(seed)
        )
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> 31)
    return 2 * (x & 1).astype(float) - 1


def projection_linearization(cam, kind, p):
    # The retained thesis PINHOLE cost divides by signed z; negative nuisance
    # observations must not cause an unrelated matched window to be dropped.
    if np.any(cam[:, 2] == 0):
        raise ValueError("zero_camera_depth")
    negative = cam[:, 2] < 0
    if np.any(negative) and kind != "PINHOLE":
        raise ValueError("unsupported_negative_depth_model")
    sign = np.where(negative, -1.0, 1.0)
    safe = cam * sign[:, None]
    J = P.jacobian(safe, kind, p) * sign[:, None, None]
    prediction = P.G.project(safe, kind, p)
    return J, prediction, int(negative.sum())


def schur_solve(H, U, W, bp, bc, free):
    inv, reg = P.inverse_psd(H)
    block = sparse.bsr_matrix(
        (inv, np.arange(len(H)), np.arange(len(H) + 1)), shape=(3 * len(H), 3 * len(H))
    )
    T = (block @ W).tocsr()
    S = U.toarray() - (W.T @ T).toarray()
    S = (S + S.T) / 2
    reduced = S[np.ix_(free, free)]
    values, V = np.linalg.eigh(reduced)
    if values[-1] <= 0 or values[0] < -1e-7 * values[-1]:
        raise ValueError("indefinite_schur_complement")
    floor = 1e-10 * values[-1]
    reg_pose = int(np.sum(values < floor))
    values = np.maximum(values, floor)
    projected = block @ bp
    rhs = bc - W.T @ projected
    pose = np.zeros_like(bc)
    pose[free] = V @ ((V.T @ rhs[free]) / values[:, None])
    point = block @ (bp - W @ pose)
    return (
        point,
        pose,
        dict(
            regularized_points=reg,
            regularized_pose_directions=reg_pose,
            pose_condition=float(values[-1] / values[0]),
        ),
    )


def normal_response(model, samples=64):
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
    idx = np.repeat(np.arange(npnt), [len(t) for t in tr])
    obs = np.array([o for t in tr for o in t], int)
    if any(i not in poses for i in set(obs[:, 0])):
        raise ValueError("unregistered_track_camera")
    ci = np.array([camera_index[i] for i in obs[:, 0]], int)
    Jp = np.empty((len(idx), 2, 3))
    Jc = np.empty((len(idx), 2, 6))
    U = np.zeros((nc, 6, 6))
    bc = np.zeros((nc, 6, samples))
    bp = np.zeros((npnt, 3, samples))
    squared_error = 0.0
    negative_depths = 0
    for iid in ids:
        at = np.where(obs[:, 0] == iid)[0]
        pointids = idx[at]
        rot = poses[iid][0]
        cam = (x[pointids] - centers[iid]) @ rot.T
        kind, p = cameras[camera_ids[iid]]
        J, prediction, negative = projection_linearization(cam, kind, p)
        negative_depths += negative
        Jp[at] = J @ rot
        Jc[at] = J @ np.concatenate(
            [-skew(cam), np.tile(np.eye(3), (len(at), 1, 1))], axis=2
        )
        k = camera_index[iid]
        U[k] = np.einsum("nai,naj->ij", Jc[at], Jc[at])
        noise = probes(obs[at, 0], obs[at, 1], samples)
        np.add.at(bp, pointids, np.einsum("nai,nas->nis", Jp[at], noise))
        bc[k] = np.einsum("nai,nas->is", Jc[at], noise)
        residual = prediction - observations[iid][obs[at, 1], :2]
        squared_error += float(np.sum(residual**2))
    H = P.scatter_gram(Jp.swapaxes(1, 2), idx, npnt)
    cross = np.einsum("nai,naj->nij", Jp, Jc)
    rows = np.broadcast_to(
        3 * idx[:, None, None] + np.arange(3)[None, :, None], cross.shape
    ).reshape(-1)
    cols = np.broadcast_to(
        6 * ci[:, None, None] + np.arange(6)[None, None, :], cross.shape
    ).reshape(-1)
    W = sparse.coo_matrix(
        (cross.reshape(-1), (rows, cols)), shape=(3 * npnt, 6 * nc)
    ).tocsr()
    U = sparse.block_diag(list(U), format="csr")
    # Gauge: first camera pose; one second-camera translation direction with nonzero scale motion.
    second = ids[1]
    direction = poses[second][0] @ (centers[ids[0]] - centers[second])
    axis = int(np.argmax(abs(direction)))
    if abs(direction[axis]) <= 1e-12:
        raise ValueError("coincident_gauge_cameras")
    fixed = list(range(6)) + [6 + 3 + axis]
    free = np.array([i for i in range(6 * nc) if i not in fixed])
    dp, dc, info = schur_solve(
        H, U, W, bp.reshape(3 * npnt, samples), bc.reshape(6 * nc, samples), free
    )
    info.update(
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
    )
    return dict(
        point=dp.reshape(npnt, 3, samples) * scale,
        pose=dc.reshape(nc, 6, samples),
        ids=ids,
        scale=scale,
        info=info,
    )


def exp_rotation(w):
    angle = float(np.linalg.norm(w))
    K = skew(np.asarray(w)[None, :])[0]
    if angle < 1e-10:
        return np.eye(3) + K + 0.5 * K @ K
    return (
        np.eye(3) + np.sin(angle) / angle * K + (1 - np.cos(angle)) / angle**2 * K @ K
    )


def perturbed_cameras(model, response, sample, step):
    result = {}
    for k, iid in enumerate(response["ids"]):
        R, C, name = model[0][iid]
        w = response["pose"][k, :3, sample]
        v = response["pose"][k, 3:, sample]
        result[iid] = (
            exp_rotation(step * w) @ R,
            C - step * response["scale"] * (R.T @ v),
            name,
        )
    return result


def response(model, samples=64):
    from .qr_response import response as solve_qr

    return solve_qr(model, samples)
