"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np


def skew_batch(w):
    # input camera x 3 x samples; output camera x 3 x 3 x samples
    z = np.zeros((len(w), 3, 3, w.shape[-1]))
    z[:, 0, 1] = -w[:, 2]
    z[:, 0, 2] = w[:, 1]
    z[:, 1, 0] = w[:, 2]
    z[:, 1, 2] = -w[:, 0]
    z[:, 2, 0] = -w[:, 1]
    z[:, 2, 1] = w[:, 0]
    return z


def pose_changes(model, response, ids):
    n = response["pose"].shape[-1]
    lookup = {i: k for k, i in enumerate(response["ids"])}
    rot = np.array([model[0][i][0] for i in ids])
    pose = response["pose"][[lookup[i] for i in ids]]
    dc = -response["scale"] * np.einsum(
        "cij,cjs->cis", rot.transpose(0, 2, 1), pose[:, 3:]
    )
    return pose[:, :3], dc


def similarity_derivative(base, local, rb, rl, alignment):
    ids = sorted(set(base[0]) & set(local[0]))
    s, R, t = alignment(base[0], local[0])
    br = np.array([base[0][i][0] for i in ids])
    lr = np.array([local[0][i][0] for i in ids])
    bc = np.array([base[0][i][1] for i in ids])
    lc = np.array([local[0][i][1] for i in ids])
    wb, db = pose_changes(base, rb, ids)
    wl, dl = pose_changes(local, rl, ids)
    M = np.einsum("cji,cjk->ik", br, lr)
    dM = np.einsum("cji,cjks,ckl->ils", br, skew_batch(wl - wb), lr)
    H = (R.T @ M + M.T @ R) / 2
    v, Q = np.linalg.eigh(H)
    den = v[:, None] + v[None, :]
    off = ~np.eye(3, dtype=bool)
    if np.min(abs(den[off])) <= 1e-12 * np.max(abs(v)):
        raise ValueError("nonunique_alignment_polar_derivative")
    A = np.einsum("ij,jks->iks", R.T, dM)
    B = A - A.transpose(1, 0, 2)
    B = np.einsum("ij,jks,kl->ils", Q.T, B, Q)
    O = np.zeros_like(B)
    O[off] = B[off] / den[off, None]
    dR = np.einsum("ij,jks,kl->ils", R @ Q, O, Q.T)
    bm = bc.mean(0)
    lm = lc.mean(0)
    dbm = db.mean(0)
    dlm = dl.mean(0)
    xb = bc - bm
    xl = lc - lm
    dxb = db - dbm
    dxl = dl - dlm
    rotated = xl @ R.T
    dr = np.einsum("ijs,cj->cis", dR, xl) + np.einsum("ij,cjs->cis", R, dxl)
    D = np.sum(xl * xl)
    dD = 2 * np.einsum("ci,cis->s", xl, dxl)
    dN = np.einsum("cis,ci->s", dxb, rotated) + np.einsum("ci,cis->s", xb, dr)
    ds = (dN - s * dD) / D
    dt = dbm - (R @ lm)[:, None] * ds - s * np.einsum("ijs,j->is", dR, lm) - s * R @ dlm
    return s, R, t, ds, dR, dt


def aligned_response(base, local, rb, rl, bi, li, alignment):
    s, R, t, ds, dR, dt = similarity_derivative(base, local, rb, rl, alignment)
    y = local[4][li]
    x = base[4][bi]
    d = s * y @ R.T + t - x
    z = (
        (y @ R.T)[:, :, None] * ds
        + s * np.einsum("ijs,nj->nis", dR, y)
        + s * np.einsum("ij,njs->nis", R, rl["point"][li])
        + dt
        - rb["point"][bi]
    )
    zero = dict(rb, pose=np.zeros_like(rb["pose"]))
    _, _, _, dsg, dRg, _ = similarity_derivative(base, base, zero, rb, alignment)
    return z + d[:, :, None] * dsg + np.einsum("ijs,nj->nis", dRg, d)
