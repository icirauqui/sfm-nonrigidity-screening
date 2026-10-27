"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
import scipy.linalg as la


def svd(a, full=False):
    try:
        return la.svd(a, full_matrices=full, lapack_driver="gesdd")
    except la.LinAlgError:
        return la.svd(a, full_matrices=full, lapack_driver="gesvd")


class Accumulator:
    def __init__(self, cameras, width=61):
        self.s = np.zeros((cameras, cameras))
        self.cross = np.zeros((cameras, width))
        self.mean = np.zeros((width, width))
        self.point_rank = 0
        self.rows = 0
        self.points = 0

    def add(self, point, camera, mean, columns):
        singular = svd(point)[1]
        if (
            len(singular) != 3
            or singular[0] <= 0
            or singular[-1] ** 2 <= 1e-12 * singular[0] ** 2
        ):
            raise ValueError("point_information_rank")
        q, _ = la.qr(point, mode="full")
        r = q[:, 3:].T
        b = r @ camera
        h = r @ mean
        self.s[np.ix_(columns, columns)] += b.T @ b
        self.cross[columns] += b.T @ h
        self.mean += h.T @ h
        self.rows += len(point)
        self.point_rank += 3
        self.points += 1

    def close(self):
        values, vectors = la.eigh((self.s + self.s.T) / 2)
        top = max(float(values[-1]), 0.0)
        if values[0] < -1e-8 * max(top, np.finfo(float).tiny):
            raise ValueError("camera_gram_indefinite")
        keep = values > 1e-10 * top if top else np.zeros(len(values), bool)
        solution = (
            (vectors[:, keep] / values[keep]) @ (vectors[:, keep].T @ self.cross)
            if keep.any()
            else np.zeros_like(self.cross)
        )
        support = float(la.norm(vectors[:, ~keep].T @ self.cross)) / max(
            np.sqrt(np.trace(self.s) * np.trace(self.mean)), np.finfo(float).tiny
        )
        if support > 1e-8:
            raise ValueError("camera_null_rhs_support")
        result = self.mean - self.cross.T @ solution
        result = (result + result.T) / 2
        return dict(
            gram=result,
            camera_rank=int(keep.sum()),
            point_rank=self.point_rank,
            points=self.points,
            observations=self.rows // 2,
            residual_dimension=self.rows - self.point_rank - int(keep.sum()),
            support_error=support,
            camera_eigenvalues=values,
            camera_gram=self.s,
            camera_cross=self.cross,
            mean_gram=self.mean,
        )


def score(result, raw_energy):
    gram = result["gram"]
    values, vectors = la.eigh((gram + gram.T) / 2)
    norm = max(float(np.max(abs(values))), np.finfo(float).tiny)
    if values[0] < -1e-7 * norm:
        raise ValueError("mean_gram_indefinite")
    root = np.sqrt(np.maximum(values, 0))[:, None] * vectors.T
    cutoff = 1e-8 * max(float(la.eigvalsh(gram[:60, :60])[-1]), 0.0)
    u, s, _ = svd(root[:, :24])
    affine = int(np.sum(s * s > cutoff))
    residual = root - u[:, :affine] @ (u[:, :affine].T @ root)
    u, s, _ = svd(residual[:, 24:60])
    q = int(np.sum(s * s > cutoff))
    datum = residual[:, -1]
    coefficients = u[:, :q].T @ datum
    e = datum - u[:, :q] @ coefficients
    gain = float(coefficients @ coefficients)
    sse0 = float(datum @ datum)
    sse1 = float(e @ e)
    nu = result["residual_dimension"] - affine - q
    if not q:
        status = "NO_ALTERNATIVE"
    elif nu <= 0:
        status = "NO_RESIDUAL_DF"
    elif np.sqrt(sse1) <= 1e-10 * np.sqrt(raw_energy):
        status = "ZERO_RESIDUAL_ENERGY"
    else:
        status = "SCORED"
    return dict(
        status=status,
        score=(gain / q) / (sse1 / nu) if status == "SCORED" else None,
        affine_rank=affine,
        quadratic_rank=q,
        residual_df=nu,
        sse0=sse0,
        sse1=sse1,
        gain=gain,
        physical_cutoff=cutoff,
        minimum_mean_eigenvalue=float(values[0]),
        mean_spectral_norm=norm,
    )
