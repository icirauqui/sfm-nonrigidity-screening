"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import global_projection as stream
from ...common import linearization as E


def prepare(base):
    if any(camera[0] != "PINHOLE" for camera in base[3].values()):
        raise ValueError("requires_signed_pinhole")
    lin = E.linearize(base)
    points = base[4]
    p = (points - points.mean(0)) / lin["scale"]
    features = np.column_stack(
        (
            np.ones(len(p)),
            p,
            p * p,
            np.sqrt(2) * p[:, 0] * p[:, 1],
            np.sqrt(2) * p[:, 0] * p[:, 2],
            np.sqrt(2) * p[:, 1] * p[:, 2],
        )
    )
    ordered = sorted(base[0], key=lambda i: base[0][i][2])
    times = {i: 2 * j / (len(ordered) - 1) - 1 for j, i in enumerate(ordered)}
    tau = np.array([times[int(i)] for i in lin["obs"][:, 0]])
    temporal = np.column_stack((tau, tau * tau))
    pi = np.repeat(np.arange(len(points)), np.diff(lin["offset"]))
    energy = np.zeros((10, 2, 3))
    for lo in range(0, len(pi), 65536):
        hi = min(lo + 65536, len(pi))
        energy += np.einsum(
            "nf,nt,na->fta",
            features[pi[lo:hi]] ** 2,
            temporal[lo:hi] ** 2,
            np.sum(lin["jp"][lo:hi] ** 2, axis=1),
            optimize=True,
        )
    rms = np.sqrt(energy / len(pi)).ravel()
    if not np.isfinite(rms).all() or np.any(rms <= 1e-12):
        raise ValueError("physical_mode_projection")
    direction = (
        base[0][lin["ids"][1]][0]
        @ (base[0][lin["ids"][0]][1] - base[0][lin["ids"][1]][1])
        / lin["scale"]
    )
    axis = int(np.argmax(abs(direction)))
    if abs(direction[axis]) <= 1e-12:
        raise ValueError("coincident_gauge_cameras")
    free = np.array(
        [i for i in range(6 * len(lin["ids"])) if i not in {*range(6), 9 + axis}]
    )
    compact = np.full(6 * len(lin["ids"]), -1, int)
    compact[free] = np.arange(len(free))
    return dict(
        lin=lin,
        features=features,
        temporal=temporal,
        rms=rms,
        compact=compact,
        free=free,
    )


def blocks(base, prepared, datum=None):
    lin = prepared["lin"]
    for i in range(len(base[4])):
        a, b = lin["offset"][i : i + 2]
        count = b - a
        point = lin["jp"][a:b].reshape(2 * count, 3)
        unique, inverse = np.unique(lin["ci"][a:b], return_inverse=True)
        columns = (6 * unique[:, None] + np.arange(6)).ravel()
        camera = np.zeros((2 * count, len(columns)))
        camera[
            np.arange(2 * count)[:, None],
            np.repeat(6 * inverse, 2)[:, None] + np.arange(6),
        ] = lin["jc"][a:b].reshape(2 * count, 6)
        mapped = prepared["compact"][columns]
        keep = mapped >= 0
        camera = camera[:, keep]
        columns = mapped[keep]
        h = (
            lin["jp"][a:b, :, None, None, :]
            * prepared["features"][i][None, None, :, None, None]
            * prepared["temporal"][a:b, None, None, :, None]
        ).reshape(2 * count, 60) / prepared["rms"]
        if datum is not None:
            y = datum[a:b].reshape(-1)
        else:
            y = []
            for iid, key in lin["obs"][a:b]:
                r, c, _ = base[0][int(iid)]
                cam = r @ (base[4][i] - c)
                intrinsics = base[3][base[2][int(iid)]][1]
                prediction = cam[:2] / cam[2] * intrinsics[:2] + intrinsics[2:]
                y.extend(base[1][int(iid)][int(key), :2] - prediction)
            y = np.asarray(y)
        yield point, camera, np.column_stack((h, y)), columns, y


def calculate(base, datum=None, progress=False):
    prepared = prepare(base)
    acc = stream.Accumulator(len(prepared["free"]))
    norms = []
    raw_energy = 0.0
    for i, (a, b, h, cols, y) in enumerate(blocks(base, prepared, datum)):
        acc.add(a, b, h, cols)
        raw_energy += float(y @ y)
        norms.extend(np.linalg.norm(y.reshape(-1, 2), axis=1))
        if progress and (i + 1) % 10000 == 0:
            print("RAW PIXEL POINTS", i + 1, "/", len(base[4]), flush=True)
    result = acc.close()
    out = stream.score(result, raw_energy)
    v = np.asarray(norms)
    scores = dict(
        pixel_median=float(np.median(v)),
        pixel_q95=float(np.quantile(v, 0.95)),
        null_rms=float(
            np.sqrt(out["sse0"] / (result["residual_dimension"] - out["affine_rank"]))
        ),
        mode_gain=out["gain"],
    )
    if out["status"] == "SCORED":
        scores["raw_pixel_F_stream_r0"] = out["score"]
    return result, out, scores, prepared["rms"], raw_energy
