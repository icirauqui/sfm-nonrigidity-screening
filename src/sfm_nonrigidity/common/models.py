"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np
from .alignment import read_cameras, read_points, alignment


def project(points, kind, p):
    if np.any(points[:, 2] <= 0):
        raise ValueError("nonpositive_camera_depth")
    uv = points[:, :2] / points[:, 2, None]
    if kind == "OPENCV_FISHEYE":
        r = np.linalg.norm(uv, axis=1)
        theta = np.arctan(r)
        scale = np.divide(theta, r, out=np.ones_like(r), where=r > 0)
        radial = 1 + sum(p[4 + k] * theta ** (2 * k + 2) for k in range(4))
        uv = uv * (scale * radial)[:, None]
    elif kind != "PINHOLE":
        raise ValueError("unsupported_camera_model:" + kind)
    return uv * p[:2] + p[2:4]


@lru_cache(maxsize=3)
def model(images_path):
    path = Path(images_path)
    poses = read_cameras(path)
    observations = {}
    camera_ids = {}
    with path.open() as f:
        width = None
        for line in f:
            if line.startswith("#"):
                if "POINTS2D[]" in line:
                    width = 4 if "POINT3DD_ID" in line else 3
                continue
            if not line.strip():
                continue
            fields = line.split()
            iid = int(fields[0])
            camera_ids[iid] = int(fields[8])
            if width is None:
                raise ValueError("missing_observation_schema_header")
            data = np.fromstring(next(f), sep=" ")
            if len(data) % width:
                raise ValueError("observation_width_mismatch")
            observations[iid] = data.reshape(-1, width)[:, :3]
    cameras = {}
    for line in path.with_name("cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            f = line.split()
            cameras[int(f[0])] = (f[1], np.array(f[4:], float))
    points, tracks = read_points(path.with_name("points3D.txt"))
    return poses, observations, camera_ids, cameras, points, tracks
