"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def read_cameras(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError("Expected COLMAP image header")
            image = int(fields[0])
            q = np.array(fields[1:5], float)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            translation = np.array(fields[5:8], float)
            result[image] = (rotation, -rotation.T @ translation, fields[9])
            next(
                stream
            )  # tracks are read from points3D.txt, preserve image row pairing
    return result


def read_points(path):
    xyz = []
    tracks = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            xyz.append([float(x) for x in fields[1:4]])
            values = list(map(int, fields[8:]))
            if len(values) % 2:
                raise ValueError("Odd track fields")
            tracks.append(tuple(zip(values[0::2], values[1::2])))
    return np.array(xyz), tracks


def alignment(base, floating):
    ids = sorted(set(base) & set(floating))
    if len(ids) < 2:
        raise ValueError("Need at least two common cameras")
    if any(base[i][2] != floating[i][2] for i in ids):
        raise ValueError("Image ID/name collision")
    total = sum((base[i][0].T @ floating[i][0] for i in ids), start=np.zeros((3, 3)))
    u, _, vt = np.linalg.svd(total)
    fix = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ fix @ vt
    bc = np.array([base[i][1] for i in ids])
    fc = np.array([floating[i][1] for i in ids])
    bm, fm = bc.mean(0), fc.mean(0)
    rotated = (fc - fm) @ rotation.T
    den = np.sum(rotated**2)
    if den <= 1e-18:
        raise ValueError("Degenerate camera baseline")
    scale = np.sum((bc - bm) * rotated) / den
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    return scale, rotation, bm - scale * rotation @ fm


def intersections(base_tracks, local_tracks):
    lookup = defaultdict(list)
    for j, track in enumerate(local_tracks):
        for obs in set(track):
            lookup[obs].append(j)
    candidates = []
    for i, track in enumerate(base_tracks):
        counts = Counter(j for obs in set(track) for j in lookup.get(obs, ()))
        candidates.extend((i, j) for j, n in sorted(counts.items()) if n >= 3)
    da = Counter(i for i, j in candidates)
    db = Counter(j for i, j in candidates)
    return [(i, j) for i, j in candidates if da[i] == db[j] == 1]
