"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np


def keys(tracks, indices):
    signatures = [tuple(sorted(set(tracks[int(i)]))) for i in indices]
    if len(set(signatures)) != len(signatures):
        raise ValueError("nonunique_observation_signature")
    order = sorted(range(len(signatures)), key=signatures.__getitem__)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank
