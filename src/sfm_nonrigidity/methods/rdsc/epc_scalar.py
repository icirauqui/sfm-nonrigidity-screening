"""Primary EPC scalar readout, preserving the exact retained arithmetic path.

SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
from . import rdsc as DET, controls as NC


def scalar_scores(x, d, index, length):
    raw = DET.detection_scores(
        x, d, neighbors=8, shuffle_count=20, seed=20260904 + index
    )
    _, ni = NC.score(x, x + d, index, DET)
    assert np.isclose(
        raw["coherent_affine_q95"], ni["amplitude"], rtol=1e-10, atol=1e-12
    )
    ratio = raw["spatial_scale"] / length
    a = raw["coherent_affine_q95"] * ratio
    b = ni["shuffle_reference"] * ratio
    return dict(
        rdsc=float(a * (a / max(b, np.finfo(float).tiny))),
        raw_median=float(raw["raw_median"] * ratio),
        raw_q95=float(raw["raw_q95"] * ratio),
        coherent_affine_q95=float(a),
        coherence_excess=float(a / max(b, np.finfo(float).tiny)),
        coherent_minus_shuffle=float(a - b),
        coherent_squared_minus_shuffle_squared=float(a * a - b * b),
    )
