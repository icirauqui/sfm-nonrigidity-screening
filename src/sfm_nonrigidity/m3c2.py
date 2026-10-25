"""Published M3C2 native distances with the retained independent PCA frontend.

Same mathematical settings as the qualified September comparator. Registration
input zero is not a physical uncertainty claim. Source lineage in the protocol.
"""

import logging
import numpy as np
from scipy.spatial import cKDTree
import py4dgeo

py4dgeo.set_num_threads(1)
logging.getLogger("py4dgeo").setLevel(logging.ERROR)
py4dgeo.Epoch.set_default_radius_search_tree("kdtree")
PARAMS = dict(
    normal_radii=[0.05, 0.10, 0.20],
    cyl_radius=0.05,
    max_distance=0.25,
    registration_error=0.0,
    robust_aggr=False,
)
METHODS = ("m3c2_abs_q95", "raw_median_m3c2_support", "raw_q95_m3c2_support")


class QualifiedM3C2(py4dgeo.M3C2):
    def directions(self):
        if self.corepoint_normals is not None:
            return self.corepoint_normals
        cloud = self.epochs[0].cloud
        core = self.corepoints
        tree = cKDTree(cloud)
        normals = np.zeros((len(core), 3))
        selected = np.zeros(len(core))
        quality = np.zeros(len(core))
        for radius in self.normal_radii:
            lists = tree.query_ball_point(core, radius, workers=1, return_sorted=True)
            cov = np.zeros((len(core), 3, 3))
            for i, ids in enumerate(lists):
                q = cloud[ids]
                delta = q - core[i]
                q = q[np.einsum("ij,ij->i", delta, delta) < radius * radius]
                if len(q) < 2:
                    continue
                z = q - q.mean(0)
                cov[i] = z.T @ z / (len(q) - 1)
            lam, V = np.linalg.eigh(cov)
            value = np.divide(
                lam[:, 1] - lam[:, 0],
                lam[:, 2],
                out=np.zeros(len(core)),
                where=lam[:, 2] > 0,
            )
            take = np.isfinite(value) & (value > quality)
            n = V[:, :, 0]
            n[n[:, 2] < 0] *= -1
            normals[take] = n[take]
            selected[take] = radius
            quality[take] = value[take]
        self.corepoint_normals = normals
        self._directions_radii = selected
        return normals


def compute(x, y):
    if (
        x.shape != y.shape
        or x.ndim != 2
        or x.shape[1] != 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid_paired_arrays")
    center = x.mean(0)
    length = float(np.quantile(np.linalg.norm(x - center, axis=1), 0.9))
    if length <= 1e-12:
        raise ValueError("degenerate_extent")
    a = np.ascontiguousarray((x - center) / length)
    b = np.ascontiguousarray((y - center) / length)
    alg = QualifiedM3C2(
        epochs=(py4dgeo.Epoch(a), py4dgeo.Epoch(b)), corepoints=a, **PARAMS
    )
    d, u = alg.run()
    normal = alg.directions()
    radii = alg.directions_radii()
    valid = (
        np.isfinite(d)
        & np.isfinite(normal).all(axis=1)
        & (abs(np.linalg.norm(normal, axis=1) - 1) <= 1e-6)
        & (u["num_samples1"] >= 5)
        & (u["num_samples2"] >= 5)
    )
    raw = np.linalg.norm(b - a, axis=1)
    reasons = []
    if len(x) < 100:
        reasons.append("fewer_than_100_pairs")
    if np.quantile(raw, 0.95) > 0.15:
        reasons.append("raw_q95_exceeds_0.15")
    if valid.sum() < 100:
        reasons.append("fewer_than_100_usable_corepoints")
    if valid.mean() < 0.8:
        reasons.append("usable_fraction_below_0.8")
    scores = (
        {}
        if reasons
        else dict(
            zip(
                METHODS,
                [
                    float(np.quantile(abs(d[valid]), 0.95)),
                    float(np.median(raw[valid])),
                    float(np.quantile(raw[valid], 0.95)),
                ],
            )
        )
    )
    return dict(
        scores=scores,
        reasons=reasons,
        usable=int(valid.sum()),
        requested=len(x),
        valid_fraction=float(valid.mean()),
        length=length,
        normal=normal,
        radii=radii,
        distance=d,
        n1=u["num_samples1"],
        n2=u["num_samples2"],
        valid=valid,
    )
