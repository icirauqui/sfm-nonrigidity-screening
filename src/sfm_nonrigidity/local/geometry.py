"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import numpy as np
from . import (
    analytic as A,
    conditioned as K,
    edges as Q,
    identity as ID,
    intervals as TV,
)
from ..common import linear as L
from ..common.alignment import alignment, intersections
from ..methods.rdsc import rdsc as DET

FIT = dict(
    max_iterations=80,
    gradient_tolerance=1e-7,
    initial_damping=1e-6,
    relative_information_floor=1e-12,
    initial_depth_floor=1e-12,
    roundoff_cost_factor=32.0,
)


def image_residual_terms(model, points, poses=None):
    """Per-point signed-pinhole SSE, gradient and Gauss–Newton information."""
    poses = model[0] if poses is None else poses
    n = len(points)
    cost = np.zeros(n)
    gradient = np.zeros((n, 3))
    information = np.zeros((n, 3, 3))
    valid = np.ones(n, bool)
    obs_count = np.zeros(n, int)
    tracks = [sorted(set(t)) for t in model[5]]
    pi = np.repeat(np.arange(n), [len(t) for t in tracks])
    obs = np.asarray([o for t in tracks for o in t], int)
    if not len(obs):
        return cost, gradient, information, np.zeros(n, bool), obs_count
    for iid in sorted(set(obs[:, 0])):
        at = np.flatnonzero(obs[:, 0] == iid)
        point_ids, keys = pi[at], obs[at, 1]
        if iid not in poses:
            valid[point_ids] = False
            continue
        kind, intrinsics = model[3][model[2][iid]]
        if kind != "PINHOLE":
            raise ValueError("conditioning_requires_pinhole")
        rotation, center, _ = poses[iid]
        camera = (points[point_ids] - center) @ rotation.T
        ok = np.isfinite(camera).all(axis=1) & (
            np.abs(camera[:, 2]) > FIT["initial_depth_floor"]
        )
        valid[point_ids[~ok]] = False
        if not ok.any():
            continue
        point_ids, keys, camera = point_ids[ok], keys[ok], camera[ok]
        fx, fy, cx, cy = intrinsics
        z = camera[:, 2]
        prediction = camera[:, :2] / z[:, None] * [fx, fy] + [cx, cy]
        residual = prediction - model[1][iid][keys, :2]
        jac = np.zeros((len(z), 2, 3))
        jac[:, 0, 0], jac[:, 1, 1] = fx / z, fy / z
        jac[:, 0, 2], jac[:, 1, 2] = (
            -fx * camera[:, 0] / z**2,
            -fy * camera[:, 1] / z**2,
        )
        jac = jac @ rotation
        np.add.at(cost, point_ids, np.sum(residual**2, axis=1))
        np.add.at(gradient, point_ids, np.einsum("nai,na->ni", jac, residual))
        np.add.at(information, point_ids, np.einsum("nai,naj->nij", jac, jac))
        np.add.at(obs_count, point_ids, 1)
    valid &= obs_count >= 2
    return cost, gradient, information, valid, obs_count


def normalized_gradient(cost, gradient, information):
    scale = np.sqrt(
        np.maximum(
            np.max(np.diagonal(information, axis1=1, axis2=2), axis=1),
            np.finfo(float).tiny,
        )
    )
    return np.max(np.abs(gradient), axis=1) / (scale * np.maximum(np.sqrt(cost), 1.0))


def accept_step(cost, new_cost, old_gradient, new_gradient, observations, pixel_scale):
    allowance = (
        FIT["roundoff_cost_factor"]
        * np.finfo(float).eps
        * max(float(pixel_scale), 1.0)
        * np.maximum(np.sqrt(2 * observations * cost), 1.0)
    )
    neutral = (new_cost <= cost + allowance) & (new_gradient < old_gradient)
    return (new_cost < cost) | neutral, neutral & (new_cost >= cost)


def conditioned_model(base, original):
    """Estimate retained local points with exactly inherited estimated cameras."""
    if not set(original[0]) <= set(base[0]):
        raise ValueError("unavailable_global_camera")
    for iid in original[0]:
        if original[0][iid][2] != base[0][iid][2]:
            raise ValueError("image_identity_mismatch")
        if original[3][original[2][iid]][0] != base[3][base[2][iid]][
            0
        ] or not np.array_equal(
            original[3][original[2][iid]][1], base[3][base[2][iid]][1]
        ):
            raise ValueError("intrinsic_mismatch")
    s, rotation, translation = alignment(base[0], original[0])
    points = s * original[4] @ rotation.T + translation
    poses = {iid: base[0][iid] for iid in original[0]}
    model = (poses, original[1], original[2], original[3], points, original[5])
    cost, gradient, information, valid, counts = image_residual_terms(model, points)
    initial_cost = cost.copy()
    damping = np.full(len(points), FIT["initial_damping"])
    accepted = np.zeros(len(points), int)
    polished = np.zeros(len(points), int)
    pixel_scale = max(
        float(np.max(np.abs(a[:, :2]))) for a in original[1].values() if len(a)
    )
    for iteration in range(FIT["max_iterations"]):
        ng = normalized_gradient(cost, gradient, information)
        active = valid & (ng > FIT["gradient_tolerance"])
        if not active.any():
            break
        ids = np.flatnonzero(active)
        h = information[ids]
        scale = np.maximum(
            np.max(np.diagonal(h, axis1=1, axis2=2), axis=1), np.finfo(float).tiny
        )
        regularized = h + damping[ids, None, None] * scale[:, None, None] * np.eye(3)
        step = np.linalg.solve(regularized, gradient[ids, :, None])[..., 0]
        candidate = points.copy()
        candidate[ids] -= step
        new_cost, new_gradient, new_information, new_valid, _ = image_residual_terms(
            model, candidate
        )
        new_ng = normalized_gradient(new_cost, new_gradient, new_information)
        improve, neutral = accept_step(
            cost[ids], new_cost[ids], ng[ids], new_ng[ids], counts[ids], pixel_scale
        )
        improve &= new_valid[ids] & np.isfinite(new_cost[ids])
        win = ids[improve]
        points[win] = candidate[win]
        cost[win], gradient[win], information[win] = (
            new_cost[win],
            new_gradient[win],
            new_information[win],
        )
        accepted[win] += 1
        polished[win] += neutral[improve]
        damping[win] = np.maximum(damping[win] / 3, 1e-15)
        damping[ids[~improve]] = np.minimum(damping[ids[~improve]] * 10, 1e15)
    values = np.linalg.eigvalsh(information)
    identifiable = (values[:, -1] > 0) & (
        values[:, 0] > FIT["relative_information_floor"] * values[:, -1]
    )
    ng = normalized_gradient(cost, gradient, information)
    converged = (
        valid
        & identifiable
        & np.isfinite(points).all(axis=1)
        & (ng <= FIT["gradient_tolerance"])
    )
    model = (poses, original[1], original[2], original[3], points, original[5])
    return model, dict(
        initial_sse=initial_cost,
        final_sse=cost,
        normalized_gradient=ng,
        converged=converged,
        identifiable=identifiable,
        observations=counts,
        accepted_steps=accepted,
        roundoff_polish_steps=polished,
        iterations=iteration + 1,
    )


def transfer_bounds(base, local, pairs, inherited):
    if inherited:
        for iid in local[0]:
            if not np.array_equal(
                base[0][iid][0], local[0][iid][0]
            ) or not np.array_equal(base[0][iid][1], local[0][iid][1]):
                raise ValueError("declared_inheritance_camera_mismatch")
            if (
                base[0][iid][2] != local[0][iid][2]
                or base[3][base[2][iid]][0] != local[3][local[2][iid]][0]
                or not np.array_equal(
                    base[3][base[2][iid]][1], local[3][local[2][iid]][1]
                )
            ):
                raise ValueError("declared_inheritance_camera_mismatch")
    n = len(pairs)
    requests = {}
    for point, (ib, il) in enumerate(pairs):
        views = TV.T.E.grouped_views(base, local, ib, il)
        if len(views) < 3:
            continue
        chosen = [views[0], views[len(views) // 2], views[-1]]
        ids = tuple(v[0] for v in chosen)
        requests.setdefault(ids, []).append((point, np.array([v[1] for v in chosen])))
    errors = np.zeros((2, n))
    known = np.zeros((2, n), bool)
    for ids, rows in requests.items():
        indexes = np.asarray([r[0] for r in rows], int)
        xy = np.asarray([r[1] for r in rows])
        for role, model in enumerate((base, local)):
            if inherited and role == 1:
                errors[1, indexes], known[1, indexes] = (
                    errors[0, indexes],
                    known[0, indexes],
                )
                continue
            rotations = np.asarray([model[0][i][0] for i in ids])
            centers = np.asarray([model[0][i][1] for i in ids])
            intrinsics = np.asarray(
                [TV.T.E.camera(*model[3][model[2][i]]) for i in ids]
            )
            try:
                prediction = TV.T.transfer(
                    xy[:, 0], xy[:, 2], rotations, centers, intrinsics
                )
                errors[role, indexes] = np.sum((prediction - xy[:, 1]) ** 2, axis=1)
                known[role, indexes] = True
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                for index, observation in zip(indexes, xy):
                    prediction, ok = TV.transfer_one(
                        observation[0], observation[2], rotations, centers, intrinsics
                    )
                    errors[role, index] = (
                        float(np.sum((prediction - observation[1]) ** 2)) if ok else 0.0
                    )
                    known[role, index] = ok
    lower, upper = TV.point_bounds(errors[0], errors[1], known[0], known[1])
    return [float(np.median(lower)), float(np.median(upper))], dict(
        points=n,
        known_global=int(known[0].sum()),
        known_local=int(known[1].sum()),
        exact_inherited_equal_errors=bool(
            np.array_equal(errors[0], errors[1]) and np.array_equal(known[0], known[1])
        )
        if inherited
        else None,
    )


def noise_scale(base, local, bi, li):
    residuals = np.zeros(len(bi))
    counts = np.zeros(len(bi), int)
    for point, (ib, il) in enumerate(zip(bi, li)):
        shared = sorted(set(base[5][ib]) & set(local[5][il]))
        for iid, key in shared:
            if np.max(np.abs(base[1][iid][key, :2] - local[1][iid][key, :2])) > 1e-8:
                raise ValueError("shared_pixels_differ")
            rotation, center, _ = local[0][iid]
            camera = ((local[4][il] - center) @ rotation.T)[None]
            prediction = L.P.G.project(camera, *local[3][local[2][iid]])[0]
            residuals[point] += float(
                np.sum((prediction - local[1][iid][key, :2]) ** 2)
            )
            counts[point] += 1
    length = np.asarray([len(set(local[5][i])) for i in li])
    value = float(np.median((residuals / counts) / (2 - 3 / length)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("invalid_pixel_variance")
    return value


def response(model):
    original = L.projection_linearization
    try:
        L.projection_linearization = K.signed_projection
        return L.response(model, 128)
    finally:
        L.projection_linearization = original


def score(base, local, index, inherited=False, global_response=None, convergence=None):
    pairs = intersections(base[5], local[5])
    record = dict(pairs=len(pairs), scores={}, reasons={}, admission=None)
    if len(pairs) < 100:
        record["reasons"]["all"] = "fewer_than_100_pairs"
        return record
    bi, li = np.asarray(pairs, int).T
    try:
        record["scores"]["transfer"], record["transfer"] = transfer_bounds(
            base, local, pairs, inherited
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        record["reasons"]["transfer"] = str(error)
    extent = float(
        np.quantile(np.linalg.norm(base[4][bi] - base[4][bi].mean(0), axis=1), 0.9)
    )
    s, rotation, translation = alignment(base[0], local[0])
    discrepancy = s * local[4][li] @ rotation.T + translation - base[4][bi]
    q95 = (
        float(np.quantile(np.linalg.norm(discrepancy, axis=1), 0.95) / extent)
        if extent > 1e-12
        else None
    )
    fit_ok = convergence is None or bool(np.all(convergence[li]))
    admitted = extent > 1e-12 and q95 <= 0.15 and fit_ok
    record["admission"] = dict(
        admitted=admitted,
        extent=extent,
        raw_q95=q95,
        matched_fit_failures=int(np.sum(~convergence[li]))
        if convergence is not None
        else 0,
    )
    if not admitted:
        record["reasons"]["point_components"] = "ordinary_support_or_fit_failure"
        return record
    values = DET.detection_scores(
        base[4][bi], discrepancy, neighbors=8, shuffle_count=20, seed=20260904 + index
    )
    for key in ("raw_median", "raw_q95", "rdsc"):
        record["scores"][key] = float(values[key])
    try:
        sigma = noise_scale(base, local, bi, li)
        rb = global_response if global_response is not None else response(base)
        rl = K.response(base, local, rb, 128) if inherited else response(local)
        delta = A.aligned_response(base, local, rb, rl, bi, li, alignment)
        covariance = sigma * np.einsum("nis,njs->nij", delta, delta) / 128
        standardized, _ = L.P.standardized(discrepancy, covariance)
        edge, _, _ = Q.scores(
            base[4][bi],
            base[4][bi] + discrepancy,
            rb["point"][bi],
            rb["point"][bi] + delta,
            sigma,
            keys=ID.keys(base[5], bi),
        )
        record["scores"].update(
            japd=float(np.median(standardized)), edge=float(edge["japd_quadratic_q95"])
        )
        record["geometry"] = dict(
            pixel_variance=sigma, local_response=rl["info"], global_response=rb["info"]
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        record["reasons"]["geometry"] = type(error).__name__ + ":" + str(error)
    return record
