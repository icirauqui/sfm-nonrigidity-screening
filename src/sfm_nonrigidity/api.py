"""Explicit-input public API. Calibration membership is always caller supplied.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path
import math
import numpy as np
from . import models, geometry, global_model, rdsc, routing, epc_scalar


def load_model(directory):
    """Read a COLMAP text triplet; return the retained six-element model tuple."""
    directory = Path(directory)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (directory / name).is_file():
            raise ValueError(f"missing_text_model_file:{name}")
    result = models.model(str((directory / "images.txt").resolve()))
    if not len(result[4]) or not np.isfinite(result[4]).all():
        raise ValueError("empty_or_nonfinite_model_points")
    return result


def rdsc_scores(
    positions,
    residuals,
    *,
    neighbors=8,
    shuffle_count=20,
    seed=20260904,
    normalization_extent=None,
):
    """Score already aligned point pairs, without labels or calibration."""
    x, d = np.asarray(positions, float), np.asarray(residuals, float)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape != d.shape or len(x) < 3:
        raise ValueError("invalid_point_pair_shapes")
    if not np.isfinite(x).all() or not np.isfinite(d).all():
        raise ValueError("nonfinite_point_pair")
    if neighbors < 1 or shuffle_count < 1:
        raise ValueError("invalid_neighborhood_or_shuffle_count")
    result = rdsc.detection_scores(
        x, d, neighbors=neighbors, shuffle_count=shuffle_count, seed=seed
    )
    if normalization_extent is not None:
        if not np.isfinite(normalization_extent) or normalization_extent <= 1e-12:
            raise ValueError("invalid_normalization_extent")
        factor = result["spatial_scale"] / normalization_extent
        for key in (
            "raw_median",
            "raw_q95",
            "affine_residual_q95",
            "coherent_affine_q95",
            "rdsc",
        ):
            result[key] *= factor
    return result


def global_scores(model, *, requested_views=90, minimum_fraction=0.8, progress=False):
    """Return scores or a whole-model abstention; this performs no SfM or fitting.

    A positive/negative decision additionally requires a supplied calibrated cutoff.
    Primary supported cameras are signed PINHOLE. Energy has squared-pixel units.
    """
    if requested_views < 2 or not 0 < minimum_fraction <= 1:
        raise ValueError("invalid_view_coverage_contract")
    count = len(model[0])
    support = dict(
        registered_views=count, requested_views=requested_views, points=len(model[4])
    )
    if count > requested_views:
        raise ValueError("registered_views_exceed_requested")
    if count < math.ceil(minimum_fraction * requested_views):
        return dict(
            status="ABSTAIN",
            reason="insufficient_registered_views",
            support=support,
            scores={},
        )
    try:
        result, statistic, scores, _, _ = global_model.calculate(
            model, progress=progress
        )
        support.update(
            {
                k: result[k]
                for k in (
                    "observations",
                    "point_rank",
                    "camera_rank",
                    "residual_dimension",
                )
            }
        )
        support.update(
            {k: statistic[k] for k in ("affine_rank", "quadratic_rank", "residual_df")}
        )
        if statistic["status"] != "SCORED":
            return dict(
                status="ABSTAIN", reason=statistic["status"], support=support, scores={}
            )
        return dict(
            status="SCORED", support=support, statistic=statistic, scores=scores
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return dict(status="ABSTAIN", reason=str(error), support=support, scores={})


def refine_points(original, reference=None):
    """Point-only EPC fit; optional reference supplies exactly inherited cameras.

    For independent fixed-camera fits reference is the original model itself.
    Returns (fitted_model, per-point convergence/solver diagnostics).
    """
    return geometry.conditioned_model(
        original if reference is None else reference, original
    )


def epc_scores(positions, residuals, *, window_index=0, normalization_extent=None):
    """Exact primary EPC scalar profile, including the elementary controls.

    The retained profile reconstructs x+d for its shuffled-reference control.
    That floating-point path is preserved, rather than substituting the direct
    residual-only primitive. Supply the original-export extent for fitted inputs.
    """
    x, d = np.asarray(positions, float), np.asarray(residuals, float)
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or x.shape != d.shape
        or len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("invalid_point_pair_arrays")
    if not isinstance(window_index, int) or window_index < 0:
        raise ValueError("invalid_window_index")
    length = (
        float(np.quantile(np.linalg.norm(x - x.mean(0), axis=1), 0.9))
        if normalization_extent is None
        else normalization_extent
    )
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("invalid_normalization_extent")
    return epc_scalar.scalar_scores(x, d, window_index, length)


def local_scores(
    base,
    local,
    *,
    window_index,
    provenance,
    convergence=None,
    normalization_extent=None,
    scalar_profile="epc",
):
    """All original local components, with explicit camera provenance.

    Models and optional convergence mask must already represent the intended
    preparation. Inherited camera poses/intrinsics are verified by the kernel.
    """
    if provenance not in ("independent", "inherit_global"):
        return dict(
            pairs=0,
            scores={},
            reasons={"all": "unqualified_provenance"},
            admission=None,
        )
    if not isinstance(window_index, int) or window_index < 0:
        raise ValueError("invalid_window_index")
    if scalar_profile not in ("original", "epc"):
        raise ValueError("unsupported_scalar_profile")
    result = geometry.score(
        base,
        local,
        window_index,
        provenance == "inherit_global",
        convergence=convergence,
    )
    if (
        scalar_profile == "epc"
        and result["admission"]
        and result["admission"]["admitted"]
    ):
        bi, li = np.asarray(geometry.intersections(base[5], local[5])).T
        scale, rotation, translation = geometry.alignment(base[0], local[0])
        x = base[4][bi]
        scalar = epc_scores(
            x,
            scale * local[4][li] @ rotation.T + translation - x,
            window_index=window_index,
            normalization_extent=normalization_extent,
        )
        for key in ("rdsc", "raw_median", "raw_q95"):
            result["scores"][key] = scalar[key]
    return result


def compose(provenance, evidence):
    """BASC-C on calibrated expert intervals; None represents abstention.

    Evidence keys: transfer_share, raw_q95, rdsc, japd_conditioned,
    japd_quadratic_q95. Each contains score/cutoff intervals or None.
    """
    if provenance not in ("independent", "inherit_global"):
        return dict(
            decision=None, selected_expert="none", reason="unqualified_provenance"
        )
    result = routing.route([provenance], evidence)
    preferred = result["old_oac"]
    decision = preferred if preferred is not None else result["components"]["rdsc"]
    if preferred is not None:
        selected = (
            (
                "transfer"
                if result["components"]["transfer_share"] is not None
                else "raw_q95"
            )
            if provenance == "independent"
            else "dual_geometry"
        )
    else:
        selected = "rdsc" if decision is not None else "none"
    return dict(
        decision=decision,
        selected_expert=selected,
        components=result["components"],
        reason="decisive" if decision is not None else "no_calibrated_decision",
    )
