"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import math

METHODS = (
    "rdsc",
    "raw_median",
    "raw_q95",
    "transfer_share",
    "japd_conditioned",
    "japd_quadratic_q95",
    "japd_f",
    "japd_f_q95",
)


def validate_interval(value):
    if value is None:
        return None
    if (
        len(value) != 2
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in value
        )
        or value[0] > value[1]
    ):
        raise ValueError("invalid_interval")
    return tuple(value)


def decide(score, cutoff):
    score = validate_interval(score)
    cutoff = validate_interval(cutoff)
    if score is None or cutoff is None:
        return None
    if score[0] > cutoff[1]:
        return True
    if score[1] <= cutoff[0]:
        return False
    return None


def either(left, right):
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def route(camera_modes, evidence):
    # Deliberately do not accept sequence IDs, populations, labels or old decisions.
    p = {
        m: decide(evidence.get(m, {}).get("score"), evidence.get(m, {}).get("cutoff"))
        for m in METHODS
    }
    modes = set(camera_modes)
    dual = (
        None
        if any(p[m] is None for m in ("japd_conditioned", "japd_quadratic_q95"))
        else p["japd_conditioned"] or p["japd_quadratic_q95"]
    )
    if modes == {"independent"}:
        preferred = p["transfer_share"]
        branch = "camera_contrast"
        old = preferred if preferred is not None else p["raw_q95"]
    elif modes == {"inherit_global"}:
        preferred = dual
        branch = "point_geometry"
        old = dual
    else:
        return dict(
            components=p,
            preferred=None,
            primary=None,
            recall=None,
            agreement=None,
            old_oac=None,
            selected="none",
            reason="unqualified_provenance",
        )
    primary = preferred if preferred is not None else p["rdsc"]
    selected = (
        branch if preferred is not None else "rdsc" if p["rdsc"] is not None else "none"
    )
    agreement = preferred if preferred is not None and preferred is p["rdsc"] else None
    return dict(
        components=p,
        preferred=preferred,
        primary=primary,
        recall=either(preferred, p["rdsc"]),
        agreement=agreement,
        old_oac=old,
        selected=selected,
        reason="preferred_available"
        if preferred is not None
        else "fallback_available"
        if primary is not None
        else "no_calibrated_decision",
    )
