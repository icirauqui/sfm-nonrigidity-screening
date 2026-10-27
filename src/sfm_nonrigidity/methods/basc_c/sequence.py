"""Inference-only reference kernels; see docs/methods.md.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations
import math
from . import routing as CORE

COMPONENTS = ("rdsc", "raw_median", "raw_q95", "transfer", "japd", "edge")
MAP = dict(
    rdsc="rdsc",
    raw_median="raw_median",
    raw_q95="raw_q95",
    transfer="transfer_share",
    japd="japd_conditioned",
    edge="japd_quadratic_q95",
)


def endpoint(rows, component):
    values = {}
    for row in rows:
        if row["index"] in values:
            raise ValueError("duplicate_window_index")
        value = row["scores"].get(component)
        if isinstance(value, (int, float)):
            value = [float(value), float(value)]
        if value is not None:
            CORE.validate_interval(value)
        values[row["index"]] = value
    valid = {i: v for i, v in values.items() if v is not None}
    persistence = 3 if component == "transfer" else 2
    runs = [
        tuple(valid[i + j] for j in range(persistence))
        for i in valid
        if all(i + j in valid for j in range(persistence))
    ]
    admitted = len(valid) >= math.ceil(0.8 * len(values)) and bool(runs)
    result = (
        [max(min(v[side] for v in run) for run in runs) for side in (0, 1)]
        if admitted
        else None
    )
    return dict(score=result, covered=len(valid), requested=len(values), runs=len(runs))


def infer(rows, calibration):
    """Calibration membership is supplied explicitly, without target labels."""
    groups = {}
    for row in rows:
        groups.setdefault((row["regime"], row["sequence"]), []).append(row)
    endpoints = {
        key: {c: endpoint(items, c) for c in COMPONENTS}
        for key, items in groups.items()
    }
    predictions = []
    for target in calibration:
        regime, sequence = target["regime"], target["sequence"]
        e = endpoints[regime, sequence]
        evidence = {}
        for component in COMPONENTS:
            values = [
                endpoints[regime, c][component]["score"] for c in target["calibration"]
            ]
            cutoff = (
                [max(v[side] for v in values) for side in (0, 1)]
                if values and all(v is not None for v in values)
                else None
            )
            evidence[MAP[component]] = dict(score=e[component]["score"], cutoff=cutoff)
        route = CORE.route([regime], evidence)
        preferred = route["old_oac"]
        decision = preferred if preferred is not None else route["components"]["rdsc"]
        selected = (
            (
                "transfer"
                if route["components"]["transfer_share"] is not None
                else "raw_q95"
            )
            if regime == "independent" and preferred is not None
            else "dual_geometry"
            if preferred is not None
            else "rdsc"
            if decision is not None
            else "none"
        )
        predictions.append(
            dict(
                id=target["id"],
                sequence=sequence,
                regime=regime,
                calibration=target["calibration"],
                endpoints=e,
                evidence=evidence,
                components=route["components"],
                basc_c=decision,
                no_r=preferred,
                no_q=route["primary"],
                selected=selected,
            )
        )
    return predictions
