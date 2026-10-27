"""Command line interface with explicit input paths and no evaluator discovery."""

from pathlib import Path
import argparse, json, sys
import numpy as np
from .api import (
    load_model,
    global_scores,
    local_scores,
    rdsc_scores,
    epc_scores,
    refine_points,
    compose,
)
from .methods.basc_c.sequence import infer


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def with_points(model, path):
    if path is None:
        return model, None
    with np.load(path, allow_pickle=False) as z:
        points = z["points"].copy()
        converged = z["converged"].copy() if "converged" in z else None
    if points.shape != model[4].shape or not np.isfinite(points).all():
        raise ValueError("invalid_point_override")
    if converged is not None and (
        converged.shape != (len(points),) or converged.dtype != np.bool_
    ):
        raise ValueError("invalid_convergence_mask")
    return (*model[:4], points, model[5]), converged


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Global modal, RDSC and BASC-C reference inference"
    )
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("global", help="Score one existing COLMAP text model")
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--points", type=Path)
    g.add_argument("--requested-views", type=int, default=90)
    r = sub.add_parser("rdsc", help="Score aligned NPZ positions/residuals")
    r.add_argument("--pairs", type=Path, required=True)
    r.add_argument("--window-index", type=int, default=0)
    r.add_argument("--normalization-extent", type=float)
    r.add_argument("--profile", choices=["epc", "original"], default="epc")
    l = sub.add_parser("local", help="Score local components from prepared model pairs")
    l.add_argument("--global-model", type=Path, required=True)
    l.add_argument("--local-model", type=Path, required=True)
    l.add_argument("--global-points", type=Path)
    l.add_argument("--local-points", type=Path)
    l.add_argument(
        "--provenance", choices=["independent", "inherit_global"], required=True
    )
    l.add_argument("--window-index", type=int, required=True)
    l.add_argument("--profile", choices=["epc", "original"], default="epc")
    f = sub.add_parser("refine", help="Fit points with fixed cameras/intrinsics")
    f.add_argument("--model", type=Path, required=True)
    f.add_argument("--reference", type=Path)
    c = sub.add_parser("compose", help="Compose already calibrated expert intervals")
    c.add_argument("--input", type=Path, required=True)
    s = sub.add_parser(
        "sequence", help="Primary temporal aggregation and BASC-C decisions"
    )
    s.add_argument("--input", type=Path, required=True)
    for q in (g, r, l, f, c, s):
        q.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.command == "global":
            model, conv = with_points(load_model(a.model), a.points)
            result = (
                global_scores(model, requested_views=a.requested_views)
                if conv is None or conv.all()
                else dict(status="ABSTAIN", reason="point_fit_failure", scores={})
            )
        elif a.command == "rdsc":
            if a.window_index < 0:
                raise ValueError("invalid_window_index")
            with np.load(a.pairs, allow_pickle=False) as z:
                result = (
                    rdsc_scores(
                        z["positions"],
                        z["residuals"],
                        seed=20260904 + a.window_index,
                        normalization_extent=a.normalization_extent,
                    )
                    if a.profile == "original"
                    else epc_scores(
                        z["positions"],
                        z["residuals"],
                        window_index=a.window_index,
                        normalization_extent=a.normalization_extent,
                    )
                )
        elif a.command == "local":
            original_base = load_model(a.global_model)
            base, base_convergence = with_points(original_base, a.global_points)
            local, conv = with_points(load_model(a.local_model), a.local_points)
            if a.provenance == "inherit_global" and a.local_points is not None:
                if any(
                    i not in base[0] or base[0][i][2] != pose[2]
                    for i, pose in local[0].items()
                ):
                    raise ValueError("inherited_camera_identity_mismatch")
                local = ({i: base[0][i] for i in local[0]}, *local[1:])
            from .common.alignment import intersections

            matched = intersections(base[5], local[5])
            extent = None
            if matched:
                bi = np.asarray(matched)[:, 0]
                if base_convergence is not None and not base_convergence[bi].all():
                    raise ValueError("matched_global_point_fit_failure")
                x = original_base[4][bi]
                extent = float(np.quantile(np.linalg.norm(x - x.mean(0), axis=1), 0.9))
            result = local_scores(
                base,
                local,
                window_index=a.window_index,
                provenance=a.provenance,
                convergence=conv,
                normalization_extent=extent,
                scalar_profile=a.profile,
            )
        elif a.command == "refine":
            model, diag = refine_points(
                load_model(a.model), load_model(a.reference) if a.reference else None
            )
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open("xb") as out:
                np.savez_compressed(out, points=model[4], **diag)
            return 0
        else:
            data = json.loads(a.input.read_text())
            if a.command == "compose":
                if set(data) != {"provenance", "evidence"}:
                    raise ValueError("unexpected_inference_fields")
                result = compose(data["provenance"], data["evidence"])
            else:
                if set(data) != {"rows", "calibration"}:
                    raise ValueError("unexpected_sequence_fields")
                result = dict(records=infer(data["rows"], data["calibration"]))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("x") as out:
            json.dump(
                result,
                out,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=json_value,
            )
            out.write("\n")
    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as error:
        p.exit(2, f"{type(error).__name__}: {error}\n")
    return 0
