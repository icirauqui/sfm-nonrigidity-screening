"""Primary model-to-score replay using only the standalone package.

Expected scores are read only in the parent, after workers finish inference.
No fresh timing/generalization claim is made. CPU/BLAS thread settings are caller
controlled; run separate processes, since the retained response kernel temporarily
sets its projection callback within a process.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, json, os, sys, time
import numpy as np
from sfm_nonrigidity import models, geometry, global_model, epc_scalar
from sfm_nonrigidity.cli import with_points, json_value
from sfm_nonrigidity.sequence import infer


def read(p):
    return json.loads(p.read_text())


def worker(task):
    root, q, mode = task
    root = Path(root).resolve()
    opened = set()
    allowed = set()
    if mode == "global":
        allowed.update(
            root / q["model"] / n for n in ["cameras.txt", "images.txt", "points3D.txt"]
        )
        if q["points"]:
            allowed.add(root / q["points"])
    else:
        for r in q:
            for role in ["global_model", "local_model"]:
                allowed.update(
                    root / r[role] / n
                    for n in ["cameras.txt", "images.txt", "points3D.txt"]
                )
            for role in ["global_points", "independent_points", "inherited_points"]:
                allowed.add(root / r[role])

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):
            if p not in allowed:
                raise PermissionError(
                    "unapproved inference input:" + str(p.relative_to(root))
                )
            opened.add(str(p.relative_to(root)))

    sys.addaudithook(guard)
    if mode == "global":
        b = models.model(str(root / q["model"] / "images.txt"))
        b, conv = with_points(b, root / q["points"] if q["points"] else None)
        if conv is not None and not conv.all():
            raise ValueError("point_fit_failure")
        try:
            _, stat, scores, _, _ = global_model.calculate(b)
            result = dict(id=q["id"], cohort=q["cohort"], scores=scores, statistic=stat)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
            result = dict(id=q["id"], cohort=q["cohort"], scores={}, reason=str(e))
        return dict(records=[result], opened=sorted(opened))
    q = sorted(q, key=lambda r: r["index"])
    raw = models.model(str(root / q[0]["global_model"] / "images.txt"))
    base, bc = with_points(raw, root / q[0]["global_points"])
    assert bc.all()
    rb = geometry.response(base)
    rows = []
    for row in q:
        original = models.model(str(root / row["local_model"] / "images.txt"))
        for regime in ["independent", "inherit_global"]:
            local, conv = with_points(
                original,
                root
                / row[
                    "independent_points"
                    if regime == "independent"
                    else "inherited_points"
                ],
            )
            assert conv.all()
            if regime == "inherit_global":
                local = ({i: base[0][i] for i in original[0]}, *local[1:])
            bi, li = np.asarray(geometry.intersections(base[5], local[5])).T
            record = geometry.score(
                base, local, row["index"], regime == "inherit_global", rb, conv
            )
            # Primary preparation retains the original-export matched-support extent.
            # Fitted coordinates supply residuals; refitting does not redefine that extent.
            length = float(
                np.quantile(
                    np.linalg.norm(raw[4][bi] - raw[4][bi].mean(0), axis=1), 0.9
                )
            )
            s, rot, t = geometry.alignment(base[0], local[0])
            x = base[4][bi]
            d = s * local[4][li] @ rot.T + t - x
            scalar = epc_scalar.scalar_scores(x, d, row["index"], length)
            for key in ["rdsc", "raw_median", "raw_q95"]:
                record["scores"][key] = scalar[key]
            rows.append(
                dict(
                    id=regime + ":" + row["id"],
                    sequence=row["sequence"],
                    index=row["index"],
                    regime=regime,
                    scores=record["scores"],
                )
            )
    return dict(records=rows, opened=sorted(opened))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduction/reference",
    )
    p.add_argument("--mode", choices=["global", "local", "all"], default="all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--comparison",
        choices=["strict", "decisions"],
        default="strict",
        help="Strict retained score tolerances, or explicit decision comparison with all numerical differences reported",
    )
    p.add_argument(
        "--recheck",
        type=Path,
        help="Evaluate saved request_*.json without rerunning inference",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = read(a.inputs / "INPUT_MANIFEST.json")
    for n, v in manifest["files"].items():
        assert hashlib.sha256((a.inputs / n).read_bytes()).hexdigest() == v["sha256"], n
    contract = read(a.inputs / "CONTRACTS.json")
    tasks = []
    if a.mode in ["global", "all"]:
        tasks.extend(
            (str(a.inputs.resolve()), q, "global") for q in contract["global_jobs"]
        )
    if a.mode in ["local", "all"]:
        groups = {}
        for r in contract["local_windows"]:
            groups.setdefault(r["sequence"], []).append(r)
        tasks.extend(
            (str(a.inputs.resolve()), rows, "local") for rows in groups.values()
        )
    if a.limit:
        tasks = tasks[: a.limit]
    start = time.monotonic()
    outputs = []
    if a.recheck:
        outputs = [read(p) for p in sorted(a.recheck.glob("request_*.json"))]
        assert len(outputs) == len(tasks), "incomplete_saved_requests"
    else:
        with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(worker, t) for t in tasks]):
                result = future.result()
                outputs.append(result)
                (a.output / f"request_{len(outputs):03d}.json").write_text(
                    json.dumps(
                        result,
                        indent=2,
                        sort_keys=True,
                        default=json_value,
                        allow_nan=False,
                    )
                    + "\n"
                )
                print("CLOSED", result["records"][0]["id"], flush=True)
    records = [r for o in outputs for r in o["records"]]
    if not a.limit:
        globals_ = [(r["cohort"], r["id"]) for r in records if "cohort" in r]
        local_ids = [r["id"] for r in records if "regime" in r]
        assert len(globals_) == len(set(globals_)) and len(local_ids) == len(
            set(local_ids)
        ), "duplicate_request_records"
        if a.mode in ("global", "all"):
            assert set(globals_) == {
                (r["cohort"], r["id"]) for r in contract["global_jobs"]
            }, "global_request_coverage"
        if a.mode in ("local", "all"):
            assert set(local_ids) == {
                regime + ":" + r["id"]
                for r in contract["local_windows"]
                for regime in ("independent", "inherit_global")
            }, "local_request_coverage"
    # Evaluation begins only after every worker has completed.
    global_expected = {
        (r["cohort"], r["id"]): r
        for r in read(a.reference / "global_scores.json")["records"]
    }
    local_expected = {
        r["id"]: r for r in read(a.reference / "local_scores.json")["records"]
    }
    score_checks = 0
    max_error = 0.0
    locals_ = []
    numerical_differences = []
    global_decisions_checked = 0
    support = {r["id"]: r for r in read(a.reference / "global_support.json")["records"]}
    for row in records:
        isglobal = "cohort" in row
        old = (
            global_expected[row["cohort"], row["id"]]
            if isglobal
            else local_expected[row["id"]]
        )
        if isglobal:
            assert set(row["scores"]) == set(old["scores"]), (
                row["id"],
                "available_scores",
            )
        else:
            assert set(row["scores"]) == set(old["scores"]), row["id"]
            locals_.append(row)
        for k, v in row["scores"].items():
            rtol, atol = (1e-7, 1e-8) if isglobal else (1e-8, 1e-10)
            if not np.allclose(v, old["scores"][k], rtol=rtol, atol=atol):
                numerical_differences.append(
                    dict(
                        id=row["id"],
                        method=k,
                        actual=v,
                        expected=old["scores"][k],
                        rtol=rtol,
                        atol=atol,
                    )
                )
            max_error = max(
                max_error,
                float(np.max(abs(np.asarray(v) - np.asarray(old["scores"][k])))),
            )
            score_checks += 1
        if isglobal:
            s = support[row["id"]]
            for method in (
                "mode_gain",
                "raw_pixel_F_stream_r0",
                "pixel_median",
                "pixel_q95",
                "null_rms",
            ):
                decision = (
                    row["scores"][method] > s[method + "_cutoff"]
                    if method in row["scores"]
                    else None
                )
                assert decision == s[method + "_decision"], (
                    row["id"],
                    method,
                    "decision_changed",
                )
                global_decisions_checked += 1
        if isglobal and row["scores"]:
            for k in ["affine_rank", "quadratic_rank", "residual_df"]:
                assert row["statistic"][k] == old["statistic"][k], (row["id"], k)
        if isglobal and not row["scores"]:
            assert all(
                v.endswith(":" + row["reason"]) for v in old["reasons"].values()
            ), (row["id"], row, old)
    if locals_ and not a.limit:
        actual = infer(locals_, contract["calibration"])
        expected = {
            r["id"]: r for r in read(a.reference / "local_predictions.json")["records"]
        }
        for r in actual:
            for key in ["basc_c", "no_r", "no_q", "selected", "components"]:
                assert r[key] == expected[r["id"]][key], (r["id"], key)
    core_scores = {r["id"]: r["scores"] for r in records if r.get("cohort") == "core"}
    global_calibration_checks = 0
    if len(core_scores) == 12 and not a.limit:
        for request in read(a.reference / "global_predictions.json")["records"]:
            for method, value in request["methods"].items():
                cutoff = max(core_scores[c][method] for c in request["calibration"])
                assert (core_scores[request["sequence"]][method] > cutoff) == value[
                    "decision"
                ], (request["id"], method, "calibration_decision_changed")
                global_calibration_checks += 1
    receipt = dict(
        status="PASS"
        if not numerical_differences
        else "DECISIONS_MATCH_NUMERICAL_REFERENCE_MISMATCH",
        requests=len(tasks),
        records=len(records),
        score_checks=score_checks,
        maximum_absolute_error=max_error,
        local_decisions=len(actual) if locals_ and not a.limit else 0,
        seconds=time.monotonic() - start,
        scope="full primary model-to-score and decision replay"
        if not a.limit
        else "limited development replay",
        inference_input_allowlist=True,
        strict_numerical_match=not numerical_differences,
        numerical_difference_count=len(numerical_differences),
        global_decisions_checked=global_decisions_checked,
        global_core_calibration_records_checked=global_calibration_checks,
        comparison_mode=a.comparison,
        rechecked_saved_inference=bool(a.recheck),
    )
    for name, value in [
        ("SCORES.json", dict(records=records)),
        ("REPLAY.json", receipt),
        ("INPUT_READS.json", outputs),
        ("NUMERICAL_DIFFERENCES.json", numerical_differences),
    ]:
        (a.output / name).write_text(
            json.dumps(
                value, indent=2, sort_keys=True, default=json_value, allow_nan=False
            )
            + "\n"
        )
    print(json.dumps(receipt, indent=2))
    if numerical_differences and a.comparison == "strict":
        raise SystemExit(
            "Retained score tolerances failed; differences saved. Decision parity does not imply numerical reference equivalence."
        )


if __name__ == "__main__":
    main()
