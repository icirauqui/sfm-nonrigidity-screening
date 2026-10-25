"""Prepare a neutral, hash-bound input bundle from the retained review capsule.

Copies only the explicit primary model/point selection. No raw imagery, evaluator
truth, reserved geometry, source-code history or upstream LOG source is copied.
"""

from pathlib import Path
import argparse, hashlib, json, shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = a.capsule
    out = a.output
    files = {}

    def copy(src, name, expected=None):
        h = sha(src)
        if expected is not None:
            assert h == expected, (src, expected, h)
        if name in files:
            assert files[name]["sha256"] == h
            return
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assert sha(dest) == h
        files[name] = dict(sha256=h, bytes=dest.stat().st_size)

    f = read(root / "aggregate/evidence/basc/FREEZE.json")
    assert len(f["rows"]) == 108 and {r["family"] for r in f["rows"]} == {
        "f01",
        "f02",
        "f03",
        "f04",
    }, "unexpected_primary_population"
    assert (
        len(f["inputs"]) == 360
        and len(f["epc_fits"]) == 120
        and len(f["inherited_fits"]) == 108
    ), "unexpected_primary_support"
    rows = []
    core = {}
    for row in f["rows"]:
        seq = row["sequence"]
        identity = row["id"]
        record = dict(
            id=identity,
            index=row["index"],
            sequence=seq,
            global_model=f"models/core/{seq}/global",
            local_model=f"models/core/{seq}/{identity}",
            global_points=f"points/global/{seq}.npz",
            independent_points=f"points/independent/{identity}.npz",
            inherited_points=f"points/inherit_global/{identity}.npz",
        )
        for role in ["global", "local"]:
            for name, source in row["models"][role].items():
                copy(
                    root / "aggregate/inputs/core" / source,
                    record[role + "_model"] + "/" + name,
                    f["inputs"][source],
                )
        for rel, dest in [
            (f"fit_global/{seq}.npz", record["global_points"]),
            (f"fit_independent/{identity}.npz", record["independent_points"]),
        ]:
            copy(root / "aggregate/inputs/epc" / rel, dest, f["epc_fits"][rel])
        copy(
            root / f"aggregate/inputs/reference/fits/{identity}.npz",
            record["inherited_points"],
            f["inherited_fits"][identity + ".npz"],
        )
        rows.append(record)
        core[seq] = dict(
            id=seq,
            cohort="core",
            model=record["global_model"],
            points=record["global_points"],
            requested_views=90,
        )
    transfer = read(root / "reproduction/evidence/transfer/FREEZE.json")["inputs"][
        "jobs"
    ]
    jobs = list(core.values())
    for row in transfer:
        identity = row["id"]
        model = f"models/transfer/{identity}"
        for name, value in row["files"].items():
            copy(
                root / f"reproduction/inputs/transfer/models/{identity}/{name}",
                model + "/" + name,
                value["sha256"],
            )
        jobs.append(
            dict(
                id=identity,
                cohort="transfer",
                model=model,
                points=None,
                requested_views=90,
            )
        )
    contracts = dict(global_jobs=jobs, local_windows=rows, calibration=f["calibration"])
    (out / "CONTRACTS.json").write_text(
        json.dumps(contracts, indent=2, sort_keys=True) + "\n"
    )
    files["CONTRACTS.json"] = dict(
        sha256=sha(out / "CONTRACTS.json"),
        bytes=(out / "CONTRACTS.json").stat().st_size,
    )
    (out / "INPUT_MANIFEST.json").write_text(
        json.dumps(
            dict(
                schema="screening_inputs_v1",
                files=files,
                global_models=len(jobs),
                local_windows=len(rows),
                inference_truth=False,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        "Prepared",
        len(files),
        "files:",
        len(jobs),
        "globals and",
        len(rows),
        "local windows",
    )


if __name__ == "__main__":
    main()
