"""Exercise all six installed CLI commands on manufactured arithmetic fixtures.

No external data, evaluator truth or private workspace files are needed.
This checks software use, not benchmark performance or physical validity.
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "fixture"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("create_fixture.py")),
         "--output", str(fixture)], check=True
    )

    def run(command, *arguments, filename):
        subprocess.run(
            [sys.executable, "-m", "sfm_nonrigidity", command,
             *map(str, arguments), "--output", str(out / filename)], check=True
        )

    model = fixture / "model"
    points = out / "refined.npz"
    run("rdsc", "--pairs", fixture / "pairs.npz", filename="rdsc.json")
    run("global", "--model", model, "--requested-views", 6, filename="global.json")
    run("compose", "--input", fixture / "compose.json", filename="composition.json")
    run("refine", "--model", model, filename=points.name)
    run("local", "--global-model", model, "--local-model", model,
        "--global-points", points, "--local-points", points,
        "--provenance", "inherit_global", "--window-index", 0, filename="local.json")
    run("sequence", "--input", fixture / "sequence.json", filename="sequence.json")

    read = lambda name: json.loads((out / name).read_text())
    if read("global.json")["status"] != "SCORED":
        raise RuntimeError("manufactured_global_refused")
    if read("composition.json")["decision"] is not True:
        raise RuntimeError("manufactured_composition_mismatch")
    with np.load(points, allow_pickle=False) as fitted:
        if not fitted["converged"].all():
            raise RuntimeError("manufactured_point_fit_failure")
    local = read("local.json")
    if local["pairs"] != 120 or local["scores"]["transfer"] != [0.5, 0.5]:
        raise RuntimeError("identical_model_transfer_mismatch")
    target = read("sequence.json")["records"][0]
    if target["basc_c"] is not True or target["selected"] != "rdsc":
        raise RuntimeError("manufactured_sequence_mismatch")
    result = dict(
        status="PASS", cli_commands=6, external_inputs=False,
        scope="Manufactured arithmetic and CLI checks; no research performance claim",
        sequence_input="Separate hand-set scores, not a sequence reconstructed from the fixture model",
    )
    (out / "EXAMPLE_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: all six CLI commands; see", out / "EXAMPLE_CHECK.json")


if __name__ == "__main__":
    main()
