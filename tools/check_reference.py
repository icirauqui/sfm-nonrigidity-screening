"""Replay projected-global scalars and all primary local sequence decisions.

This is a compact replay; full model-to-score replay is separate.
"""

from pathlib import Path
import argparse, hashlib, json
import numpy as np
from sfm_nonrigidity.global_projection import score
from sfm_nonrigidity.sequence import infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduction/reference",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.reference
    hashes = json.loads((root / "SHA256.json").read_text())
    for n, h in hashes.items():
        assert hashlib.sha256((root / n).read_bytes()).hexdigest() == h, n
    records = json.loads((root / "global_scores.json").read_text())["records"]
    checks = 0
    refused = 0
    for row in records:
        if "projected_matrix" not in row:
            refused += 1
            continue
        with np.load(root / row["projected_matrix"], allow_pickle=False) as z:
            matrix = z["gram"]
        result = score(
            dict(gram=matrix, residual_dimension=row["residual_dimension"]),
            row["raw_energy"],
        )
        assert result["status"] == row["statistic"]["status"], row["id"]
        for k, v in result.items():
            old = row["statistic"][k]
            if isinstance(v, (int, float)):
                assert np.isclose(v, old, rtol=1e-7, atol=1e-8), (row["id"], k, v, old)
            else:
                assert v == old
            checks += 1
    local = json.loads((root / "local_scores.json").read_text())["records"]
    cal = json.loads((root / "calibration.json").read_text())["calibration"]
    expected = json.loads((root / "local_predictions.json").read_text())["records"]
    actual = infer(local, cal)
    assert actual == expected
    result = dict(
        status="PASS",
        manifest_files=len(hashes),
        projected_global_models=len(records) - refused,
        retained_global_refusals=refused,
        global_statistic_checks=checks,
        local_window_scores=len(local),
        local_sequence_decisions=len(actual),
        scope="projected matrix and sequence replay; not model-to-score inference",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
