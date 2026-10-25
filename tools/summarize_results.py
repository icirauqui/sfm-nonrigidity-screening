"""Regenerate the primary comparison counts from closed full-precision records.

Evaluator labels come from the retained matched-comparison case table, after
all original predictions were frozen. This tool does not run detector inference.
"""

from pathlib import Path
import argparse, csv, json


def read(p):
    return json.loads(p.read_text())


def counts(rows, labels):
    out = dict(TP=0, TN=0, FP=0, FN=0, A=0)
    for identity, value in rows:
        positive = labels[identity]
        key = (
            "A"
            if value is None
            else "TP"
            if value and positive
            else "FP"
            if value
            else "FN"
            if positive
            else "TN"
        )
        out[key] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduction/reference",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    r = a.reference
    cases = read(r / "log_results.json")["cases"]
    labels = {x["sequence"]: x["positive"] for x in cases}
    g = {
        x["sequence"]: x
        for x in read(r / "global_predictions.json")["records"]
        if x["regime"] == "independent" and x["id"].endswith(":full")
    }
    local = read(r / "local_predictions.json")["records"]
    elementary = read(r / "elementary_predictions.json")["records"]
    result = []
    for regime in ["independent", "inherit_global"]:
        full = [x for x in local if x["regime"] == regime and x["id"].endswith(":full")]
        for method, key in [("RDSC", "rdsc"), ("BASC-C", "basc_c"), ("No-R", "no_r")]:
            result.append(
                dict(
                    regime=regime,
                    method=method,
                    **counts(
                        [
                            (
                                x["sequence"],
                                x["components"]["rdsc"] if key == "rdsc" else x[key],
                            )
                            for x in full
                        ],
                        labels,
                    ),
                )
            )
        for method, key in [
            ("Coherent amplitude", "coherent_affine_q95"),
            ("Ratio", "coherence_excess"),
            ("Median", "raw_median"),
            ("q95", "raw_q95"),
        ]:
            chosen = [
                x
                for x in elementary
                if x["regime"] == regime and x["id"].endswith(":full")
            ]
            result.append(
                dict(
                    regime=regime,
                    method=method,
                    **counts(
                        [
                            (x["sequence"], x["methods"][key]["decision"])
                            for x in chosen
                        ],
                        labels,
                    ),
                )
            )
        result.append(
            dict(
                regime=regime,
                method="Global modal energy",
                **counts(
                    [
                        (seq, x["methods"]["mode_gain"]["decision"])
                        for seq, x in g.items()
                    ],
                    labels,
                ),
            )
        )
        for file, key, label in [
            ("m3c2_results.json", "m3c2_abs_q95", "M3C2"),
            ("log_results.json", "log_lite_union_zd", "LOG-lite/ZD union"),
        ]:
            selected = [
                x
                for x in read(r / file)["cases"]
                if x["regime"] == regime and x["fold"] == "full"
            ]
            result.append(
                dict(
                    regime=regime,
                    method=label,
                    **counts(
                        [(x["sequence"], x["answers"][key]) for x in selected], labels
                    ),
                )
            )
    (a.output / "comparison.json").write_text(
        json.dumps(
            dict(
                scope="12 exposed acquisitions, dependent regimes; descriptive counts",
                rows=result,
            ),
            indent=2,
        )
        + "\n"
    )
    with (a.output / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["regime", "method", "TP", "TN", "FP", "FN", "A"]
        )
        w.writeheader()
        w.writerows(result)
    print("Recounted", len(result), "method/regime rows")


if __name__ == "__main__":
    main()
