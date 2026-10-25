"""Publication-size result figures; all values/decisions are closed retained data."""

from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm

ROOT = Path(".").resolve()
PACKAGE = ROOT / "results"
USED = {}
METHODS = (
    "mode_gain",
    "raw_pixel_F_stream_r0",
    "pixel_median",
    "pixel_q95",
    "null_rms",
)
COLORS = ["#176b84", "#7552a3", "#bd6b1a", "#8d4e85", "#16816a"]
LABELS = [
    r"Modal energy $\Delta_{\rm mod}$",
    r"Modal $F_{\rm mod}$",
    "Pixel median",
    "Pixel q95",
    "Null RMS",
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def read(p):
    USED[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)


def main():
    global OUT, ROOT, PACKAGE
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    a = p.parse_args()
    ROOT = a.inputs.resolve()
    PACKAGE = ROOT / "results"
    OUT = a.output
    OUT.mkdir(exist_ok=False)
    source = Path(__file__).read_bytes()
    (OUT / "SOURCE_RESULT_FIGURES.py").write_bytes(source)
    display = {}
    pred = read(PACKAGE / "core_predictions.json")["records"]
    full = {
        r["sequence"]: r
        for r in pred
        if r["regime"] == "independent" and r["id"].endswith(":full")
    }
    order = [
        f"f{i:02d}__{condition}"
        for i in range(1, 5)
        for condition in ("static", "fem", "nonfem")
    ]
    ratios = np.array(
        [
            [
                full[s]["methods"][m]["score"] / full[s]["methods"][m]["cutoff"]
                for m in METHODS
            ]
            for s in order
        ]
    )
    assert (ratios > 1).tolist() == [
        [full[s]["methods"][m]["decision"] for m in METHODS] for s in order
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.7},
    )
    x = np.arange(12)
    markers = ["o", "s", "^", "v", "D"]
    for ax, selected, title in [
        (axes[0], [0, 1], "(a) Modal alignment"),
        (axes[1], [2, 3, 4], "(b) Magnitude summaries"),
    ]:
        for j in selected:
            ax.plot(
                x + (j - 2) * 0.10,
                ratios[:, j],
                ls="none",
                marker=markers[j],
                ms=5,
                color=COLORS[j],
                label=LABELS[j],
            )
        ax.axhline(1, c="#293e49", ls="--", lw=0.9)
        ax.set_yscale("log")
        ax.set_ylabel("Score / static cutoff")
        ax.set_title(title, loc="left", fontsize=10, pad=32)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=len(selected),
            fontsize=8,
            frameon=False,
            borderaxespad=0,
        )
        ax.set_xlim(-0.6, 11.6)
        ax.grid(axis="y", which="major", alpha=0.16)
        for at in [2.5, 5.5, 8.5]:
            ax.axvline(at, c="#d8e0e2", lw=0.8)
    axes[0].set_ylim(0.4, 150)
    axes[1].set_ylim(0.88, 1.17)
    axes[1].set_yticks([0.9, 1, 1.1], ["0.9", "1.0", "1.1"])
    axes[1].minorticks_off()
    axes[1].set_xticks(
        x,
        [
            s.replace("__", "\n").replace("nonfem", "prescr.").replace("fem", "FE")
            for s in order
        ],
        fontsize=8,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.84)
    save(fig, "core_global_readouts")
    display["core"] = dict(
        order=order,
        methods=METHODS,
        ratios=ratios.tolist(),
        lower_panel_is_expanded_near_cutoff=True,
    )
    m3 = read(PACKAGE / "m3c2.json")
    log = read(PACKAGE / "log_zd.json")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 4.35), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    columns = [
        "Global",
        "RDSC",
        "BASC-C",
        "M3C2",
        "LOG\nloc.",
        "LOG\nori.",
        "LOG\nunion",
    ]
    seqs = sorted({r["sequence"] for r in log["cases"] if r["fold"] == "full"})
    matrices = {}
    for ax, regime, title in zip(
        axes,
        ("independent", "inherit_global"),
        ("Separate local cameras", "Inherited global cameras"),
    ):
        c = {
            r["sequence"]: r
            for r in log["cases"]
            if r["regime"] == regime and r["fold"] == "full"
        }
        m = {
            r["sequence"]: r
            for r in m3["cases"]
            if r["regime"] == regime and r["fold"] == "full"
        }
        values = np.zeros((12, 7))
        matrix = []
        for i, s in enumerate(seqs):
            assert c[s]["positive"] == m[s]["positive"]
            v = (
                [c[s]["answers"][k] for k in ("global_modal", "rdsc", "basc_c")]
                + [m[s]["answers"]["m3c2_abs_q95"]]
                + [
                    c[s]["answers"][k]
                    for k in (
                        "log_lite_graphd_zd",
                        "log_lite_graphf_zd",
                        "log_lite_union_zd",
                    )
                ]
            )
            matrix.append(v)
            for j, d in enumerate(v):
                values[i, j] = 0 if d is None else 1 if d == c[s]["positive"] else 2
        ax.pcolormesh(
            np.arange(8) - 0.5,
            np.arange(13) - 0.5,
            values,
            cmap=ListedColormap(["#d7d7d7", "#d8eaee", "#efb99f"]),
            vmin=0,
            vmax=2,
        )
        ax.set_ylim(11.5, -0.5)
        ax.set_xlim(-0.5, 6.5)
        for i, v in enumerate(matrix):
            for j, d in enumerate(v):
                ax.text(
                    j,
                    i,
                    "A" if d is None else "+" if d else "-",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        ax.set_xticks(range(7), columns, fontsize=8)
        ax.set_yticks(
            range(12),
            [
                s.replace("__fem", " / FE +")
                .replace("__nonfem", " / prescr. +")
                .replace("__static", " / static -")
                for s in seqs
            ],
            fontsize=8.5,
        )
        ax.set_title(title, fontsize=9.5, pad=10)
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        ax.grid(which="minor", c="white", lw=1)
        ax.tick_params(which="both", length=0)
        matrices[regime] = matrix
    fig.subplots_adjust(left=0.205, right=0.995, top=0.9, bottom=0.17)
    fig.text(
        0.58,
        0.055,
        "+ positive, - negative; blue correct, orange incorrect",
        ha="center",
        fontsize=8,
        color="#324b57",
    )
    fig.text(
        0.58,
        0.015,
        "Same 12 acquisitions in both panels",
        ha="center",
        fontsize=8,
        color="#697e85",
    )
    save(fig, "core_complete_decisions")
    display["decisions"] = dict(order=seqs, columns=columns, matrices=matrices)
    tf = read(PACKAGE / "transfer_contract.json")
    tr = {r["id"]: r for r in read(PACKAGE / "transfer_scores.json")["records"]}
    conditions = (
        "diffuse_symmetric",
        "coated_symmetric",
        "diffuse_asymmetric",
        "diffuse_exposure_half",
        "diffuse_exposure_double",
    )
    short = ["Diffuse", "Coated", "Asymmetric", "Half exposure", "Double exposure"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 4.3), sharey=True, gridspec_kw={"wspace": 0.10}
    )
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#c7cdd0")
    transfer = {}
    for ax, pipeline in zip(axes, ["A", "B"]):
        jobs = [
            next(
                j
                for j in tf["inputs"]["jobs"]
                if j["family"] == f
                and j["condition"] == c
                and j["pipeline"] == pipeline
            )
            for f in ["f01", "f04"]
            for c in conditions
        ]
        values = np.full((10, 5), np.nan)
        for i, j in enumerate(jobs):
            for k, m in enumerate(METHODS):
                if m in tr[j["id"]]["scores"]:
                    values[i, k] = (
                        tr[j["id"]]["scores"][m] / tf["thresholds"][j["family"]][m]
                    )
        image = ax.pcolormesh(
            np.arange(6) - 0.5,
            np.arange(11) - 0.5,
            np.log10(values),
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=1.2),
        )
        ax.set_ylim(9.5, -0.5)
        ax.set_xlim(-0.5, 4.5)
        ax.set_title("Pipeline " + pipeline, fontsize=10, pad=10)
        ax.set_xticks(
            range(5), [r"$\Delta$", r"$F$", "Median", "q95", "RMS"], fontsize=8.5
        )
        ax.set_yticks(
            range(10),
            [f + " / " + c for f in ["f01", "f04"] for c in short],
            fontsize=8.5,
        )
        ax.tick_params(length=0)
        ax.axhline(4.5, c="white", lw=1.5)
        for i in range(10):
            for k in range(5):
                v = values[i, k]
                text = (
                    "A" if not np.isfinite(v) else f"{v:.2f}" + ("+" if v > 1 else "-")
                )
                ax.text(
                    k,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    color="white"
                    if np.isfinite(v) and abs(np.log10(v)) > 0.6
                    else "#172f3a",
                )
        transfer[pipeline] = [
            [None if not np.isfinite(z) else float(z) for z in row] for row in values
        ]
    fig.subplots_adjust(left=0.23, right=0.87, top=0.90, bottom=0.16)
    cax = fig.add_axes([0.90, 0.2, 0.012, 0.61])
    bar = fig.colorbar(image, cax=cax)
    bar.solids.set_rasterized(False)
    bar.set_label("log10(score / fixed cutoff)", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    fig.text(
        0.57,
        0.055,
        "+ above cutoff, - at/below cutoff; A numerical abstention",
        ha="center",
        fontsize=8,
        color="#324b57",
    )
    save(fig, "appearance_complete_transfer")
    display["transfer"] = dict(
        methods=METHODS, ratios=transfer, decision_signs_use_unrounded_values=True
    )
    (OUT / "DISPLAY_DATA.json").write_text(
        json.dumps(display, indent=2, allow_nan=False) + "\n"
    )
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir()
        if p.is_file()
    }
    (OUT / "RESULT_FIGURE_RECEIPT.json").write_text(
        json.dumps(
            dict(
                status="GENERATED_PENDING_VISUAL_REVIEW",
                inputs=USED,
                outputs=outputs,
                source_sha256=hashlib.sha256(source).hexdigest(),
                no_inference_or_decision_change=True,
                matplotlib=mpl.__version__,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("THREE COMPLETE RESULT FIGURES CLOSED")


if __name__ == "__main__":
    main()
