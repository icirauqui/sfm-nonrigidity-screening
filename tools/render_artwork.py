"""Render the nine complete figures, graphical abstract and production panels.

Only the separately prepared display-only input bundle is accepted here.
"""

from pathlib import Path
import argparse, hashlib, json, sys, subprocess
import artwork as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.inputs.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    hashes = json.loads((root / "SHA256.json").read_text())
    for n, h in hashes.items():
        assert F.sha(root / n) == h, n
    F.ROOT = root
    F.PREP = root / "display"
    F.DATA = root / "source"
    F.OUT = out
    for fn in [
        F.atlas,
        F.motion,
        F.reconstruction,
        F.global_modes,
        F.rdsc_fields,
        F.basc_router,
        F.graphical_abstract,
    ]:
        fn()
    from matplotlib.text import Text

    def save(fig, name):
        if name == "global_modal_explanation":
            fig.set_size_inches(7.1, 4.2)
            fig.axes[1].set_position([0.10, 0.24, 0.25, 0.33])
            fig.axes[2].set_position([0.12, 0.11, 0.22, 0.018])
        if name == "bundles_geometry":
            pos = fig.axes[2].get_position()
            fig.axes[2].set_position([pos.x0 + 0.025, pos.y0, pos.width, pos.height])
            for t in fig.texts:
                x, y = t.get_position()
                if abs(y - 0.94) < 1e-6:
                    t.set_position((x, 0.925))
                if abs(y - 0.435) < 1e-6:
                    t.set_position((x, 0.42))
        minimum = 8.3 if name == "graphical_abstract" else 9.2
        for t in fig.findobj(match=Text):
            if t.get_text():
                t.set_fontsize(max(t.get_fontsize(), minimum))
        fig.canvas.draw()
        if name == "global_modal_explanation":
            renderer = fig.canvas.get_renderer()
            label = fig.axes[1].xaxis.label.get_window_extent(renderer)
            bar = fig.axes[2].get_window_extent(renderer)
            assert label.y0 - bar.y1 >= 6
            assert fig.axes[2].xaxis.label.get_window_extent(renderer).y0 >= 4
        fig.savefig(
            out / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None}
        )
        fig.savefig(out / (name + ".png"), dpi=180)
        F.plt.close(fig)

    F.save = save
    F.plt.rcParams.update(
        {"axes.labelsize": 9.2, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2}
    )
    for fn in [F.reconstruction, F.global_modes, F.rdsc_fields, F.graphical_abstract]:
        fn()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("result_artwork.py")),
            "--inputs",
            str(root),
            "--output",
            str(out / "results"),
        ],
        check=True,
    )
    for f in (out / "results").glob("*"):
        if f.suffix in [".pdf", ".png"]:
            f.replace(out / f.name)
    for n, h in F.USED.items():
        assert F.sha(root / n) == h, n
    result = dict(
        status="GENERATED",
        display_only=True,
        files={p.name: F.sha(p) for p in out.iterdir() if p.suffix in [".pdf", ".png"]},
        inputs=hashes,
    )
    (out / "ARTWORK_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("Rendered nine figures and graphical abstract")


if __name__ == "__main__":
    main()
