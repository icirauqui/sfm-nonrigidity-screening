"""Reproducible journal artwork from retained data and explicit schematics.

No inference, optimization, calibration or scientific-image synthesis. Reference
surfaces/RGB are display-only; estimated-frame panels never consume truth poses.
"""

from pathlib import Path
import argparse, hashlib, json, os
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
import pymupdf

ROOT = Path(".").resolve()
PREP = ROOT / "display"
DATA = ROOT / "source"
TEAL = "#176b84"
ORANGE = "#bd6b1a"
PURPLE = "#7552a3"
INK = "#203741"
GRAY = "#697e85"
LIGHT = "#e5edef"
USED = {}
RECORDS = {}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a9ba0",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
    }
)
horizontal = np.array([0.2, 1, 0.0])
horizontal /= np.linalg.norm(horizontal)
vertical = np.array([-0.3, 0.06, 1.0])
vertical /= np.linalg.norm(vertical)
PROJ = np.stack([horizontal, vertical], axis=1)
DEPTH = np.cross(horizontal, vertical)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def npz(p):
    USED[str(p.relative_to(ROOT))] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    assert all(
        np.isfinite(v).all()
        for v in result.values()
        if np.issubdtype(v.dtype, np.number)
    ), p
    return result


def source(name):
    expected = read(PREP / "ILLUSTRATION_ONLY_INPUTS.json")["files"][name]
    p = DATA / name
    assert sha(p) == expected["sha256"]
    USED[str(p.relative_to(ROOT))] = sha(p)
    return p


def geometry(family):
    return npz(source("benchmark/families/" + family + "/evaluation/geometry.npz"))


def state(pkg):
    return npz(source(f"benchmark/v{pkg}/evaluation/moving/states.npz"))


def rgb(pkg):
    return np.sqrt(
        np.asarray(
            Image.open(
                source(f"benchmark/v{pkg}/public/moving/rgb/000052.png")
            ).convert("RGB"),
            dtype=float,
        )
        / 255
    )


def save(fig, name):
    fig.canvas.draw()
    fig.savefig(OUT / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)


def panel_label(fig, x, y, text):
    fig.text(
        x, y, text, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK
    )


def clean(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def dot(x, y):
    result = np.einsum("...i,ij->...j", x, y)
    assert np.isfinite(result).all()
    return result


def mesh2(
    ax, vertices, faces, color=TEAL, values=None, norm=None, alpha=1.0, cutaway=False
):
    tri = vertices[faces]
    keep = np.ones(len(tri), bool)
    if cutaway:
        keep = tri.mean(1)[:, 2] <= np.median(vertices[:, 2])
        tri = tri[keep]
    order = np.argsort(np.einsum("ij,j->i", tri.mean(1), DEPTH))
    tri = tri[order]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    light = np.array([-0.5, -0.3, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(np.einsum("ij,j->i", normals, light))
    if values is None:
        colors = np.asarray(mpl.colors.to_rgb(color))[None, :] * shade[:, None]
    else:
        colors = mpl.colormaps["magma"](norm(np.asarray(values)[keep][order]))
        colors[:, :3] *= 0.72 + 0.28 * shade[:, None]
    projected = dot(tri, PROJ)
    ax.add_collection(
        PolyCollection(
            projected,
            facecolors=colors,
            edgecolors="none",
            alpha=alpha,
            rasterized=True,
            zorder=1,
        )
    )
    return projected


def wire2(ax, vertices, alpha=0.55):
    rings = vertices.reshape(112, 48, 3)
    for i in (0, 27, 55, 83, 111):
        p = dot(np.vstack([rings[i], rings[i, 0]]), PROJ)
        ax.plot(*p.T, c=GRAY, lw=0.5, alpha=alpha, zorder=4)
    for j in (0, 12, 24, 36):
        ax.plot(*dot(rings[:, j], PROJ).T, c=GRAY, lw=0.45, alpha=alpha, zorder=4)


def box(ax, x, y, w, h, text, color=TEAL, fontsize=9, face="#f1f5f6"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        edgecolor=color,
        facecolor=face,
        lw=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.35,
    )


def arrow(ax, a, b, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def glyphs(ax, centers, rotations, project, color, size=0.03, indices=None):
    indices = range(len(centers)) if indices is None else indices
    lines = []
    for i in indices:
        c = centers[i]
        corners = (
            c
            + dot(
                np.array(
                    [[-0.7, -0.5, 1], [0.7, -0.5, 1], [0.7, 0.5, 1], [-0.7, 0.5, 1]]
                ),
                rotations[i].T,
            )
            * size
        )
        lines.extend([project(np.stack([c, k])) for k in corners])
        lines.append(project(np.vstack([corners, corners[0]])))
    ax.add_collection(LineCollection(lines, colors=color, linewidths=0.65, zorder=7))


def limits(ax, points, pad=0.07):
    lo, hi = points.min(0), points.max(0)
    margin = np.maximum(hi - lo, 0.05) * pad
    ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
    ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
    ax.set_aspect("equal")


def chart(g, x):
    return dot(x - g["center"], g["chart"][:, :2]) / g["length"]


def atlas():
    fig = plt.figure(figsize=(7.1, 2.7))
    families = ["f01", "f02", "f03", "f04"]
    geos = [geometry(f) for f in families]
    R = float(np.median(geos[0]["base_radius"]))
    projected = [dot(g["vertices"] / R, PROJ) for g in geos]
    centers = [(p.min(0) + p.max(0)) / 2 for p in projected]
    span = np.max(np.stack([np.ptp(p, axis=0) for p in projected]), axis=0) + np.array(
        [0.55, 0.40]
    )
    titles = ["Straight", "Right bend", "Upward bend", "Two bends"]
    axes = []
    for j, (family, g, pkg, label, center) in enumerate(
        zip(families, geos, [1, 4, 7, 10], titles, centers)
    ):
        left = 0.01 + (j % 2) * 0.50
        center_y = 2.04 if j < 2 else 0.74
        bottom = (center_y - 0.558) / 2.7
        ax = fig.add_axes([left, bottom, 0.325, 1.116 / 2.7])
        offset = np.einsum("ij,j->i", PROJ, center)
        mesh2(ax, g["vertices"] / R - offset, g["triangles"])
        ax.set_xlim(-span[0] / 2, span[0] / 2)
        ax.set_ylim(-span[1] / 2, span[1] / 2)
        clean(ax)
        axes.append(ax)
        panel_label(
            fig,
            left,
            (2.66 if j < 2 else 1.48) / 2.7,
            f"({chr(97 + j)}) {family} · {label}",
        )
        view = fig.add_axes([left + 0.34, bottom, 0.14, 1.116 / 2.7])
        view.imshow(rgb(pkg), interpolation="none")
        view.axis("off")
    # Translation-only centering; identical limits and data-to-page scale in every mesh panel.
    ax = axes[-1]
    start = np.array([-span[0] / 2 + 0.55, -span[1] / 2 + 0.70])
    ax.plot([start[0], start[0] + 4], [start[1], start[1]], c=INK, lw=1.4)
    ax.text(
        start[0] + 2,
        start[1] - 0.23,
        r"$4R_{\mathrm{f01}}$",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.canvas.draw()
    metric = [a.transData.get_affine().get_matrix()[:2, :2] for a in axes]
    assert all(np.allclose(m, metric[0], rtol=0, atol=1e-10) for m in metric)
    assert np.isclose(metric[0][0, 0], metric[0][1, 1])
    inches_per_radius = float(metric[0][0, 0] / fig.dpi)
    save(fig, "core_tubes_views")
    RECORDS["atlas"] = dict(
        families=families,
        static_packages=[1, 4, 7, 10],
        frame=52,
        layout="two_by_two_geometry_left_rgb_right",
        canvas_inches=[7.1, 2.7],
        common_inches_per_radius=inches_per_radius,
        shared_generating_scale=R,
        projection=PROJ.tolist(),
        projected_bbox_centers=[c.tolist() for c in centers],
        common_projected_span=span.tolist(),
        centering_is_translation_only=True,
        rgb_display_gamma=0.5,
        native_rgb=[640, 480],
        native_rgb_min_dpi=640 / (7.1 * 0.14),
    )


def motion():
    g = geometry("f01")
    R = float(np.median(g["base_radius"]))
    ref = g["vertices"] / R
    faces = g["triangles"]
    states = [state(p) for p in (1, 2, 3)]
    assert all(np.array_equal(states[0]["C_world"], s["C_world"]) for s in states)
    norm = mpl.colors.Normalize(0, 0.08)
    fig = plt.figure(figsize=(7.1, 2.9))
    allv = np.concatenate([s["vertices"][52] / R for s in states])
    bounds = allv @ PROJ
    for j, (s, pkg, title) in enumerate(
        zip(states, [1, 2, 3], ["Static", "FE-derived", "Prescribed non-FE"])
    ):
        x = 0.03 + j * 0.33
        panel_label(fig, x, 0.98, f"({chr(97 + j)}) {title}")
        ax = fig.add_axes([x, 0.72, 0.29, 0.19])
        v = s["vertices"][52] / R
        values = np.linalg.norm(s["displacement"][52], axis=1) / R
        mesh2(ax, v, faces, values=values[faces].mean(1), norm=norm)
        wire2(ax, ref)
        limits(ax, bounds, 0.035)
        clean(ax)
        image = fig.add_axes([x, 0.20, 0.285, 0.49])
        image.imshow(rgb(pkg), interpolation="none")
        image.axis("off")
    barax = fig.add_axes([0.24, 0.145, 0.52, 0.025])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="magma"),
        cax=barax,
        orientation="horizontal",
    )
    bar.ax.tick_params(labelsize=8)
    bar.set_label(
        r"Source surface displacement / reference radius $R$", fontsize=8, labelpad=1
    )
    save(fig, "controlled_motion_views")
    RECORDS["motion"] = dict(
        packages=[1, 2, 3],
        frame=52,
        source_geometry_magnification=1,
        color_limits=[0, 0.08],
        radius=R,
        rgb_display_gamma=0.5,
        native_rgb_min_dpi=640 / (7.1 * 0.285),
    )


def reconstruction():
    seq = "f01__fem"
    g = npz(PREP / (seq + "_global.npz"))
    sep = npz(PREP / (seq + "_independent.npz"))
    inh = npz(PREP / (seq + "_inherit_global.npz"))
    fig = plt.figure(figsize=(7.1, 5.0))
    p = chart(g, g["points"])
    cam = chart(g, g["camera_centers"])
    lo, hi = np.quantile(p, [0.01, 0.99], axis=0)
    ax = fig.add_axes([0.08, 0.59, 0.39, 0.32])
    ax.scatter(*p.T, s=0.25, c=g["colors"], rasterized=True, zorder=2)
    ax.plot(*cam.T, c=INK, lw=1, zorder=4)
    idx = np.where(np.isin(g["camera_ids"], sep["local_ids"]))[0]
    ax.plot(*cam[idx].T, c=ORANGE, lw=2, zorder=6)
    limits(ax, np.vstack([p, cam]), 0.05)
    ax.add_patch(Rectangle(lo, *(hi - lo), facecolor="none", edgecolor=ORANGE, lw=1))
    ax.set_xlabel(r"Display coordinate $u/L_g$")
    ax.set_ylabel(r"Display coordinate $v/L_g$")
    panel_label(fig, 0.04, 0.98, "(a) Full global reconstruction")
    fig.text(
        0.04, 0.94, f"{len(p):,} points • 90 estimated cameras", fontsize=8, color=GRAY
    )
    ax = fig.add_axes([0.57, 0.59, 0.39, 0.32])
    ax.scatter(*p.T, s=0.55, c=g["colors"], rasterized=True, zorder=2)
    ax.plot(*cam.T, c=INK, lw=1, zorder=4)
    ax.plot(*cam[idx].T, c=ORANGE, lw=2, zorder=6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    glyphs(
        ax,
        g["camera_centers"],
        g["camera_rotations"],
        lambda x: chart(g, x),
        INK,
        size=0.025 * g["length"],
        indices=idx[::7],
    )
    ax.set_xlabel(r"$u/L_g$")
    ax.set_ylabel(r"$v/L_g$")
    panel_label(fig, 0.53, 0.98, "(b) Detail of the orange box")
    fig.text(
        0.53, 0.94, "Fixed 1st–99th percentile display bounds", fontsize=8, color=GRAY
    )
    pp = np.concatenate([chart(g, d[k]) for d in (sep, inh) for k in ("x", "y")])
    bounds = np.stack([pp.min(0), pp.max(0)])
    for j, (d, title) in enumerate(
        [(sep, "Separately estimated local cameras"), (inh, "Inherited global cameras")]
    ):
        x = 0.08 + j * 0.49
        ax = fig.add_axes([x, 0.13, 0.39, 0.27])
        a, b = chart(g, d["x"]), chart(g, d["y"])
        ax.scatter(
            *a.T,
            s=0.55,
            c=TEAL,
            alpha=0.75,
            rasterized=True,
            label="Global paired points",
        )
        ax.scatter(
            *b.T,
            s=0.50,
            c=ORANGE,
            alpha=0.60,
            rasterized=True,
            label="Aligned local points",
        )
        c = chart(g, d["local_centers"])
        ax.plot(*c.T, c=PURPLE, lw=1.2, zorder=6)
        glyphs(
            ax,
            d["local_centers"],
            d["local_rotations"],
            lambda x: chart(g, x),
            PURPLE,
            size=0.025 * g["length"],
            indices=[0, 7, 14],
        )
        limits(ax, bounds, 0.06)
        ax.set_xlabel(r"$u/L_g$")
        ax.set_ylabel(r"$v/L_g$")
        panel_label(
            fig,
            x - 0.04,
            0.48,
            "(c) Separate local cameras" if j == 0 else "(d) Inherited local cameras",
        )
        fig.text(
            x - 0.04,
            0.435,
            f"{len(a):,} paired points • local frames 40–54",
            fontsize=8,
            color=GRAY,
        )
    fig.legend(
        *fig.axes[2].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=8,
    )
    save(fig, "bundles_geometry")
    RECORDS["reconstruction"] = dict(
        sequence=seq,
        window=4,
        full_global_points=len(p),
        paired_points={"separate": len(sep["x"]), "inherited": len(inh["x"])},
        detail_bounds=np.stack([lo, hi]).tolist(),
        full_extent=p.min(0).tolist() + p.max(0).tolist(),
        camera_glyphs="fixed depth, orientations only; not calibrated fields of view",
        no_truth_alignment=True,
        display_chart=g["chart"].tolist(),
    )


def global_modes():
    fig = plt.figure(figsize=(7.1, 3.85))
    flow = fig.add_axes([0.01, 0.70, 0.98, 0.29])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    boxes = [
        (0.02, 0.33, 0.22, 0.53, "Global image\nresiduals $r$"),
        (0.36, 0.33, 0.28, 0.53, "Remove point/camera\nand affine temporal span"),
        (0.77, 0.33, 0.21, 0.53, "Quadratic\nmodal energy"),
    ]
    for x, y, w, h, t in boxes:
        box(flow, x, y, w, h, t)
    arrow(flow, (0.25, 0.6), (0.34, 0.6))
    arrow(flow, (0.65, 0.6), (0.75, 0.6))
    flow.text(
        0.5,
        0.06,
        "One global model; static calibration; explicit numerical abstention",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    d = npz(PREP / "f01__fem_image_plane.npz")
    ax = fig.add_axes([0.07, 0.20, 0.26, 0.38])
    res = d["residual"]
    m = np.linalg.norm(res, axis=1)
    image = ax.scatter(
        *d["observed"].T,
        c=m,
        s=0.65,
        cmap="viridis",
        norm=mpl.colors.Normalize(0, 1.5),
        rasterized=True,
    )
    ii = np.linspace(0, len(m) - 1, min(60, len(m)), dtype=int)
    ax.quiver(
        *d["predicted"][ii].T,
        *(20 * res[ii]).T,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=INK,
        width=0.0025,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Image x (pixels)")
    ax.set_ylabel("Image y (pixels)")
    ax.set_aspect("equal")
    panel_label(fig, 0.025, 0.65, "(a) Global pixel residuals")
    cax = fig.add_axes([0.09, 0.095, 0.22, 0.018])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", extend="max")
    bar.ax.tick_params(labelsize=7)
    bar.set_label("Residual norm (pixels)", fontsize=7.5, labelpad=1)
    g = geometry("f01")
    R = float(np.median(g["base_radius"]))
    ref = g["vertices"] / R
    faces = g["triangles"]
    center = ref.mean(0)
    L = float(np.quantile(np.linalg.norm(ref - center, axis=1), 0.9))
    coord = (ref - center) / L
    indices = np.asarray([ring * 48 + 12 for ring in range(4, 108, 7)])
    for j, quadratic in enumerate((False, True)):
        x = 0.385 + j * 0.31
        ax = fig.add_axes([x, 0.22, 0.29, 0.34])
        mesh2(ax, ref, faces, color="#c5d7dc")
        field = np.zeros_like(ref)
        field[:, 2] = coord[:, 1] ** (2 if quadratic else 1)
        field *= 1.1
        origin = dot(ref, PROJ)
        uv = dot(field, PROJ)
        ax.quiver(
            *origin[indices].T,
            *uv[indices].T,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=ORANGE,
            width=0.008,
        )
        bounds = np.vstack([origin, dot(ref + field, PROJ)])
        limits(ax, bounds, 0.15)
        clean(ax)
        panel_label(
            fig, x, 0.65, "(c) Quadratic mode" if quadratic else "(b) Affine nuisance"
        )
        ax.text(
            0.5,
            -0.04,
            r"$u\propto p_y^2\tau\,e_z$" if quadratic else r"$u\propto p_y\tau\,e_z$",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=INK,
        )
    fig.text(
        0.67,
        0.10,
        "Basis examples only; arbitrary amplitude",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    save(fig, "global_modal_explanation")
    RECORDS["global_modes"] = dict(
        residual_sequence="f01__fem",
        image=str(d["image_name"]),
        observations=len(m),
        residual_arrow_magnification=20,
        residual_color_max_pixels=1.5,
        mesh_basis_example_only=True,
        source_geometry_never_used_by_modal_inference=True,
    )


def rdsc_fields():
    fig = plt.figure(figsize=(7.1, 3.1))
    norm = mpl.colors.LogNorm(1e-6, 0.1)
    stages = [
        ("raw", "Aligned difference $d$"),
        ("affine_residual", "Affine residual $e$"),
        ("coherent", "Coherent mean $c$"),
    ]
    records = []
    for row, seq in enumerate(("f01__fem", "f03__fem")):
        g = npz(PREP / (seq + "_global.npz"))
        d = npz(PREP / (seq + "_independent.npz"))
        x = dot(d["x"] - d["x"].mean(0), g["chart"][:, :2]) / d["extent"]
        lo, hi = x.min(0), x.max(0)
        for col, (key, title) in enumerate(stages):
            ax = fig.add_axes([0.035 + col * 0.33, 0.59 - row * 0.39, 0.30, 0.25])
            v = d[key] / d["extent"]
            value = np.linalg.norm(v, axis=1)
            ax.scatter(
                *x.T,
                c=np.maximum(value, 1e-6),
                s=0.45,
                cmap="magma",
                norm=norm,
                rasterized=True,
            )
            limits(ax, np.stack([lo, hi]), 0.1)
            ax.axis("off")
            panel_label(
                fig,
                0.035 + col * 0.33,
                0.98 - row * 0.39,
                f"({chr(97 + row * 3 + col)}) " + title,
            )
        fig.text(
            0.035,
            0.89 - row * 0.39,
            ("f01 / FE-derived" if row == 0 else "f03 / FE-derived")
            + f" • window 4 • {len(x):,} pairs",
            fontsize=8,
            color=GRAY,
        )
        records.append(
            dict(
                sequence=seq,
                window=4,
                paired_points=len(x),
                axes_use_matched_reference_extent=float(d["extent"]),
                coordinate_limits=np.stack([lo, hi]).tolist(),
                vector_directions_not_shown=True,
            )
        )
    barax = fig.add_axes([0.25, 0.145, 0.50, 0.022])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="magma"),
        cax=barax,
        orientation="horizontal",
        extend="both",
    )
    bar.set_label(
        r"Vector magnitude / matched extent $L_w$ (log scale)", fontsize=8, labelpad=2
    )
    bar.ax.tick_params(labelsize=8)
    save(fig, "rdsc_spatial_fields")
    RECORDS["rdsc_fields"] = dict(
        cases=records,
        color_limits=[1e-6, 0.1],
        color_floor_only_not_score_floor=True,
        all_points_retained=True,
        closed_scalar_parity=True,
    )


def basc_router():
    fig = plt.figure(figsize=(7.1, 4.1))
    g = npz(PREP / "f01__fem_global.npz")
    sep = npz(PREP / "f01__fem_independent.npz")
    inh = npz(PREP / "f01__fem_inherit_global.npz")
    global_by_id = dict(zip(g["camera_ids"], g["camera_centers"]))
    ax = fig.add_axes([0.085, 0.22, 0.31, 0.55])
    vectors = {}
    for data, label, color, style in [
        (sep, "Separate cameras", TEAL, "-"),
        (inh, "Inherited cameras", ORANGE, "--"),
    ]:
        ids = data["local_ids"]
        reference = np.array([global_by_id[i] for i in ids])
        values = np.linalg.norm(data["local_centers"] - reference, axis=1) / g["length"]
        vectors[label] = values.tolist()
        ax.plot(np.arange(40, 55), values, style, color=color, label=label, lw=1.6)
    ax.set_xticks([40, 43, 46, 49, 52, 54])
    ax.set_xlabel("Registered image index")
    ax.set_ylabel(r"Camera-center difference / $L_g$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(
        frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20)
    )
    panel_label(fig, 0.03, 0.96, "(a) Camera provenance")
    fig.text(
        0.03,
        0.88,
        "f01 / FE-derived, window 4\nCamera-derived alignment",
        fontsize=8,
        color=GRAY,
    )
    flow = fig.add_axes([0.46, 0.07, 0.53, 0.78])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    panel_label(fig, 0.46, 0.96, "(b) BASC-C routing")
    box(flow, 0.025, 0.78, 0.43, 0.18, "Separate local cameras", TEAL, 8.4)
    box(flow, 0.55, 0.78, 0.43, 0.18, "Inherited global cameras", ORANGE, 8.4)
    box(
        flow,
        0.025,
        0.49,
        0.43,
        0.17,
        "Transfer decision\nthen raw-q95 if unavailable",
        TEAL,
        8.3,
    )
    box(
        flow,
        0.55,
        0.49,
        0.43,
        0.17,
        "Point JAPD OR edge\nonly if both decisive",
        ORANGE,
        8.3,
    )
    arrow(flow, (0.24, 0.77), (0.24, 0.68), TEAL)
    arrow(flow, (0.765, 0.77), (0.765, 0.68), ORANGE)
    box(
        flow,
        0.17,
        0.22,
        0.66,
        0.18,
        "RDSC fallback\nif preferred answer unavailable",
        PURPLE,
        8.5,
    )
    arrow(flow, (0.24, 0.48), (0.35, 0.41), PURPLE)
    arrow(flow, (0.765, 0.48), (0.65, 0.41), PURPLE)
    box(
        flow, 0.17, 0.025, 0.66, 0.12, "Unknown / mixed provenance:\nabstain", GRAY, 8.2
    )
    flow.text(
        0.5,
        0.17,
        "Decisive preferred answer is preserved",
        ha="center",
        fontsize=7.6,
        color=GRAY,
    )
    save(fig, "basc_provenance_routing")
    RECORDS["basc"] = dict(
        sequence="f01__fem",
        window=4,
        normalized_camera_center_differences=vectors,
        no_physical_pose_accuracy_claim=True,
        router_schematic_matches_original=True,
    )


def graphical_abstract():
    fig = plt.figure(figsize=(5.5, 2.2))
    g = npz(PREP / "f01__fem_global.npz")
    p = chart(g, g["points"])
    lo, hi = np.quantile(p, [0.01, 0.99], axis=0)
    ax = fig.add_axes([0.01, 0.26, 0.25, 0.53])
    ax.scatter(*p.T, c=g["colors"], s=0.6, rasterized=True)
    c = chart(g, g["camera_centers"])
    ax.plot(*c.T, c=INK, lw=0.8)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    clean(ax)
    fig.text(
        0.135,
        0.94,
        "Estimated SfM\nevidence",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.135,
        0.19,
        "Global model\n+ optional locals",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    flow = fig.add_axes([0.29, 0.16, 0.70, 0.73])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    for y, name, color in [
        (0.7, "Global modal residuals", TEAL),
        (0.4, "RDSC spatial coherence", PURPLE),
        (0.1, "BASC-C expert composition", ORANGE),
    ]:
        box(flow, 0.04, y, 0.59, 0.19, name, color, 8.5)
        arrow(flow, (0.65, y + 0.095), (0.78, 0.495), color)
    box(flow, 0.80, 0.35, 0.18, 0.29, "Screen /\nabstain", GRAY, 8.2)
    fig.text(
        0.65, 0.95, "Three alternative strategies", ha="center", fontsize=8.5, color=INK
    )
    fig.text(
        0.5,
        0.07,
        "12 synthetic acquisitions • conditional trade-offs",
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.text(
        0.5,
        0.015,
        "Retained-model detail; Codex-assisted visualization code",
        ha="center",
        fontsize=7,
        color=GRAY,
    )
    save(fig, "graphical_abstract")
    RECORDS["graphical_abstract"] = dict(
        canvas_inches=[5.5, 2.2],
        minimum_preview_width=1328,
        detail_bounds=np.stack([lo, hi]).tolist(),
        not_a_full_extent_reconstruction_panel=True,
    )
    # Dedicated journal raster export at 450dpi, exceeding minimum1328px width.
    with pymupdf.open(OUT / "graphical_abstract.pdf") as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(450 / 72, 450 / 72), alpha=False)
        pix.set_dpi(450, 450)
        pix.save(OUT / "graphical_abstract.png")


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    OUT = a.output
    OUT.mkdir(exist_ok=False)
    frozen_source = Path(__file__).read_bytes()
    (OUT / "SOURCE_FIGURES.py").write_bytes(frozen_source)
    assert read(PREP / "MODEL_DISPLAY_CLOSED.json")["status"] == "PASS"
    assert read(PREP / "ILLUSTRATION_ONLY_INPUTS.json")["not_inference_inputs"]
    for function in (
        atlas,
        motion,
        reconstruction,
        global_modes,
        rdsc_fields,
        basc_router,
        graphical_abstract,
    ):
        function()
        print("FIGURE CLOSED", function.__name__, flush=True)
    # Every used scientific/display input is rechecked, all final outputs hash-bound.
    for n, h in USED.items():
        assert sha(ROOT / n) == h, n
    receipt = dict(
        status="GENERATED_PENDING_VISUAL_REVIEW",
        source_sha256=hashlib.sha256(frozen_source).hexdigest(),
        inputs=USED,
        outputs={
            p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in OUT.iterdir()
            if p.is_file()
        },
        records=RECORDS,
        versions=dict(
            python=__import__("sys").version,
            numpy=np.__version__,
            matplotlib=mpl.__version__,
            pymupdf=pymupdf.VersionBind,
        ),
        raster_layer_dpi=600,
        reference_geometry_display_only=True,
        source_rgb_gamma=0.5,
        source_rgb_values_not_overwritten=True,
        no_new_inference_or_optimization=True,
    )
    (OUT / "FIGURE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
