"""Small manufactured arithmetic example; no anatomical/physical validation."""

from pathlib import Path
import argparse, json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    model = out / "model"
    model.mkdir()
    rng = np.random.default_rng(923)
    n = 120
    x = np.column_stack(
        [rng.uniform(-1, 1, n), rng.uniform(-0.6, 0.6, n), rng.uniform(4, 7, n)]
    )
    k = np.array([450.0, 452.0, 320.0, 240.0])
    poses = {i: np.array([0.15 * i, 0.05 * np.sin(i), 0.0]) for i in range(1, 7)}
    (model / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 450 452 320 240\n"
    )
    lines = ["# Image list\n", "# POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    for i, c in poses.items():
        pixel = (
            (x - c)[:, :2] / (x - c)[:, 2, None] * k[:2]
            + k[2:]
            + rng.normal(0, 0.2, (n, 2))
        )
        lines.append(f"{i} 1 0 0 0 {-c[0]} {-c[1]} {-c[2]} 1 {i:04d}.png\n")
        lines.append(
            " ".join(f"{uv[0]:.17g} {uv[1]:.17g} {j + 1}" for j, uv in enumerate(pixel))
            + "\n"
        )
    (model / "images.txt").write_text("".join(lines))
    lines = ["# 3D point list\n"]
    for j, point in enumerate(x):
        lines.append(
            f"{j + 1} "
            + " ".join(format(float(v), ".17g") for v in point)
            + " 128 128 128 0 "
            + " ".join(f"{i} {j}" for i in poses)
            + "\n"
        )
    (model / "points3D.txt").write_text("".join(lines))
    d = 0.002 * rng.normal(size=x.shape)
    d[:, 2] += 0.01 * np.sin(x[:, 0] * 4)
    np.savez_compressed(out / "pairs.npz", positions=x, residuals=d)
    (out / "compose.json").write_text(
        json.dumps(
            dict(
                provenance="independent",
                evidence=dict(rdsc=dict(score=[2.0, 2.0], cutoff=[1.0, 1.0])),
            ),
            indent=2,
        )
        + "\n"
    )
    print("Created arithmetic examples at", out)


if __name__ == "__main__":
    main()
