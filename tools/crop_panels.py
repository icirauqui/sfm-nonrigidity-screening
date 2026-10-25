"""Crop production panels from complete figures; captions remain authoritative."""

from pathlib import Path
import argparse, hashlib, json
import pymupdf

p = argparse.ArgumentParser()
p.add_argument("--figures", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
FIGURES = a.figures
OUT = a.output
OUT.mkdir(parents=True, exist_ok=False)
CROPS = {
    "core_tubes_views": [
        (0, 0, 0.5, 0.43),
        (0.5, 0, 1, 0.43),
        (0, 0.43, 0.5, 1),
        (0.5, 0.43, 1, 1),
    ],
    "controlled_motion_views": [
        (0, 0, 0.33, 0.82),
        (0.33, 0, 0.66, 0.82),
        (0.66, 0, 1, 0.82),
        (0, 0.83, 1, 1),
    ],
    "bundles_geometry": [
        (0, 0, 0.5, 0.50),
        (0.5, 0, 1, 0.50),
        (0, 0.50, 0.5, 0.93),
        (0.5, 0.50, 1, 0.93),
        (0, 0.93, 1, 1),
    ],
    "global_modal_explanation": [
        (0, 0, 1, 0.31),
        (0, 0.31, 0.365, 1),
        (0.365, 0.31, 0.69, 0.85),
        (0.69, 0.31, 1, 0.85),
    ],
    "rdsc_spatial_fields": [(i / 3, 0, (i + 1) / 3, 0.40) for i in range(3)]
    + [(i / 3, 0.40, (i + 1) / 3, 0.82) for i in range(3)]
    + [(0, 0.82, 1, 1)],
    "basc_provenance_routing": [(0, 0, 0.43, 1), (0.43, 0, 1, 1)],
    "core_global_readouts": [(0, 0, 1, 0.47), (0, 0.47, 1, 1)],
    "core_complete_decisions": [(0, 0, 0.60, 1), (0.605, 0, 1, 1)],
    "appearance_complete_transfer": [(0, 0, 0.54, 1), (0.56, 0, 1, 1)],
}
manifest = {}
for name, rects in CROPS.items():
    src = FIGURES / f"{name}.pdf"
    with pymupdf.open(src) as doc:
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        for i, (a, b, c, d) in enumerate(rects, 1):
            clip = pymupdf.Rect(a * w, b * h, c * w, d * h)
            out = pymupdf.open()
            q = out.new_page(width=clip.width, height=clip.height)
            q.show_pdf_page(q.rect, doc, 0, clip=clip)
            dest = OUT / f"{name}_panel_{i:02d}.pdf"
            out.save(dest, garbage=4, deflate=True, no_new_id=True)
            out.close()
            manifest[dest.name] = dict(
                source=src.name,
                source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                normalized_clip=[a, b, c, d],
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                interpret_only_with_complete_figure_caption=True,
            )
(OUT / "PANEL_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
(OUT / "README.txt").write_text(
    "Production panel crops. The complete figures and their captions are authoritative. Crops preserve vector text and original image streams; shared legends/scales may be separate. They are assets for layout, not new scientific views. Never infer denominators or meaning from a crop alone.\n"
)
print(len(manifest), "production panels")
