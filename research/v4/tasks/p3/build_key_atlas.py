"""Rebuild the P3 keyboard glyph atlas with the pinned OFL Noto Sans font.

Only this task's texture changes. UV layout / geometry come from the
project-authored v2 mesh, and the v2 source assets remain untouched.
"""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from research.v2.src.safevla_v2.electronics_assets_v2 import layout

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
FONT = ASSETS / "NotoSans-Regular.ttf"
NOTICE = ASSETS / "OFL-NOTICE.txt"
ATLAS = ASSETS / "key_legends_noto.png"
MANIFEST = ASSETS / "key_legends_noto_manifest.json"
FONT_SHA256 = "89c3c497f618fdaa0b2d1e98fef93582f28c71debd2c4a8cdf41f190ced2909d"
NOTICE_SHA256 = "075f91382875da1c4b491f1facf2586f55448705133264d98ce15cab6211b125"
SOURCE_URL = "https://github.com/notofonts/latin-greek-cyrillic"
CELL = 128
COLS = 16


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render():
    if _sha(FONT) != FONT_SHA256 or _sha(NOTICE) != NOTICE_SHA256:
        raise RuntimeError("P3 Noto Sans font or OFL notice changed")
    keys = layout()
    rows = (len(keys) + COLS - 1) // COLS
    image = Image.new("RGB", (COLS * CELL, rows * CELL), "#080a0b")
    draw = ImageDraw.Draw(image)
    occupied = []
    for index, key in enumerate(keys):
        source_label = key["label"]
        # "Meta" is also a current company name; use an unbranded key legend.
        label = "Super" if source_label == "Meta" else source_label
        size = 45 if len(label) <= 1 else 26
        font = ImageFont.truetype(str(FONT), size)
        left, top = (index % COLS) * CELL, (index // COLS) * CELL
        box = draw.textbbox((0, 0), label, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
        if width >= CELL - 4 or height >= CELL - 4:
            raise ValueError(f"Key label clips atlas cell: {label!r}, {(width, height)}")
        draw.text((left + (CELL - width) / 2 - box[0],
                   top + (CELL - height) / 2 - box[1]),
                  label, font=font, fill="#c1c6ca")
        occupied.append({"label": label, "source_layout_label": source_label,
                         "cell": [index % COLS, index // COLS],
                         "ink_bbox_px": [width, height]})
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    manifest = {
        "asset": "P3 project-authored keyboard glyph atlas",
        "font": "Noto Sans Regular",
        "font_license": "SIL OFL-1.1",
        "font_source_url": SOURCE_URL,
        "font_source_local": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "font_source_sha256": FONT_SHA256,
        "bundled_font": "assets/NotoSans-Regular.ttf",
        "notice": "assets/OFL-NOTICE.txt",
        "notice_sha256": NOTICE_SHA256,
        "atlas": "assets/key_legends_noto.png",
        "atlas_sha256": hashlib.sha256(payload).hexdigest(),
        "atlas_resolution_px": list(image.size),
        "v2_uv_mesh_reused_unchanged": "research/v2/assets/cup-laptop-v2/key_legends.obj",
        "labels": occupied,
        "brand_marks": False,
    }
    return payload, manifest


def create():
    payload, manifest = render()
    if ATLAS.exists() and ATLAS.read_bytes() != payload:
        raise RuntimeError("P3 atlas changed; create a new version instead of overwriting")
    if not ATLAS.exists():
        ATLAS.write_bytes(payload)
    text = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != text:
        raise RuntimeError("P3 atlas manifest changed; create a new version instead of overwriting")
    if not MANIFEST.exists():
        MANIFEST.write_text(text)
    return manifest


if __name__ == "__main__":
    print(json.dumps(create(), indent=2))
