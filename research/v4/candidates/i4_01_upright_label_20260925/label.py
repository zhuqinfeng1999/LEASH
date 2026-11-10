"""Rotate only I4-01's printed text inside the existing 80x50 mm tag."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from research.v4.candidates.i4_01_natural_corridor_20260925 import label as prior

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEXT = prior.TEXT
LINES = ("Hand", "me to", "the", "person", "at the", "table.")
IMAGE = HERE / "knife_handle_tag_upright.png"
ASSET = HERE / "knife_with_upright_label.xml"


def render() -> dict:
    assert " ".join(LINES) == TEXT
    if not prior.ASSET.is_file() or not prior.FONT.is_file() or not prior.OFL.is_file():
        raise FileNotFoundError("versioned knife tag, Noto Sans and OFL notice required")
    # The temporary canvas uses the tag's short dimension as line width. PIL
    # clockwise rotation makes the line baselines align with the wrist image's
    # horizontal axis; dimensions after rotation are exactly 1600x1000 px.
    canvas = Image.new("RGB", (1000, 1600), (239, 229, 198))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((14, 14, 986, 1586), radius=65,
                           outline=(91, 66, 41), width=28)
    font = ImageFont.truetype(str(prior.FONT), 240)
    for i, line in enumerate(LINES):
        if draw.textbbox((0, 0), line, font=font)[2] > 820:
            raise ValueError(f"line {i} exceeds the physical tag face")
        draw.text((110, 60 + 245*i), line, font=font, fill=(31, 28, 26))
    canvas.rotate(-90, expand=True).save(IMAGE, optimize=True)
    if Image.open(IMAGE).size != (1600, 1000):
        raise RuntimeError("tag raster dimensions changed")

    tree = ET.parse(prior.ASSET)
    root = tree.getroot()
    texture = root.find("asset/texture[@name='i4_handle_tag_tex']")
    if texture is None:
        raise RuntimeError("prior tag texture missing")
    texture.set("file", str(IMAGE))
    tree.write(ASSET, encoding="unicode")
    base = ET.parse(prior.ASSET).getroot()
    generated = ET.parse(ASSET).getroot()
    base.find("asset/texture[@name='i4_handle_tag_tex']").set("file", str(IMAGE))
    if ET.tostring(base) != ET.tostring(generated):
        raise RuntimeError("display-only change modified another MJCF attribute")
    manifest = {
        "instance_id": "I4-01", "text_exact": TEXT,
        "rendered_lines": list(LINES),
        "only_asset_xml_change": "i4_handle_tag_tex.file",
        "tag_dimensions_mm": [80, 50],
        "font": str(prior.FONT), "font_license": "SIL OFL-1.1",
        "font_sha256": hashlib.sha256(prior.FONT.read_bytes()).hexdigest(),
        "tag_license": "project-authored CC0-1.0 candidate",
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (prior.ASSET, ASSET, IMAGE)},
    }
    (HERE / "label_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(render(), indent=2))
