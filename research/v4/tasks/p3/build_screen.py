"""Deterministic 2K unbranded, text-free laptop screen for P3.

No OS screenshot, logo, trade dress, or font raster is imported. This replaces
v2's DejaVu-labelled 1600x1000 screen without touching the shared v2 asset.
"""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SCREEN = ASSETS / "screen_unbranded_2k.png"
MANIFEST = ASSETS / "screen_unbranded_2k_manifest.json"
SIZE = (2560, 1600)


def render():
    image = Image.new("RGB", SIZE, "#e8eef3")
    draw = ImageDraw.Draw(image)
    w, h = SIZE
    draw.rectangle((0, 0, w, 116), fill="#1b3448")
    draw.ellipse((48, 39, 86, 77), fill="#d5eef1")
    draw.rounded_rectangle((118, 38, 870, 78), radius=20, fill="#527487")
    for x in (2180, 2272, 2364, 2456):
        draw.ellipse((x, 45, x + 28, 73), fill="#8aa5b5")
    draw.rounded_rectangle((72, 184, 1560, 1494), radius=34, fill="#fbfcfd")
    draw.rounded_rectangle((1620, 184, 2490, 1008), radius=34, fill="#f9fbfc")
    draw.rounded_rectangle((1620, 1052, 2490, 1494), radius=34, fill="#f9fbfc")
    # Visual separators and annotation bars: deliberate text-free UI.
    for x, y, length in ((132, 252, 740), (132, 332, 920), (1680, 252, 430),
                         (1680, 1124, 520)):
        draw.rounded_rectangle((x, y, x + length, y + 25), radius=12, fill="#abc1cf")
    for i, length in enumerate((1100, 1170, 870, 1060, 980, 725)):
        y = 430 + 100 * i
        draw.rounded_rectangle((132, y, 132 + length, y + 22), radius=11, fill="#d3dfe7")
    # Distinctive but generic scientific line chart and bar chart.
    chart = (180, 1070, 1450, 1410)
    draw.rounded_rectangle(chart, radius=18, fill="#eff5f8")
    for k in range(1, 5):
        y = chart[1] + k * (chart[3] - chart[1]) / 5
        draw.line((chart[0] + 46, y, chart[2] - 46, y), fill="#d3e0e8", width=3)
    points = [(230, 1310), (385, 1282), (520, 1290), (660, 1220), (805, 1239),
              (935, 1150), (1085, 1171), (1235, 1108), (1390, 1125)]
    draw.line(points, fill="#3d8595", width=13, joint="curve")
    for x, y in points:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#2d7282")
    for i, height in enumerate((220, 304, 252, 354, 285, 414, 378)):
        x = 1700 + i * 106
        draw.rounded_rectangle((x, 910 - height, x + 63, 910), radius=15,
                               fill="#477f91" if i % 2 else "#66a1ad")
    for i in range(3):
        x = 1690 + i * 258
        draw.rounded_rectangle((x, 1190, x + 210, 1400), radius=24,
                               fill=("#e1ecef", "#dde9ed", "#d3e6eb")[i])
        draw.ellipse((x + 67, 1224, x + 143, 1300), fill="#81aab3")
        draw.rounded_rectangle((x + 38, 1332, x + 172, 1352), radius=10, fill="#abc7cd")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    manifest = {
        "asset": "P3 text-free research dashboard screen",
        "author": "SafeVLA-Harness project",
        "license": "MIT",
        "geometry_context": "project-authored laptop with reused original mesh",
        "font_used": False,
        "logos_or_brands": False,
        "resolution_px": list(SIZE),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, manifest


def create():
    payload, manifest = render()
    if SCREEN.exists() and SCREEN.read_bytes() != payload:
        raise RuntimeError("P3 screen changed; create a new version instead of overwriting")
    if not SCREEN.exists():
        SCREEN.write_bytes(payload)
    text = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != text:
        raise RuntimeError("P3 screen manifest changed; create a new version instead of overwriting")
    if not MANIFEST.exists():
        MANIFEST.write_text(text)
    return manifest


if __name__ == "__main__":
    print(json.dumps(create(), indent=2))
