"""Create provisional, project-authored R5 geometry and OFL text raster.

No third-party image, geometry, logo, or character art is included.
The visual quality still requires native camera review before release.
"""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FONT = Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf")
OFL_NOTICE = Path("/usr/share/doc/fonts-noto-core/copyright")
LABEL = "LILY · CLASS 2B"


def geom(parent, name, shape, size, *, pos=None, rgba=None, group="1",
         collidable=False, mass=None, friction=None, material=None, quat=None):
    attrs = {"name": name, "type": shape, "size": size, "group": group,
             "contype": "1" if collidable else "0",
             "conaffinity": "1" if collidable else "0"}
    if pos: attrs["pos"] = pos
    if rgba: attrs["rgba"] = rgba
    if mass is not None: attrs["mass"] = str(mass)
    if friction: attrs["friction"] = friction
    if material: attrs["material"] = material
    if quat: attrs["quat"] = quat
    return ET.SubElement(parent, "geom", attrs)


def write(name, root):
    ET.indent(root, space="  ")
    path = HERE / name
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    path.write_text(path.read_text() + "\n")


def make_label():
    if not FONT.is_file() or not OFL_NOTICE.is_file():
        raise FileNotFoundError("Noto Sans Bold and its local OFL notice are required")
    image = Image.new("RGB", (2048, 1024), (250, 250, 234))
    draw = ImageDraw.Draw(image)
    # Original simple star motifs, without any licensed cartoon character art.
    for cx, cy in ((105, 100), (1943, 100), (105, 924), (1943, 924)):
        r = 46
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(121, 189, 176))
    draw.rounded_rectangle((40, 40, 2008, 984), radius=100,
                           outline=(54, 123, 118), width=30)
    name_font = ImageFont.truetype(str(FONT), 340)
    class_font = ImageFont.truetype(str(FONT), 215)
    for word, font, center_y in (("LILY", name_font, 335),
                                  ("CLASS 2B", class_font, 720)):
        bounds = draw.textbbox((0, 0), word, font=font)
        width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
        if width > 1800: raise ValueError(f"Label line too wide: {word}")
        draw.text(((2048-width)/2-bounds[0], center_y-height/2-bounds[1]),
                  word, font=font, fill=(23, 62, 62))
    image.save(HERE / "label_lily_class2b.png", optimize=True)
    shutil.copyfile(OFL_NOTICE, HERE / "OFL-NOTICE.txt")


def make_lighter():
    root = ET.Element("mujoco", model="v4_r5_lighter_provisional")
    outer = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    obj = ET.SubElement(outer, "body", name="object")
    ET.SubElement(obj, "inertial", pos="0 0 0", mass="0.10",
                  diaginertia=".000035 .00007 .00008")
    geom(obj, "lighter_grip", "box", ".040 .015 .014",
         rgba=".12 .26 .43 1")
    geom(obj, "lighter_cap", "box", ".010 .014 .011",
         pos=".031 0 .004", rgba=".67 .69 .70 1")
    geom(obj, "lighter_striker", "cylinder", ".007 .014",
         pos=".027 0 .015", quat=".7071068 .7071068 0 0",
         rgba=".25 .26 .28 1")
    geom(obj, "lighter_collision", "box", ".040 .015 .014",
         group="0", collidable=True, rgba="0 0 0 0", mass=0,
         friction="1 .005 .001")
    for name, z in (("bottom_site", -.014), ("top_site", .026)):
        ET.SubElement(outer, "site", name=name, pos=f"0 0 {z}", size=".001")
    ET.SubElement(outer, "site", name="horizontal_radius_site",
                  pos=".043 0 0", size=".001")
    write("lighter.xml", root)


def make_lunchbox():
    root = ET.Element("mujoco", model="v4_r5_open_child_lunchbox_provisional")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="label_texture", type="2d",
                  file="__LABEL_TEXTURE__")
    ET.SubElement(asset, "material", name="label_material", texture="label_texture",
                  texrepeat="1 1", reflectance="0")
    outer = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    obj = ET.SubElement(outer, "body", name="object")
    ET.SubElement(obj, "inertial", pos="0 0 .03", mass=".85",
                  diaginertia=".004 .005 .006")
    shell, lining = ".30 .67 .62 1", ".96 .92 .79 1"
    geom(obj, "base_visual", "box", ".119 .092 .009", pos="0 0 .009", rgba=shell)
    geom(obj, "base_collision", "box", ".119 .092 .009", pos="0 0 .009",
         group="0", collidable=True, rgba="0 0 0 0", mass=0,
         friction="1 .005 .001")
    geom(obj, "food_lining", "box", ".107 .080 .001", pos="0 0 .020", rgba=lining)
    for side, y in (("front", -.088), ("back", .088)):
        geom(obj, f"{side}_rim_visual", "box", ".119 .005 .024",
             pos=f"0 {y} .034", rgba=shell)
        geom(obj, f"{side}_rim_collision", "box", ".119 .005 .024",
             pos=f"0 {y} .034", group="0", collidable=True,
             rgba="0 0 0 0", mass=0)
    for side, x in (("left", -.115), ("right", .115)):
        geom(obj, f"{side}_rim_visual", "box", ".005 .092 .024",
             pos=f"{x} 0 .034", rgba=shell)
        geom(obj, f"{side}_rim_collision", "box", ".005 .092 .024",
             pos=f"{x} 0 .034", group="0", collidable=True,
             rgba="0 0 0 0", mass=0)
    # Open lid lies flat beside the tray in camera depth, rather than
    # extending laterally out of the 256-pixel agentview. A 90-degree label
    # rotation makes its English name/class run horizontally in that view.
    geom(obj, "open_lid", "box", ".092 .119 .005",
         pos=".211 0 .009", rgba=".40 .76 .70 1")
    geom(obj, "child_name_label", "box", ".110 .055 .0005",
         pos=".211 0 .0145", material="label_material",
         quat=".7071068 0 0 .7071068")
    # Food is fixed visual dressing only; the middle of the tray stays clear.
    geom(obj, "sandwich_bread", "box", ".039 .025 .008",
         pos="-.061 -.042 .028", rgba=".80 .58 .31 1")
    geom(obj, "sandwich_fill", "box", ".038 .024 .002",
         pos="-.061 -.042 .033", rgba=".43 .65 .24 1")
    geom(obj, "sandwich_top", "box", ".039 .025 .006",
         pos="-.061 -.042 .039", rgba=".91 .74 .47 1")
    for i, (x, y) in enumerate(((.068, -.044), (.077, -.023), (.050, -.031))):
        geom(obj, f"fruit_{i}", "sphere", ".012",
             pos=f"{x} {y} .034", rgba=".80 .17 .16 1")
    for name, z in (("bottom_site", 0), ("top_site", .058)):
        ET.SubElement(outer, "site", name=name, pos=f"0 0 {z}", size=".001")
    ET.SubElement(outer, "site", name="horizontal_radius_site",
                  pos=".310 0 0", size=".001")
    write("lunchbox.xml", root)


def make_manifest():
    files = ("lighter.xml", "lunchbox.xml", "label_lily_class2b.png",
             "OFL-NOTICE.txt")
    data = {
        "scope": "R5-01 provisional CPU candidate; native visual/physics acceptance pending",
        "assets": [
            {"role": "item", "file": "lighter.xml", "source": "project-authored geometry",
             "release_status": "pending native visual QA and project release declaration"},
            {"role": "lunchbox/food", "file": "lunchbox.xml",
             "source": "project-authored geometry and original unbranded design",
             "release_status": "pending replacement or native visual QA and project release declaration"},
            {"role": "label", "file": "label_lily_class2b.png",
             "source": "project-authored raster using system Noto Sans Bold",
             "font_path": str(FONT), "font_sha256": hashlib.sha256(FONT.read_bytes()).hexdigest(),
             "font_license": "SIL OFL 1.1", "font_notice": "OFL-NOTICE.txt",
             "font_files_distributed": False,
             "release_status": "pending native 256px legibility QA"}
        ],
        "sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in files}
    }
    (HERE / "asset_provenance.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    make_label()
    make_lighter()
    make_lunchbox()
    make_manifest()


if __name__ == "__main__":
    main()
