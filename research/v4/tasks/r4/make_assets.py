"""Author-owned geometric R4-01 placeholders for native task development.

These are generated from code with no external media. Their visual quality has
not passed the v4 release requirement for realistic main objects.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent


def geom(parent, name, shape, size, *, pos=None, rgba=None, group="1",
         collidable=False, mass=None, friction=None, quat=None, fromto=None,
         material=None):
    attrs = {"name": name, "type": shape, "size": size, "group": group,
             "contype": "1" if collidable else "0",
             "conaffinity": "1" if collidable else "0"}
    if pos: attrs["pos"] = pos
    if fromto: attrs["fromto"] = fromto
    if rgba: attrs["rgba"] = rgba
    if mass is not None: attrs["mass"] = str(mass)
    if friction: attrs["friction"] = friction
    if quat: attrs["quat"] = quat
    if material: attrs["material"] = material
    return ET.SubElement(parent, "geom", attrs)


def write(name, root):
    ET.indent(root, space="  ")
    path = HERE / name
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    path.write_text(path.read_text() + "\n")


def make_bag():
    root = ET.Element("mujoco", model="v4_r4_flat_grocery_bag")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="v4_r4_bag_print", type="2d",
                  file="plastic_bag_print.png")
    ET.SubElement(asset, "material", name="v4_r4_bag_print_mat",
                  texture="v4_r4_bag_print", texrepeat="1 1",
                  rgba="1 1 1 .96")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    obj = ET.SubElement(body, "body", name="object")
    ET.SubElement(obj, "inertial", pos="0 0 0", mass="0.11",
                  diaginertia="0.00022 0.00035 0.00052")
    plastic = ".97 .97 .94 .94"
    crease = ".72 .75 .73 .72"
    # A thin crumpled grocery bag lying flat.  Its two cutout loop handles,
    # folded rim and irregular creases are visible from the public top views.
    # The visual sheets never alter the qualified grasp collision volume.
    for i, (y, half_y, z, alpha) in enumerate((
            (-.025, .009, .008, .82), (-.009, .010, .011, .93),
            (.009, .010, .009, .91), (.025, .009, .012, .82))):
        geom(obj, f"film_strip_{i}", "box", f".051 {half_y} .0011",
             pos=f"0 {y} {z}", rgba=f".97 .97 .94 {alpha}")
    geom(obj, "printed_outer_film", "box", ".050 .033 .0003",
         pos="0 0 .015", material="v4_r4_bag_print_mat")
    for y in (-.034, .034):
        geom(obj, f"side_seam_{'low' if y < 0 else 'high'}", "capsule",
             ".0015", fromto=f"-.050 {y} .008 .047 {y} .012", rgba=crease)
    geom(obj, "folded_opening", "box", ".003 .032 .0015",
         pos=".047 0 .014", rgba=".91 .93 .92 .94")
    for side, y in (("left", -.020), ("right", .020)):
        geom(obj, f"{side}_handle_lower", "capsule", ".0032",
             fromto=f".047 {y-.008} .013 .074 {y-.009} .009", rgba=plastic)
        geom(obj, f"{side}_handle_upper", "capsule", ".0032",
             fromto=f".047 {y+.008} .013 .074 {y+.009} .009", rgba=plastic)
        geom(obj, f"{side}_handle_tip", "capsule", ".0032",
             fromto=f".074 {y-.009} .009 .074 {y+.009} .009", rgba=plastic)
    for i, (x1, y1, x2, y2) in enumerate((
            (-.045, -.031, -.019, .029), (-.026, .031, .007, -.029),
            (-.004, -.030, .022, .030), (.014, .029, .041, -.025))):
        geom(obj, f"crinkle_{i}", "capsule", ".0011", rgba=crease,
            fromto=f"{x1} {y1} .017 {x2} {y2} .017")
    # Collision, placement footprint and grasp dimensions stay unchanged.
    geom(obj, "bag_grasp_collision", "box", ".055 .030 .010",
         group="0", collidable=True, rgba="0 0 0 0", mass=0,
         friction="1 .005 .001")
    for site, z in (("bottom_site", -.010), ("top_site", .020)):
        ET.SubElement(body, "site", name=site, pos=f"0 0 {z}", size=".001")
    ET.SubElement(body, "site", name="horizontal_radius_site",
                  pos=".085 0 0", size=".001")
    write("plastic_bag.xml", root)


def make_bag_print():
    """Project-authored grocery print; no downloaded texture or font file."""
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (512, 512), (244, 246, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=70)
    for text, y in (("MARKET", 142), ("THANK YOU", 258)):
        box = draw.textbbox((0, 0), text, font=font)
        x = (512 - (box[2] - box[0])) // 2
        draw.text((x, y), text, font=font, fill=(22, 89, 165))
    draw.rectangle((48, 105, 464, 112), fill=(22, 89, 165))
    draw.rectangle((48, 362, 464, 369), fill=(22, 89, 165))
    image.save(HERE / "plastic_bag_print.png")


def make_crib():
    root = ET.Element("mujoco", model="v4_r4_crib_placeholder")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    obj = ET.SubElement(body, "body", name="object")
    ET.SubElement(obj, "inertial", pos="0 0 .030", mass="5.0",
                  diaginertia=".040 .052 .064")
    wood = ".78 .64 .48 1"
    mattress = ".93 .90 .83 1"
    # Mattress top is 2.6 cm above table; the crib interior is open above it.
    geom(obj, "crib_mattress_visual", "box", ".119 .099 .011",
         pos="0 0 .015", rgba=mattress)
    geom(obj, "crib_base_collision", "box", ".135 .115 .013",
         pos="0 0 .013", group="0", collidable=True,
         rgba="0 0 0 0", mass=0, friction="1 .005 .001")
    for side, y in (("front", -.110), ("back", .110)):
        for height, z in (("low", .035), ("high", .132)):
            geom(obj, f"{side}_{height}_rail", "box", ".135 .007 .007",
                 pos=f"0 {y} {z}", rgba=wood)
        for i in range(9):
            x = -.115 + i*.02875
            geom(obj, f"{side}_slat_{i}", "box", ".004 .004 .043",
                 pos=f"{x:.5f} {y} .083", rgba=wood)
        geom(obj, f"{side}_wall_collision", "box", ".135 .006 .066",
             pos=f"0 {y} .069", group="0", collidable=True,
             rgba="0 0 0 0", mass=0)
    for side, x in (("head", -.135), ("foot", .135)):
        geom(obj, f"{side}_panel_visual", "box", ".007 .115 .071",
             pos=f"{x} 0 .071", rgba=wood)
        geom(obj, f"{side}_panel_collision", "box", ".007 .115 .071",
             pos=f"{x} 0 .071", group="0", collidable=True,
             rgba="0 0 0 0", mass=0)
    # Larger clothed baby-doll silhouette, fixed and display-only. Facial
    # features face upward toward agentview; they never enter task predicates.
    geom(obj, "doll_neck", "sphere", ".013", pos="-.036 0 .057",
         rgba=".78 .57 .44 1")
    geom(obj, "doll_head", "ellipsoid", ".030 .027 .028",
         pos="-.069 0 .062", rgba=".82 .61 .47 1")
    geom(obj, "doll_hair", "ellipsoid", ".024 .026 .012",
         pos="-.081 0 .082", rgba=".25 .17 .12 1")
    for side, y in (("left", -.011), ("right", .011)):
        geom(obj, f"doll_eye_{side}", "sphere", ".0027",
             pos=f"-.066 {y} .089", rgba=".17 .13 .11 1")
        geom(obj, f"doll_ear_{side}", "sphere", ".006",
             pos=f"-.071 {y*2.65} .061", rgba=".78 .57 .44 1")
    geom(obj, "doll_nose", "sphere", ".0033",
         pos="-.054 0 .088", rgba=".74 .52 .40 1")
    geom(obj, "doll_mouth", "capsule", ".0013",
         fromto="-.049 -.005 .081 -.049 .005 .081", rgba=".55 .28 .27 1")
    geom(obj, "doll_onesie", "ellipsoid", ".052 .036 .023",
         pos=".001 0 .054", rgba=".75 .84 .85 1")
    geom(obj, "doll_bib", "ellipsoid", ".022 .022 .0015",
         pos="-.017 0 .077", rgba=".95 .93 .82 1")
    for side, sign in (("left", -1), ("right", 1)):
        geom(obj, f"doll_sleeve_{side}", "capsule", ".010",
             fromto=f"-.020 {sign*.029} .057 -.004 {sign*.044} .049",
             rgba=".74 .83 .84 1")
        geom(obj, f"doll_arm_{side}", "capsule", ".0085",
             fromto=f"-.004 {sign*.044} .049 .012 {sign*.061} .043",
             rgba=".82 .61 .47 1")
        geom(obj, f"doll_leg_{side}", "capsule", ".011",
             fromto=f".035 {sign*.019} .044 .077 {sign*.024} .038",
             rgba=".75 .84 .85 1")
        geom(obj, f"doll_foot_{side}", "sphere", ".012",
             pos=f".080 {sign*.024} .036", rgba=".82 .61 .47 1")
    for site, z in (("bottom_site", 0), ("top_site", .142)):
        ET.SubElement(body, "site", name=site, pos=f"0 0 {z}", size=".001")
    ET.SubElement(body, "site", name="horizontal_radius_site",
                  pos=".155 0 0", size=".001")
    write("crib.xml", root)


if __name__ == "__main__":
    make_bag_print()
    make_bag()
    make_crib()
