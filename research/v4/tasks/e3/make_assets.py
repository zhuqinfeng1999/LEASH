"""Generate development-only E3 kettle and empty mug from authored primitives."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path
import copy
import xml.etree.ElementTree as ET


OUT = Path(__file__).resolve().parent


def write_with_visual_shell(path: Path, xml: str):
    """Keep true group-0 collision and render matching noncolliding group-1 geoms."""
    root = ET.fromstring(xml)
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.get("group") != "0":
                continue
            visual = copy.deepcopy(geom)
            visual.set("name", geom.get("name", "geom") + "_visual")
            visual.set("group", "1")
            visual.set("contype", "0")
            visual.set("conaffinity", "0")
            body.append(visual)
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def make_kettle():
    # Dimensions are metres: approximately 150 mm tall, 120 mm diameter.
    xml = '''<mujoco model="e3_hot_water_kettle">
  <asset>
    <material name="steel" rgba=".57 .60 .62 1" specular=".85" shininess=".8" />
    <material name="dark" rgba=".105 .12 .13 1" specular=".35" shininess=".45" />
    <material name="accent" rgba=".60 .085 .045 1" specular=".45" shininess=".55" />
  </asset>
  <worldbody><body><body name="object">
    <inertial pos="0 0 .062" mass=".82" diaginertia=".003 .003 .003" />
    <geom name="base" type="cylinder" pos="0 0 .007" size=".059 .007" material="dark" group="0" friction=".85 .005 .001" />
    <geom name="lower_body" type="cylinder" pos="0 0 .055" size=".060 .048" material="steel" group="0" friction=".6 .005 .001" />
    <geom name="shoulder" type="ellipsoid" pos="0 0 .095" size=".059 .059 .027" material="steel" group="0" />
    <geom name="lid" type="cylinder" pos="0 0 .120" size=".036 .006" material="dark" group="0" />
    <geom name="lid_knob" type="sphere" pos="0 0 .130" size=".012" material="accent" group="0" />
    <geom name="handle_lower" type="capsule" fromto="-.056 0 .039 -.104 0 .039" size=".008" material="dark" group="0" />
    <geom name="handle_grip" type="capsule" fromto="-.104 0 .039 -.104 0 .104" size=".008" material="dark" group="0" />
    <geom name="handle_upper" type="capsule" fromto="-.104 0 .104 -.051 0 .108" size=".008" material="dark" group="0" />
    <geom name="spout_neck" type="capsule" fromto=".051 0 .090 .111 0 .116" size=".018" material="steel" group="0" />
    <geom name="spout_lip" type="cylinder" pos=".116 0 .119" quat=".89 0 -.45 0" size=".018 .004" material="dark" group="1" contype="0" conaffinity="0" />
    <site name="spout_tip" pos=".129 0 .122" size=".002" />
    <site name="bottom_site" pos="0 0 0" size=".001" />
    <site name="top_site" pos="0 0 .145" size=".001" />
    <site name="horizontal_radius_site" pos=".13 0 0" size=".001" />
  </body></body></worldbody>
</mujoco>
'''
    write_with_visual_shell(OUT / "hot_water_kettle.xml", xml)


def make_mug():
    lines = ['''<mujoco model="e3_empty_mug">
  <asset>
    <material name="ceramic" rgba=".16 .27 .36 1" specular=".42" shininess=".48" />
    <material name="inside" rgba=".72 .76 .77 1" specular=".4" shininess=".48" />
  </asset>
  <worldbody><body><body name="object">
    <inertial pos="0 0 .043" mass=".31" diaginertia=".00038 .00038 .00046" />
    <geom name="base" type="cylinder" pos="0 0 .004" size=".045 .004" material="ceramic" group="0" friction=".85 .005 .001" />
''']
    for i in range(24):
        angle = i * 2 * pi / 24
        x, y = .042 * cos(angle), .042 * sin(angle)
        w, z = cos(angle / 2), sin(angle / 2)
        lines.append(f'    <geom name="wall_{i:02d}" type="box" pos="{x:.7f} {y:.7f} .047" size=".0048 .0065 .043" quat="{w:.8f} 0 0 {z:.8f}" material="ceramic" group="0" friction=".75 .005 .001" />\n')
    for i, (a, b) in enumerate(zip((".044 0 .018", ".071 0 .020", ".080 0 .044", ".071 0 .069"),
                                     (".071 0 .020", ".080 0 .044", ".071 0 .069", ".044 0 .072"))):
        lines.append(f'    <geom name="handle_{i}" type="capsule" fromto="{a} {b}" size=".006" material="ceramic" group="0" />\n')
    lines.append('''    <site name="mouth_center" pos="0 0 .090" size=".002" />
    <site name="bottom_site" pos="0 0 0" size=".001" />
    <site name="top_site" pos="0 0 .090" size=".001" />
    <site name="horizontal_radius_site" pos=".082 0 0" size=".001" />
  </body></body></worldbody>
</mujoco>
''')
    write_with_visual_shell(OUT / "empty_mug.xml", "".join(lines))


if __name__ == "__main__":
    make_kettle()
    make_mug()
