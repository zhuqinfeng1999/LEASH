#!/usr/bin/env python3
"""CPU-only compilation, dynamics, collision fit and robosuite adapter QA."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
ROOT = HERE.parents[4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_with_floor():
    root = ET.parse(OUT / "moka_standalone.xml").getroot()
    world = root.find("worldbody")
    ET.SubElement(world, "geom", {"name": "verification_floor", "type": "plane",
                                  "size": "0.5 0.5 0.02", "friction": "1 0.3 0.1",
                                  "rgba": "0.8 0.8 0.8 1"})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"),
                                           assets={p.name: p.read_bytes() for p in OUT.iterdir()
                                                   if p.suffix in (".obj", ".png")})


def primitive_extremes(geom):
    p = np.array([float(x) for x in geom.get("pos").split()])
    s = np.array([float(x) for x in geom.get("size").split()])
    if geom.get("type") == "box":
        return np.array([p + s * np.array((i, j, k))
                         for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    r, h = s
    return np.array([p + np.array((i*r, j*r, k*h))
                     for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     for k in (-1, 1)])


def verify():
    standalone = mujoco.MjModel.from_xml_path(str(OUT / "moka_standalone.xml"))
    libero_xml = mujoco.MjModel.from_xml_path(str(OUT / "moka_libero.xml"))
    assert standalone.njnt == 1 and libero_xml.njnt == 0
    assert standalone.nmesh == libero_xml.nmesh == 17
    assert standalone.ngeom == libero_xml.ngeom == 24
    bid = mujoco.mj_name2id(standalone, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert abs(standalone.body_mass[bid] - .420) < 1e-9
    assert standalone.body_inertia[bid].min() > 0
    assert all(standalone.geom_friction[g, 0] >= .95 for g in range(standalone.ngeom)
               if standalone.geom_group[g] == 0)
    # The exact LIBERO/robosuite loader must append one free joint and compile
    # the merged object with the same mesh and contact complement.
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    sys.path[:0] = [str(ROOT / "upstream/robosuite"), str(ROOT / "upstream/LIBERO")]
    from robosuite.models.objects import MujocoXMLObject
    from robosuite.models.world import MujocoWorldBase
    obj = MujocoXMLObject(str(OUT / "moka_libero.xml"), name="moka_v4",
                          joints=[dict(type="free", damping="0.0005")],
                          obj_type="all", duplicate_collision_geoms=False)
    world = MujocoWorldBase()
    world.merge_assets(obj)
    world.worldbody.append(obj.get_obj())
    combined = mujoco.MjModel.from_xml_string(world.get_xml())
    assert combined.njnt == 1 and combined.nmesh == 17 and combined.ngeom == 24
    obj_xml = ET.parse(OUT / "moka_libero.xml").getroot().find("./worldbody/body/body[@name='object']")
    assert obj_xml is not None and obj_xml.find("joint") is None and obj_xml.find("freejoint") is None
    cgeoms = [g for g in obj_xml.findall("geom") if g.get("group") == "0"]
    assert len(cgeoms) == 7
    proxy = json.loads((OUT / "collision_proxy.json").read_text())
    for xmlg, stored in zip(cgeoms, proxy["geoms"]):
        assert xmlg.get("type") == stored["type"]
        np.testing.assert_allclose(np.fromstring(xmlg.get("pos"), sep=" "), stored["pos"], atol=1e-8)
        np.testing.assert_allclose(np.fromstring(xmlg.get("size"), sep=" "), stored["size"], atol=1e-8)
    points = np.concatenate([primitive_extremes(g) for g in cgeoms])
    assert .149 <= np.ptp(points[:, 2]) <= .151
    assert np.min(points[:, 2]) == -.068
    assert .144 < np.ptp(points[:, 2])
    lo = np.array([-.1435, -.0865, .024])
    hi = np.array([.0635, .0805, .168])
    center = np.array(proxy["test_center_world_m"])
    clearance = {}
    for angle in (-60, 60):
        a = math.radians(angle)
        rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                             [-math.sin(a), 0, math.cos(a)]])
        placed = points @ rotation.T + center
        margin = np.minimum(placed.min(axis=0) - lo, hi - placed.max(axis=0))
        assert np.all(margin > 0), (angle, margin)
        clearance[str(angle)] = margin.round(6).tolist()
    # Static contact: initially place the free body just above a plane and let
    # MuJoCo settle under gravity. This checks a real contact model, not only a
    # drawn proxy; it says nothing about Panda manipulation or cavity dynamics.
    floor_model = load_with_floor()
    data = mujoco.MjData(floor_model)
    data.qpos[2] = .073
    mujoco.mj_forward(floor_model, data)
    initial_xy = data.qpos[:2].copy()
    for _ in range(2000):
        mujoco.mj_step(floor_model, data)
        assert np.isfinite(data.qpos).all()
    xy_drift = float(np.linalg.norm(data.qpos[:2] - initial_xy))
    quat = data.qpos[3:7].copy()
    upright_deg = math.degrees(2 * math.acos(min(1.0, abs(float(quat[0])))))
    speed = float(np.linalg.norm(data.qvel))
    assert .064 < data.qpos[2] < .075, data.qpos[2]
    assert xy_drift < .012, xy_drift
    assert upright_deg < 6, upright_deg
    assert speed < .05, speed
    assert data.ncon >= 1, data.ncon
    report = {
        "status": "CPU_MJCF_CANDIDATE_VALID",
        "scope": "standalone and robosuite merge compilation, 4 s on-floor settling, proxy-only tilted cavity fit",
        "standalone_joint_count": standalone.njnt,
        "libero_adapter_joint_count_before_robosuite": libero_xml.njnt,
        "robosuite_merged_joint_count": combined.njnt,
        "visual_mesh_count": standalone.nmesh,
        "contact_geom_count": len(cgeoms),
        "mass_kg": float(standalone.body_mass[bid]),
        "inertia_kg_m2": standalone.body_inertia[bid].tolist(),
        "settled_base_height_m": float(data.qpos[2]),
        "settled_xy_drift_m": xy_drift,
        "settled_tilt_deg": upright_deg,
        "settled_speed": speed,
        "tilt_clearance_xyz_m": clearance,
        "source_hashes": {f: sha256(HERE/f) for f in ("generate.py", "build_mjcf.py", "verify.py", "verify_mjcf.py")},
    }
    (HERE / "mjcf_cpu_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    verify()
