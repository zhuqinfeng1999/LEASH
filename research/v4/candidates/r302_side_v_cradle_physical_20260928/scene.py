"""Opt-in R3-02 original can laid on a real, fixed wooden V; no tool changes."""
import math
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.accelerated import tabletop
from research.v4.candidates.r3_02_oven_only_prefix_20260928 import scene as own_front

RADIUS = .0365
VERTEX = .004
SOFT_SETTLE = .0002
CENTER_ABOVE_TABLE = VERTEX + RADIUS * math.sqrt(2) - SOFT_SETTLE


class RealVCradleCartEnv(own_front.OwnCartEnv):
    def _load_model(self):
        super()._load_model()
        p, yaw = tabletop.sample_slot("L", self.layout_seed, object_index=0)
        p[:2] += np.asarray((0., .08))  # Exactly the original own source shift.
        p[2] = tabletop.TABLE_Z
        b = ET.SubElement(
            self.model.worldbody, "body", name="r302_v_wood_cradle",
            pos=" ".join(map(str, p)),
            quat=f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}",
        )
        # Two genuine, visible/colliding 45-degree surfaces and a table-backed
        # plate. Nominal can center is 0.2 mm below exact rigid tangency to
        # compensate for the old saved empty-contact analytic-tangent version.
        for sign in (-1, 1):
            z = VERTEX + .025 - .003 * math.sqrt(2)
            ET.SubElement(
                b, "geom", name=f"r302_v_rail_{sign}", type="box",
                pos=f"0 {sign * .025} {z}", size=".075 .04 .003",
                euler=f"{sign * math.pi / 4} 0 0", rgba=".52 .28 .11 1",
                contype="1", conaffinity="1", friction="1 .005 .0001",
            )
        ET.SubElement(
            b, "geom", name="r302_v_base", type="box", pos="0 0 .002",
            size=".075 .058 .002", rgba=".42 .22 .09 1",
            contype="1", conaffinity="1", friction="1 .005 .0001",
        )


class SideCanTopdownScene(own_front.OvenOnlySoupScene):
    layout_version = "R302-original-can-localzone42mm-realV-topdown-source-v1"

    def __init__(self, *args, **kwargs):
        # The existing own constructor still compiles the original can XML,
        # normal front appliance/cart, Panda/controller, assist and scorer.
        # Its existing fixture patch selects this subclass only in this scope.
        with patch.object(own_front, "OwnCartEnv", RealVCradleCartEnv):
            super().__init__(*args, **kwargs)
        sim = self.backend.sim
        obj = self.backend.objects_dict[self.item_name]
        q = np.asarray(sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        old = Rotation.from_quat(q[[4, 5, 6, 3]])
        q[2] = tabletop.TABLE_Z + CENTER_ABOVE_TABLE
        q[3:] = (old * Rotation.from_euler("y", math.pi / 2)).as_quat()[[3, 0, 1, 2]]
        sim.data.set_joint_qpos(obj.joints[0], q)
        sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        sim.forward()
        self.source_initial_qpos = q.copy()
        if kwargs.get("camera_observations", False):
            self.raw = self.backend._get_observations(force_update=True)
