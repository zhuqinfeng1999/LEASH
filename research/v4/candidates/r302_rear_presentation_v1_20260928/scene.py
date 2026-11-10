"""Opt-in R3-02 source presentation: original horizontal can + V, both yaw +180°.

The physical can, +42 mm authored grasp zone, normal microwave, cart, scorer,
assist, controller and default scenes are inherited without modification.
"""

import math
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.candidates.r302_side_v_cradle_physical_20260928 import scene as parent


def _turn_wxyz(text):
    w, x, y, z = map(float, text.split())
    original = Rotation.from_quat([x, y, z, w])
    changed = Rotation.from_euler("z", math.pi) * original
    x, y, z, w = changed.as_quat()
    return f"{w:.17g} {x:.17g} {y:.17g} {z:.17g}"


class TurnedVCradleCartEnv(parent.RealVCradleCartEnv):
    def _load_model(self):
        super()._load_model()
        body = self.model.worldbody.find(".//body[@name='r302_v_wood_cradle']")
        if body is None:
            raise ValueError("exact original physical V source is missing")
        body.set("quat", _turn_wxyz(body.get("quat")))


class RearPresentedCanScene(parent.SideCanTopdownScene):
    layout_version = "R302-rear-material-presentation-same-plus42-zone-v1"

    def __init__(self, *args, **kwargs):
        with patch.object(parent, "RealVCradleCartEnv", TurnedVCradleCartEnv):
            super().__init__(*args, **kwargs)
        sim = self.backend.sim
        obj = self.backend.objects_dict[self.item_name]
        q = np.asarray(sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        old = Rotation.from_quat(q[[4, 5, 6, 3]])
        x, y, z, w = (Rotation.from_euler("z", math.pi) * old).as_quat()
        q[3:] = [w, x, y, z]
        sim.data.set_joint_qpos(obj.joints[0], q)
        sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        sim.forward()
        self.source_initial_qpos = q.copy()
        if kwargs.get("camera_observations", False):
            self.raw = self.backend._get_observations(force_update=True)
