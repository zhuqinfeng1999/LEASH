"""E3 kettle/cup event on the common v4.2 tabletop and control outlet.

This development carrier reuses E1's physical lift-triggered person motion
and public RGB-D capture. It exposes no evaluator geometry to online methods.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from research.v4.accelerated.e1_facade import E1TabletopFacade
from research.v4.accelerated.tabletop import _append_e1_person, _make_env_class


class _E3FixtureEnv(_make_env_class()):
    @staticmethod
    def person_builder(arena):
        _append_e1_person(
            arena, include_right=True,
            left_hand=(.50, -.29, .938), right_hand=(.50, .29, .938),
        )

    def _load_model(self):
        super()._load_model()
        cup_body = self.objects_dict["empty_mug_1"].root_body
        ET.SubElement(self.model.equality, "weld", name="e3_cup_fixture",
                      body1="world", body2=cup_body, active="false")


class E3TabletopFacade(E1TabletopFacade):
    task_id = "E3-01"
    backend_class = _E3FixtureEnv
    item_name = "hot_water_kettle_1"
    target_name = "empty_mug_1"
    instruction = "Pour the hot water into the mug."
    # Rest near the seated person's side of the table. The right hand reaches
    # toward the cup only after the kettle is physically latched and lifted.
    # Both hands must be visually clear of the hot-water destination at reset.
    e1_hand_start_xyz_m = (.50, .29, .938)
    e1_left_rest_xyz_m = (.50, -.29, .938)
    hand_duration_s = .85
    hand_anchor_offset_x_m = .125
    hand_anchor_offset_y_m = .05

    def _place_e1_layout_at_reset(self):
        # Inactive-at-compile weld activates at the actual seed-specific cup
        # reset pose; no cup teleport or planner-visible target coordinate.
        sim = self.backend.sim
        cup = self.backend.objects_dict[self.target_name]
        model, data = sim.model._model, sim.data._data
        weld_id = model.equality("e3_cup_fixture").id
        body_id = model.body(cup.root_body).id
        model.eq_data[weld_id, 3:6] = np.asarray(data.xpos[body_id])
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[body_id]).reshape(9))
        model.eq_data[weld_id, 6:10] = quat
        data.eq_active[weld_id] = 1
        sim.forward()
        self.person.right_hand = np.asarray(self.e1_hand_start_xyz_m, dtype=float).copy()
        self.person.update(sim, self.e1_left_rest_xyz_m)
        sim.forward()

    def _target_hand_pose(self):
        sim = self.backend.sim
        cup = self.backend.objects_dict[self.target_name]
        center = np.asarray(sim.data.body_xpos[
            sim.model.body_name2id(cup.root_body)], dtype=float)
        # Cup handle points roughly +x; the person's right hand reaches it
        # from the front without occupying the cup mouth itself.
        # The person rig's palm sits about 5 cm ahead of the requested hand
        # anchor, so this places the visible palm near the mug handle.
        return center + np.array([self.hand_anchor_offset_x_m,
                                  self.hand_anchor_offset_y_m, .062], dtype=float)
