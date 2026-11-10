"""v4.2 private, source-agnostic MuJoCo grasp assistance.

The scene compiler must insert one *inactive* weld equality per graspable
object.  Slot metadata names its object body, grasp-zone site, weld, and
metric tolerances.  This module runs only at the shared low-level control
outlet, after the commit gate has passed an instruction and before physics is
advanced.  It never changes the visible finger command or public observation.

The class is an isolated CPU implementation, not a claim that the v4 bridge
is integrated or that any task has passed native acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


@dataclass(frozen=True)
class GraspZone:
    """Private slot metadata; all distances are in meters.

    ``tolerance_xyz`` is expressed in the gripper-center site frame.  The
    grasp-zone site must also lie geometrically between the two pad sites.
    """

    object_body: str
    zone_site: str
    weld_name: str
    tolerance_xyz: tuple[float, float, float]

    @classmethod
    def from_metadata(cls, value: Mapping) -> "GraspZone":
        return cls(
            object_body=str(value["object_body"]),
            zone_site=str(value["zone_site"]),
            weld_name=str(value["weld_name"]),
            tolerance_xyz=tuple(float(x) for x in value["tolerance_xyz"]),
        )


@dataclass(frozen=True)
class AssistEvent:
    """Safe control-outlet event; deliberately omits object identifiers."""

    kind: str  # attached | detached | ambiguous | no_match | unchanged
    candidate_count: int = 0


def inactive_weld_element(*, name: str, hand_body: str, object_body: str) -> ET.Element:
    """MJCF equality to add before compilation, never after model creation."""

    return ET.Element("weld", {"name": name, "body1": hand_body,
                               "body2": object_body, "active": "false"})


class GraspAssist:
    """Switch precompiled welds when a control command closes or opens fingers.

    The caller supplies the same scalar gripper channel for analytic tools and
    VLA chunks: positive closes, negative opens, zero keeps the current state.
    ``source`` is accepted for integration logging but never affects matching.
    An ambiguous close attaches nothing; it is not arbitrarily assigned to
    whichever object happens to be first in a scene list.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, *,
        hand_body: str, center_site: str, left_pad_site: str,
        right_pad_site: str, slot_metadata: Mapping[str, Mapping],
    ) -> None:
        self.model = model
        self.data = data
        self._hand = model.body(hand_body).id
        self._center = model.site(center_site).id
        self._left = model.site(left_pad_site).id
        self._right = model.site(right_pad_site).id
        if len({self._center, self._left, self._right}) != 3:
            raise ValueError("gripper center and pad sites must be distinct")
        if model.site_bodyid[self._center] != self._hand:
            raise ValueError("center site must be on the weld hand body")
        self._entries: list[tuple[GraspZone, int, int, int]] = []
        object_ids: set[int] = set()
        weld_ids: set[int] = set()
        for slot_name, metadata in slot_metadata.items():
            for raw in metadata.get("grasp_zones", ()):
                zone = GraspZone.from_metadata(raw)
                tol = zone.tolerance_xyz
                if len(tol) != 3 or any(not math.isfinite(x) or x <= 0 for x in tol):
                    raise ValueError(f"invalid grasp tolerance in slot {slot_name}")
                body_id = model.body(zone.object_body).id
                site_id = model.site(zone.zone_site).id
                weld_id = model.equality(zone.weld_name).id
                if body_id in object_ids or weld_id in weld_ids:
                    raise ValueError("each object and weld must occur only once")
                if model.site_bodyid[site_id] != body_id:
                    raise ValueError("grasp-zone site must be on its object body")
                if model.body_jntnum[body_id] != 1 or model.jnt_type[model.body_jntadr[body_id]] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError("assisted object body must have a free joint")
                if (model.eq_type[weld_id] != mujoco.mjtEq.mjEQ_WELD
                    or model.eq_obj1id[weld_id] != self._hand
                    or model.eq_obj2id[weld_id] != body_id):
                    raise ValueError("inactive weld must join hand and declared object")
                if model.eq_active0[weld_id] or data.eq_active[weld_id]:
                    raise ValueError("assist weld must start inactive")
                object_ids.add(body_id)
                weld_ids.add(weld_id)
                self._entries.append((zone, body_id, site_id, weld_id))
        self._held_index: int | None = None

    @property
    def held(self) -> bool:
        """Public-safe Boolean; never an asset name or body index."""

        return self._held_index is not None

    def reset(self) -> None:
        """Call after each simulator reset, before any controller instruction."""

        for _, _, _, weld_id in self._entries:
            self.data.eq_active[weld_id] = 0
        self._held_index = None
        mujoco.mj_forward(self.model, self.data)

    def _eligible(self, site_id: int, tolerance_xyz: tuple[float, float, float]) -> bool:
        left = np.asarray(self.data.site_xpos[self._left])
        right = np.asarray(self.data.site_xpos[self._right])
        gap_vector = right - left
        gap = float(np.linalg.norm(gap_vector))
        if not math.isfinite(gap) or gap <= 1e-6:
            return False
        zone_pos = np.asarray(self.data.site_xpos[site_id])
        along = float(np.dot(zone_pos - left, gap_vector / gap))
        if along < 0.0 or along > gap:
            return False
        center = np.asarray(self.data.site_xpos[self._center])
        frame = np.asarray(self.data.site_xmat[self._center]).reshape(3, 3)
        local = frame.T @ (zone_pos - center)
        return bool(np.all(np.abs(local) <= np.asarray(tolerance_xyz)))

    def _attach(self, index: int) -> None:
        _, body_id, _, weld_id = self._entries[index]
        hand_rotation = np.asarray(self.data.xmat[self._hand]).reshape(3, 3)
        body_rotation = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        relative_position = hand_rotation.T @ (
            np.asarray(self.data.xpos[body_id]) - np.asarray(self.data.xpos[self._hand])
        )
        relative_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(relative_quaternion,
                           (hand_rotation.T @ body_rotation).reshape(9))
        # MuJoCo weld eq_data: anchor[0:3], relpos[3:6], relquat[6:10],
        # torquescale[10].  Compute relpose at the actual grasp instant, so
        # assistance never teleports an object to its initial MJCF pose.
        self.model.eq_data[weld_id, 3:6] = relative_position
        self.model.eq_data[weld_id, 6:10] = relative_quaternion
        self.data.eq_active[weld_id] = 1
        self._held_index = index
        mujoco.mj_forward(self.model, self.data)

    def before_control(self, gripper_command: float, *, source: str = "") -> AssistEvent:
        """Apply assist only for a commit-gate-approved low-level instruction.

        The caller still passes its unmodified 7D action to the controller;
        this method edits only the private MuJoCo equality activation.
        """

        command = float(gripper_command)
        if not math.isfinite(command):
            raise ValueError("gripper command must be finite")
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if command < 0:
            if self._held_index is None:
                return AssistEvent("unchanged")
            _, _, _, weld_id = self._entries[self._held_index]
            self.data.eq_active[weld_id] = 0
            self._held_index = None
            mujoco.mj_forward(self.model, self.data)
            return AssistEvent("detached")
        if command == 0 or self._held_index is not None:
            return AssistEvent("unchanged")
        mujoco.mj_forward(self.model, self.data)
        eligible = [i for i, (zone, _, site_id, _) in enumerate(self._entries)
                    if self._eligible(site_id, zone.tolerance_xyz)]
        if len(eligible) != 1:
            return AssistEvent("ambiguous" if eligible else "no_match", len(eligible))
        self._attach(eligible[0])
        return AssistEvent("attached", 1)
