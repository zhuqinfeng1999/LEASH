"""One explicit P1 development layout; no accepted/default scene is changed.

The left arm remains visible, collidable and protected, but its actual posed
bones and attached skin rest beyond the far x edge and below the tabletop.
Only the right hand/forearm enters the operation area. Table widening is
local to this backend instance, before the original ManipulationTask merge.
This file defines configuration only; it starts no Scene or check.
"""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from research.v4.accelerated.tabletop import SLOTS, sample_slot
from research.v4.candidates.p1_natural_left_rest_20260925.candidate_table_wrist import (
    TableWristPerson,
)


VERSION = "p1_single_right_arm_dev1"
INSTANCES = ("P1-01", "P1-02", "P1-03")
SOURCE_XY = (.02, -.34)
TARGET_XY = (.18, .36)
RIGHT_HAND = (.10, .00, .963)
LEFT_HAND = (.58, -.30, .720)
TABLE_FULL_SIZE = (.90, 1.20, .05)


class RetractedLeftArmPerson(TableWristPerson):
    """Use one real left-arm pose for skin, collisions and every update.

    The caller's old tabletop left-hand target cannot restore the previous
    obstacle: augment(), update() and surface_vertices() all use pose().
    The existing complete skeleton, collision sizes and protected list are
    retained, as are the right-arm solver and the left-wrist half-blend.
    """

    def pose(self, hand, right_hand=None):
        return super().pose(np.asarray(LEFT_HAND, dtype=float),
                            right_hand=right_hand)


def _append_scoped_person(arena):
    """Resize this P1 arena, then install the full person at the new pose."""
    # The original table's x length also places the Panda base. Keep x/z
    # unchanged: only extra lateral support is needed for full receiver rims.
    # Reuse TableArena.configure_location so its visual/collision sizes,
    # tabletop site and leg positions stay mutually consistent.
    arena.table_full_size = np.asarray(TABLE_FULL_SIZE, dtype=float)
    arena.table_half_size = arena.table_full_size / 2.
    arena.configure_location()

    # Use the explicit class, rather than _append_e1_person's module-global
    # Person. The inherited accepted P1 constructor temporarily patches that
    # global to TableWristPerson; it must not select the old left-arm pose.
    person = RetractedLeftArmPerson(include_right=True)
    person.right_hand = np.asarray(RIGHT_HAND, dtype=float).copy()
    hand = np.asarray(LEFT_HAND, dtype=float).copy()
    transformed = ET.fromstring(person.augment(arena.get_xml(), hand))
    arena.root = transformed
    arena.tree = ET.ElementTree(transformed)
    arena.worldbody = transformed.find("worldbody")
    arena.asset = transformed.find("asset")
    arena.person = person
    arena.person_hand_start = hand


def configuration(configured, instance, version):
    """Select the single development candidate; preserve own scorer/assets."""
    if version == "accepted":
        return configured
    if version != VERSION or instance not in INSTANCES:
        raise ValueError("Unsupported explicit single-right-arm P1 layout")

    original_backend = configured.Scene.backend_class

    class Backend(original_backend):
        person_builder = staticmethod(_append_scoped_person)

    class Scene(configured.Scene):
        backend_class = Backend
        layout_version = VERSION
        e1_plate_xy_m = TARGET_XY
        e1_hand_start_xyz_m = RIGHT_HAND
        e1_left_rest_xyz_m = LEFT_HAND

        def _place_e1_layout_at_reset(self):
            # Original E1 placement fixes the receiver at the actual pose and
            # updates e1_plate_fixture. Class attributes above are consumed
            # there, so this does not move an already fixed receiving body.
            super()._place_e1_layout_at_reset()
            sim = self.backend.sim
            item = self.backend.objects_dict[self.item_name]
            pose = np.asarray(sim.data.get_joint_qpos(item.joints[0]),
                              dtype=float).copy()
            sampled, _yaw = sample_slot("L", self.backend.layout_seed,
                                        object_index=0)
            # Replace only nominal XY, retaining the original +/-2 cm source
            # perturbation, yaw, body height and exact object geometry.
            pose[:2] = (np.asarray(SOURCE_XY, dtype=float) + sampled[:2]
                        - np.asarray(SLOTS["L"].xyz[:2], dtype=float))
            sim.data.set_joint_qpos(item.joints[0], pose)
            sim.data.set_joint_qvel(item.joints[0], np.zeros(6))
            sim.forward()

    return SimpleNamespace(**{**vars(configured), "Scene": Scene})
