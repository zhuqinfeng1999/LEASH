"""Versioned I1 common-tabletop candidate with a naturally extended right arm.

Only the seated person's RIGHT arm target changes from accepted I1 safe-slot
v3. All object bindings, assets, cameras, note text, hazards and grasp physics
remain inherited. This candidate requires new native visual/method acceptance.
"""
from __future__ import annotations

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.i1_v42_t1t7_20260925 import candidate as base
from research.v4.candidates.i1_v42_t1t7_20260925 import candidate_safe_slot_v3 as safe_slot

LEFT_HAND = (.20, -.38, .938)
RIGHT_HAND = (.125, .32, .938)
BINDINGS = safe_slot.BINDINGS


class I1StraightRightEnv(base._I1Env):
    @staticmethod
    def person_builder(arena):
        tabletop._append_e1_person(
            arena, include_right=True,
            left_hand=LEFT_HAND, right_hand=RIGHT_HAND,
        )
        base._append_microwave(arena)
        # Match the accepted I1 scene's rendered English note exactly.
        from research.v4.tasks.i1.scene import _append_note
        from research.v4.tasks.i1.template import load_instance
        _append_note(arena.root, load_instance("I1-01"))
        arena.root.find(".//texture[@name='v4_i1_print']").set(
            "file", str(base.HERE / "note_large_print.png"))
        arena.root.find(".//body[@name='v4_i1_paper_note']").set("pos", "0 .34 1.24")
        stand = arena.root.find(".//body[@name='v4_i1_note_stand']")
        stand.set("pos", "0 .34 .900")
        stand.find("geom[@name='v4_i1_note_post']").set("pos", "0 0 .15")
        stand.find("geom[@name='v4_i1_note_post']").set("size", ".008 .008 .15")
        import xml.etree.ElementTree as ET
        arena.tree = ET.ElementTree(arena.root)
        arena.worldbody = arena.root.find("worldbody")
        arena.asset = arena.root.find("asset")


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    """Return the same common tabletop with only versioned person pose changed."""
    key = "E1-01"
    original = tabletop.PILOTS.get(key)
    tabletop.PILOTS[key] = BINDINGS
    try:
        scene = I1StraightRightEnv(
            key, seed, camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        scene.reset()
        assist = GraspAssist(scene.sim.model._model, scene.sim.data._data,
                             **scene.grasp_assist_kwargs)
        return scene, assist
    finally:
        if original is None:
            del tabletop.PILOTS[key]
        else:
            tabletop.PILOTS[key] = original
