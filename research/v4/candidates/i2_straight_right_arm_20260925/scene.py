"""I2 note pixels on the validated naturally extended right-arm common scene."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.i1_straight_right_arm_20260925.scene import (
    I1StraightRightEnv, BINDINGS, LEFT_HAND, RIGHT_HAND,
)
from research.v4.candidates.i1_v42_t1t7_20260925.candidate import INSTRUCTION
from research.v4.candidates.i2_v42_overlay_20260925.note import NOTE, OUTPUT

HERE = Path(__file__).resolve().parent


class I2StraightRightEnv(I1StraightRightEnv):
    @staticmethod
    def person_builder(arena):
        I1StraightRightEnv.person_builder(arena)
        if not OUTPUT.is_file():
            raise FileNotFoundError("Source-pinned English I2 paper raster missing")
        texture = arena.root.find(".//texture[@name='v4_i1_print']")
        if texture is None:
            raise RuntimeError("Inherited paper texture missing")
        texture.set("file", str(OUTPUT))
        arena.tree = ET.ElementTree(arena.root)
        arena.worldbody = arena.root.find("worldbody")
        arena.asset = arena.root.find("asset")


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    key = "E1-01"
    original = tabletop.PILOTS.get(key)
    tabletop.PILOTS[key] = BINDINGS
    try:
        scene = I2StraightRightEnv(
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
