"""I2 note overlay on the versioned I1 natural-arm basket corridor.

Only the basket R-slot y offset changes from -0.08 to -0.18 m relative to
the previously qualified I2 natural-arm scene. The note, arm, assets,
cameras, grasp assist, and evaluator remain unchanged.
"""
from __future__ import annotations

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.i1_natural_corridor_20260925.scene import (
    BINDINGS, LEFT_HAND, RIGHT_HAND,
)
from research.v4.candidates.i2_straight_right_arm_20260925.scene import (
    I2StraightRightEnv, INSTRUCTION, NOTE,
)


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
