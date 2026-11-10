"""I4-01 display-only upright label on the same close-wrist tabletop."""
from __future__ import annotations

from dataclasses import replace

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.i4_01_natural_corridor_20260925 import scene_close_start as prior
from research.v4.candidates.i4_01_natural_corridor_20260925.scene import I4HandleLabelEnv
from .label import ASSET, IMAGE, TEXT

LEFT_HAND = prior.LEFT_HAND
RIGHT_HAND = prior.RIGHT_HAND
INSTRUCTION = prior.INSTRUCTION
INSPECTION_PARK_QPOS = prior.INSPECTION_PARK_QPOS
BINDINGS = tuple(
    replace(item, xml=str(ASSET.relative_to(tabletop.ROOT)))
    if item.name == "blunt_tool_1" else item
    for item in prior.BINDINGS
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    if not (ASSET.is_file() and IMAGE.is_file()):
        raise FileNotFoundError("render upright tag asset first")
    key = "E1-01"
    previous_bindings = tabletop.PILOTS.get(key)
    previous_park = tabletop.PARK_QPOS
    tabletop.PILOTS[key] = BINDINGS
    tabletop.PARK_QPOS = INSPECTION_PARK_QPOS
    try:
        env = I4HandleLabelEnv(key, seed,
                               camera_observations=camera_observations,
                               render_gpu_device_id=render_gpu_device_id)
        env.reset()
        assist = GraspAssist(env.sim.model._model, env.sim.data._data,
                             **env.grasp_assist_kwargs)
        return env, assist
    finally:
        if previous_bindings is None:
            del tabletop.PILOTS[key]
        else:
            tabletop.PILOTS[key] = previous_bindings
        tabletop.PARK_QPOS = previous_park
