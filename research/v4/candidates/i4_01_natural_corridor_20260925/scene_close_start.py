"""I4-01 candidate with a fixed, inspected wrist start pose.

The pose was obtained by a collision-free open-gripper path from the common
park. It is identical for every seed and uses the same two public cameras,
resolution, FOV and physics. This is an alternative observation protocol, not
an already accepted task or method result.
"""
from __future__ import annotations

from research.v4.accelerated import tabletop
from . import scene as base

LEFT_HAND = base.LEFT_HAND
RIGHT_HAND = base.RIGHT_HAND
TEXT = base.TEXT
INSTRUCTION = base.INSTRUCTION
BINDINGS = base.BINDINGS

INSPECTION_PARK_QPOS = (
    -0.0811678886357895, 0.8887429775816174,
    -0.06466321592357861, -1.5561648492671123,
    0.05317469672369221, 2.5525812118425364,
    0.618759343095488,
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    previous = tabletop.PARK_QPOS
    tabletop.PARK_QPOS = INSPECTION_PARK_QPOS
    try:
        return base.create(seed, camera_observations=camera_observations,
                           render_gpu_device_id=render_gpu_device_id)
    finally:
        tabletop.PARK_QPOS = previous
