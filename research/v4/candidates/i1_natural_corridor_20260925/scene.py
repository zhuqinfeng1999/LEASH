"""I1 common tabletop: keep the natural arm, move only the basket 10 cm toward L.

The existing right hand remains at (0.125, 0.320, 0.938) m. The basket's
R-slot y offset changes from -0.08 to -0.18 m. Nothing about the evaluator,
gate, grasp zones, objects, or camera protocol changes here.
"""
from __future__ import annotations

from dataclasses import replace

from research.v4.candidates.i1_straight_right_arm_20260925 import scene as prior

LEFT_HAND = prior.LEFT_HAND
RIGHT_HAND = prior.RIGHT_HAND
BINDINGS = tuple(
    replace(item, xy_offset=(item.xy_offset[0], -.18))
    if item.name == "basket_1" else item
    for item in prior.BINDINGS
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    before = prior.BINDINGS
    prior.BINDINGS = BINDINGS
    try:
        return prior.create(seed, camera_observations=camera_observations,
                            render_gpu_device_id=render_gpu_device_id)
    finally:
        prior.BINDINGS = before
