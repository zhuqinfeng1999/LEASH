"""I1 shared-layout v3: move the bottle out of the public left-hand sweep.

The v2 source remains frozen for #240/#241. This changes only the bottle L-slot
XY offset from (+.18,+.04) to (+.03,+.04), a 15 cm shift toward the robot. It
retains true bottle size, collision, public cameras, human pose and risk rules.
"""
from __future__ import annotations
from dataclasses import replace
from . import candidate as prior

BINDINGS = tuple(
    replace(item, xy_offset=(.03, .04)) if item.name == 'ketchup_1' else item
    for item in prior.BINDINGS
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    # The existing scene builder reads its module-level binding table during
    # construction. Each native episode is a separate process and restores it.
    before=prior.BINDINGS
    prior.BINDINGS=BINDINGS
    try:
        return prior.create(seed,camera_observations=camera_observations,
                            render_gpu_device_id=render_gpu_device_id)
    finally:
        prior.BINDINGS=before
