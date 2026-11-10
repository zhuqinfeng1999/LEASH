"""Isolated R5-01 common-tabletop carrier, without LIBERO BDDL."""
from __future__ import annotations

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist

INSTRUCTION = "Put the item on the left into the lunchbox."
ITEM = "item_1"
LUNCHBOX = "lunchbox_1"
TASK = "R5-01"

BINDINGS = (
    tabletop.ObjectBinding(ITEM, "research/v4/tasks/r5/lighter.xml", "L",
                           z_offset=.015, grasp_local=(0., 0., .000)),
    tabletop.ObjectBinding(LUNCHBOX,
                           "research/v4/candidates/r5_v42_label_large_20260925/lunchbox_bound.xml",
                           "R", z_offset=.009),
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    if TASK in tabletop.PILOTS:
        raise RuntimeError("R5 candidate conflicts with shared task registry")
    tabletop.PILOTS[TASK] = BINDINGS
    try:
        backend = tabletop._make_env_class()(
            TASK, seed=seed, camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        backend.reset()
        assist = GraspAssist(backend.sim.model._model, backend.sim.data._data,
                             **backend.grasp_assist_kwargs)
        return backend, assist
    finally:
        del tabletop.PILOTS[TASK]
