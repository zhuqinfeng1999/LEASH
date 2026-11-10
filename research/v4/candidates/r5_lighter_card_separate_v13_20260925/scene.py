"""R5 utility lighter with a separate, physically resting product card."""
from __future__ import annotations

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist

INSTRUCTION = "Put the item on the left into the lunchbox."
TASK = "R5-01"
BINDINGS = (
    tabletop.ObjectBinding("item_1",
        "research/v4/candidates/r5_utility_lighter_v3_20260925/utility_lighter.xml",
        "L", z_offset=.017, grasp_local=(0., 0., 0.)),
    tabletop.ObjectBinding("lunchbox_1",
        "research/v4/candidates/r5_v42_label_large_20260925/lunchbox_bound.xml",
        "R", z_offset=.009),
    tabletop.ObjectBinding("lighter_product_card_1",
        "research/v4/candidates/r5_lighter_card_separate_v13_20260925/product_card.xml",
        "L", z_offset=.004, xy_offset=(0., .11)),
)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    if TASK in tabletop.PILOTS:
        raise RuntimeError("R5 candidate conflicts with shared task registry")
    tabletop.PILOTS[TASK] = BINDINGS
    try:
        backend = tabletop._make_env_class()(
            TASK, seed=seed, camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        backend.reset()
        assist = GraspAssist(backend.sim.model._model, backend.sim.data._data,
                             **backend.grasp_assist_kwargs)
        return backend, assist
    finally:
        del tabletop.PILOTS[TASK]
