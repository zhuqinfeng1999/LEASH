"""Opt-in normal reset sampling seam; not an episode-time body write."""
from contextlib import contextmanager
import math
from unittest.mock import patch

from research.v4.accelerated import tabletop

ORIGINAL_SAMPLE=tabletop.sample_slot
SOURCE_XY=(-.08,-.312266002685539)
ROBOT_BASE_XY=(-.61,0.)
YAW_OFFSET=math.atan2(ROBOT_BASE_XY[1]-SOURCE_XY[1],ROBOT_BASE_XY[0]-SOURCE_XY[0])


@contextmanager
def robot_facing_source_reset_scope(bindings):
    # Explicit own binding order guards the sampler's index, rather than use a
    # scene-name substring or intercept unrelated L-slot objects.
    expected=(('fork_bowl_1','L'),('microwave_1','R'),('bowl_board_1','L'))
    if tuple((b.name,b.slot)for b in bindings)!=expected:
        raise RuntimeError('Exact own reset bindings/order differ')
    if tuple(bindings[0].grasp_local)!=(.080,0.,.022):
        raise RuntimeError('Exact physical ceramic-ear grasp annotation differs')
    if tabletop.sample_slot is not ORIGINAL_SAMPLE:
        raise RuntimeError('Original sampler scope already modified')
    def selected(slot,seed,*,object_index=0):
        xyz,yaw=ORIGINAL_SAMPLE(slot,seed,object_index=object_index)
        if object_index==0:
            if slot!='L':raise RuntimeError('Actual own source reset slot differs')
            yaw+=YAW_OFFSET
        return xyz,yaw
    try:
        with patch.object(tabletop,'sample_slot',selected):yield selected
    finally:
        if tabletop.sample_slot is not ORIGINAL_SAMPLE:
            raise RuntimeError('Normal reset sampler scope failed to restore')


# This is only the proposed placement seam. There is no qualified actual
# Y80+cart constructor here; the geometry candidate must pass before root may
# bind this scope around its own validated normal construction/reset stages.
