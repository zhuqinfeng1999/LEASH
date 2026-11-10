"""One own R305 reset-pose version at its last already-consumed PARK override."""
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_05_oven_only_prefix_20260928 import scene as parent

NEW_PARK = (0., -.17536149667384343, 0., -1.7895761989061578,
            0., 1.7416096266501226, .7853981633974483)
OLD_PARK = tuple(parent.own.NEUTRAL_QPOS)


class RaisedNeutralOwnScene(parent.OvenOnlyPrefixScene):
    layout_version = 'R305-original-source-front-oven-raised-neutral-reset-v1'

    def __init__(self, *args, **kwargs):
        if tuple(tabletop.PARK_QPOS) != NEW_PARK or tuple(parent.own.NEUTRAL_QPOS) != OLD_PARK:
            raise RuntimeError('Exact original own reset seam or approved PARK changed')
        if OLD_PARK == NEW_PARK:
            raise RuntimeError('Own source already has requested PARK; no new version needed')
        # Parent then scopes tabletop.PARK_QPOS to this value and normal Robot
        # reset initializes both actual qpos and OSC's initial_joint reference.
        # No episode-time qpos/pose/controller write is added here.
        with patch.object(parent.own, 'NEUTRAL_QPOS', NEW_PARK):
            super().__init__(*args, **kwargs)
