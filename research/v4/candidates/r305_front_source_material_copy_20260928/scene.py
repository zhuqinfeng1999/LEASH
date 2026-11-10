"""One source+board placement offset, consumed by the original reset binding."""
from dataclasses import replace
from unittest.mock import patch

from research.v4.accelerated import tabletop
from research.v4.candidates.r305_raised_neutral_copy_20260928 import scene as parent

SOURCE_DY = -.152266002685539
NEW_NOMINAL_Y = -.312266002685539
ORIGINAL_BINDING = tabletop.ObjectBinding
SOURCE_NAMES = ('fork_bowl_1', 'bowl_board_1')


class FrontSourceMaterialScene(parent.RaisedNeutralOwnScene):
    layout_version = 'R305-positive-material-rim-source-board-front-slot-v1'

    def __init__(self, *args, **kwargs):
        if tabletop.ObjectBinding is not ORIGINAL_BINDING:
            raise RuntimeError('Original placement binding class changed')
        if tabletop.SLOTS['L'].xyz != (-.08, -.24, .9):
            raise RuntimeError('Actual original source slot changed')
        if parent.parent.own.configuration()['layout']['source_dy_m'] != .08:
            raise RuntimeError('Original own post-reset source layout changed')
        counts = {name: 0 for name in SOURCE_NAMES}
        receipts = []

        def selected_binding(*a, **kw):
            binding = ORIGINAL_BINDING(*a, **kw)
            if binding.name in SOURCE_NAMES:
                if binding.slot != 'L' or binding.xy_offset != (0., 0.):
                    raise RuntimeError('Exact own source/board binding changed')
                counts[binding.name] += 1
                binding = replace(binding, xy_offset=(0., SOURCE_DY))
                receipts.append({'name': binding.name, 'slot': binding.slot,
                                 'xy_offset': binding.xy_offset,
                                 'z_offset': binding.z_offset,
                                 'grasp_local': binding.grasp_local})
            return binding

        # Only normal reset placement receives the additional source offset.
        # The parent retains its old constructor/reset/fixture protocol. This
        # wrapper adds no live qpos, pose, controller or equality write.
        with patch.object(tabletop, 'ObjectBinding', selected_binding):
            super().__init__(*args, **kwargs)
        if counts != {name: 1 for name in SOURCE_NAMES}:
            self.close()
            raise RuntimeError('Source/board placement factory count changed')
        self.front_source_binding_receipt = receipts
        self.front_source_binding_scope_restored = tabletop.ObjectBinding is ORIGINAL_BINDING
