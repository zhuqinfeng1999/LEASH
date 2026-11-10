"""One source+board+real wood-plinth table-edge reset layout; CPU candidate."""
from unittest.mock import patch

from research.v4.candidates.r304_source_plinth60_mirrored_20260928 import scene as parent

TABLE_NEGATIVE_Y = -.450
EDGE_MARGIN = .005
SOURCE_Y = TABLE_NEGATIVE_Y + parent.PLINTH_HALF[1] + EDGE_MARGIN
SOURCE_SHIFT_Y = SOURCE_Y - parent.PLINTH_CENTER[1]


def configuration():
    c = parent.configuration()
    c['layout']['source_dy_m'] += SOURCE_SHIFT_Y
    return c


class TableEdgePlinthCartEnv(parent.SourcePlinthCartEnv):
    def _load_model(self):
        super()._load_model()
        body = self.model.worldbody.find("body[@name='r304_fixed_wood_source_plinth60']")
        if body is None:
            raise RuntimeError('Original physical source support not compiled')
        original = tuple(float(v) for v in body.get('pos').split())
        expected = (*parent.PLINTH_CENTER, .9 + parent.RISE / 2)
        if original != expected:
            raise RuntimeError('Original physical source support pose changed')
        body.set('pos', f'{original[0]} {SOURCE_Y} {original[2]}')


class TableEdgeFoilScene(parent.donor.HorizontalSideFoilScene):
    layout_version = 'R304-table-edge-real-source-board-plinth60-v1'

    def __init__(self, *args, **kwargs):
        if tuple(parent.own.NEUTRAL_QPOS) != parent.OLD_Q:
            raise RuntimeError('Original own initializer changed')
        with patch.object(parent.donor, 'configuration', configuration), \
             patch.object(parent.donor, 'OwnCartEnv', TableEdgePlinthCartEnv), \
             patch.object(parent.own, 'NEUTRAL_QPOS', parent.NEW_Q):
            super().__init__(*args, **kwargs)
