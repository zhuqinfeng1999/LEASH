"""One nominal mouth-centered source; unchanged original fully-open device."""
from unittest.mock import patch
from dataclasses import replace
from research.v4.accelerated import tabletop
from research.v4.candidates.user_rework_r3_20260927.soup_low_table_v8.scene import SoupLowTableScene

class MouthCenteredScene(SoupLowTableScene):
    layout_version = 'R302-mouth-centered-normal-top-source-readonly-v1'
    def __init__(self, *args, **kwargs):
        original = tabletop.ObjectBinding
        def binding(*a, **k):
            value = original(*a, **k)
            if value.name == self.item_name:
                if value.xy_offset != (-.14, 0.):
                    raise ValueError('Qualified source-binding seam changed')
                # Nominal oven (.02,0), yaw -90; mouth lateral midpoint
                # local X=-.072 maps to world Y=+.072. Front source X=-.32.
                value = replace(value, xy_offset=(-.24, .312))
            return value
        with patch.object(tabletop, 'ObjectBinding', binding):
            super().__init__(*args, **kwargs)
