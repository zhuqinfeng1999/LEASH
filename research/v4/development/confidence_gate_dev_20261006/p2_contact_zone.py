"""P2-01-only private contact-zone correction for future development scenes.

The actual accepted constructor and every physical/control parameter remain.
Only its object-local zone z changes during construction; this is not a TCP
target, an EMPTY fact, an assist-eligibility change or a scoring change.
Root selects this factory for a new version after the active run has ended.
"""
from dataclasses import replace
from unittest.mock import patch

from research.v4.candidates.user_rework_p2_20260927.scene_v3 import Scene as ParentScene
from research.v4.candidates.p2_v42_t3_20260925 import scene_safe_slot_v2 as physical


VERSION = 'p201-physical-handle-zone-dev1-20261007'
AFFECTED_INSTANCES = ('P2-01',)
ORIGINAL_ZONE = (-.045, 0., .028)
CONTACT_ZONE = (-.045, 0., 0.)
INITIAL_RESET_RECIPE = __name__ + ':Scene:P2-01'


def corrected_binding(original):
    """Replace one known private annotation; retain every other binding field."""
    if original.name != 'blunt_tool_1' or original.grasp_local != ORIGINAL_ZONE:
        raise ValueError('Expected the original P2-01 object-local contact zone')
    return replace(original, grasp_local=CONTACT_ZONE)


class Scene(ParentScene):
    contact_zone_version = VERSION

    def __init__(self, *args, **kwargs):
        binding = corrected_binding(physical.KNIFE_BINDING)
        # P2Facade's own PILOTS scope consumes this module binding during its
        # normal compilation. patch restores it even if the parent raises.
        with patch.object(physical, 'KNIFE_BINDING', binding):
            super().__init__(*args, **kwargs)
        # Name the actual factory, not a fabricated simulator reset revision.
        self.initial_reset_recipe = INITIAL_RESET_RECIPE
