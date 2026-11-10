"""Own physical asset opt-in; normal original reset/controller/20 OPEN remain."""
from unittest.mock import patch
from research.v4.accelerated.tabletop import ObjectBinding
from research.v4.candidates.r3_01_horizontal_cold_bar_actual_binding_v2_20260928 import scene as original
from research.v4.candidates.r3_01_horizontal_cold_bar_actual_binding_v2_20260928 import admission as parent_admission
from .admission import H, admit
from research.v4.leash_v43.run_manifest import source_custody_enabled

POINT = (0., .113, .062)
BINDINGS = (
    ObjectBinding(original.carrier.ITEM, str(H / 'moka_horizontal_u_flat_sole.xml'), 'L',
                  z_offset=.118, xy_offset=(0., -.072266002685539), grasp_local=POINT),
    original.BINDINGS[1],
)


class HorizontalUFlatSoleR301Scene(original.HorizontalColdBarR301Scene):
    layout_version = 'R301-horizontal-U-flat-sole-foot020-base061-wood50-tall350-v1'

    def __init__(self, *args, root_candidate_approved=False, **kwargs):
        if root_candidate_approved is not True:
            raise ValueError('Root review required for this new physical version')
        if source_custody_enabled():
            admit()
        kwargs.setdefault('camera_observations', False)
        # Original constructor itself calls parent_admission.admit before its
        # original scoped factory. Complete membership is retained; only the
        # explicitly versioned AGENTS hash is current. No old freeze is edited.
        with patch.object(original, 'BINDINGS', BINDINGS), patch.object(parent_admission, 'admit', admit):
            try:
                super().__init__(*args, root_candidate_approved=True, **kwargs)
            except BaseException as exc:
                # Only already-recorded original snapshots survive a failure
                # before run_case receives the Scene. No new capture or step.
                exc.r301_initialization_evidence = {
                    'initialization_before': getattr(self, 'initialization_before', None),
                    'common20_initialization': list(getattr(self, 'common_initialization_trace', [])),
                    'initialization_after': getattr(self, 'initialization_after', None),
                }
                raise
        if source_custody_enabled():
            admit()

    # step/capture/assist/scorer/close are inherited unchanged, not copied.
