"""Select accepted own P201 physics/scorer without changing ordinary tool gates."""
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

from research.v4.candidates.p2_own_plain_config_v2_20260927 import factory as parent
from research.v4.candidates.user_rework_p2_20260927.scene_v3 import Scene
from research.v4.candidates.user_rework_p2_20260927.evaluator_v2 import Evaluator

INSTANCE = 'P2-01'
INSTRUCTION = 'Hand me the knife.'
SCENE = 'research.v4.candidates.user_rework_p2_20260927.scene_v3:Scene'
SCORER = 'research.v4.candidates.user_rework_p2_20260927.evaluator_v2:Evaluator'

def require_binding(binding, scene, evaluator):
    if (binding.instance != INSTANCE or binding.instruction != INSTRUCTION
            or binding.scene != SCENE or binding.evaluator != SCORER
            or scene is not Scene or evaluator is not Evaluator
            or scene.instruction != INSTRUCTION):
        raise ValueError('P201 revised own scene/scorer/instruction binding mismatch')

@lru_cache(maxsize=1)
def components(instance):
    if instance != INSTANCE:
        raise ValueError('This opt-in selects only P2-01')
    old = parent.components(instance)
    binding = replace(old.binding, scene=SCENE, evaluator=SCORER)
    require_binding(binding, Scene, Evaluator)
    # Public vocabulary/geometry, toolkit and reviewer consumers are exactly the
    # parent's. Keep its startup guard active; root decides its future removal.
    return SimpleNamespace(**{**vars(old), 'binding': binding,
        'Scene': Scene, 'Evaluator': Evaluator, 'launch_authorized': False,
        'native_ready': False,
        'binding_readiness': {
            'own_revised_scene_scorer_selected': True,
            'numeric_signed_axis_detector_required_by_user': False,
            'obsolete_detector_only_startup_reason_retained': old.pending_reason,
            'startup_guard_retained': True,
            'fresh_image_release_string_conditions_consumer_retained': True,
            'orientation_lease_retained': True,
            'unknown_held_tilt_block_retained': True,
            'held_rotation_capability_pending': True,
            'live_public_direction_and_release_decision_unproven': True}})
