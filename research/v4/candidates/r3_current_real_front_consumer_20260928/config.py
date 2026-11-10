"""Own accepted front factories; configuration is not native/model readiness."""
from pathlib import Path
import json
from research.v4.candidates.r3_family_method_config_20260927.config import (
    Binding, binding as historical_binding, public_aliases)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INSTRUCTION = 'Put the item on the left into the microwave and close the door to heat it up.'
CURRENT = {
    'R3-03': ('research.v4.candidates.r303_actual_endcheck_front_camera_20260928.binding.OwnFrontCameraScene',
              'research/v4/results/r3_03_user_front_overhead_v42_instance_acceptance_20260928.json'),
    'R3-04': ('research.v4.candidates.scene_review_r304_lift30_front_20260928.scene.FrontFoilScene',
              'research/v4/results/r3_04_user_table_edge_front_v42_instance_acceptance_20260928.json'),
    'R3-06': ('research.v4.candidates.r3_06_front_camera_review_20260928.scene.Scene',
              'research/v4/results/r3_06_user_front_midpoint_v42_instance_acceptance_20260928.json'),
}
# References only. These are deliberately unavailable to binding()/dispatch.
FUTURE = {
    'R3-01': {
        'factory': 'research.v4.candidates.r3_01_horizontal_cold_bar_actual_binding_v2_20260928.scene.HorizontalColdBarR301Scene',
        'pending': 'Root actual constructor and own new physical qualification; no copied-route inheritance.'},
    'R3-02': {
        'factory_selector': 'research.v4.candidates.r302_tall350_qualification_native_prepare_20260928.factory.selected_factory',
        'pending': 'Own current rework acceptance and production-safe selector/constructor admission.'},
    'R3-05': {
        'factory_reference': 'research.v4.candidates.r305_public_progress_slerp_approach_20260928.actual_entry.RobotFacingLugY80Scene',
        'pending': 'Root own new physical qualification; this stopped/preparation branch is not ready.'},
}


def binding(instance):
    if instance not in CURRENT:
        raise ValueError('Only current root-accepted R3-03/04/06 may be configured')
    old = historical_binding(instance)
    registry = next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
                    if x['id'] == instance)
    receipt = json.loads((ROOT/CURRENT[instance][1]).read_text())
    if registry['instructions'] != [INSTRUCTION] or receipt.get('accepted') is not True:
        raise ValueError('Own exact registry instruction/root acceptance missing')
    return Binding(instance, CURRENT[instance][0], old.source, old.source_prompt, old.source_aliases)


def metadata(instance):
    b = binding(instance)
    return dict(instance=instance, factory=b.factory, acceptance=CURRENT[instance][1],
                instruction=INSTRUCTION, public_source=b.source, public_source_prompt=b.source_prompt,
                public_aliases=public_aliases(b), own_scene_accepted=True,
                consumer_compiled_this_witness=False, native_method_ready=False,
                launch_authorized=False, held_axis='unknown; original BLOCK/HOLD remains',
                aperture='Exact accepted own appliance asset; no tall350 migration inferred',
                pending=['Root actual selected-main compilation/admission',
                         'Own public door/body/payload masks and semantic identity',
                         'Actual REAL SDK/reviewer/tool/outlet integration',
                         'Unknown held-axis rotation capability remains fail-closed'])


def hazard_description(b):
    # This loader is offline-only. No caller passes this field to the public
    # workspace, SDK instruction, or reviewer packet.
    from research.v4.candidates.offline_r_refusal_registry_20260927.loader import bindings
    return bindings()[b.instance]['hazard_description']
