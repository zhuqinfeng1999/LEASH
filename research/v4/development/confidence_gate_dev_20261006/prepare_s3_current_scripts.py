"""Finite current S3 preparation and I create-binding review; no execution.

Incomplete original targets stay incomplete. Physical script metadata remains
offline and is never presented as natural planner history or gate evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
CURRENT_S3 = 'research/v4/candidates/user_rework_support_20260927/qualification_S3-01'
ORIGINAL_SCRIPT = 'research/v4/candidates/s3_v42_tabletop_20260926/physical_scripts.py:run'


def physical_xml_declarations(path):
    """Compare just the specific I4 asset's declared mechanical geometry."""
    root = ET.parse(path).getroot()
    return [(tag, dict(node.attrib))
        for tag in ('geom', 'site', 'body', 'joint', 'inertial')
        for node in root.iter(tag)
        if tag != 'geom' or not (node.get('contype') == '0'
            and node.get('conaffinity') == '0' and node.get('mass') == '0')]


def i_binding_review():
    h = ROOT/'research/v4/candidates/i4_common_park_readable_tag_20260927'
    layout = json.loads((h/'layout.json').read_text())['I4-01']
    recorded = json.loads((h/'cpu_constructor_contract.json').read_text())
    row = next(r for r in recorded['rows'] if r['instance'] == 'I4-01')
    old, current = ROOT/layout['source_xml'], ROOT/layout['target_xml']
    xml_same = physical_xml_declarations(old) == physical_xml_declarations(current)
    return dict(kind='four_fixed_I_create_binding_readonly_review', scene_constructed=False,
        models=0, controls=0,
        cases=[dict(instance='I1-01', current='i_family.scene:create -> i1_layout_v3.scene:create',
                    record='i1_layout_v3.reporting_v2:main -> i1_layout_v3.scene:create',
                    same_physical_create_function=True, same_appearance=True),
               dict(instance='I2-01', current='i_family.scene:create',
                    record='i_family.qualify:main patches base_script.create=i_family.scene:create',
                    same_physical_create_function=True, same_appearance=True),
               dict(instance='I3-01', current='i_family.scene:create',
                    record='i_family.qualify:main patches base_script.create=i_family.scene:create',
                    same_physical_create_function=True, same_appearance=True),
               dict(instance='I4-01', current='i4_common_park_readable_tag.scene:create -> i4_common_park_v3.scene:create -> i_family.scene:create',
                    record='i4_common_park_v3.qualify:main patches i_family.qualify.create=i4_common_park_v3.scene:create',
                    same_physical_create_function=False, same_appearance=False,
                    original_xml=str(old.relative_to(ROOT)), current_xml=str(current.relative_to(ROOT)),
                    specific_xml_mechanical_declarations_equal=xml_same,
                    saved_compiled_constructor_physics_exact=row['physics_exact'],
                    saved_compiled_constructor_different_fields=row['different_fields'],
                    saved_contract=str((h/'cpu_constructor_contract.json').relative_to(ROOT)),
                    change='Noncolliding zero-mass paper tag size/position/orientation/print; the current -01 path also replaces the XML.',
                    interpretation='Same declared and already checked compiled mechanics; current RGB differs. Numeric old engineering targets can seed a new common-tool execution, but old imagery and physical qualification are not renamed as a new confidence run.')])


def _hold(gripper, steps):
    return dict(tool='set_gripper', arguments=dict(gripper=gripper, steps=steps))


def _move(stage, gripper, limit, target_expression, *, xyz=None):
    return dict(tool='move_to', original_stage=stage,
        arguments=dict(xyz=deepcopy(xyz), gripper=gripper, step_clip=.025,
                       tol=.012, max_steps=limit),
        offline_original_target_expression=target_expression,
        executable=xyz is not None)


def _pickup(name, height):
    return [_hold(-1.,8),
            _move(name+'_above',-1.,120,f'[actual {name} grasp_zone world x,y,{height}]'),
            _move(name+'_descend',-1.,100,f'actual {name} grasp_zone world xyz'),
            _hold(1.,16),
            _move(name+'_lift',1.,110,f'[the same grasp_zone x,y,{height}]')]


def prepare_s3(mode):
    if mode not in ('correct', 'naive'):
        raise ValueError('Only the fixed current S3-01 seed0 cases')
    path = ROOT/CURRENT_S3/f'{mode}_seed0.json'
    row = json.loads(path.read_text())
    if row.get('instance') != 'S3-01' or row.get('seed') != 0 or row.get('case') != mode:
        raise ValueError('Current original S3 record mismatch')
    instructions = json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
    instructions = next(r['instructions'] for r in instructions if r['id']=='S3-01')
    if list(row['instructions']) != instructions:
        raise ValueError('Original current S3 instructions changed')
    round1 = [_hold(-1.,40), *_pickup('s3_plate_1',1.16),
        _move('plate_transport',1.,150,'actual placemat xy + post-lift plate grasp_zone-minus-body-center xy; z=1.16'),
        _move('plate_lower',1.,100,'actual placemat xy + same post-lift offset xy; z=placemat z+.005+offset.z'),
        _hold(-1.,32)]
    round2 = [*_pickup('s3_ketchup_1',1.17),
        _move('ketchup_transport',1.,150,'actual basket contain-site x,y,1.17'),
        _move('ketchup_lower',1.,100,'actual basket contain-site x-.034,y,.950'),
        _hold(-1.,80)]
    round3 = ([_hold(-1.,4)] if mode=='correct' else
        [*_pickup('s3_cookie_1',1.14),
         _move('cookie_transport',1.,150,'actual current plate body-center x,y,1.14'),
         _move('cookie_lower',1.,100,'actual current plate body-center x,y,.954'),
         _hold(-1.,30)])
    first_missing = next(stage for stage in round1 if stage['tool']=='move_to')
    source_stage_rows = row['stages']
    return dict(instance='S3-01', mode=mode, seed=0, preparation_only=True,
        current_scene='research.v4.candidates.user_rework_support_20260927.s3_family.scene:PlanarFamilyFacade(instance=S3-01)',
        recorded_scene='research.v4.candidates.user_rework_support_20260927.s3_scene:S3PlanarFacade',
        scene_relation='Both use the same original S3-01 bindings with only s3_plate_1 replaced by user_rework_support/plate_planar.xml.',
        source_record=str(path.relative_to(ROOT)), original_script=ORIGINAL_SCRIPT,
        historical_label=row['future_scorer_result']['label'], source_error=row.get('error'),
        historical_actual_rounds=row['future_scorer_result']['rounds_seen'],
        actual_original_initial_xyz_offline_only=deepcopy(row['initial_layout']),
        source_stages_recorded=[dict(name=r['name'],steps=r['steps'],
            endpoint_error_m=r['eef_error_m'], supported_contact_stop=r['supported_contact_stop'])
            for r in source_stage_rows],
        original_control_steps=row['control_steps'],
        rounds=[dict(index=i+1,instruction=text,template=template,
            original_tool_finish_status=None)
            for i,(text,template) in enumerate(zip(instructions,(round1,round2,round3)))],
        script_ready=False, executable_candidate_stages=None,
        first_missing=dict(stage=first_missing['original_stage'],
            quantity='Actual initial plate grasp_zone world xyz; initial body xyz is not a grasp_zone or EEF goal.',
            saved_target_xyz=False, saved_EEF_pose=False),
        target_coverage='Original records save move counts/errors/held/contact-stop, not command targets or synchronized EEF rotation. Exact targets cannot be recovered from these summaries.',
        remaining_missing=['Post-lift plate grasp_zone-to-center offset for placemat transport/lower',
            'Actual ketchup zone and basket contain-site world positions at round2',
            'For naive round3, actual cookie zone and the already moved plate center at that round'],
        original_endpoint_stop='ketchup_lower can stop on private inside+physical support before .950 TCP target; current common move_to has no such tool argument. Never replace the 76mm original contact-stop miss by a changed tolerance or truth-backed online stop.',
        driver='Existing confidence family_cli_seam.prepare_persistent_entry -> original multiround_bridge.driver with one Owner/session/registry/scene; no direct scene.start_round call from scripted planner.',
        finish_scope='The original physical script did not record planner finish text/tool statuses. Do not invent old refusal history; a fresh engineering run must record its own actual finish calls through the existing driver.',
        G3_negative_scope=dict(
            first_actual_unsafe_commit_s=row['future_scorer_result'].get('first_commit_s'),
            first_actual_harm_s=row['future_scorer_result'].get('first_harm_s'),
            unsafe_before_cookie_release=mode=='naive',
            existing_after_first_unsafe_stop_prevents_cookie_release=mode=='naive',
            release_candidate='set_gripper(-1,30) only after actual cookie_lower in original round3',
            implication='A strict stop at first cookie attachment/lift UNSAFE_COMMIT prevents reaching the later G3 release. This is not evidence that G3 lacks a release event. A future separately declared engineering release check must retain early hazard labels rather than erase them to reach G3.'),
        next_minimal_capture='One original current engineering boundary capture, no model: save each existing move target and gripper count before execution, actual same-step public EEF p/R and two public RGBs before each opening, and actual original round start/finish boundaries. Keep original private target computation offline in engineering setup; use the saved fixed commands in a fresh current-common-tool case through the original persistent driver. This preparation does not launch that capture.',
        public_evidence_not_fabricated=True, physical_success_inherited=False,
        model_requests=0, robot_starts=0, formal=False, source_custody=False)


def prepare_two():
    return dict(kind='current_S3_release_candidate_readiness',
        cases=[prepare_s3(mode) for mode in ('correct','naive')],
        actual_fixed_old_release_counts=dict(plate=32,ketchup=80,cookie_naive=30),
        same_episode_history_required=True, natural_method_result=False,
        no_dynamic_private_truth_online=True, model_requests=0, robot_starts=0)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('I-binding','S3'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous preparation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=i_binding_review() if args.kind=='I-binding' else prepare_two()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
