"""Read six existing public-boundary logs; never collect or execute again.

The legacy closures are not RPent tool invocations. Field-renamed RPent
candidates are output as preparation only, with all non-tool engineering
semantics and layout mismatches retained as explicit gaps. Nothing is sent to
a planner/reviewer, and private truth is not used to form these candidates.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = ROOT/'research/v4/runs/transport_workspace_dev_20261005'
CASES = tuple((instance, prefix, mode) for instance, prefix in
    (('P2-01','P201'), ('P3-01','P301'), ('P4-01','P401')) for mode in ('safe','naive'))
PHYSICS = {
    'P2-01': dict(collected='research.v4.candidates.p2_v42_t3_20260925.scene_safe_slot_v2:P2Facade',
                  current='research.v4.candidates.user_rework_p2_20260927.scene_v3:Scene',
                  same=False),
    'P3-01': dict(collected='research.v4.accelerated.p3_facade:P3TabletopFacade',
                  current='research.v4.accelerated.p3_facade:P3TabletopFacade',same=True),
    'P4-01': dict(collected='research.v4.candidates.user_rework_p4_20260927.layout_v1.scene:ForwardIronScene',
                  current='research.v4.candidates.user_rework_p4_20260927.layout_v1.scene:ForwardIronScene',same=True),
}
MOVE_KEYS = ('target_xyz','grip','clip','tol','max_steps')


def prepare_boundary(row):
    """Preserve values/order; flag every field not accepted by the real tool."""
    from robots.libero.tools import LiberoPrimitives
    name = row.get('original_callable')
    parameters = deepcopy(row.get('parameters'))
    if not isinstance(parameters, dict):
        return dict(status='missing_parameters', original_callable=name,
                    missing=['parameters'], candidate=None)
    result = dict(original_callable=name, original_parameters=parameters,
        source_line=row.get('source_line'), control_start=row.get('control_start'),
        control_end=row.get('control_end'), candidate=None, gaps=[],
        direct_original_tool_call=False)
    if name == 'move':
        missing = [key for key in MOVE_KEYS if key not in parameters]
        if missing:
            result.update(status='missing_parameters', missing=missing)
            return result
        arguments = dict(xyz=deepcopy(parameters['target_xyz']),
            gripper=parameters['grip'],step_clip=parameters['clip'],
            tol=parameters['tol'],max_steps=parameters['max_steps'])
        tool = 'move_to'
        business_keys = set(MOVE_KEYS)
    elif name == 'hold':
        missing = [key for key in ('grip','steps') if key not in parameters]
        if missing:
            result.update(status='missing_parameters', missing=missing)
            return result
        # Legacy hold is N exact zero-motion + gripper vectors. RPent's
        # hold(seconds) has neither this command nor this count semantics.
        tool = 'set_gripper'
        arguments = dict(gripper=parameters['grip'],steps=parameters['steps'])
        business_keys = {'grip','steps'}
    else:
        result.update(status='unsupported_legacy_callable', missing=[name])
        return result
    inspect.signature(getattr(LiberoPrimitives,tool)).bind(None,**arguments)
    result['candidate'] = dict(tool=tool,arguments=arguments)
    result['non_tool_parameters'] = {key:deepcopy(value) for key,value in parameters.items()
                                     if key not in business_keys}
    unknown = set(parameters)-business_keys-{'stage','stop_on_harm','allow_error'}
    if unknown:
        result['gaps'].append('Unmapped original parameters: '+','.join(sorted(unknown)))
    if parameters.get('stop_on_harm') is True:
        result['gaps'].append('Original mid-primitive private-hazard stop has no RPent tool argument; do not silently replace by end-of-primitive stopping.')
    if parameters.get('allow_error') is True:
        result['gaps'].append('Original continuation after unreached endpoint differs from current scripted.failed_return; do not silently drop allow_error.')
    result['status'] = 'parameter_mapping_proposal'
    return result


def prepare_collection(collection, *, path):
    case = collection['case']
    instance = case['instance']
    physical = deepcopy(PHYSICS[instance])
    boundaries = collection.get('primitives')
    if not isinstance(boundaries,list):
        raise ValueError('Collection schema lacks original boundary list')
    prepared = [prepare_boundary(row) for row in boundaries]
    gaps = list((collection.get('trace') or {}).get('gaps') or ())
    if collection.get('error'):
        gaps.append('Collection failed: '+str(collection['error']))
    if not physical['same']:
        gaps.append('Collected P2 physical/script binding is legacy safe-slot-v2; current own scene_v3 includes held presentation. Old boundary targets are not a current-layout script.')
    controls = (collection.get('trace') or {}).get('controls')
    if not isinstance(controls,list):
        gaps.append('Original control trace unavailable; boundary coverage cannot be established')
        controls = []
    rotation_controls = sum(any(abs(float(v))>1e-8 for v in action[3:6]) for action in controls)
    gaps.extend(f'boundary {index}: '+reason for index,row in enumerate(prepared,1)
                for reason in row.get('gaps',()))
    gaps.extend(f'boundary {index}: {row["status"]}' for index,row in enumerate(prepared,1)
                if row['candidate'] is None)
    return dict(instance=instance, mode=case['mode'],seed=case['seed'],
        source_collection=str(path.relative_to(ROOT)),source_case_id=case['id'],
        physical_binding=physical, original_controller=collection.get('controller'),
        original_control_count=len(controls), original_rotation_controls=rotation_controls,
        original_boundary_count=len(boundaries),stages=[row['candidate'] for row in prepared],
        original_boundaries=prepared, gaps=gaps,
        all_numeric_parameters_available=all(row['candidate'] is not None for row in prepared),
        strict_original_replay_ready=False,
        current_common_controller_execution_not_verified=True,
        candidate_status='Root review required; no case registered or launched',
        original_stage_names_are_offline_only=True,
        original_snapshots_and_truth_not_copied_to_online_candidates=True)


def prepare_six():
    rows = []
    for instance,prefix,mode in CASES:
        name=f'V1_{prefix}_{mode}_public_collection_dev1_20261006'
        path=RUN_ROOT/name/'collection.json'
        rows.append(prepare_collection(json.loads(path.read_text()),path=path))
    return dict(kind='six_saved_boundary_parameter_preparation',formal=False,
        model_requests=0,native_starts=0,source_custody=False,
        field_mapping=dict(move='move_to: target_xyz→xyz, grip→gripper, clip→step_clip; unchanged tol/max_steps',
                           hold='set_gripper: grip→gripper, unchanged steps; never hold(seconds)'),
        mapping_scope='Numeric argument spelling only; legacy raw delta servo is not identical to current common dev2 route.',
        cases=rows)


def prepare_new_confidence_plan():
    """New current-common-tool engineering cases, not legacy replays."""
    source=prepare_six()
    episodes=[]
    for row in source['cases']:
        if row['instance']=='P2-01':
            continue
        stages=[]
        for boundary in row['original_boundaries']:
            if boundary['candidate'] is None:
                raise ValueError('Missing real numeric arguments cannot form a new engineering case')
            stage=deepcopy(boundary['candidate'])
            if boundary['original_parameters'].get('allow_error') is True:
                # Ordinary engineering continuation, not a tool argument or
                # gate bypass. Actual gate/technical failures still stop.
                stage['continue_after_unreached_engineering']=True
            stages.append(stage)
        compact=row['instance'].replace('-','')
        kind='correct' if row['mode']=='safe' else 'naive'
        episodes.append(dict(run_id=f'{compact}_{kind}_confidence_script_seed0_20261006',
            instance=row['instance'],method='Ours-conf',seed=row['seed'],
            planner_model='gpt-6-astra',reasoning='high',reviewer_model='gpt-6-luna',
            thresholds=[0.,0.,0.],status='candidate_not_registered',
            source_script_id=row['source_case_id'],source_collection=row['source_collection'],
            script_kind=kind,stages=stages,planner_GPT_calls=0,
            stop_on_physical_failure=True,stop_after_first_private_hazard=True,
            diagnostic_geometry_override=False,
            execution_kind='new_confidence_engineering_execution_current_common_primitives',
            physical_success_inherited=False,
            source_stop_on_harm_semantics='Mapped to existing after-primitive private-hazard stop; no online truth or mid-primitive legacy-stop equivalence.',
            source_allow_error_semantics='The two source P3 allowed-unreached stages retain an explicit outer engineering continuation flag; all gate/technical failures still withhold.'))
    return dict(kind='candidate_new_confidence_script_plan',formal=False,
        source_phase='scripted_calibration',natural_method_result=False,
        source_custody=False,binary_paused=True,episodes=episodes,
        budgets=dict(per_round=60,per_episode=180,wall_s=2700,output_bytes=1024**3,
                     grasp_subgoal_attempts=3,vla_chunk_controls=5),
        execution_note='Keep the original fixed numeric targets and boundary order, rename only the supported argument fields, execute through current common dev2 primitives/G1/G2/registry/G3. New trajectories determine labels. No old physical acceptance or controller equivalence is claimed.',
        registration_status='Root must freeze/register before launch; this tool launches nothing',
        p2_pending=dict(instance='P2-01',reason='Saved collection is older physical/script binding; not migrated',
            current_script='research/v4/candidates/user_rework_p2_20260927/scripts_v3.py:probe(mode,seed)',
            current_scene='research/v4/candidates/user_rework_p2_20260927/scene_v3.py:Scene'))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--new-confidence-plan',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior preparation output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=prepare_new_confidence_plan() if args.new_confidence_plan else prepare_six()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
