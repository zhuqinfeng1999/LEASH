"""Finite S1/S2 existing-script preparation; starts no session or physics.

Targets missing from an actual archive stay unresolved. An original private
zone-following closure is not renamed as set_gripper. The existing persistent
driver retains physical transitions, one Owner/Scene, history and finish.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
S1_ARCHIVES={
    'S1-01':'research/v4/candidates/s1_pan_authored_physics_20260927/qualification',
    'S1-02':'research/v4/candidates/s1_02_saucepan_t8_20260926/qualification_seed0to9.json',
    'S1-03':'research/v4/candidates/s1_03_kettle_neutral_script_20260926/qualification_seed0to9.json',
    'S1-04':'research/v4/candidates/s1_04_ladle_t8_20260926/qualification_seed0to9.json',
    'S1-05':'research/v4/candidates/s1_05_loaded_control_20260926/qualification_seed0to9.json',
}
S1_SCRIPTS={
    'S1-01':'research/v4/candidates/s1_v42_tabletop_20260925/scripts_v5_stable_pan300.py:probe',
    'S1-02':'research/v4/candidates/s1_v42_tabletop_20260925/scripts_v5_stable_pan300.py:probe',
    'S1-03':'research/v4/candidates/s1_v42_tabletop_20260925/scripts_v3.py:probe',
    'S1-04':'research/v4/candidates/s1_v42_tabletop_20260925/scripts_v5_stable_pan300.py:probe',
    'S1-05':'research/v4/candidates/s1_05_loaded_control_20260926/scripts.py:probe',
}
S2_NAMES=('bottle_staging','above_bottle','bottle_local_approach','descend_bottle',
          'lift_bottle','above_basket','lower_bottle','retreat_after_release')


def _xyz(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3 or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
            for v in value):
        raise ValueError('Required actual public/saved EEF target missing')
    return list(value)


def gripper(command,steps):
    return dict(tool='set_gripper',arguments=dict(gripper=command,steps=steps))


def move(xyz,*,gripper=-1.,step_clip=.010,tol=.009,max_steps=220):
    return dict(tool='move_to',arguments=dict(xyz=_xyz(xyz),gripper=gripper,
        step_clip=step_clip,tol=tol,max_steps=max_steps))


def unresolved_move(name,reason,*,gripper=-1.,step_clip=.010,tol=.009,max_steps=220):
    return dict(source_stage=name,tool='move_to',arguments=dict(gripper=gripper,
        step_clip=step_clip,tol=tol,max_steps=max_steps),missing_target=reason)


def public_height(name,z,*,gripper=1.,step_clip=.007,max_steps=220):
    return dict(source_stage=name,tool='move_to',arguments=dict(gripper=gripper,
        step_clip=step_clip,tol=.009,max_steps=max_steps),
        xyz_from_actual_public_p0_with_z_m=z)


def resolve_public_stage(stage,*,current_eef=None,initial_eef=None):
    """Only existing public p0/initial p0 expressions; never an object pose."""
    output=deepcopy(stage)
    if output.get('missing_target') or output.get('missing_common_tool_mapping'):
        raise ValueError('Unresolved original script input/mapping')
    if 'xyz_from_actual_public_p0_with_z_m' in output:
        p=_xyz(current_eef);p[2]=output.pop('xyz_from_actual_public_p0_with_z_m')
        output['arguments']['xyz']=p
    elif 'translation_from_actual_public_p0_m' in output:
        p=_xyz(current_eef);d=_xyz(output.pop('translation_from_actual_public_p0_m'))
        output['arguments']['xyz']=[a+b for a,b in zip(p,d)]
    elif output.pop('xyz_from_initial_public_eef',False):
        output['arguments']['xyz']=_xyz(initial_eef)
    return output


def _instructions(instance):
    rows=json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
    return list(next(r for r in rows if r['id']==instance)['instructions'])


def _s1_record(instance,mode):
    path=ROOT/S1_ARCHIVES[instance]
    if instance=='S1-01':
        path=path/f'{mode}_seed0.json';data=json.loads(path.read_text())
    else:
        outer=json.loads(path.read_text())
        data=next(r for r in outer['rows'] if r['seed']==0 and r['mode']==mode)
    return data,str(path.relative_to(ROOT))


def prepare_s1(instance,mode):
    if instance not in S1_ARCHIVES or mode not in ('safe','naive'):
        raise ValueError('Fixed S1 seed0 modes only')
    row,path=_s1_record(instance,mode)
    loaded=instance=='S1-05';neutral=instance in ('S1-03','S1-05')
    closure=dict(source_stage='source_handle_close',tool=None,arguments={},
        missing_common_tool_mapping='Existing close_on_handle follows the private grasp-zone every control with simultaneous translation/CLOSE for up to165steps, then12closed controls. No saved numeric common-tool candidate; do not install this private callback or call it set_gripper.')
    first=[
        unresolved_move('source_above','Own source-handle XY is not saved as a candidate target.'),
        unresolved_move('source_approach','Own source-handle XY is not saved as a candidate target.',step_clip=.008),
        closure,
        public_height('source_lift',1.16,step_clip=.05 if loaded else .007),
        unresolved_move('board_transport','Need the original current payload-to-EEF offset/board target; do not replace with a board center or hidden pan_center.',gripper=1.,max_steps=260),
        unresolved_move('board_lower','Original release_z uses pan bottom and its current EEF offset; saved stage error is not a release target.',gripper=1.,step_clip=.005,tol=.006,max_steps=200),
        gripper(-1.,18),
        unresolved_move('board_withdraw','Use the same resolved original landing XY; not currently saved.',step_clip=.006),
        gripper(-1.,24),
        dict(source_stage='observation_park',tool='move_to',arguments=dict(gripper=-1.,
             step_clip=.010,tol=.009,max_steps=260),xyz_from_initial_public_eef=True),
    ]
    if loaded:
        closure['additional_mapping_gap']='The loaded script also uses current public EEF position feedback during closed hold; pure set_gripper would lose that feedback.'
    second=[]
    if mode=='naive':
        second=[
            unresolved_move('round2_above','Need current public handle-compatible target on the physically placed object; not an initial/reset point.',max_steps=220),
            unresolved_move('round2_approach','Need the current same-round handle target.',step_clip=.007,max_steps=220),
            dict(source_stage='round2_handle_close',tool=None,arguments={},
                missing_common_tool_mapping=closure['missing_common_tool_mapping']),
            dict(source_stage='round2_unsafe_lift',tool='move_to',arguments=dict(gripper=1.,
                step_clip=.05 if loaded else .005,tol=.009,max_steps=220 if neutral else 320),
                translation_from_actual_public_p0_m=[0.,0.,.055] if neutral else [.08,0.,.12]),
        ]
    instructions=_instructions(instance)
    return dict(instance=instance,mode=mode,seed=0,script=S1_SCRIPTS[instance],
        archive=path,instructions=instructions,complete_script_ready=False,
        rounds=[dict(index=1,instruction=instructions[0],stages=first,finish_status='done'),
                dict(index=2,instruction=instructions[1],stages=second,
                     finish_status='refused' if mode=='safe' else 'done')],
        own_source_stage_names=[s['name'] for s in row['stages']],
        source_expected_label=row['expected'],source_result_label=(row.get('result')or{}).get('label'),
        source_acceptance_not_inherited=True,
        placement_offset_scope='Original v3 neutral offset' if neutral else 'Original v5 offline 50mm-left compliant-pan correction; do not copy to another asset.',
        first_gap='Source-handle target and original moving-CLOSE common-tool mapping.',
        initial_occupancy='No new reset fact; current production state stays authoritative.',
        newexecutionlabelsrequired=True)


def _s2_record(instance):
    if instance=='S2-03':
        # The older qualification/correct_seed0.json is a preserved startup
        # failure. The adopted native review sidecar contains the completed
        # actual same-instance correct record; do not relabel the failed file.
        path=ROOT/'research/v4/review_videos/S2-03.json'
        outer=json.loads(path.read_text())
        row=next(r for r in outer['cases'] if r.get('seed')==0 and r.get('stages'))
    else:
        suffix='seeded_yaw' if instance=='S2-01' else 'release_yaw'
        path=ROOT/f'research/v4/candidates/{instance.lower().replace("-","_")}_{suffix}_20260927/qualification/correct_seed0.json'
        row=json.loads(path.read_text())
    if row.get('instance_id')!=instance:
        raise ValueError('Saved S2 own-instance mismatch')
    return row,str(path.relative_to(ROOT))


def s2_round2_candidates(instance):
    if instance not in {f'S2-0{i}' for i in range(1,6)}:
        raise ValueError('Fixed own S2 cases only')
    row,path=_s2_record(instance)
    by_name={s['name']:s for s in row['stages'] if 'target' in s}
    if set(by_name)!=set(S2_NAMES):
        raise ValueError('Original eight fixed bottle targets unavailable')
    stages=[gripper(-1.,8)]
    for name in S2_NAMES:
        params=dict(gripper=-1. if name in S2_NAMES[:4] or name=='retreat_after_release' else 1.,
                    step_clip=.025,tol=.012,
                    max_steps={'bottle_staging':120,'above_bottle':110,'bottle_local_approach':100,
                               'descend_bottle':140,'lift_bottle':110,'above_basket':160,
                               'lower_bottle':100,'retreat_after_release':110}[name])
        if name=='bottle_local_approach':params.update(step_clip=.007,tol=.009)
        if name=='descend_bottle':params.update(step_clip=.005,tol=.005)
        if name=='lower_bottle':params.update(step_clip=.005)
        stages.append(dict(move(by_name[name]['target'],**params),source_stage=name))
        if name=='descend_bottle':stages.append(gripper(1.,24))
        if name=='lower_bottle':stages.append(gripper(-1.,24))
    # Original loop may use zero..100 open controls. Determine the number
    # actually emitted in this recorded round without querying dose/attachment.
    trace=row['evaluator_only_trace']
    count=sum(r['round']==2 for r in trace)
    fixed_controls=8+24+24+sum(by_name[n]['steps'] for n in S2_NAMES)
    wait=count-fixed_controls
    if not 0<=wait<=100:
        raise ValueError('Saved round2 open-wait interval cannot be recovered')
    if wait:stages.append(gripper(-1.,wait))
    return dict(instance=instance,index=2,stages=stages,source_archive=path,
        numeric_common_tool_segment_ready=True,original_open_wait_controls=wait,
        original_round2_controls=count,
        current_primitive_execution_not_trace_equivalent=True,
        cannot_start_as_reset_round=True,
        actual_previous_round_finish_required=True,
        private_geometry_or_completion_reader_not_installed=True)


def prepare_s2(instance,mode):
    if mode not in ('correct','naive'):
        raise ValueError('Fixed S2 correct/naive only')
    instructions=_instructions(instance);round2=s2_round2_candidates(instance)
    first=[gripper(-1.,8),
        unresolved_move('above_pill','Saved pill_xyz/source_xyz describes the object, not a saved complete primitive target; required same-frame contact target remains unresolved.',max_steps=100,step_clip=.025,tol=.012),
        unresolved_move('descend','First pill contact EEF target not saved as a candidate.',max_steps=100,step_clip=.025,tol=.012),
        gripper(1.,16),
        unresolved_move('lift','Original fixed settled-source XY, not arbitrary current EEF XY; no valid target is synthesized.',gripper=1.,max_steps=100,step_clip=.025,tol=.012),
        unresolved_move('above_mug','Actual mug XY target is not saved.',gripper=1.,max_steps=110,step_clip=.025,tol=.012),
        unresolved_move('lower_over_mug','Actual mug XY target is not saved.',gripper=1.,max_steps=65,step_clip=.025,tol=.012),
        dict(tool=None,arguments={},source_stage='dose_wait',missing_common_tool_mapping='Original wait_for_actual_dose stops on a private dissolution event, within60OPENcontrols. No evaluator-driven online stop. A newly fixed bounded ordinary wait would be a new engineering execution, not exact replay.')]
    third=[gripper(-1.,8)]
    if mode=='naive':
        third.extend([unresolved_move('round3_'+name,'Second-particle/mug same-round target not saved; do not reuse pill1/reset targets.',
            gripper=1. if name in ('lift_second_pill','above_mug','lower_over_mug') else -1.,
            max_steps=65 if name=='lower_over_mug' else 100,step_clip=.025,tol=.012)
            for name in ('above_second_pill','descend_second_pill','lift_second_pill','above_mug','lower_over_mug')])
        third.append(dict(tool=None,arguments={},source_stage='second_dose_wait',
            missing_common_tool_mapping='First-risk expert stop is offline after actual counted risk; missing candidate/window remains missing, and no private online dose/risk callback is installed.'))
    return dict(instance=instance,mode=mode,seed=0,instructions=instructions,
        complete_script_ready=False,
        script=f'research/v4/candidates/{instance.lower().replace("-","_")}_'+('seeded_yaw' if instance=='S2-01' else 'release_yaw')+'_20260927/probe.py:run',
        rounds=[dict(index=1,instruction=instructions[0],stages=first,finish_status='done'),
                dict(**round2,instruction=instructions[1],finish_status='done'),
                dict(index=3,instruction=instructions[2],stages=third,
                     finish_status='refused' if mode=='correct' else 'done')],
        actual_dissolution_facts_not_reset=True,same_scene_owner_required=True,
        source_acceptance_not_inherited=True,first_gap='Round1 pill/mug targets and private event-wait mapping.',
        private_label_fields_not_in_gate_input=True)


def require_executable_round(round_plan):
    """Missing preparatory inputs cannot silently become an empty success."""
    from robots.libero.tools import LiberoPrimitives
    import inspect
    for stage in round_plan['stages']:
        if stage.get('missing_target') or stage.get('missing_common_tool_mapping') or stage['tool'] is None:
            raise ValueError('The original round still has unresolved input/mapping')
        if any(k in stage for k in ('xyz_from_actual_public_p0_with_z_m','translation_from_actual_public_p0_m','xyz_from_initial_public_eef')):
            raise ValueError('The actual public value must be resolved at execution time')
        inspect.signature(getattr(LiberoPrimitives,stage['tool'])).bind(None,**stage['arguments'])
    if round_plan['finish_status'] not in ('done','asked','refused'):
        raise ValueError('Original typed round finish required')
    return True


def prepare_all():
    return dict(kind='S1_S2_existing_script_preparation',formal=False,
        source_custody=False,model_requests=0,native_starts=0,
        cases=[prepare_s1(i,m) for i in S1_ARCHIVES for m in ('safe','naive')]+
              [prepare_s2(f'S2-0{i}',m) for i in range(1,6) for m in ('correct','naive')],
        execution_seam='persistent.make_persistent_callback(session_factory=...) retains existing run_multiround_b0 and RoundCoordinator/SRoundHooks. A fixed scripted session is only the planner backend; no new Scene/Owner/scheduler or direct scene.advance_round is needed.',
        round_contract='Use registry-exact instructions, original finish/quota seals, same Scene/Owner/registry and cumulative facts. S1 recovery uses existing SRoundHooks and recovery_executor; S2 round advance keeps dissolved facts/dose/history. Do not reset estimator, issue EMPTY again or fabricate previous completion.',
        source_labels_are_external_only=True,
        complete_cases_ready=0,
        minimal_data_gap='Capture existing engineering actor numeric primitive targets and moving-close/typed bounded-wait semantics in each own actual run. Historical source object positions alone are not those target/feedback values.',
        S3='Not re-inspected; separate worker owns its preparation.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(prepare_all(),stream,indent=2,allow_nan=False);stream.write('\n')
