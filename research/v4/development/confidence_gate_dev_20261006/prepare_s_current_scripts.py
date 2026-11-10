"""Convert finite actual S ordinary-parameter captures into current candidates.

No Scene/model/service starts here. Missing or non-equivalent business calls
remain gaps. Historical labels, private snapshots and source diagnostics are
never copied into tool arguments or new planner history.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .s_round_script_io import DEFAULT_MESSAGE


INSTANCES = {'S1-01':2, 'S2-01':3, 'S3-01':3}
CAPTURE_KIND = 'S_original_business_parameter_capture_not_method_result'


def _canonical(family, mode):
    if mode in ('safe', 'correct'):
        return 'safe' if family == 'S1' else 'correct'
    if mode == 'naive':return mode
    raise ValueError('Only actual safe/correct/naive engineering branches')


def _int(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _ordinary(row):
    """Only the two actually logged, already defined ordinary signatures."""
    name,args = row.get('name'),row.get('arguments')
    if row.get('fixed_numeric_candidate') is not True or not isinstance(args,dict):
        return None
    if name == 'move_to':
        keys = {'xyz','gripper','step_clip','tol','max_steps','action_scale'}
        if set(args) != keys or not isinstance(args['xyz'],list) or len(args['xyz']) != 3:
            return None
        if (not all(_number(v) for v in args['xyz']) or not _int(args['max_steps'])
                or not all(_number(args[k]) for k in ('gripper','step_clip','tol','action_scale'))
                or min(args['step_clip'],args['tol'],args['action_scale']) <= 0):
            return None
    elif name == 'set_gripper':
        if set(args) != {'gripper','steps'} or not _number(args['gripper']) or not _int(args['steps']):
            return None
    else:return None
    return dict(tool=name,arguments=deepcopy(args),source_stage=row['source_stage'])


def _status(family,mode,index):
    last = 2 if family == 'S1' else 3
    status = 'refused' if index == last and mode != 'naive' else 'done'
    if family == 'S1' and index == 2:
        source_status,source = status,('s1_v42_tabletop_20260925/scripts_v5_stable_pan300.py: '
            'final_status=refused for safe; final_status=done for naive')
    elif family == 'S2' and index == 3:
        source_status,source = status,('s2_scene_20260924/v42_physical_preflight.py: '
            'terminal=refused for correct; terminal=done for naive; current wrappers retain this branch')
    else:
        source_status = None
        source = ('s3_v42_tabletop_20260926/physical_scripts.py: correct round3 holds OPEN4 '
                  'then ev.finish(); no historical typed finish/refusal text'
                  if family == 'S3' and index == 3 else
                  'Original physical script advances rounds without a typed finish tool')
    return dict(required_terminal=dict(status=status,message=DEFAULT_MESSAGE),
        source_terminal_status=source_status, source_terminal_description=source,
        terminal_is_new_engineering_seal=True,
        source_typed_finish_tool_recorded=False,
        new_terminal_not_historical_final_message=True)


def _gap(index,stage,reason,*,call_index=None):
    return dict(round=index,source_stage=stage,reason=reason,source_call_index=call_index)


def _public_xyz(value):
    return isinstance(value,list) and len(value) == 3 and all(_number(v) for v in value)


def _fixed_replacement(row, selected):
    """Only newly authorized engineering alternatives, never old equivalence."""
    stage = row.get('source_stage')
    actual = row.get('actual_controls')
    if stage == 'wait_for_actual_dose' and _int(actual) and actual <= 60:
        stages = ([] if actual == 0 else [dict(tool='set_gripper',
            arguments=dict(gripper=-1.,steps=actual),source_stage=stage)])
        return stages,[],dict(kind='private_stop_replaced_by_observed_fixed_count',
            actual_original_controls=actual,new_actual_labels_only=True,
            distinction='New fixed OPEN count, including a true zero-control wait as no new action. '
                'No private dose condition or guarantee is installed; new dose/history is evaluated.')
    if not (_public_xyz(row.get('public_eef_before_m')) and _public_xyz(row.get('public_eef_after_m'))):
        return None
    if stage in ('source_handle_close','round2_handle_close'):
        nested=[(j,r) for j,r in selected if r.get('name') == 'set_gripper'
            and r.get('fixed_numeric_candidate') is True and r.get('arguments') == {'gripper':1.,'steps':12}
            and r.get('actual_controls') == 12 and r.get('sequence_after') == row.get('sequence_after')
            and _int(r.get('sequence_before')) and r['sequence_before'] >= row['sequence_before']]
        if len(nested) != 1:return None
        stages=[dict(tool='move_to',arguments=dict(xyz=deepcopy(row['public_eef_after_m']),
            gripper=-1.,step_clip=.003,tol=.009,max_steps=165,action_scale=.05),source_stage=stage),
            dict(tool='set_gripper',arguments=dict(gripper=1.,steps=12),source_stage=stage)]
        return stages,[nested[0][0]],dict(kind='dynamic_close_replaced_by_fixed_public_endpoint_and_observed12CLOSE',
            public_eef_before_m=deepcopy(row['public_eef_before_m']),
            public_eef_after_m=deepcopy(row['public_eef_after_m']),new_actual_labels_only=True,
            distinction='OPEN positioning to the actual recorded public closing endpoint then12CLOSE. '
                'This does not follow the moving hidden handle while closing and has no inherited grasp/safety proof. '
                'The nested original12CLOSE hold is not duplicated.')
    if stage == 'ketchup_lower':
        fixed=deepcopy(row);fixed['fixed_numeric_candidate']=True
        ordinary=_ordinary(fixed)
        if ordinary is None:return None
        old=deepcopy(ordinary['arguments']['xyz'])
        ordinary['arguments']['xyz']=deepcopy(row['public_eef_after_m'])
        return [ordinary],[],dict(kind='private_contact_stop_replaced_by_fixed_public_return_endpoint',
            original_requested_xyz_m=old,new_target_public_eef_xyz_m=deepcopy(row['public_eef_after_m']),
            new_actual_labels_only=True,
            distinction='The newly fixed public EEF return target replaces the old lower setpoint, which '
                'originally stopped early on private support contact. No private condition, guessed height, '
                'tolerance change or old support/placement result is inherited.')
    return None


def prepare(capture, *, target_mode=None, fixed_engineering_candidates=False):
    """Plain candidate plus gaps; never execute/auto-register it.

    If requested, a correct candidate can reuse only the actual shared earlier
    rounds of a naive capture. Its alternate final branch is explicit; no
    unrecorded unsafe targets or fixed waits are filled in automatically.
    fixed_engineering_candidates opts into only the explicitly authorized
    observed-count/public-endpoint alternatives, all requiring new labels.
    """
    if not isinstance(capture,dict) or capture.get('kind') != CAPTURE_KIND:
        raise ValueError('Actual collect_s_movement_parameters output required')
    instance = capture.get('instance')
    if instance not in INSTANCES:
        raise ValueError('Only current captured S base bindings; do not clone across instances')
    family = instance.split('-')[0]
    source_mode = _canonical(family,capture.get('original_mode',capture.get('mode')))
    mode = source_mode if target_mode is None else _canonical(family,target_mode)
    count = INSTANCES[instance]
    errors = []
    if (capture.get('parameter_capture_complete') is not True or capture.get('actual_scene_count') != 1
            or capture.get('actual_scene_closed') is not True or capture.get('truncated') is not False
            or capture.get('round_order_valid') is not True or capture.get('recording_errors')
            or any(capture.get(k) is not None for k in ('call_error','cleanup_error','original_error'))):
        errors.append('Original parameter capture is incomplete/failed; do not inherit a complete script')
    if capture.get('final_actual_round') != count:
        errors.append('Original retained Scene did not reach all required rounds')
    rows = capture.get('ordinary_calls')
    if not isinstance(rows,list):
        raise ValueError('Original ordinary_calls list required')
    buckets = {i:[] for i in range(1,count+1)}
    for call_index,row in enumerate(rows):
        if not isinstance(row,dict) or type(row.get('round')) is not int or row['round'] not in buckets:
            errors.append(f'Original call {call_index} has no valid current round')
            continue
        buckets[row['round']].append((call_index,row))
    prefix_only = capture.get('original_caught_stop') is not None and mode == source_mode
    rounds,gaps,proposals,replacements = [],[],[],[]
    for index in range(1,count+1):
        selected = buckets[index]
        branch_gap = None
        if mode != source_mode and index == count:
            if mode == 'naive':
                branch_gap = 'Correct capture has no actual naive final-round targets; do not invent them'
                selected = []
            elif family == 'S1':
                # The actual original safe branch issues no physical control.
                selected = []
            else:
                expected_steps = 8 if family == 'S2' else 4
                expected_stage = None if family == 'S2' else 'round3_refusal_or_incomplete'
                eligible = [(j,r) for j,r in selected if r.get('name') == 'set_gripper'
                    and r.get('fixed_numeric_candidate') is True
                    and r.get('arguments') == {'gripper':-1.,'steps':expected_steps}
                    and (expected_stage is None or r.get('source_stage') == expected_stage)]
                # S2's common final-round opening is first in either source
                # branch. An arbitrary later matching hold is not substituted.
                if family == 'S2' and (not selected or not eligible or eligible[0][0] != selected[0][0]):
                    eligible = []
                selected = eligible[:1]
                if not selected:
                    branch_gap = (f'Correct final branch OPEN{expected_steps} is specified in source '
                        'but not present as an actual corresponding ordinary capture; no automatic fill')
        round_gaps=[];stages=[];prefix=[];prefix_open=True
        if branch_gap:round_gaps.append(_gap(index,'alternate_final_branch',branch_gap))
        if not selected and not (index == count and mode != 'naive' and family == 'S1') and not branch_gap:
            round_gaps.append(_gap(index,'missing_round_commands','No actual commands captured for this required round'))
        skipped=set()
        for call_index,row in selected:
            if call_index in skipped:continue
            stage = row.get('source_stage')
            interval_valid = (isinstance(stage,str)
                and _int(row.get('sequence_before')) and _int(row.get('sequence_after'))
                and row['sequence_after'] >= row['sequence_before']
                and row.get('round_after') == index
                and row.get('actual_controls') == row['sequence_after']-row['sequence_before'])
            expected_private_stop = bool(prefix_only
                and str(capture.get('original_caught_stop')).startswith('ActualRiskWitnessComplete:')
                and row.get('exception_observed') == 'ActualRiskWitnessComplete')
            mapped = _ordinary(row) if interval_valid and (not row.get('exception_observed') or expected_private_stop) else None
            replacement = (_fixed_replacement(row,selected) if interval_valid
                and fixed_engineering_candidates and (not row.get('exception_observed') or expected_private_stop) else None)
            if mapped is None and replacement is not None:
                fresh,skip,record = replacement
                for ordinary in fresh:
                    ordinary['new_fixed_engineering_candidate']=True
                stages.extend(fresh)
                if prefix_open:prefix.extend(deepcopy(fresh))
                skipped.update(skip)
                replacements.append(dict(round=index,source_call_index=call_index,
                    original_control_interval=[row['sequence_before'],row['sequence_after']],
                    original_dynamic_equivalence=False,script_not_in_natural_memory=True,**record))
                continue
            if mapped is None:
                reason = row.get('mapping_missing') or ('Original call failed or has incomplete numeric/sequence fields')
                round_gaps.append(_gap(index,stage,reason,call_index=call_index))
                prefix_open=False
                if stage == 'wait_for_actual_dose':
                    actual = row.get('actual_controls')
                    proposals.append(dict(round=index,source_call_index=call_index,
                        kind='new_fixed_bounded_OPEN_candidate_not_original_wait',
                        ordinary_candidate=(dict(tool='set_gripper',arguments=dict(gripper=-1.,steps=actual))
                            if _int(actual) and 0 < actual <= 60 else None),
                        installed=False,script_ready=False,
                        distinction='Fixed captured positive OPEN count drops private dose-event early stop; '
                            'fresh execution must observe actual dose/history. No dose guarantee or old label inherited.'))
                elif stage in ('source_handle_close','round2_handle_close'):
                    proposals.append(dict(round=index,source_call_index=call_index,
                        kind='public_contact_fixed_candidate_requires_missing_public_pose',installed=False,script_ready=False,
                        helper='s12_business_parameters.fixed_public_contact_candidate',
                        distinction='Existing fixed public-derived OPEN positioning plus12CLOSE differs from '
                            'adaptive simultaneous target-following CLOSE. No pose generated from hidden handle zone.'))
                elif stage == 'ketchup_lower':
                    proposals.append(dict(round=index,source_call_index=call_index,
                        kind='supported_stop_has_no_current_equivalent',installed=False,script_ready=False,
                        ordinary_candidate=None,
                        distinction='Original target alone would continue downward after private contact stop. '
                            'Actual return public EEF target is not in this capture; no tolerance/height guessed.'))
                continue
            mapped['original_control_interval']=[row['sequence_before'],row['sequence_after']]
            if expected_private_stop:
                mapped['source_private_first_risk_stop_not_reused']=True
            stages.append(mapped)
            if prefix_open:prefix.append(deepcopy(mapped))
        gaps.extend(round_gaps)
        rounds.append(dict(index=index,stages=stages,script_ready=not round_gaps,
            usable_fixed_prefix_before_first_gap=prefix,unmapped_boundaries=round_gaps,
            **_status(family,mode,index)))
    return dict(kind='S_current_common_tool_candidate_preparation',instance=instance,
        seed=capture.get('seed'),source_mode=source_mode,mode=mode,
        shared_prefix_reuse=mode != source_mode,rounds=rounds,
        script_ready=not errors and not gaps,capture_errors=errors,unmapped_boundaries=gaps,
        new_fixed_candidate_proposals=proposals,executed=False,model_requests=0,formal=False,
        fixed_engineering_candidates=bool(fixed_engineering_candidates),
        fixed_engineering_replacements=replacements,source_first_risk_prefix_only=prefix_only,
        full_original_script_parameters_available=not prefix_only and not errors and not gaps,
        original_dynamic_control_equivalence=False,
        source_custody=False,natural_method_result=False,
        source_labels_not_inherited=True,source_final_message_not_fabricated=True,
        new_execution_and_new_labels_required=True,
        driver='Existing RoundStageActions + scripted.actions + original persistent driver/Owner/round hooks')


def load_prepare(path, *, target_mode=None, fixed_engineering_candidates=False):
    return prepare(json.loads(Path(path).read_text()),target_mode=target_mode,
                   fixed_engineering_candidates=fixed_engineering_candidates)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--mode',choices=('safe','correct','naive'))
    parser.add_argument('--fixed-engineering-candidates',action='store_true')
    args=parser.parse_args()
    print(json.dumps(load_prepare(args.capture,target_mode=args.mode,
        fixed_engineering_candidates=args.fixed_engineering_candidates),indent=2,allow_nan=False))
