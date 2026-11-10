"""Pure own P302–305 capture conversion, no collector/Scene/model import.

The output is a NEW common-tool engineering candidate, never the old
physical acceptance. Missing inline controls are not reconstructed.
"""
from copy import deepcopy
import json
from pathlib import Path
from .s12_business_parameters import _count,_number,_xyz

OWN={
    'P3-02':'research.v4.candidates.p3_02_keyboard_coffee_20260926',
    'P3-03':'research.v4.candidates.p3_03_tablet_juice_20260926',
    'P3-04':'research.v4.candidates.p3_04_album_wine_v2_20260926',
    'P3-05':'research.v4.candidates.p3_05_book_soup_20260926',
}


def ordinary_arguments(name,args):
    expected={'xyz','gripper','step_clip','tol','max_steps','action_scale'} if name=='move_to' else {'gripper','steps'}
    if name not in ('move_to','set_gripper') or set(args)!=expected:
        raise ValueError('Only exact recorded current numeric move/hold fields')
    if name=='move_to':
        _xyz(args['xyz']);_number(args['gripper']);_count(args['max_steps'])
        if _number(args['step_clip'])<=0 or _number(args['tol'])<=0 or _number(args['action_scale'])!=.05:
            raise ValueError('Recorded original movement resource/scale invalid')
    else:
        _number(args['gripper']);_count(args['steps'])
    return deepcopy(args)


def prepare(capture,*,fixed_engineering_candidates=False):
    instance=capture.get('instance');mode=capture.get('mode')
    if capture.get('kind')!='ordinary_existing_derived_P3_movement_parameter_collection' or instance not in OWN or mode not in ('safe','naive') or capture.get('seed')!=0:
        raise ValueError('Exact own P302–305 fixed seed0 capture required')
    if (capture.get('original_probe'),capture.get('original_scene'))!=(OWN[instance]+'.qualify:probe',OWN[instance]+'.scene:Scene'):
        raise ValueError('Own original probe/Scene mismatch; no cross-instance mapping')
    if type(fixed_engineering_candidates) is not bool:
        raise ValueError('Explicit new-fixed-candidate option must be boolean')
    gaps=[];stages=[];differences=[];windows=[]
    if capture.get('actual_original_script_error') is not None:
        gaps.append('Original script ended with error; partial boundary list is not a complete candidate')
    if capture.get('recording_errors') or capture.get('truncated') is not False:
        gaps.append('Ordinary recording errored, truncated or completeness flag unavailable')
    calls=capture.get('business_calls');boundaries=capture.get('boundaries')
    if not isinstance(calls,list) or not isinstance(boundaries,list):
        raise ValueError('Actual business_calls and ordinary boundaries lists required')
    if len(calls)!=len(boundaries):gaps.append('Ordinary business/boundary row counts differ')
    previous=0;move_count=0
    for index,row in enumerate(calls):
        if row.get('instance')!=instance or row.get('round')!=1:
            raise ValueError('Actual own-instance one-round row required')
        name=row.get('name')
        if name not in ('move_to','set_gripper'):
            gaps.append(f'Unmapped business row{index}');continue
        arguments=ordinary_arguments(name,row['arguments'])
        required=('sequence_before','sequence_after','actual_controls')
        if not all(k in row for k in required):
            gaps.append(f'Incomplete actual interval at row{index}');continue
        before=_count(row['sequence_before']);after=_count(row['sequence_after'])
        count=_count(row['actual_controls'])
        if after<before or count!=after-before:
            raise ValueError('Actual original control range is inconsistent')
        if before!=previous:
            gaps.append(f'Unrecorded or overlapping controls before row{index}: {previous}→{before}')
        previous=after
        if row.get('fixed_numeric_candidate') is not True:
            gaps.append(f'Original row{index} is not a recorded fixed numeric candidate')
        if row.get('exception_observed'):
            gaps.append(f'Original business exception at row{index}')
        if index>=len(boundaries) or boundaries[index].get('tool')!=name or boundaries[index].get('arguments')!=arguments:
            gaps.append(f'Ordinary boundary arguments disagree at row{index}')
        stage=dict(tool=name,arguments=arguments,source_stage=row['source_stage'])
        if name=='move_to':
            move_count+=1
            for flag in ('original_stop_on_harm','original_allow_unreached_engineering'):
                if type(row.get(flag)) is not bool:raise ValueError('Original move stop/scope flags unavailable')
            if row['original_allow_unreached_engineering']:
                stage['continue_after_unreached_engineering']=True
            if row['original_stop_on_harm']:
                if not fixed_engineering_candidates:
                    gaps.append(f'Private mid-control footprint stop at row{index} lacks explicit new candidate selection')
                differences.append(dict(source_stage=row['source_stage'],
                    change='Original private held/expanded-footprint early stop is not installed in the common tool. Existing engineering first-hazard stopping remains.',
                    original_control_equivalence=False,new_actual_labels_required=True))
        stages.append(stage)
        windows.append(dict(source_stage=row['source_stage'],sequence_before=before,
                            sequence_after=after,actual_controls=count))
    expected=15 if mode=='safe' else 6
    if move_count!=expected or capture.get('complete_original_move_list') is not True:
        gaps.append(f'Actual original move list incomplete: {move_count}/{expected}')
    receipt=capture.get('actual_scene_close_receipt')
    if not isinstance(receipt,dict) or receipt.get('called') is not True or 'executed_sequence' not in receipt:
        gaps.append('Actual original Scene close/control endpoint unavailable')
    else:
        total=_count(receipt['executed_sequence'])
        if previous!=total:gaps.append(f'Unrecorded trailing controls: {previous}→{total}')
        if receipt.get('secondary_error'):gaps.append('Original Scene cleanup failed')
    return dict(instance=instance,seed=0,script_kind='correct' if mode=='safe' else 'naive',
        stages=stages,script_ready=bool(stages and not gaps),gaps=gaps,differences=differences,
        source_phase='scripted_calibration',formal=False,natural_method_result=False,
        source=dict(original_probe=capture['original_probe'],original_scene=capture['original_scene'],
                    captured_original_intervals=windows),
        original_acceptance_inherited=False,original_control_equivalence=False,
        new_execution_determines_labels=True,original_label_not_copied=True,
        terminal=dict(status='done',scope='New engineering terminal; no source label is supplied'),
        source_reach_scope='Only original per-move allow_error survives. Original per-move reached/error_m is not recorded here; current common-tool execution must supply its own result.',
        source_rest_scope='Recorded finite hold list is fixed for the new candidate; private source placement-complete early termination is not rerun online.',
        original_metadata_not_in_gate_input=True,no_new_empty_reset_fact=True,
        registration='Root registers only actual-ready candidates and derives labels from fresh executed windows. No P301 coordinates or asset truth is supplied.')


def load_prepare(path,*,fixed_engineering_candidates=False):
    return prepare(json.loads(Path(path).read_text()),fixed_engineering_candidates=fixed_engineering_candidates)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--new-fixed-engineering-candidate',action='store_true')
    args=parser.parse_args()
    result=load_prepare(args.capture,fixed_engineering_candidates=args.new_fixed_engineering_candidate)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
