"""Convert actual own P1 business captures into unaccepted fixed candidates."""
from copy import deepcopy
from .s12_business_parameters import _count,_number,_xyz


def prepare(capture,*,fixed_engineering_candidates=False,p105_fixed40_rest=False):
    instance=capture.get('instance')
    if instance not in ('P1-04','P1-05','P1-06') or capture.get('mode') not in ('safe','naive'):
        raise ValueError('Exact own capture instance/mode required')
    stages=[];gaps=[];differences=[]
    if not capture.get('parameter_capture_complete'):
        gaps.append('Original finite capture failed or incomplete; no complete script registered')
    previous=None
    for row in capture.get('ordinary_calls',[]):
        if row.get('instance')!=instance:raise ValueError('Cross-instance business row')
        before=_count(row['sequence_before']);after=_count(row['sequence_after'])
        if after<before or row['actual_controls']!=after-before:raise ValueError('Actual sequence range invalid')
        if previous is not None and before!=previous:
            gaps.append(f'Unrecorded controls before {row["source_stage"]}')
        previous=after
        if row.get('exception_observed'):
            gaps.append(f'Original exception at {row["source_stage"]}');continue
        name=row['name'];arguments=deepcopy(row['arguments'])
        if name is None:
            public=row.get('public_position_after')
            if not fixed_engineering_candidates or public is None or public.get('sequence')!=after:
                gaps.append(f'{row["source_stage"]}: private dynamic control not mapped');continue
            if not row.get('original_private_center_payload'):
                gaps.append(f'{row["source_stage"]}: unknown unmapped primitive');continue
            arguments=deepcopy(row['original_numeric_arguments'])
            arguments['xyz']=_xyz(public['eef_pos']);name='move_to'
            differences.append(dict(stage=row['source_stage'],
                change='private per-control centering replaced by actual recorded public EEF endpoint',
                original_control_equivalence=False,new_actual_labels_required=True))
        if name not in ('move_to','set_gripper'):raise ValueError('Unsupported original business tool')
        if name=='move_to':
            _xyz(arguments['xyz']);_number(arguments['gripper']);_number(arguments['step_clip'])
            _number(arguments['tol']);_count(arguments['max_steps'])
            if row.get('original_stop_on_harm'):
                if not fixed_engineering_candidates:
                    gaps.append(f'{row["source_stage"]}: original private early-stop not mapped')
                differences.append(dict(stage=row['source_stage'],
                    change='Private per-control hazard stop replaced by existing engineering runner first-hazard stopping',
                    original_control_equivalence=False,new_actual_labels_required=True))
        else:
            _number(arguments['gripper']);_count(arguments['steps'])
        stage=dict(tool=name,arguments=arguments,source_stage=row['source_stage'])
        if row.get('original_lower_support_unreached_exception'):
            stage['continue_after_unreached_engineering']=True
        stages.append(stage)
    if instance=='P1-05' and capture['mode']=='safe':
        if p105_fixed40_rest:
            stages.append(dict(tool='set_gripper',arguments=dict(gripper=-1.,steps=40),
                source_stage='new_fixed40_open_rest_candidate'))
            differences.append(dict(stage='new_fixed40_open_rest_candidate',
                change='Historical close-side post-rest witness expressed as new ordinary fixed40 OPEN suffix',
                original_control_equivalence=False,new_actual_labels_required=True))
        else:
            gaps.append('Historical P105 later40OPEN rest witness omitted; opt-in new fixed suffix not selected')
    if previous is not None and capture.get('actual_control_count')!=previous:
        gaps.append('Actual original controls not fully represented by ordinary boundaries')
    return dict(instance=instance,seed=capture['seed'],script_kind='correct' if capture['mode']=='safe' else 'naive',
        stages=stages,script_ready=bool(stages and not gaps),gaps=gaps,differences=differences,
        source_phase='scripted_calibration',formal=False,natural_method_result=False,
        original_acceptance_inherited=False,new_execution_determines_labels=True,
        no_new_empty_reset_fact=True,no_private_metadata_in_gate_input=True,
        registration='Preparation only; Root must register/execute and retain fresh actual labels.')
