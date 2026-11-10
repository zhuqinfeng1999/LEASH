"""Pure own R201/R401 capture conversion; no old label/control inheritance."""
from copy import deepcopy
import math

from .collect_r24_movement_parameters import OWN
from .convert_p2_captured_candidates import _count, _ordinary_arguments
from .s12_business_parameters import _xyz


def prepare(capture, *, fixed_engineering_candidates=False):
    if not isinstance(capture, dict) or capture.get('instance') not in OWN:
        raise ValueError('Own actual R2-01/R4-01 capture required')
    instance = capture['instance']
    if capture.get('case') != 'naive' or type(capture.get('seed')) is not int or capture['seed'] != 0:
        raise ValueError('Only fixed naive seed0; correct refusal has no G2 actions')
    expected = OWN[instance]['module'] + ':' + OWN[instance]['function']
    if capture.get('original_probe') != expected:
        raise ValueError('Capture belongs to another own business entry')
    missing = []
    if capture.get('complete_numeric_boundaries') is not True:
        missing.append('Original numeric boundaries incomplete')
    if capture.get('actual_original_script_error') is not None:
        missing.append('Original script/capture failure retained')
    if capture.get('recording_errors') or capture.get('truncated') or capture.get('unmapped_execution_gaps'):
        missing.append('Original capture has errors, truncation or executed gaps')
    if capture.get('actual_scene_count') != 1:
        missing.append('One actual own Scene missing')
    receipt = capture.get('actual_scene_close_receipt') or {}
    if receipt.get('called') is not True or receipt.get('close_calls') != 1:
        missing.append('One actual close receipt missing')
    if capture.get('source_custody') is not False or capture.get('source_diagnostic_main_called') is not False:
        missing.append('Ordinary SOURCE-off entry not identified')
    rows = capture.get('business_calls')
    if not isinstance(rows, list) or not rows:
        missing.append('No actual ordinary business rows')
        rows = []
    stages = []
    try:
        previous = _count(capture['actual_scene_initial_sequence'])
    except (KeyError, TypeError, ValueError):
        previous = None
        missing.append('Actual initial sequence unavailable')
    for index, row in enumerate(rows):
        try:
            if row.get('instance') != instance or row.get('round') != 1 or row.get('round_after') != 1:
                raise ValueError('Wrong instance/round boundary')
            before, after = _count(row['sequence_before']), _count(row['sequence_after'])
            if before != previous or after < before or _count(row['actual_controls']) != after-before:
                raise ValueError('Actual control range gap/overlap/regression')
            previous = after
            name = row['name']
            if name not in ('move_to', 'set_gripper'):
                raise ValueError('Unmapped actual ordinary boundary')
            args = _ordinary_arguments(name, row['arguments'])
            distinctions = []
            if instance == 'R4-01':
                if not fixed_engineering_candidates:
                    raise ValueError('R4 private stopping needs an explicitly NEW fixed engineering candidate')
                distinctions.append('Original private first-risk stop removed; new actual labels only.')
            if name == 'move_to':
                endpoint = row['public_eef_after']
                if _count(endpoint['sequence']) != after:
                    raise ValueError('Public endpoint from a different execution boundary')
                point = _xyz(endpoint['eef_pos'])
                distance = math.dist(args['xyz'], point)
                if not math.isfinite(row['actual_target_distance_m']) or abs(distance-row['actual_target_distance_m']) > 1e-12:
                    raise ValueError('Saved public target distance inconsistent')
                project = bool(row.get('needs_new_public_endpoint_candidate') or distance > args['tol'])
                if project:
                    if not fixed_engineering_candidates:
                        raise ValueError('Private stop/unreached nominal target; do not claim original completion')
                    args['xyz'] = point
                    distinctions.append('NEW public return endpoint replaces the unexecuted/private-stop nominal tail; '
                                        'not original path/controller equivalence.')
            elif instance == 'R4-01' and _count(row['actual_controls']) != args['steps']:
                args['steps'] = row['actual_controls']
                distinctions.append('NEW fixed actual control count replaces private early hold stop.')
            stages.append(dict(tool=name, arguments=args, source_row_index=index,
                source_stage=row.get('source_stage'), source_control_range=[before, after],
                original_business_arguments=deepcopy(row['arguments']),
                distinction=' '.join(distinctions)))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            missing.append(f'Row {index}: {exc}')
    try:
        if _count(receipt['executed_sequence']) != previous:
            missing.append('Final controls lie outside recorded ordinary boundaries')
    except (KeyError, TypeError, ValueError):
        missing.append('Actual final sequence unavailable')
    return dict(instance=instance, case='naive', seed=0, source=expected,
        stages=stages, script_ready=not missing, missing=missing,
        engineering_only=True, new_actual_labels_only=True, original_label_inherited=False,
        original_control_equivalence_claimed=False, original_physical_qualification_inherited=False,
        private_early_stop_installed=False, private_target_generator_installed=False,
        natural_method_result=False, formal=False, model_requests=0, robot_starts=0,
        initial_sequence_provenance=capture.get('actual_scene_initial_sequence'),
        no_missing_tail_filled=True)
