"""Pure R1/R5 numeric capture -> NEW ordinary engineering candidates.

No source/schema self-proof, Scene, model, actor, or launch is installed here.
Only the existing actual business arguments and executed ranges are checked.
Correct refusals have no manufactured motion. Original labels are provenance
in the source capture only and are never used to approve a new candidate.
"""
from copy import deepcopy

from .collect_r15_movement_parameters import OWN
from .convert_p2_captured_candidates import _count, _ordinary_arguments


def convert_capture(capture):
    if not isinstance(capture, dict) or capture.get('instance') not in OWN:
        raise ValueError('Actual fixed R1-01/R5-01 capture mapping required')
    instance = capture['instance']
    if (capture.get('case') != 'naive' or type(capture.get('seed')) is not int
            or capture['seed'] != 0):
        raise ValueError('Only the two original naive seed0 captures; correct refusal has no motion')
    expected = OWN[instance]['module'] + ':run_case'
    if capture.get('original_probe') != expected:
        raise ValueError('Capture source differs from its own instance')
    missing = []
    if capture.get('complete_numeric_boundaries') is not True:
        missing.append('Actual original numeric boundary capture is incomplete')
    if capture.get('actual_original_script_error') is not None:
        missing.append('Original script/capture failed; preserve the partial rows')
    if capture.get('recording_errors') or capture.get('truncated'):
        missing.append('Boundary recording failed or was truncated')
    if capture.get('unmapped_execution_gaps'):
        missing.append('Original executed controls contain an unmapped boundary')
    if (type(capture.get('actual_scene_count')) is not int
            or capture['actual_scene_count'] != 1):
        missing.append('One actual own Scene was not identified')
    receipt = capture.get('actual_scene_close_receipt') or {}
    if receipt.get('called') is not True:
        missing.append('Actual original Scene close boundary missing')
    if capture.get('source_diagnostic_main_called') is not False or capture.get('source_custody') is not False:
        missing.append('Capture does not identify the ordinary SOURCE-off entry')
    rows = capture.get('business_calls')
    if not isinstance(rows, list) or not rows:
        rows = []
        missing.append('Actual ordinary business arguments missing')
    stages, previous = [], 0
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or row.get('instance') != instance:
                raise ValueError('Boundary belongs to another/missing own instance')
            before, after = _count(row['sequence_before']), _count(row['sequence_after'])
            if before != previous:
                raise ValueError('Executed business ranges have a gap, overlap, or regression')
            previous = after
            if after < before or _count(row['actual_controls']) != after - before:
                raise ValueError('Executed sequence/control count inconsistent')
            if row.get('round') != 1 or row.get('round_after') != 1:
                raise ValueError('Boundary is not the original single-round execution')
            name = row.get('name')
            if name not in ('move_to', 'set_gripper'):
                raise ValueError('Unmapped original business function; no new primitive is inferred')
            args = _ordinary_arguments(name, row['arguments'])
            stages.append(dict(tool=name, arguments=args, source_row_index=index,
                               source_stage=row.get('source_stage'),
                               source_control_range=[before, after]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            missing.append(f'Row {index}: {exc}')
    try:
        if _count(receipt['executed_sequence']) != previous:
            missing.append('Final actual controls lie outside captured business boundaries')
    except (KeyError, TypeError, ValueError):
        missing.append('Final actual sequence unavailable')
    return dict(instance=instance, source=expected, case='naive', seed=0,
        stages=deepcopy(stages), script_ready=not missing, missing=missing,
        execution_kind='new_confidence_engineering_execution_current_common_primitives',
        engineering_only=True, natural_method_result=False, formal=False,
        new_actual_labels_only=True, original_label_inherited=False,
        original_control_equivalence_claimed=False, original_physical_qualification_inherited=False,
        private_target_generator_installed=False, private_holding_or_score_reader_installed=False,
        source_custody=False, model_requests=0, native_starts=0)
