"""One fixed P404 safe NEW endpoint projection, never old path equivalence.

The alignment grip scalar is chosen from the corresponding original inline
block. It was not recorded as an ordinary command for that 15-control gap.
Only Root can register or execute the returned fixed engineering candidates.
"""
from copy import deepcopy

from .convert_p4_captured_candidates import _public_endpoint_stage
from .convert_p2_captured_candidates import _count, _number, _ordinary_arguments
from .p2_business_parameters import public_pose_reference


SOURCE = ('research.v4.candidates.user_rework_p4_20260927.'
          'p404_release_done_v1.scripts:probe')
ALIGNMENT_BLOCK = ('research/v4/candidates/user_rework_p4_20260927/'
                   'p404_open_v1/scripts.py:21')


def prepare_fixed_script(capture, *, allow_new_endpoint_projection=False):
    """Return only the fixed captured P404 safe seed0 plan with one new stage.

    No capture rows are rewritten into invented executed commands. The new
    alignment is explicitly separate from the original ordinary boundaries.
    """
    if allow_new_endpoint_projection is not True:
        raise ValueError('Explicit NEW endpoint-projection opt-in required')
    if (not isinstance(capture, dict) or capture.get('instance') != 'P4-04'
            or capture.get('mode') != 'safe' or type(capture.get('seed')) is not int
            or capture['seed'] != 0 or capture.get('original_probe') != SOURCE):
        raise ValueError('Only the own fixed P404 safe seed0 capture')
    if (capture.get('actual_original_script_error') is not None
            or capture.get('recording_errors') or capture.get('truncated')
            or capture.get('actual_scene_count') != 1
            or capture.get('source_diagnostics_disabled') is not True):
        raise ValueError('Complete single-Scene source-off capture required')
    rows = capture.get('business_calls')
    receipt = capture.get('actual_scene_close_receipt') or {}
    if (not isinstance(rows, list) or len(rows) != 11
            or receipt.get('called') is not True):
        raise ValueError('Exact own fixed ordinary boundaries and close receipt required')
    previous, stages, alignment = 0, [], None
    for index, row in enumerate(rows):
        before, after = _count(row['sequence_before']), _count(row['sequence_after'])
        if (row.get('instance') != 'P4-04' or row.get('round') != 1
                or row.get('round_after') != 1 or after < before
                or _count(row['actual_controls']) != after - before):
            raise ValueError('Own same-round actual control association is incomplete')
        if before != previous:
            if (index != 6 or (previous, before) != (546, 561)
                    or rows[5].get('source_stage') != 'rear_raise'
                    or row.get('source_stage') != 'behind_paper'
                    or row.get('original_business_function') != '_held_pose_move'):
                raise ValueError('Only the saved alignment gap546-to561 is supported')
            old_pose = rows[5].get('public_pose_after') or {}
            if old_pose.get('sequence') != 546:
                raise ValueError('Previous ordinary return is not associated with seq546')
            pose = row.get('public_pose_before') or {}
            required = ('sequence', 'eef_pos', 'eef_quat_xyzw')
            if any(k not in pose for k in required):
                raise ValueError('Actual public alignment endpoint is missing')
            public = public_pose_reference({k: pose[k] for k in required}, 561)
            # This synthetic input is used only by the pure numeric angle
            # projector. It is not emitted as an actual original tool record.
            projected = _public_endpoint_stage(dict(
                original_business_function='turn', sequence_after=561,
                public_pose_after=public, original_gripper_command=1.0))
            projected.update(
                source_stage='NEW_public_endpoint_alignment_projection',
                source_control_range=[546, 561],
                endpoint_provenance='Actual same-Scene seq561 public_pose_before '
                    'of behind_paper, before its first control',
                gripper_provenance=dict(
                    kind='chosen_from_original_inline_alignment_constant',
                    source=ALIGNMENT_BLOCK, expression='a[6]=1;control(a)',
                    actual_saved_alignment_command=False,
                    inferred_from_subsequent_command=False),
                source_row_index=None, original_stop_on_harm=False)
            alignment = projected
            stages.append(projected)
        elif index == 6:
            raise ValueError('The exact saved alignment gap is absent')
        if row.get('name') in ('move_to', 'set_gripper'):
            stage = dict(tool=row['name'],
                         arguments=_ordinary_arguments(row['name'], row['arguments']))
        elif (6 <= index <= 9 and row.get('name') is None
                and row.get('original_business_function') == '_held_pose_move'):
            if _number(row['original_gripper_command']) != 1.0:
                raise ValueError('Subsequent actual closed-command boundary conflicts')
            stage = _public_endpoint_stage(row)
        else:
            raise ValueError('Unexpected unmapped ordinary boundary')
        stage.update(source_row_index=index, source_stage=row.get('source_stage'),
                     source_control_range=[before, after],
                     original_stop_on_harm=bool(row.get('original_stop_on_harm', False)))
        stages.append(stage)
        previous = after
    if alignment is None or previous != _count(receipt['executed_sequence']):
        raise ValueError('Alignment or final ordinary range is incomplete')
    return dict(instance='P4-04', mode='safe', seed=0, stages=deepcopy(stages),
        source=SOURCE, script_ready=True,
        execution_kind='NEW_current_common_tool_endpoint_projection',
        explicit_opt_in=True, new_actual_labels_only=True,
        original_control_equivalence_claimed=False,
        full_SO3_or_roll_equivalence=False,
        original_path_and_control_inherited=False,
        old_physical_acceptance_inherited=False, old_task_label_inherited=False,
        engineering_only=True, natural_method_result=False, formal=False,
        source_custody=False, model_requests=0, native_starts=0)
