"""Two fixed NEW current-tool plans from actual ordinary CALL endpoints.

The preceding original inline nose-down path is not reconstructed. Its grip
scalar is chosen from that identified original block, not labelled a saved
ordinary command. Missing old CALL endpoints remain missing.
"""
from copy import deepcopy

from .convert_p4_captured_candidates import _public_endpoint_stage
from .convert_p2_captured_candidates import _count, _number, _ordinary_arguments
from .p2_business_parameters import public_pose_reference
from .p4_business_parameters import P4_SCRIPTS


INLINE_BLOCK = ('research/v4/candidates/p4_03_gluegun_20260926/'
                'scripts.py:18-21')
PREFIX = ('staging', 'above_handle', 'handle_grasp', 'hold', 'lift',
          'rear_raise', 'above_destination', 'lower_to_cloth')
STAGES = {'P4-03': PREFIX + ('hold', 'clear_gripper', 'hold'),
          'P4-04': PREFIX}


def prepare_fixed_naive_script(capture, *, allow_new_endpoint_projection=False):
    """Prepare only a complete own P403/P404 naive seed0 capture.

    Root registers and obtains new actual labels. This function starts no
    Scene/model, modifies no saved capture, and grants no execution permit.
    """
    if allow_new_endpoint_projection is not True:
        raise ValueError('Explicit NEW inline endpoint-projection opt-in required')
    instance = capture.get('instance') if isinstance(capture, dict) else None
    if (instance not in STAGES or capture.get('mode') != 'naive'
            or type(capture.get('seed')) is not int or capture['seed'] != 0
            or capture.get('original_probe') != P4_SCRIPTS[instance] + ':probe'):
        raise ValueError('Only own fixed P403/P404 naive seed0 captures')
    if (capture.get('actual_original_script_error') is not None
            or capture.get('recording_errors') or capture.get('truncated')
            or capture.get('actual_scene_count') != 1
            or capture.get('source_diagnostics_disabled') is not True):
        raise ValueError('Complete single-Scene source-off capture required')
    rows, receipt = capture.get('business_calls'), capture.get('actual_scene_close_receipt') or {}
    if (not isinstance(rows, list)
            or tuple(r.get('source_stage') for r in rows) != STAGES[instance]
            or receipt.get('called') is not True):
        raise ValueError('The fixed own ordinary stage list and close receipt are incomplete')
    previous, new_stages, alignment = 0, [], None
    for index, row in enumerate(rows):
        before, after = _count(row['sequence_before']), _count(row['sequence_after'])
        if (row.get('instance') != instance or row.get('round') != 1
                or row.get('round_after') != 1 or after < before
                or _count(row['actual_controls']) != after - before):
            raise ValueError('Actual own single-round boundary association is incomplete')
        if before != previous:
            if (index != 6 or before <= previous
                    or row.get('original_business_function') != 'move'
                    or row.get('name') != 'move_to'):
                raise ValueError('An additional/unknown inline control gap is not supported')
            if (rows[5].get('public_pose_after') or {}).get('sequence') != previous:
                raise ValueError('Pre-inline rear_raise return is not associated with its sequence')
            pose = row.get('public_pose_before') or {}
            required = ('sequence', 'eef_pos', 'eef_quat_xyzw')
            if any(k not in pose for k in required):
                raise ValueError('Actual same-CALL inline endpoint p/xyzw is missing')
            public = public_pose_reference({k: pose[k] for k in required}, before)
            subsequent = _ordinary_arguments('move_to', row['arguments'])
            if (_number(row['original_gripper_command']) != 1.0
                    or subsequent['gripper'] != 1.0):
                raise ValueError('Subsequent captured closed-command boundary conflicts')
            # Only a pure numeric projection input; no synthetic executed
            # command is written into original business_calls.
            projected = _public_endpoint_stage(dict(
                original_business_function='turn', sequence_after=before,
                public_pose_after=public, original_gripper_command=1.0))
            projected.update(source_stage='NEW_public_inline_nose_down_endpoint_projection',
                source_row_index=None, source_control_range=[previous, before],
                original_stop_on_harm=False,
                endpoint_provenance='Actual same-Scene public_pose_before at '
                    'above_destination CALL, before its first control',
                gripper_provenance=dict(
                    kind='chosen_from_identified_original_inline_constant',
                    source=INLINE_BLOCK, expression='a[4]=.08;a[6]=1;control(a)',
                    actual_saved_inline_command=False,
                    inferred_from_subsequent_command=False),
                preceding_original_rotation_axis_not_mapped=True)
            alignment = projected
            new_stages.append(projected)
        elif index == 6:
            raise ValueError('The known actual inline nose-down gap is absent')
        name = row.get('name')
        if name not in ('move_to', 'set_gripper'):
            raise ValueError('Unexpected unmapped original ordinary motion')
        stage = dict(tool=name, arguments=_ordinary_arguments(name, row['arguments']),
            source_row_index=index, source_stage=row['source_stage'],
            source_control_range=[before, after],
            original_stop_on_harm=bool(row.get('original_stop_on_harm', False)))
        new_stages.append(stage)
        previous = after
    if alignment is None or previous != _count(receipt['executed_sequence']):
        raise ValueError('Inline endpoint or final ordinary control coverage is incomplete')
    return dict(instance=instance, mode='naive', seed=0,
        source=P4_SCRIPTS[instance] + ':probe', stages=deepcopy(new_stages),
        script_ready=True, explicit_opt_in=True,
        execution_kind='NEW_current_common_tool_inline_endpoint_projection',
        new_actual_labels_only=True, old_task_label_inherited=False,
        full_SO3_or_roll_equivalence=False,
        original_control_equivalence_claimed=False,
        original_path_and_control_inherited=False,
        old_physical_acceptance_inherited=False,
        engineering_only=True, natural_method_result=False, formal=False,
        source_custody=False, model_requests=0, native_starts=0)
