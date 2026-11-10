"""Pure conversion of complete original P2 business captures.

Only the capture's actual ordinary arguments and same-boundary public EEF
pose are used. The resulting fixed script is a NEW common-tool engineering
candidate, never an equivalent object-yaw servo or inherited physical result.
This module imports no Scene, scorer, model, controller, or launch entry.
"""
from copy import deepcopy
import math

from .p2_business_parameters import P2_SCRIPTS, public_pose_reference


_MOVE_KEYS = {'xyz', 'gripper', 'step_clip', 'tol', 'max_steps', 'action_scale'}
_GRIP_KEYS = {'gripper', 'steps'}


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Actual nonnegative integer sequence/count required')
    return value


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean cannot be a motion number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite motion number')
    return result


def _ordinary_arguments(name, args):
    """Validate without rounding, clipping, replacing, or fitting parameters."""
    if not isinstance(args, dict):
        raise ValueError('Actual ordinary argument mapping missing')
    expected = _MOVE_KEYS if name == 'move_to' else _GRIP_KEYS
    if set(args) != expected:
        raise ValueError('Unexpected/missing ordinary tool parameters')
    if name == 'move_to':
        if len(args['xyz']) != 3:
            raise ValueError('Actual move target missing')
        for value in args['xyz']:
            _number(value)
        for key in ('gripper', 'step_clip', 'tol', 'action_scale'):
            _number(args[key])
        if min(args['step_clip'], args['tol'], args['action_scale']) <= 0:
            raise ValueError('Original positive move limits required')
        _count(args['max_steps'])
    else:
        _number(args['gripper'])
        _count(args['steps'])
    return deepcopy(args)


def _presentation_endpoint(row):
    """Recompute yaw from the actual saved public quaternion, never object yaw."""
    snapshot = row.get('public_pose_after')
    if not isinstance(snapshot, dict):
        raise ValueError('Missing actual public EEF pose at present() return')
    required = ('sequence', 'eef_pos', 'eef_quat_xyzw')
    if any(k not in snapshot for k in required):
        raise ValueError('Incomplete actual public EEF pose at present() return')
    pose = public_pose_reference({k: snapshot[k] for k in required}, row['sequence_after'])
    if 'body_world_yaw_rad' in snapshot:
        supplied = _number(snapshot['body_world_yaw_rad'])
        difference = math.atan2(math.sin(supplied - pose['body_world_yaw_rad']),
                                math.cos(supplied - pose['body_world_yaw_rad']))
        if abs(difference) > 1e-12:
            raise ValueError('Saved public yaw disagrees with its actual quaternion')
    return pose


def convert_capture(capture):
    """Return mapped rows plus explicit readiness; never execute/retry anything.

    A presentation endpoint produces only world-z yaw, using the existing
    rotate_wrist defaults and original closed-gripper command. Its observed
    xyz/full quaternion are provenance, not a claim that that tool reproduces
    full SO(3), object orientation, or the old rotation path. All other stages
    use this very capture's original consumed move/hold numeric arguments.
    """
    if not isinstance(capture, dict):
        raise ValueError('Actual P2 capture mapping required')
    instance = capture.get('instance')
    if instance not in P2_SCRIPTS or capture.get('mode') not in ('safe', 'naive'):
        raise ValueError('Only the finite original own P2 bindings/branches')
    seed = _count(capture.get('seed'))
    source = P2_SCRIPTS[instance] + ':probe'
    if capture.get('source') != source:
        raise ValueError('Capture source differs from its own instance')
    missing = []
    if capture.get('parameter_capture_complete') is not True:
        missing.append('Original parameter capture did not complete')
    if type(capture.get('actual_scene_count')) is not int or capture['actual_scene_count'] != 1:
        missing.append('Capture does not identify one actual original Scene')
    if capture.get('original_error') is not None or capture.get('call_error') is not None:
        missing.append('Original probe/capture has an error; retain the partial record')
    if capture.get('recording_errors') or capture.get('truncated'):
        missing.append('Capture has recording errors or was truncated')
    rows = capture.get('ordinary_calls')
    if not isinstance(rows, list) or not rows:
        rows = []
        missing.append('Actual ordinary boundaries missing')
    stages, previous_after = [], None
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or row.get('instance') != instance:
                raise ValueError('Business row belongs to another/missing instance')
            before, after = _count(row['sequence_before']), _count(row['sequence_after'])
            if after < before or _count(row['actual_controls']) != after - before:
                raise ValueError('Actual boundary/control range is inconsistent')
            if previous_after is None and before != 0:
                raise ValueError('Initial executed controls are outside captured business boundaries')
            if previous_after is not None and before != previous_after:
                raise ValueError('Business boundaries have an unrecorded gap, overlap, or regression')
            previous_after = after
            if row.get('round') != 1 or row.get('round_after') != 1:
                raise ValueError('P2 boundary is not the actual single original round')
            name = row.get('name')
            if name in ('move_to', 'set_gripper'):
                stage = dict(tool=name, arguments=_ordinary_arguments(name, row['arguments']))
                if row.get('original_lower_table_unreached_exception'):
                    stage['continue_after_unreached_engineering'] = True
            elif name is None and row.get('source_stage') == 'actual_held_presentation':
                if row.get('arguments') != {}:
                    raise ValueError('Presentation must not supply a private yaw target')
                pose = _presentation_endpoint(row)
                stage = dict(tool='rotate_wrist',
                    arguments=dict(target_yaw=pose['body_world_yaw_rad'], gripper=1.0),
                    public_endpoint_reference=pose,
                    current_tool_defaults_preserved=True,
                    full_SO3_or_object_orientation_equivalence=False,
                    original_rotation_path_inherited=False,
                    observed_endpoint_position_is_not_a_new_move_target=True)
            else:
                raise ValueError(row.get('mapping_missing') or 'Unsupported actual business boundary')
            stage.update(source_row_index=index, source_stage=row.get('source_stage'),
                         source_control_range=[before, after],
                         original_stop_on_harm=bool(row.get('original_stop_on_harm', False)))
            stages.append(stage)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            missing.append(f'Row {index}: {exc}')
    if rows:
        try:
            if previous_after != _count(capture['actual_control_count']):
                missing.append('Final capture control count differs from last complete boundary')
        except (KeyError, TypeError, ValueError):
            missing.append('Final actual control count missing')
    return dict(instance=instance, mode=capture['mode'], seed=seed,
        execution_kind='new_confidence_engineering_execution_current_common_primitives',
        script_ready=not missing, stages=stages, missing=missing,
        source=source, original_capture_complete=capture.get('parameter_capture_complete') is True,
        engineering_only=True, natural_method_result=False, formal=False,
        original_physical_acceptance_inherited=False,
        original_control_equivalence_claimed=False,
        offline_numeric_actor_only=True, hidden_dynamic_reader_installed=False,
        source_custody=False, model_requests=0, native_starts=0)
