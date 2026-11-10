"""Pure current-tool candidates; optional public endpoint projection only.

The projection is an explicitly NEW pitch/yaw/xyz engineering trajectory.
It is not original full-SO3/roll/control/path or object-to-EEF equivalence.
Missing actual endpoint/gripper or uncaptured inline controls remain gaps.
"""
from copy import deepcopy
import math
import sys

from .convert_p2_captured_candidates import _count, _number, _ordinary_arguments
from .p2_business_parameters import public_pose_reference
from .p4_business_parameters import P4_SCRIPTS


def _public_endpoint_stage(row):
    function = row.get('original_business_function')
    if function not in ('turn', '_held_pose_move', 'move'):
        raise ValueError('No captured ordinary boundary identifies this endpoint projection')
    if function == 'move' and 'ordinary_translation_arguments' not in row:
        raise ValueError('An unspecified move cannot be replaced by an endpoint pose')
    snapshot = row.get('public_pose_after')
    if not isinstance(snapshot, dict):
        raise ValueError('Actual same-return public EEF endpoint is missing')
    required = ('sequence', 'eef_pos', 'eef_quat_xyzw')
    if any(k not in snapshot for k in required):
        raise ValueError('Actual public EEF endpoint fields are incomplete')
    pose = public_pose_reference({k: snapshot[k] for k in required}, row['sequence_after'])
    grip = _number(row['original_gripper_command'])
    if 'ordinary_translation_arguments' in row:
        original = _ordinary_arguments('move_to', row['ordinary_translation_arguments'])
        if grip != original['gripper']:
            raise ValueError('Captured actual gripper parameters disagree')
    q = pose['eef_quat_xyzw']
    norm = math.sqrt(sum(v * v for v in q))
    x, y, z, w = (v / norm for v in q)
    # Exactly the installed RPent move_pose/rotate_pitch/rotate_wrist angle
    # definitions, not object yaw or a hidden full-SO3 inverse controller.
    r00, r10 = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
    r12, r22 = 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    numerical_zero = 64 * sys.float_info.epsilon
    if math.hypot(r00, r10) <= numerical_zero or math.hypot(r12, r22) <= numerical_zero:
        raise ValueError('Public orientation has an undefined current-tool pitch/yaw projection')
    pitch, yaw = math.atan2(r12, -r22), math.atan2(r10, r00)
    for key, actual in (('body_world_yaw_rad', yaw), ('original_rpent_world_X_pitch_rad', pitch)):
        if key in snapshot:
            given = _number(snapshot[key])
            error = math.atan2(math.sin(given - actual), math.cos(given - actual))
            if abs(error) > 1e-12:
                raise ValueError('Saved projected angle differs from its actual public quaternion')
    return dict(tool='move_pose', arguments=dict(xyz=deepcopy(pose['eef_pos']),
        target_pitch=pitch, target_yaw=yaw, gripper=grip),
        public_endpoint_reference=pose, public_endpoint_projection=True,
        current_tool_defaults_preserved=True, full_SO3_or_roll_equivalence=False,
        original_path_and_control_inherited=False, old_task_label_inherited=False,
        new_actual_labels_only=True,
        projection_scope='Actual observed public EEF xyz plus existing tool pitch/yaw; '
            'uncontrolled orientation/roll and the new trajectory are not the original full pose/path.')


def convert_capture(capture, *, project_public_endpoints=False):
    if type(project_public_endpoints) is not bool:
        raise ValueError('Explicit boolean endpoint-projection opt-in required')
    if not isinstance(capture, dict) or capture.get('instance') not in P4_SCRIPTS:
        raise ValueError('Actual own P4-02/03/04 capture required')
    instance = capture['instance']
    if (capture.get('mode') not in ('safe', 'naive')
            or type(capture.get('seed')) is not int or capture['seed'] != 0):
        raise ValueError('Only fixed safe/naive seed0 engineering cases')
    if capture.get('original_probe') != P4_SCRIPTS[instance] + ':probe':
        raise ValueError('Capture source differs from its own physical instance')
    missing = []
    if capture.get('actual_original_script_error') is not None:
        missing.append('Original script did not finish normally; retain its original failure')
    if capture.get('recording_errors') or capture.get('truncated'):
        missing.append('Incomplete business parameter recording')
    receipt = capture.get('actual_scene_close_receipt') or {}
    if (type(capture.get('actual_scene_count')) is not int
            or capture['actual_scene_count'] != 1 or receipt.get('called') is not True):
        missing.append('One original Scene close boundary was not confirmed')
    if capture.get('source_diagnostics_disabled') is not True:
        missing.append('Original pure source diagnostics have not been disabled')
    rows = capture.get('business_calls')
    if not isinstance(rows, list) or not rows:
        rows = []
        missing.append('Actual business parameter rows missing')
    stages, previous = [], 0
    for index, row in enumerate(rows):
        try:
            if row.get('instance') != instance:
                raise ValueError('Boundary belongs to another/missing instance')
            before, after = _count(row['sequence_before']), _count(row['sequence_after'])
            if before != previous:
                raise ValueError('Unmapped inline controls or overlapping/regressed boundaries')
            previous = after
            if after < before or _count(row['actual_controls']) != after - before:
                raise ValueError('Actual boundary/control range inconsistent')
            if row.get('round') != 1 or row.get('round_after') != 1:
                raise ValueError('Boundary is not the original single round')
            name = row.get('name')
            if name is None and project_public_endpoints:
                stage = _public_endpoint_stage(row)
            elif name in ('move_to', 'set_gripper'):
                stage = dict(tool=name, arguments=_ordinary_arguments(name, row['arguments']))
            else:
                raise ValueError(row.get('mapping_missing') or 'Unmapped original motion')
            stage.update(source_row_index=index, source_stage=row.get('source_stage'),
                         source_control_range=[before, after],
                         original_stop_on_harm=bool(row.get('original_stop_on_harm', False)))
            stages.append(stage)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            missing.append(f'Row {index}: {exc}')
    try:
        if previous != _count(receipt['executed_sequence']):
            missing.append('Final original controls are outside captured boundaries')
    except (KeyError, TypeError, ValueError):
        missing.append('Final actual original sequence missing')
    return dict(instance=instance, mode=capture['mode'], seed=0,
        source=P4_SCRIPTS[instance] + ':probe', stages=deepcopy(stages),
        script_ready=not missing, missing=missing,
        project_public_endpoints=project_public_endpoints,
        execution_kind='new_confidence_engineering_execution_current_common_primitives',
        engineering_only=True, natural_method_result=False, formal=False,
        source_custody=False, model_requests=0, native_starts=0,
        old_physical_acceptance_inherited=False, original_control_equivalence_claimed=False,
        private_stopper_to_online_input=False, private_pose_reader_installed=False)
