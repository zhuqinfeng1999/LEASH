"""Pure preparation from the finite saved R301 U two-stage public prefix.

Only saved public EEF xyzw/position and actually executed 7D commands enter
this projection. It is a PARTIAL NEW common-tool candidate, not a complete
naive script or a full-SO(3) equivalent. No Scene/model/imported actor starts.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULT_RECORD = Path('research/v4/candidates/r301_horizontal_u_physical_prepare_20260928/'
                      'root_naive_seed0/raw_before_final_admission.json')
ACTUAL_U_STAGES = ('ordinary_world_down_home_to_above_handle', 'real_upper_bridge_approach')


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Actual nonnegative integer sequence/count required')
    return value


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError('Required saved finite public vector missing')
    if any(isinstance(v, bool) for v in value):
        raise ValueError('Boolean is not a public motion number')
    numbers = [float(v) for v in value]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError('Saved public vector nonfinite')
    return numbers


def public_pitch_yaw(quat_xyzw):
    """Exactly the two scalar definitions consumed by existing move_pose.

    This does NOT assert a complete orientation target: its omitted degree of
    freedom and its simultaneous world-X/Z servo differ from the old rotvec.
    """
    x, y, z, w = _vector(quat_xyzw, 4)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError('Invalid saved public quaternion')
    x, y, z, w = (v / norm for v in (x, y, z, w))
    R10, R00 = 2*(x*y + z*w), 1-2*(y*y + z*z)
    R12, R22 = 2*(y*z - x*w), 1-2*(x*x + y*y)
    return dict(target_pitch=math.atan2(R12, -R22), target_yaw=math.atan2(R10, R00))


def _public_row(row):
    if not isinstance(row, dict):
        raise ValueError('Saved actual public/control row missing')
    return dict(sequence=_count(row['sequence']),
        eef_pos=_vector(row['eef_pos'], 3), eef_quat_xyzw=_vector(row['eef_quat'], 4),
        executed_action=_vector(row['executed_action'], 7))


def prepare(record):
    if (not isinstance(record, dict) or record.get('case') != 'naive'
            or type(record.get('seed')) is not int or record['seed'] != 0):
        raise ValueError('Only the actual finite U naive seed0 record')
    task = record.get('task_result')
    if not isinstance(task, dict):
        raise ValueError('Actual task result/public trace unavailable')
    trace = [_public_row(r) for r in task['actual_control_trace']]
    initialization = task['common20_initialization']
    if not isinstance(initialization, list) or len(initialization) != 20:
        raise ValueError('Actual twenty setup controls unavailable; do not assume reset sequence0')
    start = _count(initialization[-1]['sequence'])
    for index, row in enumerate(trace):
        if row['sequence'] != start + index + 1:
            raise ValueError('Actual public/control sequence has a gap or is from another step')
    stages = task.get('stages')
    if not isinstance(stages, list) or tuple(r.get('name') for r in stages) != ACTUAL_U_STAGES:
        raise ValueError('Actual U two-stage record required; do not substitute cold-bar stages')
    if sum(_count(r['steps']) for r in stages) != len(trace):
        raise ValueError('Actual stage ranges do not match saved control rows')
    if _count(task['control_steps']) != start + len(trace):
        raise ValueError('Actual setup/task/final control counts inconsistent')
    if _count(task['post_initialization_control_steps']) != len(trace):
        raise ValueError('Actual task control count inconsistent')
    candidates, offset = [], 0
    for stage in stages:
        count = _count(stage['steps'])
        if not count:
            raise ValueError('No actual endpoint for a zero-control stage')
        rows = trace[offset:offset + count]
        end = rows[-1]
        grip = rows[0]['executed_action'][6]
        if any(r['executed_action'][6] != grip for r in rows):
            raise ValueError('No single common-tool gripper argument for this stage')
        arguments = dict(xyz=deepcopy(end['eef_pos']), gripper=grip,
                         **public_pitch_yaw(end['eef_quat_xyzw']))
        candidates.append(dict(tool='move_pose', arguments=arguments,
            source_stage=stage['name'], actual_controls=count,
            source_control_range=[rows[0]['sequence']-1, end['sequence']],
            endpoint_public_state=dict(sequence=end['sequence'], eef_pos=deepcopy(end['eef_pos']),
                                       eef_quat_xyzw=deepcopy(end['eef_quat_xyzw'])),
            original_reported_reached=stage.get('reached'),
            distinction='NEW projection of actual public return endpoint into existing pitch/yaw tool. '
                        'Not the original full-SO(3) target/controller/path; no original label is inherited.',
            parameters_not_reconstructed=['original full-SO(3) target', 'original callable phase limits'],
            invoke_ready=False))
        offset += count
    return dict(instance='R3-01', case='naive', seed=0,
        actual_record_kind='historical_U_partial_194_task_controls_plus_20_setup',
        actual_source='r301_horizontal_u_physical_prepare_20260928 original U selected actor',
        old_cold_bar_expert_not_mixed=True, initial_actual_sequence=start,
        task_controls=len(trace), final_actual_sequence=trace[-1]['sequence'],
        partial_new_candidates=candidates, script_ready=False, full_naive_prepared=False,
        missing=['No actual CLOSE/contact/lift/transport/critical tail in this record. '
                 'The upper-bridge approach ended at its original phase cap (reached=False).',
                 'Original arbitrary SO(3) goal is not an existing move_pose parameter. '
                 'Current tool controls pitch/world-X and yaw/world-Z only; endpoint projection is a NEW candidate.',
                 'Original full callable motion parameters were not saved in these public control rows; '
                 'no omitted parameter/default is fabricated as original.'],
        original_terminal_label_not_inherited=True, original_control_equivalence_claimed=False,
        original_physical_acceptance_inherited=False, future_tail_filled=False,
        holding_or_empty_state_issued=False, source_custody=False,
        preparation_only=True, models=0, Scene_starts=0, robot_starts=0, formal=False)


def load_prepare(path=DEFAULT_RECORD):
    return prepare(json.loads(Path(path).read_text()))
