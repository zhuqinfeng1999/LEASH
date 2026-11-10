"""Finite, public-only reconstruction of one saved E3 G3 review input.

No service, Scene, robot, evaluator or retired Consumer is imported. The raw
old command is retained: unavailable historical OSC scales are not invented.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .evidence import EvidenceBuilder

ROOT = Path(__file__).resolve().parents[4]
OLD_ROOT = ROOT / 'research/v4/runs/lite_20261003'
OUT = ROOT / 'research/v4/results/confidence_gate_dev_20261006/E3_saved_G3_20261007'
RUN = OLD_ROOT / 'E3_01_workpackage_pour_hold_public'


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def rgb(path):
    value = np.asarray(Image.open(path).convert('RGB'))
    if value.shape != (480, 640, 3):
        raise ValueError('Saved public RGB is not 640x480')
    return value


def prepare(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episode = json.loads((RUN / 'episode.json').read_text())
    tools = json.loads((RUN / 'scripted_tools.json').read_text())
    states = json.loads((RUN / 'rpent/states.json').read_text())['steps']
    executed = lines(RUN / 'public_executed.jsonl')
    old_reviews = lines(RUN / 'reviewer.jsonl')
    dispatch = episode['commit_gate']['logs']

    # Actual G3 call20 was candidate442 BEFORE any tilt control was sent.
    review = next(x for x in old_reviews if x.get('call_index') == 20)
    gate = next(x for x in dispatch if x.get('kind') == 'review'
        and x.get('events') == ['POUR_ONSET'] and x.get('primitive') == 7)
    seq = gate['candidate_sequence'] - 1
    if seq != 441 or not gate['evidence_version'].endswith(':442:441'):
        raise ValueError('Saved POUR candidate/frame mapping differs')
    state = states[6]['state']
    row = next(x for x in executed if x['executed_sequence'] == seq
        and x.get('candidate_dispatched'))
    blocked = next(x for x in executed if x['executed_sequence'] == seq
        and not x.get('candidate_dispatched'))
    current = row['after_public']
    if blocked['before_public'] != current or blocked['after_public'] != current:
        raise ValueError('Blocked command did not use the saved current public state')
    if current['eef_xyz'] != state['robot0_eef_pos'] or current['eef_quat_xyzw'] != state['robot0_eef_quat']:
        raise ValueError('Current primitive-end public state and execution state differ')

    packet = {'frame_id': seq, 'cameras': {}}
    images = {}
    for view, old_name in (('agentview', 'agentview'), ('robot0_eye_in_hand', 'wrist')):
        path = RUN / next(p for p in review['images'] if p.endswith(f'_{view}_rgb.png'))
        image = rgb(path)
        same_step = rgb(RUN / 'rpent' / f'{old_name}.png' / '06.png')
        if not np.array_equal(image, same_step):
            raise ValueError('Actual reviewer RGB differs from the corresponding saved public step')
        camera = json.loads((RUN / 'rpent' / f'{old_name}_metadata.json' / '06.json').read_text())
        packet['cameras'][view] = {'frame_id': seq, 'rgb': image,
            'intrinsics': camera['intrinsic_K'], 'extrinsics': camera['extrinsic_cam2world']}
        images[view] = str(path.resolve())

    # R_grasp really equals the public EEF rotation at the executed CLOSE
    # boundary (235 -> 236), not a default matrix or a hidden object pose.
    close_before = next(x for x in executed if x['executed_sequence'] == 236
        and x.get('candidate_dispatched'))
    grasp_rotation = np.asarray(row['R_grasp'], float)
    baseline_rotation = Rotation.from_quat(close_before['before_public']['eef_quat_xyzw']).as_matrix()
    if not np.allclose(grasp_rotation, baseline_rotation, atol=1e-12):
        raise ValueError('Saved grasp baseline is not the executed CLOSE public rotation')
    close_controls = [x for x in dispatch if x.get('kind') == 'dispatched' and x.get('primitive') == 5]
    if [x['executed_sequence'] for x in close_controls] != list(range(236, 262)):
        raise ValueError('Actual CLOSE interval differs')

    history = []
    for primitive in range(2, 7):
        sent = [x for x in dispatch if x.get('kind') == 'dispatched' and x.get('primitive') == primitive]
        if not sent:
            raise ValueError('Requested history primitive was not actually executed')
        history.append({'name': tools[primitive - 1]['tool'],
            'arguments': tools[primitive - 1]['arguments'],
            'executed_start': sent[0]['executed_sequence'] - 1,
            'executed_end': sent[-1]['executed_sequence'],
            'gate_outcomes': [{'gate': 'G2', 'decision': episode['action_gate']['logs'][primitive - 1]['result'],
                'comment': episode['action_gate']['logs'][primitive - 1]['reason']}]
                + [{'gate': 'G3', 'decision': x['decision'], 'comment': x['reason']}
                   for x in dispatch if x.get('kind') == 'review' and x.get('primitive') == primitive]})

    robot = {'eef_pos': current['eef_xyz'], 'eef_quat': current['eef_quat_xyzw'],
        'eef_rotation': Rotation.from_quat(current['eef_quat_xyzw']).as_matrix().tolist(),
        'gripper_qpos': current['gripper_qpos'],
        'gripper_aperture_m': abs(current['gripper_qpos'][0] - current['gripper_qpos'][1]),
        'holding_estimate': row['holding_estimate'], 'last_gripper_command': 'CLOSE'}
    candidate = {'name': tools[6]['tool'], 'arguments': tools[6]['arguments'],
        'control': blocked['command'], 'events': ['POUR_ONSET']}
    first = next(x for x in old_reviews if x.get('call_index') == 1)
    first_image = RUN / next(p for p in first['images'] if p.endswith('_agentview_rgb.png'))
    if not np.array_equal(rgb(first_image), rgb(RUN / 'rpent/agentview.png/00.png')):
        raise ValueError('Initial workspace image is not the saved initial state')
    builder = EvidenceBuilder(output)
    builder.first_frame = {'view': 'agentview', 'frame_id': 0,
        'rgb_path': str(first_image.resolve()), 'width': 640, 'height': 480}
    event = {'events': ['POUR_ONSET'], 'gripper': 'CLOSE',
        'translation_m': [0., 0., 0.],
        'rotation_world_axis_angle_rad': None,
        'rotation_conversion_availability': 'Historical OSC scaling arrays were not recorded; raw actual control and original target_pitch are retained.'}
    evidence = builder.capture(packet, robot, episode['instruction'], 'G3', candidate,
        event=event, history=history)
    (output / 'public_G3_candidate442.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    report = {'runs_examined': [str(OLD_ROOT/'E3_01_workpackage_pour_fixed'), str(RUN)],
        'fixed_run': {'actual_pour_onset': False,
            'blocker': "rotate_pitch was rejected by the old public whitelist before any POUR_ONSET candidate."},
        'hold_public_run': {'actual_pour_onset': True, 'candidate_sequence': 442,
            'executed_sequence': seq, 'simulation_time_s': row['sim_time_s'],
            'public_two_view_synchrony': True, 'reviewer_images': images,
            'public_close_executed_range': [236, 261], 'R_grasp': row['R_grasp'],
            'baseline_source': 'Public rotation before executed CLOSE control236; equals after control235.',
            'first_record_with_grasp_baseline_sequence': 247,
            'latest_five_executed_primitives': history,
            'original_results': {'first_scripted_G3': 'BLOCK', 'fresh_scripted_G3': 'PASS',
                'controls_after_pass': 1, 'episode': 'SYSTEM_FAILURE',
                'model_finding': False},
            'builder_output': str((output/'public_G3_candidate442.json').resolve()),
            'full_current_project_equivalence': False,
            'missing_exact_historical_project_field': 'OSC-scaled rotation_world_axis_angle_rad',
            'scope': 'Saved public input availability; no new reviewer request or robot execution.'},
        'calibration_limitation': 'The candidate was above the source, not the mug mouth. One sent control followed by a service failure does not establish the safety of the full intended tilt. There is no second independently executed unsafe POUR candidate in these two runs.',
        'new_model_calls': 0, 'new_robot_starts': 0}
    (output/'availability.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
