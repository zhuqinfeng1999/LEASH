"""RGB-only gate projection and commanded-path annotation from public inputs."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from research.v4.development.confidence_gates_20261006.config import policy, PROMPT_VERSION

VIEWS = ('agentview', 'robot0_eye_in_hand')
ROBOT_KEYS = ('eef_pos', 'eef_quat', 'gripper_qpos', 'eef_rotation',
              'gripper_aperture_m', 'holding_estimate', 'last_gripper_command', 'initial_occupancy')


def review_scope(gate, candidate, event=None):
    """Identify the existing decision boundary; never infer safety or holding."""
    if gate == 'G1':
        return dict(kind='task_goal', scored_field='trusted_instruction',
                    context_only_fields=['candidate'])
    if gate == 'G2':
        return dict(kind='primitive_call', scored_field='candidate',
                    context_only_fields=['trusted_instruction'])
    if gate != 'G3':
        raise ValueError('Unknown gate review scope')
    control = np.asarray(candidate.get('control'), dtype=float)
    if control.shape != (7,) or not np.isfinite(control).all() or not isinstance(event, dict):
        raise ValueError('G3 requires its exact pending control and decoded event')
    return dict(kind='pending_commit_transition', scored_field='candidate.control + event',
                context_only_fields=['trusted_instruction', 'parent_primitive_context'],
                excluded_scope='unexecuted remainder of the parent primitive and hypothetical later steps')


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def _point(value):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Public endpoint must be a finite world three-vector')
    return result


def _gripper_description(value):
    # The existing Panda/RPent low-level decoder uses the sign, not a
    # 0..1 aperture target. Explain that public command without inferring
    # occupancy or the outcome of closing.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A finite numeric gripper command is required')
    meaning = 'OPEN' if value < 0 else 'CLOSE' if value > 0 else 'HOLD'
    return f'{value} ({meaning}; Panda negative=open, positive=close, zero=hold)'


def candidate_description(candidate, robot):
    name = candidate.get('name', candidate.get('primitive', 'unknown'))
    args = candidate.get('arguments', candidate.get('parameters', {}))
    if name == 'move_to':
        delta = _point(args['xyz']) - _point(robot['eef_pos'])
        route = candidate.get('execution_route', {})
        return ('Move the documented EEF reference point by world XYZ '
                f'{delta.tolist()} metres to {plain(args["xyz"])}; '
                f'gripper command {_gripper_description(args.get("gripper", -1.))}; '
                + ('the common controller commands a straight translation with initial orientation.'
                   if route.get('route') == 'common_persistent_world_line' and route.get('path_known')
                   else 'this call uses the original RPent servo branch; no guaranteed straight path is drawn.'))
    if name in ('pi0_pick', 'pi0_doubled', 'vla_act'):
        return 'Execute the existing local learned skill; its path is not known in advance. Prompt: ' + str(args.get('prompt', args.get('instruction', '')))
    if name == 'set_gripper':
        return ('Keep the EEF pose and drive gripper command '
                f'{_gripper_description(args.get("gripper", -1.))} '
                f'for {args.get("steps", 5)} existing control steps. '
                'This command does not establish whether an object is grasped.')
    if name == 'release':
        return ('Keep the EEF pose and command OPEN (Panda gripper -1) '
                f'for at most {args.get("max_steps", 20)} existing control steps.')
    if name in ('move_pose', 'rotate_pitch', 'rotate_wrist'):
        return (f'Execute exactly {name} with parameters {json.dumps(plain(args), allow_nan=False)}; '
                f'gripper command {_gripper_description(args.get("gripper", -1.))}.')
    return f'Execute exactly {name} with parameters {json.dumps(plain(args), allow_nan=False)}.'


def annotate(rgb, camera, robot, candidate):
    """No depth, segmentation, obstacle or risk calculations are used."""
    name = candidate.get('name', candidate.get('primitive'))
    args = candidate.get('arguments', candidate.get('parameters', {}))
    if name not in ('move_to', 'move_pose', 'rotate_pitch', 'rotate_wrist'):
        return None, {'drawn': False, 'reason': 'path unavailable or no spatial motion'}
    route = candidate.get('execution_route')
    if name == 'move_to' and (not isinstance(route, dict)
            or route.get('route') != 'common_persistent_world_line' or route.get('path_known') is not True):
        return None, {'drawn': False, 'reason': 'original RPent servo branch; no preknown straight path',
                      'execution_route': plain(route)}
    k, t = np.asarray(camera['intrinsics'], float), np.asarray(camera['extrinsics'], float)
    if (k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all()
            or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)):
        raise ValueError('Public path overlay requires valid camera calibration')
    world_to_camera = np.linalg.inv(t)

    def project(world):
        c = world_to_camera @ np.r_[_point(world), 1.]
        if c[2] <= 1e-8:
            return None
        p = k @ c[:3]
        xy = p[:2] / p[2]
        return tuple(float(v) for v in xy) if np.isfinite(xy).all() else None

    start = _point(robot['eef_pos'])
    goal = _point(args.get('xyz', start))
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    a, b = project(start), project(goal)
    colour = (0, 220, 255)
    if a is not None:
        draw.ellipse((a[0]-2, a[1]-2, a[0]+2, a[1]+2), fill=colour)
    if a is not None and b is not None and np.linalg.norm(np.array(b)-a) > .1:
        draw.line((a, b), fill=colour, width=1)
        d = (np.array(b)-a); d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        for sign in (-1, 1):
            edge = np.array(b) - 6*d + sign*2*n
            draw.line((b, tuple(edge)), fill=colour, width=1)
    metadata = {'drawn': a is not None or b is not None, 'kind': 'commanded path',
                'world_start': start.tolist(), 'world_endpoint': goal.tolist(),
                'calibration_source': 'same-step public K/T, camera-to-world'}
    if name in ('move_pose', 'rotate_pitch', 'rotate_wrist'):
        quat = robot.get('eef_quat')
        r = np.asarray(robot.get('eef_rotation'), float) if quat is None else Rotation.from_quat(quat).as_matrix()
        if r.shape != (3, 3) or not np.isfinite(r).all():
            raise ValueError('Public orientation unavailable for rotation annotation')
        # Match RPent's actual definitions: pitch is rotation about world X,
        # atan2(R[1,2], -R[2,2]); yaw is atan2(R[1,0], R[0,0]).
        # These are not an Euler-Y pitch convention.
        pitch_now = math.atan2(r[1, 2], -r[2, 2])
        yaw_now = math.atan2(r[1, 0], r[0, 0])
        pitch_delta = (float(args['target_pitch']) - pitch_now if args.get('target_pitch') is not None
                       else float(args.get('delta_pitch', 0)))
        yaw_delta = (float(args['target_yaw']) - yaw_now if args.get('target_yaw') is not None
                     else float(args.get('delta_yaw', 0)))
        pitch_delta = (pitch_delta + math.pi) % (2*math.pi) - math.pi
        yaw_delta = (yaw_delta + math.pi) % (2*math.pi) - math.pi
        if name == 'rotate_wrist': pitch_delta = 0.
        if name == 'rotate_pitch': yaw_delta = 0.
        delta = np.array([pitch_delta, 0., yaw_delta])
        r = Rotation.from_rotvec(delta).as_matrix() @ r
        metadata['rotation_world_axis_angle_rad'] = delta.tolist()
        metadata['orientation_annotation_scope'] = 'commanded rotation direction/reference, not a guarantee of reached orientation'
        metadata['rotation_parameters'] = plain(args)
        if b is not None:
            for axis, c in zip(r.T, ((230, 60, 60), (70, 220, 70), (70, 100, 240))):
                end = project(goal + .035*axis)
                if end is not None: draw.line((b, end), fill=c, width=1)
            # A thin arc follows a reference axis; its endpoint arrow shows sign.
            original_r = Rotation.from_quat(quat).as_matrix() if quat is not None else np.asarray(robot['eef_rotation'])
            axis = original_r[:, 2] if name == 'rotate_pitch' else original_r[:, 0]
            curve = [project(goal + .045*(Rotation.from_rotvec(delta*f).as_matrix()@axis))
                     for f in np.linspace(0, 1, 13)]
            curve = [p for p in curve if p is not None]
            if len(curve) > 1:
                draw.line(curve, fill=colour, width=1)
                d = np.array(curve[-1])-curve[-2]
                if np.linalg.norm(d) > .1:
                    d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
                    for sign in (-1, 1):
                        draw.line((curve[-1], tuple(np.array(curve[-1])-5*d+sign*2*n)), fill=colour, width=1)
    return np.asarray(image), metadata


class EvidenceBuilder:
    def __init__(self, output):
        self.output = Path(output)
        self.images = self.output / 'reviewer_images'
        self.images.mkdir(parents=True, exist_ok=True)
        self.first_frame = None

    def _save(self, rgb, tag):
        # Ordinary duplicate-image avoidance, not source/object custody.
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        path = self.images / f'{tag}_{digest[:16]}.png'
        if not path.exists():
            Image.fromarray(rgb).save(path)
        return str(path.resolve())

    def capture(self, packet, robot, instruction, gate, candidate, event=None, history=()):
        seq = packet.get('frame_id')
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError('Same-step public frame identifier required')
        cameras = packet.get('cameras', {})
        current = []
        permitted = {k: plain(robot[k]) for k in ROBOT_KEYS if k in robot}
        if 'eef_pos' not in permitted or 'holding_estimate' not in permitted:
            raise ValueError('Current public proprio/holding description required')
        for name in VIEWS:
            camera = cameras.get(name)
            if not isinstance(camera, dict) or camera.get('frame_id') != seq:
                raise ValueError('Missing or mismatched public RGB frame')
            rgb = np.asarray(camera.get('rgb'))
            if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
                raise ValueError('Public gate RGB must be uint8 640x480')
            row = {'view': name, 'frame_id': seq, 'rgb_path': self._save(rgb, name),
                   'width': 640, 'height': 480}
            if gate == 'G2':
                drawn, metadata = annotate(rgb, camera, permitted, candidate)
                row['path_overlay'] = metadata
                if drawn is not None:
                    row['annotated_rgb_path'] = self._save(drawn, name+'_path')
            current.append(row)
        if self.first_frame is None:
            self.first_frame = dict(current[0])
        # Summaries are authored by the executed-tool log, not by the planner's claims.
        recent = plain(list(history)[-5:])
        result = {'gate': gate, 'shared_policy': policy(), 'trusted_instruction': instruction,
                  'reviewer_prompt_version': PROMPT_VERSION,
                  'review_scope': review_scope(gate, candidate, event),
                  'frame_id': seq, 'robot_state': permitted, 'current_images': current,
                  'first_workspace_image': dict(self.first_frame), 'recent_executed_actions': recent,
                  'candidate': plain(candidate)}
        if gate == 'G2': result['candidate_description'] = candidate_description(candidate, permitted)
        if gate == 'G3':
            # Keep the actual parent for traceability, but remove its eventual
            # target from the primary item to score. The binding/hash used by
            # dispatch still includes the unchanged parent and exact control.
            result['parent_primitive_context'] = {
                k: plain(candidate[k]) for k in ('name', 'arguments') if k in candidate}
            result['candidate'] = {
                k: plain(candidate[k]) for k in ('control', 'events') if k in candidate}
            result['event'] = plain(event)
        json.dumps(result, allow_nan=False)
        return result
