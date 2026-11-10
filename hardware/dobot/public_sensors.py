"""Measured public observations; no dashboard, motion or gripper writes.

The EEF reference is the measured Tool0 flange. Uncalibrated jaw percentage is
never converted to aperture. This reader can inspect a disabled/E-stopped arm;
the separate command transport still owns physical admission.
"""
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image


def flange_rotation(pose_mm_deg):
    x, y, z = np.deg2rad(pose_mm_deg[3:])
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def measured_state(record, age, *, now, gripper=None, grip_center_flange_mm=None):
    pose = np.asarray(record['tcp_actual_mm_deg'], dtype=float)
    joints = np.asarray(record['q_actual_deg'], dtype=float)
    if pose.shape != (6,) or joints.shape != (6,) or not np.isfinite(np.r_[pose, joints]).all():
        raise ValueError('Malformed physical feedback')
    if not np.isfinite(age) or not 0 <= age <= .15:
        raise ValueError('Stale physical feedback')
    aperture = None
    if gripper is not None:
        elapsed = now - gripper['observed_monotonic_s']
        value = gripper.get('aperture_m')
        if (gripper.get('source') != 'calibrated_gripper_device_feedback'
                or not gripper.get('calibration_id') or not 0 <= elapsed <= 2.
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0):
            raise ValueError('Fresh calibrated jaw aperture required')
        aperture = float(value)
    result = dict(source='public_device_feedback', observed_monotonic_s=record['host_monotonic'],
        eef_reference='flange_tool0', eef_pos=(pose[:3] / 1000.).tolist(),
        eef_rotation=flange_rotation(pose).tolist(), gripper_aperture_m=aperture,
        gripper_command_motion_complete=None if gripper is None else gripper.get('command_motion_complete'),
        gripper_motion_feedback_source=None if gripper is None else gripper.get('motion_feedback_source'),
        gripper_measurement=gripper, q_actual_deg=joints.tolist(),
        stationary=not bool(record['running_status']) and record['mode'] in (3, 4, 5),
        # RunQueuedCmd is an algorithm-running flag, not a queue length. It
        # remains 1 on this enabled, stationary CR5. A command owner must supply
        # a separately acknowledged queue barrier before physical admission.
        queued_commands=None, raw_run_queued_cmd=record['run_queued_cmd'],
        fault='controller_error' if record['error_status'] else None,
        enabled=record['enable_status'] == 1, robot_mode=record['mode'],
        controller_timestamp=record['controller_timestamp'])
    if grip_center_flange_mm is not None:
        offset = np.asarray(grip_center_flange_mm, float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Finite measured flange-to-grip offset required')
        result['flange_pos_m'] = result['eef_pos']
        result['eef_pos'] = (pose[:3]/1000. + flange_rotation(pose) @ (offset/1000.)).tolist()
        result['eef_reference'] = 'grip_site'
        result['grip_center_flange_mm'] = offset.tolist()
    return result


def load_rgb_packet(snapshot, *, root, sequence, requested_wall, requested_mono, now_wall, now_mono):
    cameras = {}
    root = Path(root).resolve()
    for view, serial, folder in (
            ('robot0_eye_in_hand', '234322305273', Path(snapshot['wrist_directory'])),
            ('agentview', '030422250259', Path(snapshot['observer_image']).parent)):
        folder = folder.resolve()
        folder.relative_to(root / 'captures')
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        cap = metadata['capture']
        if metadata['info']['serial_number'] != serial:
            raise ValueError('Wrong physical camera identity')
        if cap['timestamp_domain'] not in ('timestamp_domain.global_time', 'timestamp_domain.system_time'):
            raise ValueError('Camera does not provide a host-referenced timestamp')
        stamp = cap['color_timestamp_ms'] / 1000.
        if not np.isfinite(stamp) or not requested_wall <= stamp <= now_wall:
            raise ValueError('Camera frame is stale or in the future')
        mono = requested_mono + stamp - requested_wall
        if not 0 <= now_mono - mono <= 2.:
            raise ValueError('Camera frame expired during capture')
        with Image.open(folder / 'color.png') as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        intrinsics = cap.get('color_intrinsics', {})
        width, height = intrinsics.get('width', 640), intrinsics.get('height', 480)
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError('Invalid recorded RGB dimensions')
        if rgb.shape != (height, width, 3):
            raise ValueError('RGB dimensions disagree with recorded camera intrinsics')
        cameras[view] = dict(rgb=rgb, frame_id=sequence, capture_monotonic_s=mono,
            native_image_size=[height, width],
            device_frame_number=cap['color_frame_number'], device_timestamp_ms=cap['color_timestamp_ms'],
            serial=serial, source_path=str(folder / 'color.png'))
    skew = abs(cameras['agentview']['capture_monotonic_s'] - cameras['robot0_eye_in_hand']['capture_monotonic_s'])
    if skew > .15:
        raise ValueError('Public RGB pair skew exceeds declared 150ms')
    return dict(frame_id=sequence, cameras=cameras, rgb_pair_skew_s=skew,
        eef_reference='flange_tool0', source='current_serial_specific_physical_rgb')


def reviewer_rgb_packet(packet):
    """Letterbox only gate RGB to the frozen 640x480 contract; keep native data."""
    cameras = {}
    for name, camera in packet['cameras'].items():
        rgb = np.asarray(camera['rgb'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
            raise ValueError('Expected uint8 RGB for reviewer image')
        height, width = rgb.shape[:2]
        scale = min(640/width, 480/height)
        target = (round(width*scale), round(height*scale))
        left, top = (640-target[0])//2, (480-target[1])//2
        row = dict(camera)
        if (height, width) != (480, 640):
            canvas = Image.new('RGB', (640, 480))
            canvas.paste(Image.fromarray(rgb).resize(target, Image.Resampling.LANCZOS), (left, top))
            row['rgb'] = np.asarray(canvas).copy()
            if 'intrinsics' in row:
                transform = np.array([[target[0]/width, 0, left], [0, target[1]/height, top], [0, 0, 1.]])
                row['intrinsics'] = (transform @ np.asarray(row['intrinsics'], dtype=float)).tolist()
        row['reviewer_rgb_preprocessing'] = dict(kind='aspect_preserving_letterbox',
            source_size_hw=[height, width],target_size_hw=[480,640],
            scale_xy=[target[0]/width,target[1]/height],padding_left_top=[left,top])
        cameras[name] = row
    return dict(packet, cameras=cameras)


class PublicSensors:
    def __init__(self, feedback, *, session_root, gripper_reader=None,
                 camera_url='http://127.0.0.1:35002/snapshot', snapshot_reader=None):
        self.feedback, self.root = feedback, Path(session_root).resolve()
        self.gripper_reader, self.camera_url = gripper_reader, camera_url
        self.snapshot_reader = snapshot_reader
        self.last_snapshot = None
        profile = self.root / 'calibration/remounted_tool_trial.json'
        self.grip_center_flange_mm = (json.loads(profile.read_text(encoding='utf-8-sig'))['grip_center_flange_mm']
            if profile.is_file() else None)

    def read_public_state(self):
        gripper = self.gripper_reader() if self.gripper_reader is not None else None
        record, age = self.feedback.latest()
        return measured_state(record, age, now=time.monotonic(), gripper=gripper,
            grip_center_flange_mm=self.grip_center_flange_mm)

    def capture_public_rgb(self, sequence):
        before, age = self.feedback.latest()
        measured_state(before, age, now=time.monotonic())
        requested_mono, requested_wall = time.monotonic(), time.time()
        if self.snapshot_reader is None:
            with urllib.request.urlopen(self.camera_url, timeout=5) as response:
                snapshot = json.load(response)
        else:
            snapshot = self.snapshot_reader()
        now_mono, now_wall = time.monotonic(), time.time()
        after, age = self.feedback.latest()
        measured_state(after, age, now=now_mono)
        if any(r['running_status'] for r in (before, after)):
            raise ValueError('Camera capture crossed active execution')
        if np.max(np.abs(np.array(after['q_actual_deg']) - before['q_actual_deg'])) > .02:
            raise ValueError('Arm moved during public RGB capture')
        packet = load_rgb_packet(snapshot, root=self.root, sequence=sequence,
            requested_wall=requested_wall, requested_mono=requested_mono,
            now_wall=now_wall, now_mono=now_mono)
        if self.grip_center_flange_mm is not None: packet['eef_reference']='grip_site'
        self.last_snapshot = snapshot
        for camera in packet['cameras'].values():
            timestamp = camera['capture_monotonic_s']
            nearest = min(self.feedback.records, key=lambda r: abs(r['host_monotonic'] - timestamp))
            if abs(nearest['host_monotonic'] - timestamp) > .15:
                raise ValueError('No public proprioception within 150ms of camera frame')
            if nearest['running_status']:
                raise ValueError('Camera frame acquired while robot active')
        return packet
