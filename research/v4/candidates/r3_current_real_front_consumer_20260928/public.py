"""Existing SAM RGB-D boxes/intent, with a rotation-invariant door heuristic."""
from dataclasses import replace
from math import dist
import numpy as np
from research.v4.candidates.r3_family_method_config_20260927 import public as parent
from research.v4.candidates.r3_family_door_intent_v2_20260927.public import is_public_close_intent
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox, PublicWorkspace
from .signals import NoPublicMask

MIN_DOOR_BODY_SEPARATION_M = .18  # Original mask-disambiguation heuristic, not a hazard threshold.
proposal = parent.proposal


def center(box):
    return tuple((lo+hi)/2 for lo, hi in zip(box.lower, box.upper))


def door_pair_separated(door, oven):
    if (not isinstance(door, PublicBox) or not isinstance(oven, PublicBox) or
        door.name != 'microwave_door' or oven.name != 'microwave' or
        door.source != 'public_rgbd_perception' or oven.source != 'public_rgbd_perception'):
        return False
    a, b = center(door), center(oven)
    return bool(np.isfinite(a).all() and np.isfinite(b).all() and
                dist(a[:2], b[:2]) >= MIN_DOOR_BODY_SEPARATION_M)


def locate(b, prompt, workspace):
    if not is_public_close_intent(prompt):
        return parent.locate(b, prompt, workspace)
    door, oven = workspace.box('microwave_door'), workspace.box('microwave')
    if not door_pair_separated(door, oven):
        return None
    return 'microwave', center(door)


class DetectionClient:
    """Keep RPC/schema errors distinct from a legitimate decoded no-detection."""
    def __init__(self, original):
        self.original = original

    def segment(self, *args, **kwargs):
        from rpent.utils.sam3_client import Sam3Result
        result = self.original.segment(*args, **kwargs)
        if not isinstance(result, Sam3Result) or type(result.found) is not bool:
            raise TypeError('Expected actual decoded Sam3Result')
        if result.found is False:
            if result.mask is not None:
                raise ValueError('No-detection result has a contradictory mask')
            raise NoPublicMask()
        return result


def workspace(b, entry, sam_client, logs):
    """Same two public captures, existing world-box helper, no private fallback."""
    try:
        packet = entry.scene.capture_agent()
        if set(packet['cameras']) != {'agentview', 'robot0_eye_in_hand'}:
            raise ValueError('Exactly two public RGB-D views required')
        frame = packet['frame_id']
        if type(frame) is not int or frame < 0:
            raise ValueError('Public frame acquisition identifier invalid')
        for cam in packet['cameras'].values():
            rgb, depth = np.asarray(cam['rgb']), np.asarray(cam['depth_m'])
            k, t = np.asarray(cam['intrinsics']), np.asarray(cam['extrinsics'])
            if (cam['frame_id'] != frame or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or
                depth.shape != (480, 640) or not np.isfinite(depth).all() or not (depth > 0).all() or
                k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or
                not np.isfinite(t).all() or abs(np.linalg.det(k)) < 1e-9):
                raise ValueError('Required public 640 RGB-D/calibration invalid')
        boxes = []
        client = DetectionClient(sam_client)
        for name, prompt in [('microwave', 'microwave'), ('microwave_door', 'open microwave door panel'),
                             (b.source, b.source_prompt)]:
            detected = []
            for view in ('agentview', 'robot0_eye_in_hand'):
                try:
                    box, details = sam_public_box(name=name, prompt=prompt,
                        camera=packet['cameras'][view], sam3_client=client, min_score=.2)
                    detected.append(box)
                    logs.append(dict(frame_id=frame, camera=view, **details))
                except NoPublicMask:
                    logs.append(dict(frame_id=frame, camera=view, name=name,
                                     cause='decoded_sam_no_detection'))
            if detected:
                boxes.append(replace(detected[0],
                    lower=tuple(min(x.lower[i] for x in detected) for i in range(3)),
                    upper=tuple(max(x.upper[i] for x in detected) for i in range(3))))
        return PublicWorkspace(entry.adapter.capture_commit_evidence(),
            tuple(float(x) for x in packet['robot_state']['eef_pos']), tuple(boxes))
    except Exception as exc:
        # Original toolkit catches unavailable proposals. Latch a real fault so
        # that catch cannot silently turn RPC/calibration/schema failure into a miss.
        entry.adapter._latch('public_workspace', exc)
        raise
