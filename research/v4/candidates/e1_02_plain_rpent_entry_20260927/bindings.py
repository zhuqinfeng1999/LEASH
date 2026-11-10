"""Scoped neutral public evidence over existing SAM and proposal helpers."""
from dataclasses import replace
import math
import numpy as np
from research.v4.candidates.e1_future_public_binding_20260927 import public_names as names
from research.v4.candidates.e2_cat_corrected_method_20260926.phase_adapter import LandingBoundary
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.accelerated.e1_gate_boundary import requires_descent_review
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicWorkspace

INSTANCE = 'E1-02'
SOURCE = 'cleaver'
TARGET = 'cutting board'
PROMPTS = names.perception_queries(INSTANCE)
NAMES = frozenset(PROMPTS)


def task_names(decision):
    return names.bind_task(INSTANCE, decision)


def motion_names(raw):
    return names.bind_motion(INSTANCE, raw)


class PublicWorkspaceFactory:
    """Only declared capture_agent + adapter evidence + frozen SAM client.

    The union uses both existing public RGB-D views when segmentation works.
    No color fallback, grasp site, evaluator state, or fixed silhouette size.
    A box is a visible-cloud bound; hidden shape/occlusion remains a limitation.
    """
    def __init__(self, scene, adapter, sam_client):
        self.scene, self.adapter, self.sam_client = scene, adapter, sam_client
        self.log = []

    def __call__(self):
        packet = self.scene.capture_agent()
        if set(packet['cameras']) != {'agentview', 'robot0_eye_in_hand'}:
            raise ValueError('exact two public RGB-D views required')
        frame = packet['cameras']['agentview'].get('frame_id')
        if not isinstance(frame, int) or frame < 0 or packet['cameras']['robot0_eye_in_hand'].get('frame_id') != frame:
            raise ValueError('public frame mismatch')
        for camera in packet['cameras'].values():
            rgb = np.asarray(camera['rgb']); depth = np.asarray(camera['depth_m'])
            k = np.asarray(camera['intrinsics']); e = np.asarray(camera['extrinsics'])
            if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or
                depth.shape != (480, 640) or not np.isfinite(depth).all() or
                not np.all(depth > 0) or k.shape != (3, 3) or e.shape != (4, 4) or
                not np.isfinite(k).all() or not np.isfinite(e).all() or
                abs(np.linalg.det(k)) < 1e-9):
                raise ValueError('required public 640x480 RGB-D/calibration invalid')
        boxes = []
        for noun in PROMPTS:
            detected = []
            for view in ('agentview', 'robot0_eye_in_hand'):
                try:
                    box, audit = names.public_box(INSTANCE, noun,
                        camera=packet['cameras'][view], sam3_client=self.sam_client)
                    detected.append(box)
                    self.log.append({'frame_id': frame, 'camera': view, **audit})
                except Exception as exc:
                    self.log.append({'frame_id': frame, 'camera': view, 'name': noun,
                                     'error': f'{type(exc).__name__}: {exc}'})
                    if not (isinstance(exc, ValueError) and
                            str(exc) == 'SAM3 did not locate the named public entity'):
                        raise  # Malformed input/calibration/SDK errors are not low confidence.
            if detected:
                boxes.append(replace(detected[0],
                    lower=tuple(min(b.lower[i] for b in detected) for i in range(3)),
                    upper=tuple(max(b.upper[i] for b in detected) for i in range(3))))
        eef = tuple(float(x) for x in packet['robot_state']['eef_pos'])
        return PublicWorkspace(self.adapter.capture_commit_evidence(), eef, tuple(boxes))

    @staticmethod
    def locate_vla_target(prompt, workspace):
        return names.locate_goal(INSTANCE, prompt, workspace)


class ObservedPayloadSweep:
    gripper_radius_m = .035
    def __init__(self, workspace):
        box = workspace.box(SOURCE)
        if box is None or box.source != 'public_rgbd_perception':
            raise ValueError('held cleaver full public box unavailable')
        # Reuse actual existing corner/EEF computation. Do NOT reuse its E2
        # landing predicate, pet policy, or evaluator values.
        boundary = LandingBoundary()
        boundary.update_workspace(None, box, empty=True)
        boundary.classify(held=False, xyz=workspace.eef_xyz, delta=(0., 0., 0.))
        boundary.classify(held=True, xyz=workspace.eef_xyz, delta=(0., 0., 0.))
        radius = boundary.radius_m
        if radius is None or not math.isfinite(radius) or radius <= 0:
            raise ValueError('public payload bound invalid')
        self.total_radius_m = max(self.gripper_radius_m, radius)
        self.held_extra_radius_m = self.total_radius_m - self.gripper_radius_m


def proposal(name, arguments, workspace, locate_vla_target):
    hold = workspace.evidence.robot.hold_estimate
    if hold not in {'held', 'empty'}:
        raise ValueError('public hold state unknown')
    held = hold == 'held'
    sweep = ObservedPayloadSweep(workspace) if held else None
    result = proposal_for_p3(name, arguments, workspace, locate_vla_target,
                             held_sweep=sweep)
    entity = SOURCE if held else result.operated_entity
    if not held and name in {'move_to', 'move_pose', 'set_gripper'}:
        box = workspace.box(SOURCE)
        point = result.motion_waypoints[-1] if result.motion_waypoints else workspace.eef_xyz
        if box is not None:
            nearest = tuple(min(max(p, lo), hi) for p, lo, hi in zip(point, box.lower, box.upper))
            if math.dist(point, nearest) <= .055:
                entity = SOURCE
    # VLA targets stay source/target-specific; manipulation while already
    # holding the cleaver operates that payload, not the supporting board.
    result = replace(result, operated_entity=entity)
    path = result.motion_waypoints
    if held and len(path) >= 2 and requires_descent_review(
        held_public=True, primitive_name='move_to', current_eef_xyz=path[0],
        proposed_eef_xyz=path[-1]):
        result = replace(result, geometry_ambiguous=True,
            parameters=(*result.parameters, ('phase', 'held_descent_precritical')))
    if held and result.name == 'vla_act':
        result = replace(result, geometry_ambiguous=True)
    return result
