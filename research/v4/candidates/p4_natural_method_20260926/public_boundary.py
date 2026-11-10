"""P4 public geometry adapter over the existing action-gate lifecycle.

Only public two-view RGB-D, calibration and declared robot state enter here.
No simulator model/data, asset metadata, evaluator or target oracle is read.
"""
from dataclasses import replace
from io import BytesIO
import itertools
import math
import re
from unittest.mock import patch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from research.v4.accelerated import e1_gated_toolkit as common
from research.v4.accelerated.public_geometry import world_box_from_mask
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox, PublicWorkspace

PROMPTS = {'iron': 'clothes iron', 'shirt': 'shirt',
           'stand': 'iron stand', 'board': 'ironing board'}
ALIASES = {'iron': 'iron', 'clothes iron': 'iron', 'steam iron': 'iron',
           'shirt': 'shirt', 'clothing': 'shirt', 'cloth': 'shirt',
           'garment': 'shirt', 'stand': 'stand', 'iron stand': 'stand',
           'iron rest': 'stand', 'support stand': 'stand',
           'board': 'board', 'ironing board': 'board'}


def bind_name(value):
    cleaned = re.sub(r'[^a-z0-9]+', ' ', value.casefold()).strip()
    if cleaned not in ALIASES:
        raise ValueError('unbound public P4 entity: ' + repr(value))
    return ALIASES[cleaned]


def bind_task(decision):
    if decision.result != 'PROCEED':
        return decision
    c = decision.constraints
    return replace(decision, constraints=replace(c,
        allowed_operated_entities=frozenset(bind_name(x) for x in c.allowed_operated_entities),
        forbidden_entities=frozenset(bind_name(x) for x in c.forbidden_entities),
        orientation_rules=tuple(replace(x, operated_entity=bind_name(x.operated_entity))
                                for x in c.orientation_rules)))


class PublicWorkspaceReader:
    def __init__(self, scene, adapter, sam3_client):
        self.scene, self.adapter, self.sam = scene, adapter, sam3_client
        self.log = []
        self.static = {}
        self.packet = None

    def __call__(self):
        packet = self.scene.capture_agent()
        if set(packet['cameras']) != {'agentview', 'robot0_eye_in_hand'}:
            raise ValueError('P4 requires exactly two public cameras')
        self.packet = packet
        boxes = []
        for name, prompt in PROMPTS.items():
            if name != 'iron' and name in self.static:
                boxes.append(self.static[name])
                continue
            views = []
            for camera_name in ('agentview', 'robot0_eye_in_hand'):
                camera = packet['cameras'][camera_name]
                try:
                    rgb, depth = np.asarray(camera['rgb']), np.asarray(camera['depth_m'])
                    if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
                        raise ValueError('native 640x480 public RGB-D required')
                    buffer = BytesIO()
                    Image.fromarray(rgb).save(buffer, format='PNG')
                    result = self.sam.segment(buffer.getvalue(), text_prompt=prompt, min_score=.2)
                    if not result.found or result.mask is None or not math.isfinite(float(result.score)) or result.score < .2:
                        raise ValueError('SAM3 did not locate the named public entity')
                    # Local close-placement protocol: 5 mm public depth-box
                    # margin rather than a 3 cm sphere around every object.
                    # This is disclosed uncertainty, not a truth correction.
                    box = world_box_from_mask(name=name, mask=result.mask, depth_m=depth,
                        intrinsic_k=camera['intrinsics'],
                        extrinsic_cam2world=camera['extrinsics'], margin_m=.005)
                    views.append(box)
                    self.log.append({'frame_id': packet['frame_id'], 'camera': camera_name,
                        'name': name, 'prompt': prompt, 'score': result.score,
                        'mask_pixels': int(np.asarray(result.mask).sum()),
                        'box_lower': box.lower, 'box_upper': box.upper,
                        'margin_m': .005, 'source': 'public_rgbd_perception'})
                except Exception as exc:
                    self.log.append({'frame_id': packet['frame_id'], 'camera': camera_name,
                                     'name': name, 'error': str(exc)})
            if views:
                box = replace(views[0], lower=tuple(min(x.lower[i] for x in views) for i in range(3)),
                              upper=tuple(max(x.upper[i] for x in views) for i in range(3)))
                boxes.append(box)
                if name != 'iron':
                    self.static[name] = box
        return PublicWorkspace(self.adapter.capture_commit_evidence(),
            tuple(float(x) for x in packet['robot_state']['eef_pos']), tuple(boxes))

    def locate_vla_target(self, prompt, workspace):
        if not isinstance(prompt, str):
            return None
        choices = [x for x in PROMPTS if re.search(r'\b'+x+r'\b', prompt.casefold())]
        if not choices:
            return None
        name = max(choices, key=lambda x: prompt.casefold().rfind(x))
        box = workspace.box(name)
        return None if box is None else (name, tuple((a+b)/2 for a,b in zip(box.lower,box.upper)))


def configuration_box(zone, payload_lower, payload_upper):
    """Exact translation AABB Minkowski difference in EEF coordinates."""
    return replace(zone,
        lower=tuple(a-b for a,b in zip(zone.lower, payload_upper)),
        upper=tuple(a-b for a,b in zip(zone.upper, payload_lower)))


class PublicMotionReviewer:
    def __init__(self, session):
        self.session = session
        self.public_workspace = None

    def review_motion(self, *, request, workspace, lease, deadline_monotonic_ns):
        # Configuration-space boxes are code-only; reviewer sees original
        # observed entity boxes and the fresh public image evidence.
        raw = dict(self.session.review_motion(request=request,
            workspace=self.public_workspace or workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns))
        raw['forbidden_entities'] = [bind_name(x) for x in raw.get('forbidden_entities', [])]
        return raw


class P4ActionBoundary(common.E1GatedToolkit):
    """Reuse all tested DENY/lease/release/interception behavior unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload_local_corners = None
        self.pending_payload = None
        self.public_workspace = None
        self.geometry_workspace = None
        self.current_tool = None
        self.payload_events = []
        self.last_unheld_span = None

    def _snapshot(self):
        return self.geometry_workspace if self.geometry_workspace is not None else self.workspace()

    def _proposal(self, name, arguments, workspace, locate):
        held = workspace.evidence.robot.hold_estimate == 'held'
        public = self.public_workspace
        proposal = proposal_for_p3(name, arguments, public, locate,
            held_sweep=type('PublicEnvelope', (), {'gripper_radius_m': 0.,
                            'held_extra_radius_m': 0.})() if held else None)
        proposal = replace(proposal, operated_entity='iron' if held or name in
            {'pi0_pick', 'pi0_doubled', 'set_gripper'} else None,
            gripper_radius_m=0. if held else .035, held_object_radius_m=0.)
        if name == 'move_pose' or name in {'rotate_wrist', 'rotate_pitch'}:
            proposal = replace(proposal, geometry_ambiguous=True)
        # Orientation rules remain enforced by the common gate. A payload
        # yaw not explicitly estimated from public geometry remains unknown
        # and fails closed, rather than substituting a simulator quaternion.
        if held and name == 'move_pose' and proposal.placement_xyz is not None:
            proposal = replace(proposal, placement_yaw_deg=None)
        return proposal

    def execute_tool(self, name, arguments):
        if name in common.OBSERVATION_OR_FILE_TOOLS or name == 'finish':
            return super().execute_tool(name, arguments)
        public = self.workspace()
        self.public_workspace = public
        packet = self.workspace.packet
        robot = packet['robot_state']
        held = public.evidence.robot.hold_estimate == 'held'
        eef = np.asarray(public.eef_xyz)
        rotation = Rotation.from_quat(np.asarray(robot['eef_quat'])).as_matrix()
        iron = public.box('iron')
        if not held:
            self.payload_local_corners = None
            self.pending_payload = None
            if iron is not None:
                corners = np.asarray(list(itertools.product(*zip(iron.lower, iron.upper))))
                self.last_unheld_span = np.asarray(iron.upper)-np.asarray(iron.lower)
                # Only a stationary close has a known pre-binding EEF pose.
                # A moving/VLA pickup must be grounded again after attachment;
                # do not pretend its initial far-away pose was the binding pose.
                self.pending_payload = ((corners-eef) @ rotation if name == 'set_gripper'
                    and arguments.get('gripper', -1) > 0 else None)
        elif self.payload_local_corners is None:
            self.payload_local_corners = self.pending_payload
            if self.payload_local_corners is None and iron is not None and self.last_unheld_span is not None:
                span = np.asarray(iron.upper)-np.asarray(iron.lower)
                if np.all(span >= .8*self.last_unheld_span):
                    corners = np.asarray(list(itertools.product(*zip(iron.lower, iron.upper))))
                    self.payload_local_corners = (corners-eef) @ rotation
            self.payload_events.append({'kind': 'latch_public_payload_envelope',
                'frame_id': packet['frame_id'], 'available': self.payload_local_corners is not None})
        transformed = public
        zero_motion = name in {'release', 'set_gripper', 'hold'}
        if held and not zero_motion:
            if self.payload_local_corners is None:
                # Common gate's ambiguous-motion review has no substitute
                # for missing required geometric evidence; deny before move.
                return self._missing_payload(name)
            offsets = self.payload_local_corners @ rotation.T
            lo = np.minimum(offsets.min(axis=0), -.035)
            hi = np.maximum(offsets.max(axis=0), .035)
            if name in {'rotate_wrist', 'rotate_pitch', 'move_pose', 'pi0_pick', 'pi0_doubled'}:
                # Bound arbitrary local rotation by the largest observed
                # corner radius; translation-only calls retain their shape.
                radius = float(np.linalg.norm(offsets, axis=1).max())
                lo, hi = np.full(3, -max(radius,.035)), np.full(3, max(radius,.035))
            transformed = replace(public, visible_entities=tuple(
                configuration_box(box, lo, hi) for box in public.visible_entities))
        self.geometry_workspace = transformed
        self._motion_reviewer.public_workspace = public
        # Scoped function selection reuses the common lifecycle and does not
        # mutate any file or shared global outside this synchronous owner call.
        try:
            with patch.object(common, 'proposal_for_e1', self._proposal), \
                 patch.object(self, 'workspace', lambda: transformed):
                return super().execute_tool(name, arguments)
        finally:
            self.geometry_workspace = None

    def _missing_payload(self, name):
        from research.v3.bridge.b0 import _tool_result
        decision = self.gate._deny('held iron envelope not grounded in public RGB-D',
            'refresh public perception before transport', reviewer_called=False)
        self.proposals.append({'tool': name, 'decision': decision.result, 'reason': decision.reason})
        return _tool_result(name, {'error': decision.reason, 'code': 'action_denied',
                                  'gate': 2, 'constraint': decision.constraint})
