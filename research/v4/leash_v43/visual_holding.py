"""Episode-local public grasp observations; opt-in development fusion.

No environment, assist, evaluator or object-truth access. Acquisition is owned
by the existing public-input boundary. A reviewer assertion is one input, never
the proof of motion or permission to dispatch. Full-corner margins are retained.
"""
from copy import deepcopy
from itertools import product
import os
import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'public-visual-holding-dev1-20261005'
# Fixed before validation: reuse the existing RGB-D box uncertainty margin.
# This is a coarse consistency bound, not a claim of millimetre calibration.
PUBLIC_BOX_MARGIN_M = .03  # selected P1 PublicWorkspaceFactory's existing margin
MOTION_CONSISTENCY_M = 2 * PUBLIC_BOX_MARGIN_M
MIN_RESOLVED_RISE_M = 2 * PUBLIC_BOX_MARGIN_M
ROLE = ('Proprioceptive UNKNOWN means that this channel is inconclusive, not that grasp failure is established. '
        'Use the supplied, sourced visual and executed-action evidence to assess current holding. '
        'Planned actions and tool success flags are not proof. A supported holding estimate may persist '
        'through temporary occlusion while its validity conditions hold, but it does not approve the '
        'current route, orientation, receiving support, or release.')


def enabled():
    return os.environ.get('SAFEVLA_VISUAL_HOLDING', '') == VERSION


def pose(packet):
    state = packet['robot_state']
    xyz = np.asarray(state['eef_pos'], float)
    r = Rotation.from_quat(np.asarray(state['eef_quat'], float)).as_matrix()
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('public EEF position unavailable')
    return xyz, r


def box_corners(box):
    lo, hi = np.asarray(box['lower'], float), np.asarray(box['upper'], float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
            or np.any(hi < lo) or box['source'] != 'public_rgbd_perception'):
        raise ValueError('invalid sourced public box')
    return np.asarray(list(product(*zip(lo, hi))))


class PublicGraspObservations:
    """A small fixed-event buffer, reset only with a new physical episode."""
    def __init__(self, episode):
        self.episode = episode
        self.last = None
        self.before = None
        self.after = []
        self.close_start = None
        self.close_end = None
        self.last_gripper = None
        self.executed_controls = 0
        self.invalidated = None
        self.tool_returns = []

    def executed(self, vector, sequence):
        vector = np.asarray(vector, float)
        grip = 'OPEN' if vector[6] < 0 else 'CLOSE' if vector[6] > 0 else 'HOLD'
        if grip == 'OPEN':
            self.before = None
            self.after = []
            self.close_start = self.close_end = None
            self.invalidated = 'actual OPEN executed'
        elif grip == 'CLOSE':
            if self.last_gripper != 'CLOSE':
                self.before = self.last
                self.after = []
                self.close_start = sequence
                self.invalidated = None
            if not self.after:
                self.close_end = sequence
        self.last_gripper = grip if grip != 'HOLD' else self.last_gripper
        self.executed_controls = sequence

    def observe(self, packet, boxes=(), *, payload_boxes=None, checkpoint=True):
        seq = packet['frame_id']
        if type(seq) is not int or any(c['frame_id'] != seq for c in packet['cameras'].values()):
            raise ValueError('grasp observation contains mixed executed frames')
        existing = next((r for r in self.after if r['sequence'] == seq), None)
        if existing is None and self.last is not None and self.last['sequence'] == seq:
            existing = self.last
        if existing is None:
            existing = dict(sequence=seq, packet=deepcopy(packet), boxes={}, payload_boxes={}, tool_return=None)
        for box in boxes:
            row = dict(name=box.name, lower=list(box.lower), upper=list(box.upper), source=box.source)
            box_corners(row)
            existing['boxes'][box.name] = row
        for box in boxes if payload_boxes is None else payload_boxes:
            row = dict(name=box.name,lower=list(box.lower),upper=list(box.upper),source=box.source)
            box_corners(row)
            existing.setdefault('payload_boxes',{})[box.name] = row
        self.last = existing
        if checkpoint and self.close_start is not None and seq >= self.close_start and not any(r is existing for r in self.after):
            # CLOSE boundary, first verification return, latest verification,
            # current: at most four distinct executed observations.
            self.after.append(existing)
            if len(self.after) > 4:
                self.after = [self.after[0], self.after[1], *self.after[-2:]]
        return existing

    def tool_return(self, log):
        if self.last is not None:
            self.last['tool_return'] = deepcopy(log)

    def rows(self):
        rows = ([self.before] if self.before is not None else []) + self.after
        return list({r['sequence']: r for r in rows}.values())

    def metadata(self, *, include_geometry=True):
        return dict(protocol=VERSION, episode=self.episode,
            close_executed_range=None if self.close_start is None else [self.close_start, self.close_end],
            invalidated=self.invalidated,
            selection='last available before actual CLOSE; CLOSE/verification primitive returns; current',
            motion_consistency_bound_m=MOTION_CONSISTENCY_M,
            resolved_rise_bound_m=MIN_RESOLVED_RISE_M,
            observations=[dict(sequence=r['sequence'],
                evidence_version=f'{self.episode}:executed-frame{r["sequence"]}',
                robot_state={k:np.asarray(v).tolist() for k,v in r['packet']['robot_state'].items()
                    if k in ('eef_pos','eef_quat','gripper_qpos')},
                boxes=list(r['boxes'].values()) if include_geometry else [],
                payload_boxes=list(r.get('payload_boxes',{}).values()) if include_geometry else [],
                tool_return=r['tool_return'], artifact_step=r.get('artifact_step')) for r in self.rows()])

    def history(self):
        from .method_inputs import PublicFrame, PublicHistory
        import json
        frames = []
        for row in self.rows():
            for name, c in row['packet']['cameras'].items():
                frames.append(PublicFrame(name,c['rgb'],c['depth_m'],c['intrinsics'],c['extrinsics'],
                    f'{self.episode}:executed-frame{row["sequence"]}'))
        if not frames:
            return ()
        return (PublicHistory('This episode: actual executed grasp observations, not a planned route.',
            (json.dumps(self.metadata(),allow_nan=False),), keyframes=tuple(frames)),)


ASSESSMENT_KEYS = {'estimate','entity','category','frame_sequences','identity_supported',
    'left_support','comoving','current_contradiction','current_occluded','full_extent_supported','reason'}


def validate_assessment(value):
    if not isinstance(value, dict) or set(value) != ASSESSMENT_KEYS:
        raise ValueError('holding assessment schema mismatch')
    if (value['estimate'] not in ('HELD','UNKNOWN') or value['category'] not in
            ('non_container','container','unknown') or not isinstance(value['entity'],str)
            or not isinstance(value['reason'],str) or not value['reason']
            or not isinstance(value['frame_sequences'],(list,tuple))
            or any(type(s) is not int or s < 0 for s in value['frame_sequences'])
            or any(type(value[k]) is not bool for k in ASSESSMENT_KEYS -
                {'estimate','entity','category','frame_sequences','reason'})):
        raise ValueError('invalid holding assessment values')
    return deepcopy(value)


def supported_record(observations, assessment, previous=None):
    """Require actual closure, public motion, identity and current consistency."""
    a = validate_assessment(assessment)
    if a['current_contradiction']:
        return None, 'current visual contradiction'
    if (a['estimate'] != 'HELD' or not a['identity_supported'] or not a['left_support']
            or not a['comoving'] or not a['entity']):
        return None, 'reviewer did not support the observed grasp relationship'
    if observations.close_start is None or observations.before is None:
        return None, 'actual CLOSE or pre-CLOSE public observation missing'
    refs = set(a['frame_sequences'])
    rows = [r for r in observations.after if r['sequence'] in refs and a['entity'] in r['boxes']]
    current = observations.last
    if current is None or current['sequence'] not in refs:
        return None, 'assessment does not refer to current executed observation'
    if len(rows) < 2 or len({r['sequence'] for r in rows}) < 2:
        return None, 'two distinct sourced object-motion observations missing'
    base = rows[0]
    bxyz, br = pose(base['packet'])
    bc = box_corners(base['boxes'][a['entity']])
    center = bc.mean(axis=0)
    relative_center = br.T @ (center-bxyz)
    relative_corners = []
    checks = []
    for row in rows[1:]:
        xyz, r = pose(row['packet'])
        corners = box_corners(row['boxes'][a['entity']])
        observed = corners.mean(axis=0)
        residual = float(np.linalg.norm(observed-(xyz+r@relative_center)))
        checks.append(dict(sequence=row['sequence'],residual_m=residual,
            object_rise_m=float(observed[2]-center[2]),eef_rise_m=float(xyz[2]-bxyz[2])))
        if residual > MOTION_CONSISTENCY_M:
            return None, 'public object motion conflicts with rigid EEF-relative relationship'
    if not any(c['object_rise_m'] > MIN_RESOLVED_RISE_M and c['eef_rise_m'] >
               MIN_RESOLVED_RISE_M for c in checks):
        return None, 'object rise is unresolved within existing public box uncertainty'
    if a['current_occluded']:
        # One unconfirmed primitive return only, never an indefinite certificate.
        if (previous is None or previous['entity'] != a['entity']
                or previous.get('occluded_returns',0) >= 1):
            return None, 'occlusion has no valid prior association or exceeded one return'
        old = deepcopy(previous)
        old.update(last_sequence=current['sequence'],occluded_returns=1,frame_sequences=sorted(refs))
        return old, 'bounded occlusion retains prior public association'
    if a['entity'] not in current['boxes']:
        return None, 'current object geometry unavailable without a supported occlusion'
    # Numeric motion uses one comparable calibrated camera. Payload extent
    # separately retains the original accepted two-view full-corner unions.
    for row in rows:
        payload = row.get('payload_boxes',{}).get(a['entity'])
        if payload is not None:
            xyz,r = pose(row['packet'])
            relative_corners.extend((r.T @ (box_corners(payload)-xyz).T).T.tolist())
    extent = a['full_extent_supported'] and bool(relative_corners)
    record = dict(protocol=VERSION,entity=a['entity'],category=a['category'],
        source='public_visual_and_executed_records',frame_sequences=sorted(refs),
        close_executed_range=[observations.close_start,observations.close_end],
        last_sequence=current['sequence'],occluded_returns=0,valid=True,
        full_extent_supported=extent,
        # Union of every confirmed full-corner box; retain its original margins
        # and add the fixed observation consistency uncertainty, never shrink.
        relative_corners=relative_corners if extent else None,
        relative_uncertainty_m=MOTION_CONSISTENCY_M,public_motion_checks=checks,
        reason=a['reason'],R_grasp=None)
    return record, 'public temporal association validated; route permission remains independent'
