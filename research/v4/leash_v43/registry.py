"""Two fixed commit events, using only decoded controls and public proprioception.

No scene, evaluator, assist/latch, object pose, image classifier or model is
imported here.  The registry decides *when* to review, never whether it is safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np


class Match(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    UNRESOLVED = "UNRESOLVED"


class Holding(str, Enum):
    HELD = "HELD"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"


RELEASE = "HELD_OBJECT_RELEASE"
POUR = "POUR_ONSET"
VERSION = "leash-v4.3-revised-no-door-v1"


@dataclass(frozen=True)
class EventEntry:
    event_id: str
    version: str
    applicability: str
    required_sources: tuple[str, ...]
    trigger: str
    parameters: str
    dedup_rearm: str
    evidence: str
    commit_conditions: str
    pass_scope: str
    block: str
    known_gaps: str


EVENTS = (
    EventEntry(RELEASE, VERSION, "held or unknown",
               ("decoded gripper request", "calibrated aperture", "executed records"),
               "begin or continue opening without a valid same-release permit",
               "backend gripper sign, aperture in metres; local calibration",
               "consecutive same opening target, no added motion, same primitive/context; finite completion bound",
               "method-permitted fresh evidence", "valid own lease or policy defaults",
               "one transition only", "withhold whole vector and cancel primitive tail",
               "thin/assisted full-close grasps can be indistinguishable from empty"),
    EventEntry(POUR, VERSION, "held/unknown unless own live binding reliably says non-container",
               ("decoded full orientation path", "public EEF pose", "reliable grasp-start EEF pose"),
               "cross 15 degrees; or remain tilted without valid same-phase permit",
               "15 degree onset, strictly below 10 degree rearm; world vertical; method heuristics",
               "same holding/primitive/location/direction/context; end/invalid binding/rearm expires",
               "method-permitted fresh evidence", "valid own lease or policy defaults",
               "continuous same pouring phase only", "withhold whole vector and cancel primitive tail",
               "upright-at-grasp/stable-relative-pose assumption, tracking error and later unobserved hazards"),
)


def rotation_matrix(value: Any) -> np.ndarray:
    r = np.asarray(value, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-6)):
        raise ValueError("public orientation must be an SO(3) local-to-world matrix")
    return r.copy()


def axis_rotation(vector: Any) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("rotation delta must be a finite world-axis angle in radians")
    a = float(np.linalg.norm(v))
    if a == 0:
        return np.eye(3)
    x, y, z = v / a
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(a) * k + (1. - math.cos(a)) * (k @ k)


def tilt_deg(r: np.ndarray, grasp: np.ndarray) -> float:
    vertical = np.array([0., 0., 1.])
    u = r @ grasp.T @ vertical
    return math.degrees(math.acos(float(np.clip(u @ vertical, -1., 1.))))


def path_tilt(r: np.ndarray, grasp: np.ndarray, delta: np.ndarray) -> tuple[float, float, bool]:
    """Exact extrema for the backend's fixed world-axis rotation segment.

    This is control-semantic geometry, not prediction of an object's trajectory.
    Extrema include intermediate orientations (not just the endpoint).
    """
    vertical = np.array([0., 0., 1.])
    u = r @ grasp.T @ vertical
    a = float(np.linalg.norm(delta))
    start = tilt_deg(r, grasp)
    if a < 1e-12:
        return start, start, True
    k = delta / a
    c = float((vertical @ k) * (k @ u))
    aa = float(vertical @ u) - c
    bb = float(vertical @ np.cross(k, u))
    phi = math.atan2(bb, aa)
    candidates = [0., a]
    for n in range(math.floor(-phi / math.pi) - 1, math.ceil((a - phi) / math.pi) + 2):
        t = phi + n * math.pi
        if 0. < t < a:
            candidates.append(t)
    z_values = [aa * math.cos(t) + bb * math.sin(t) + c for t in candidates]
    peak = math.degrees(math.acos(float(np.clip(min(z_values), -1., 1.))))
    end = tilt_deg(axis_rotation(delta) @ r, grasp)
    # d(z projection)/dt >= 0 is equivalent to non-increasing tilt.
    derivative_points = [0., a]
    offset = math.atan2(-aa, bb)
    for n in range(-4, math.ceil(a / math.pi) + 4):
        t = offset + n * math.pi
        if 0. < t < a:
            derivative_points.append(t)
    decreasing = min(-aa * math.sin(t) + bb * math.cos(t) for t in derivative_points) >= -1e-10
    return end, peak, decreasing


@dataclass(frozen=True)
class Proprio:
    aperture_m: float | None
    eef_xyz: np.ndarray | None
    eef_rotation: np.ndarray | None
    executed_sequence: int


@dataclass(frozen=True)
class GripperCalibration:
    """Explicit measured/local mechanical calibration; never assume empty.

    Assist can retain a thin object at the closed hard stop.  Such installations
    MUST keep closed_gap_proves_empty=False, even if empty closure is calibrated.
    """
    source: str
    empty_closed_gap_m: float
    uncertainty_margin_m: float
    open_complete_gap_m: float
    stable_samples: int
    stable_gap_range_m: float
    closed_gap_proves_empty: bool = False
    close_completion_controls: int = 10

    def __post_init__(self):
        vals = (self.empty_closed_gap_m, self.uncertainty_margin_m,
                self.open_complete_gap_m, self.stable_gap_range_m)
        if (not self.source or not np.isfinite(vals).all() or min(vals) < 0
                or self.uncertainty_margin_m <= 0 or self.stable_samples < 2
                or type(self.close_completion_controls) is not int or self.close_completion_controls < 1
                or self.open_complete_gap_m <= self.empty_closed_gap_m + self.uncertainty_margin_m):
            raise ValueError("explicit local gripper calibration is required")


class HoldingEstimator:
    def __init__(self, calibration: GripperCalibration, *, grasp_assumption: str = "unverified",
                 capture_grasp_pose_on_close: bool = False):
        if grasp_assumption not in {"verified_upright", "unverified", "known_violated", "recorded_development"}:
            raise ValueError("invalid grasp assumption")
        self.calibration = calibration
        self.grasp_assumption = grasp_assumption
        self.capture_grasp_pose_on_close = capture_grasp_pose_on_close
        self.kind = Holding.UNKNOWN
        self.last_gripper: str | None = None
        self.grasp_rotation: np.ndarray | None = None
        self.generation = 0
        self._grip_active = False
        self._samples: list[tuple[int, float, str | None]] = []
        self._sequence = -1
        self._close_command_count = 0
        self._executed_close_rotation: np.ndarray | None = None
        self.initial_occupancy = None
        self._initial_empty_active = False
        self.visual_record = None
        self.visual_updates = []

    @property
    def kind(self):
        # R3 (2026-10-06): public visual assessment is diagnostic evidence,
        # not a second state estimator or an admission prerequisite. UNKNOWN
        # still triggers possible-holding commit checks in the registry.
        return self._proprio_kind

    @kind.setter
    def kind(self, value):
        self._proprio_kind = value

    @property
    def holding_proprio(self):
        return self._proprio_kind

    def invalidate_visual(self, reason):
        # A current contradiction must remain distinguishable from the known
        # OPEN's expected posterior invalidation, including an already-cleared
        # visual record. This is state semantics, not an extra evidence log.
        self.last_visual_invalidation_reason = reason
        if self.visual_record is not None:
            self.visual_updates.append(dict(estimate='UNKNOWN',reason=reason,
                previous_entity=self.visual_record['entity']))
            self.visual_record = None

    def accept_visual(self, observations, assessment):
        from .visual_holding import supported_record
        record, reason = supported_record(observations, assessment, self.visual_record)
        if record is None:
            self.invalidate_visual(reason)
        else:
            self.visual_record = record
            # Retain the unchanged temporal diagnostic calculation, including
            # its 60 mm rule, without synthesizing an executed grip or permit.
        self.visual_updates.append(dict(sequence=observations.last['sequence'],
            holding_proprio=self.holding_proprio.value,holding_estimate=self.kind.value,
            visual_record=record,reason=reason))
        return record

    def initialize_empty(self, fact, p: Proprio, *, episode):
        """Consume trusted initializer evidence once, never an OPEN record."""
        from .initial_occupancy import validate_initial_fact
        checked = validate_initial_fact(fact, episode=episode)
        if (self.initial_occupancy is not None or self._sequence != -1
                or self.last_gripper is not None or self._samples
                or p.executed_sequence != 0 or p.aperture_m is None
                or not math.isfinite(p.aperture_m) or p.aperture_m < 0):
            raise ValueError('initial occupancy cannot reset an active estimator')
        self.initial_occupancy = checked
        self._initial_empty_active = True
        self.kind = Holding.EMPTY

    def observe(self, p: Proprio) -> None:
        gap = p.aperture_m
        if gap is None or not math.isfinite(gap) or gap < 0:
            self.invalidate_visual('public aperture missing or invalid')
            self._initial_empty_active = False
            self.kind = Holding.UNKNOWN
            return
        if p.executed_sequence < self._sequence:
            raise ValueError("public executed sequence regressed")
        if p.executed_sequence == self._sequence:
            return  # Rendering again cannot fabricate stability samples.
        self._sequence = p.executed_sequence
        if self._initial_empty_active:
            # Passive reads do not fabricate execution or stability samples.
            self.kind = Holding.EMPTY
            return
        self._samples.append((p.executed_sequence, float(gap), self.last_gripper))
        self._samples = self._samples[-self.calibration.stable_samples:]
        stable = (len(self._samples) == self.calibration.stable_samples
                  and max(x[1] for x in self._samples) - min(x[1] for x in self._samples)
                  <= self.calibration.stable_gap_range_m)
        if not stable:
            self.kind = Holding.UNKNOWN
            return
        targets = {x[2] for x in self._samples}
        if targets == {"OPEN"} and gap >= self.calibration.open_complete_gap_m:
            self.invalidate_visual('actual completed opening')
            self.kind = Holding.EMPTY
            self._grip_active = False
            self.grasp_rotation = None
            self._executed_close_rotation = None
        elif targets == {"CLOSE"} and self._close_command_count >= self.calibration.close_completion_controls:
            closed = self.calibration.empty_closed_gap_m
            margin = self.calibration.uncertainty_margin_m
            if gap > closed + margin:
                self.kind = Holding.HELD
                if not self._grip_active:
                    self.generation += 1
                    self._grip_active = True
                    # Public baseline belongs to an actually executed CLOSE,
                    # not a later stability sample or a proposed command.
                    # Uprightness remains a separate requirement in classify.
                    self.grasp_rotation = (None if self._executed_close_rotation is None
                                           else self._executed_close_rotation.copy())
            elif self.calibration.closed_gap_proves_empty and gap < closed + margin * .5:
                self.kind = Holding.EMPTY
                self.grasp_rotation = None
                self._grip_active = False
            else:
                self.kind = Holding.UNKNOWN
        else:
            self.kind = Holding.UNKNOWN

    def record_command(self, gripper: str, at_execution: Proprio | None = None) -> None:
        """Call only after a known environment send; capture can fail afterward."""
        if gripper in {"OPEN", "CLOSE"}:
            if gripper == 'OPEN' or (gripper == 'CLOSE' and self.last_gripper != 'CLOSE'):
                self.invalidate_visual('actual OPEN or new CLOSE executed')
            if gripper == 'CLOSE' and self._initial_empty_active:
                self._initial_empty_active = False
                self.kind = Holding.UNKNOWN
            if gripper == "CLOSE" and self.last_gripper != "CLOSE":
                self._executed_close_rotation = (
                    None if at_execution is None or at_execution.eef_rotation is None
                    else rotation_matrix(at_execution.eef_rotation))
                if self.capture_grasp_pose_on_close:
                    # Confidence protocol: preserve the actual executed CLOSE
                    # baseline even when occupancy remains UNKNOWN. This is a
                    # public pose record, not evidence that a grasp succeeded.
                    self.grasp_rotation = (None if self._executed_close_rotation is None
                                           else self._executed_close_rotation.copy())
            elif gripper == "OPEN":
                self._executed_close_rotation = None
            self._close_command_count = (self._close_command_count + 1
                                         if gripper == "CLOSE" and self.last_gripper == "CLOSE"
                                         else 1 if gripper == "CLOSE" else 0)
            self.last_gripper = gripper

    def execution_uncertain(self) -> None:
        """An unknown-consumption fault cannot leave a reliable grasp baseline."""
        self._initial_empty_active = False
        self.invalidate_visual('execution consumption uncertain')
        self.kind = Holding.UNKNOWN
        self.grasp_rotation = None
        self._samples.clear()
        # Do not manufacture a new grasp generation from subsequent rendering.
        self._grip_active = True
        self.last_gripper = None
        self._close_command_count = 0
        self._executed_close_rotation = None

    def executed(self, gripper: str, after: Proprio) -> None:
        self.record_command(gripper, after)
        self.observe(after)


@dataclass(frozen=True)
class Command:
    payload: np.ndarray
    gripper: str
    translation: np.ndarray
    rotation_delta: np.ndarray
    position_target: np.ndarray | None
    orientation_target: np.ndarray | None

    @property
    def moving(self) -> bool:
        return bool(np.any(self.translation != 0.) or np.any(self.rotation_delta != 0.))


@dataclass(frozen=True)
class LeaseContext:
    owner: str
    version: str
    conditions: tuple[str, ...]
    valid: bool
    holding_generation: int | None = None
    reliable_non_container: bool = False


@dataclass(frozen=True)
class Context:
    episode: str
    round_id: int
    primitive: int
    binding_version: str
    completion_controls: int


@dataclass
class Permit:
    event: str
    context: Context
    generation: int
    issued_sequence: int
    expires_sequence: int
    gripper: str
    location: np.ndarray | None
    direction: np.ndarray | None
    evidence_version: str
    eef_rotation: np.ndarray | None


@dataclass(frozen=True)
class Classification:
    state: Match
    events: tuple[str, ...]
    reason: str
    current_tilt_deg: float | None = None
    candidate_tilt_deg: float | None = None
    maximum_path_tilt_deg: float | None = None
    # Ordinary execution bookkeeping: which existing permit actually avoided
    # another review. This has no effect on matching or permission validity.
    permit_events: tuple[str, ...] = ()


class CommitEventRegistry:
    entries = EVENTS

    def __init__(self, estimator: HoldingEstimator, *, position_tolerance_m: float,
                 direction_tolerance_rad: float, method: str = "Ours",
                 require_grasp_assumption: bool = True,
                 semantic_unknown_pour: bool = False):
        if position_tolerance_m <= 0 or direction_tolerance_rad <= 0:
            raise ValueError("existing backend completion tolerances required")
        self.estimator = estimator
        self.position_tolerance_m = position_tolerance_m
        self.direction_tolerance_rad = direction_tolerance_rad
        self.method = method
        # The confidence protocol retains the executed public pose baseline,
        # but removes the former geometry/profile qualification prerequisite.
        # Historical paths retain their original default and behaviour.
        self.require_grasp_assumption = require_grasp_assumption
        # Confidence-only: unknown occupancy/grasp history is semantic
        # uncertainty for the reviewer, not missing current robot telemetry.
        # The historical registry retains its original default.
        self.semantic_unknown_pour = semantic_unknown_pour
        self.permits: dict[str, Permit] = {}
        self._semantic_pour_permit = False
        self._known_tilted = False
        self._lease_signature = None

    def invalidate(self) -> None:
        self.permits.clear()
        self._semantic_pour_permit = False

    def _pour_direction(self, cmd):
        # Compare the proposed rotation direction, not the changing absolute
        # tip vector along that rotation. Onset itself still uses the complete
        # current/target pose and R_grasp. Pure yaw is not a tilt direction.
        direction = np.cross(cmd.rotation_delta, np.array([0., 0., 1.]))
        norm = float(np.linalg.norm(direction))
        if norm > 1e-10:
            return direction / norm
        if cmd.orientation_target is None or self.estimator.grasp_rotation is None:
            return None
        direction = cmd.orientation_target @ self.estimator.grasp_rotation.T @ np.array([0., 0., 1.])
        direction[2] = 0.
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-10 else None

    def _valid_permit(self, event, cmd, p, context, direction=None) -> bool:
        permit = self.permits.get(event)
        if (event == POUR and self._semantic_pour_permit
                and self.estimator.grasp_rotation is not None):
            # An actual CLOSE establishes a new public grasp-pose baseline.
            # A pre-grasp possible-pour review cannot authorize that new phase.
            return False
        if (permit is None or permit.context != context
                or permit.generation != self.estimator.generation
                or p.executed_sequence > permit.expires_sequence
                or p.executed_sequence < permit.issued_sequence):
            return False
        if event == RELEASE:
            if (cmd.moving or cmd.gripper != permit.gripper or permit.location is None
                    or p.eef_xyz is None or permit.eef_rotation is None or p.eef_rotation is None):
                return False
            angle = math.acos(float(np.clip((np.trace(p.eef_rotation @ permit.eef_rotation.T) - 1.) / 2., -1., 1.)))
            return bool(np.linalg.norm(p.eef_xyz - permit.location) <= self.position_tolerance_m
                        and angle <= self.direction_tolerance_rad)
        if (permit.location is None or p.eef_xyz is None or cmd.position_target is None
                or np.linalg.norm(p.eef_xyz - permit.location) > self.position_tolerance_m
                or np.linalg.norm(cmd.position_target - permit.location) > self.position_tolerance_m
                or direction is None or permit.direction is None):
            return False
        return bool(math.acos(float(np.clip(direction @ permit.direction, -1., 1.)))
                    <= self.direction_tolerance_rad)

    def classify(self, cmd: Command, p: Proprio, context: Context,
                 lease: LeaseContext | None) -> Classification:
        self.estimator.observe(p)
        signature = None if lease is None else (lease.owner, lease.version, lease.valid,
                                                lease.holding_generation, lease.reliable_non_container,
                                                lease.conditions)
        if signature != self._lease_signature:
            self.invalidate()
            self._lease_signature = signature
        if cmd.moving or cmd.gripper == "CLOSE":
            self.permits.pop(RELEASE, None)
        matched: list[str] = []
        continued: list[str] = []
        unresolved: list[str] = []
        current = end = peak = None
        if self.estimator.kind == Holding.EMPTY:
            self.invalidate()
            self._known_tilted = False
            return Classification(Match.NO_MATCH, (), "calibrated empty/open-complete gripper")
        # Relevant release unknown semantics means MATCH, not a technical failure.
        opening = (cmd.gripper == "OPEN" or
                   (cmd.gripper == "HOLD" and self.estimator.last_gripper == "OPEN"))
        if opening:
            if p.aperture_m is None or not math.isfinite(p.aperture_m):
                unresolved.append("opening requires available public gripper aperture")
            elif not self._valid_permit(RELEASE, cmd, p, context):
                matched.append(RELEASE)
            else:
                continued.append(RELEASE)
        own_live = (lease is not None and lease.valid and lease.owner == self.method
                    and lease.version == context.binding_version
                    and (lease.holding_generation is None
                         or lease.holding_generation == self.estimator.generation))
        non_container = bool(own_live and lease.reliable_non_container)
        rotating = bool(np.any(cmd.rotation_delta != 0.))
        # No new rotation and no recorded tilted phase: missing unrelated axis
        # semantics must not turn pure gripper closure into a pour event.
        relevant = rotating or self._known_tilted
        if not non_container and self.estimator.grasp_rotation is not None and p.eef_rotation is not None:
            current = tilt_deg(p.eef_rotation, self.estimator.grasp_rotation)
            relevant = relevant or current >= 15.
        if not non_container and relevant:
            unknown_baseline = (self.semantic_unknown_pour
                and self.estimator.kind == Holding.UNKNOWN
                and self.estimator.grasp_rotation is None
                and p.aperture_m is not None and math.isfinite(p.aperture_m)
                and p.aperture_m >= 0
                and p.eef_rotation is not None)
            if unknown_baseline:
                # No invented R_grasp, tilt number, EMPTY or HELD state.
                # Ask about this exact possible transition with current RGB
                # and actual public pose. Only the usual finite permit may
                # avoid another review; otherwise conservatively MATCH.
                direction = self._pour_direction(cmd)
                if not self._valid_permit(POUR, cmd, p, context, direction):
                    matched.append(POUR)
                else:
                    continued.append(POUR)
            elif (p.aperture_m is None or not math.isfinite(p.aperture_m)
                    or p.eef_rotation is None or self.estimator.grasp_rotation is None
                    or (self.require_grasp_assumption and
                        self.estimator.grasp_assumption not in {"verified_upright", "recorded_development"})):
                unresolved.append("relevant rotation/tilted phase lacks reliable public grasp pose basis")
            else:
                current = tilt_deg(p.eef_rotation, self.estimator.grasp_rotation)
                end, peak, decreasing = path_tilt(p.eef_rotation, self.estimator.grasp_rotation, cmd.rotation_delta)
                # A predictive onset permit can be issued before the physical
                # controller reaches 15 degrees. Rearm only after an observed
                # tilted phase returns below 10, not during its initial rise.
                if self._known_tilted and current < 10.:
                    self.permits.pop(POUR, None)
                    self._known_tilted = False
                direction = self._pour_direction(cmd)
                self._known_tilted = self._known_tilted or current >= 15.
                recovery = (rotating and not np.any(cmd.translation != 0.) and decreasing and end < current)
                if peak >= 15. and not recovery:
                    if not self._valid_permit(POUR, cmd, p, context, direction):
                        matched.append(POUR)
                    else:
                        continued.append(POUR)
        if unresolved:
            return Classification(Match.UNRESOLVED, tuple(matched), "; ".join(unresolved), current, end, peak)
        return Classification(Match.MATCH if matched else Match.NO_MATCH, tuple(matched),
                              "registered conversion requires review" if matched else "no registered conversion or valid same-transition permit",
                              current, end, peak, tuple(continued))

    def grant(self, events: tuple[str, ...], cmd: Command, p: Proprio,
              context: Context, evidence_version: str) -> None:
        for event in events:
            if event == POUR:
                self._semantic_pour_permit = bool(self.semantic_unknown_pour
                    and self.estimator.grasp_rotation is None)
            direction = None
            if event == POUR and ((cmd.orientation_target is not None
                    and self.estimator.grasp_rotation is not None)
                    or (self.semantic_unknown_pour and self.estimator.grasp_rotation is None)):
                direction = self._pour_direction(cmd)
            self.permits[event] = Permit(event, context, self.estimator.generation,
                                         p.executed_sequence, p.executed_sequence + context.completion_controls,
                                         cmd.gripper, None if p.eef_xyz is None else p.eef_xyz.copy(),
                                         direction, evidence_version,
                                         None if p.eef_rotation is None else p.eef_rotation.copy())
