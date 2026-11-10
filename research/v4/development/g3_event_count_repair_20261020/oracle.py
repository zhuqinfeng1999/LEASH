"""Independent, offline count oracle for public G3 demo traces.

No production registry, dispatcher, holding estimator or scope helper is used.
This is a declarative acceptance contract, not a safety reviewer or controller.
In particular, PASS/BLOCK is supplied by a CPU fixture or a preserved response;
the oracle never decides that a real candidate is safe.

``evaluate_trace`` accepts JSON-compatible rows with these fields:

* ``sequence`` is the number of controls already executed; ``primitive`` and
  optional ``episode``, ``round_id``, ``binding_version`` identify its context.
* ``holding_before`` is EMPTY / HELD / UNKNOWN, with ``holding_generation`` and
  optional ``grasp_rotation``.  These are explicit public phase facts.  An
  archived trace without every public aperture cannot independently recreate
  its holding history: callers must preserve that limitation and provenance.
* ``aperture_m``, ``eef_position`` and ``eef_rotation`` are public telemetry.
* ``candidate`` has ``gripper`` (OPEN / CLOSE / HOLD), ``position_target``,
  ``orientation_target``, ``rotation_delta`` (world axis-angle radians), and
  optional ``payload`` / ``gripper_target``.  A full vector is withheld on stop.
* Optional ``source`` discloses a finite immutable continuation: ``id``,
  ``kind`` (prepared_translation / prepared_rotation / known_chunk), ``index``,
  ``remaining_controls``, ``position_path``, ``rotation_anchor`` and
  ``orientation_radius_rad``.  The radius describes the disclosed rotation,
  not a replacement for the unchanged 0.06 rad controller allowance.  Optional
  ``expected_position``, ``expected_rotation``, ``expected_payload`` verify the
  current indexed command; ``payloads`` binds a known chunk exactly.  Fixture
  sources may additionally declare ``origin_sequence`` / ``origin_index`` and
  a fixed numeric ``gripper_target``.  Those are provenance constraints:
  repeated/skipped indices or a same-sign changed target cannot become a new
  permission merely by asking again.  Existing windows also enforce exact
  index/sequence lockstep independently of those optional origin fields.
  Prepared sources use the disclosed polyline with its 8mm tube.  A known
  chunk uses the explicitly disclosed axis-aligned extent of its generated
  prefix plus the same 8mm margin, with exact payload/index binding.  Those
  two spatial contracts are deliberately different.
  authors must derive this contract from the immutable prepared descriptor,
  never from actual review/grant records or a production coverage decision.
* ``review_decision`` is PASS / BLOCK ONLY when a review is independently
  required.  Missing/invalid responses stop; no implicit PASS is manufactured.
  Optional ``dispatched`` checks actual consumption, and ``owner_cancelled``
  always forbids consumption, including after a late PASS.

The output contains required *positions and event sets*, not just a count.
``compare_reviews`` reports extra / missed / wrong-event calls separately.
Source binding, public phase inputs and response provenance are assumptions of
this small oracle.  It is not an image judge, pose predictor, complete backend
decoder, simulator, causal performance comparison or formal safety proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Iterable

import numpy as np


RELEASE = "HELD_OBJECT_RELEASE"
POUR = "POUR_ONSET"


@dataclass(frozen=True)
class OracleConfig:
    position_tolerance_m: float = .008
    direction_tolerance_rad: float = .06
    onset_deg: float = 15.
    rearm_deg: float = 10.
    open_complete_gap_m: float = .075
    stable_samples: int = 3
    stable_gap_range_m: float = .001

    def __post_init__(self):
        if (self.position_tolerance_m != .008 or self.direction_tolerance_rad != .06
                or self.onset_deg != 15. or self.rearm_deg != 10.):
            raise ValueError("this acceptance oracle fixes the authorized 8mm/.06rad/15/10 contract")
        if self.stable_samples < 2 or self.open_complete_gap_m <= 0 or self.stable_gap_range_m <= 0:
            raise ValueError("explicit positive public OPEN calibration required")


@dataclass
class OracleResult:
    reviews: list[dict] = field(default_factory=list)
    technical_stops: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    rows_checked: int = 0

    @property
    def required_calls(self):
        return len(self.reviews)

    def as_dict(self):
        return {**asdict(self), "required_calls": self.required_calls}


def _vec(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("finite public 3-vector required")
    return a


def _rot(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (3, 3) or not np.isfinite(a).all()
            or not np.allclose(a.T @ a, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(a), 1., atol=1e-6)):
        raise ValueError("proper public SO(3) orientation required")
    return a


def _angle(a, b):
    return math.acos(float(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.)))


def _line_distance(x, points):
    """Independent scalar segment projection, including segment end caps."""
    pts = [_vec(p) for p in points]
    if not pts:
        raise ValueError("disclosed position path is empty")
    best = float(np.linalg.norm(x - pts[0]))
    for a, b in zip(pts, pts[1:]):
        axis = b - a
        length2 = float(axis @ axis)
        fraction = 0. if length2 == 0 else min(1., max(0., float((x - a) @ axis) / length2))
        best = min(best, float(np.linalg.norm(x - (a + fraction * axis))))
    return best


def _tilt_and_segment(rotation, grasp, delta):
    """Tilt extrema of a world-axis rotation, using its sinusoid coefficients."""
    up = rotation @ grasp.T @ np.array([0., 0., 1.])
    current = math.degrees(math.acos(float(np.clip(up[2], -1., 1.))))
    theta = float(np.linalg.norm(delta))
    if theta < 1e-14:
        return current, current, current, True
    axis = delta / theta
    constant = axis[2] * float(axis @ up)
    cosine = up[2] - constant
    sine = float(np.cross(axis, up)[2])
    values = [constant + cosine, constant + cosine * math.cos(theta) + sine * math.sin(theta)]
    # d(z)/dt = -A sin(t)+B cos(t); all stationary points are pi apart.
    zero = math.atan2(sine, cosine)
    first = math.ceil(-zero / math.pi)
    last = math.floor((theta - zero) / math.pi)
    for k in range(first, last + 1):
        t = zero + k * math.pi
        if 0. < t < theta:
            values.append(constant + cosine * math.cos(t) + sine * math.sin(t))
    end = math.degrees(math.acos(float(np.clip(values[1], -1., 1.))))
    peak = math.degrees(math.acos(float(np.clip(min(values), -1., 1.))))
    # A non-increasing tilt has a non-negative z derivative throughout.
    checks = [0., theta]
    critical = math.atan2(-cosine, sine)
    for k in range(math.ceil(-critical / math.pi), math.floor((theta - critical) / math.pi) + 1):
        t = critical + k * math.pi
        if 0. < t < theta:
            checks.append(t)
    monotone_return = min(-cosine * math.sin(t) + sine * math.cos(t) for t in checks) >= -1e-10
    return current, end, peak, monotone_return


class OpenCompletionFacts:
    """Independent narrow OPEN completion automaton, never a default EMPTY.

    A previously established completed opening remains a public EMPTY fact
    under actual continuous OPEN and an aperture still at/above open-complete.
    Missing/invalid/below-threshold telemetry, actual CLOSE, or uncertain
    consumption invalidates that fact.  A review/PASS alone changes nothing.
    """
    def __init__(self, config=OracleConfig()):
        self.config = config
        self.empty = False
        self.last_executed_gripper = None
        self.last_sequence = -1
        self.samples = []

    def executed(self, gripper, *, consumption_known=True):
        if gripper not in {"OPEN", "CLOSE", "HOLD"}:
            raise ValueError("decoded public gripper command required")
        if not consumption_known or gripper == "CLOSE":
            self.empty = False
            self.samples.clear()
        if not consumption_known:
            self.last_executed_gripper = None
        elif gripper != "HOLD":
            self.last_executed_gripper = gripper

    def observe(self, sequence, aperture_m):
        if sequence < self.last_sequence:
            raise ValueError("public sequence regressed")
        valid = (isinstance(aperture_m, (int, float)) and math.isfinite(aperture_m)
                 and aperture_m >= 0)
        if not valid or self.last_executed_gripper != "OPEN":
            self.empty = False
            self.samples.clear()
            return "UNKNOWN"
        if self.empty and aperture_m < self.config.open_complete_gap_m:
            self.empty = False
            self.samples.clear()
        if sequence == self.last_sequence:
            return "EMPTY" if self.empty else "UNKNOWN"
        self.last_sequence = sequence
        if self.empty and aperture_m >= self.config.open_complete_gap_m:
            return "EMPTY"
        self.empty = False
        self.samples.append(float(aperture_m))
        self.samples = self.samples[-self.config.stable_samples:]
        if (len(self.samples) == self.config.stable_samples
                # Establishment keeps the existing calibration: a stable
                # executed-OPEN sample window and its *current* aperture at
                # open-complete.  Earlier members may be just below it.
                and aperture_m >= self.config.open_complete_gap_m
                and max(self.samples) - min(self.samples) <= self.config.stable_gap_range_m):
            self.empty = True
        return "EMPTY" if self.empty else "UNKNOWN"


def _context(row):
    return (row.get("episode", "demo"), row.get("round_id", 1), row["primitive"],
            row.get("binding_version", "declared-demo-binding"),
            # A lease revision is not its only semantic content.  A changed
            # condition, validity or non-container finding also ends the old
            # permission, even if a producer mistakenly reuses its version.
            row.get("binding_valid", True), row.get("reliable_non_container", False),
            json.dumps(row.get("binding_conditions", []), sort_keys=True))


def _source_problem(source, cmd, sequence):
    if source is None:
        return None
    if (source.get("kind") not in {"prepared_translation", "prepared_rotation", "known_chunk"}
            or not isinstance(source.get("id"), str) or not source["id"]
            or type(source.get("index")) is not int or source["index"] < 0
            or type(source.get("remaining_controls")) is not int or source["remaining_controls"] < 1):
        return "malformed finite source identity/index/bound"
    if "origin_sequence" in source:
        origin = source["origin_sequence"]
        first_index = source.get("origin_index", 0)
        if (type(origin) is not int or type(first_index) is not int
                or source["index"] - first_index != sequence - origin):
            return "finite source index/executed sequence lockstep broken"
    if "gripper_target" in source:
        target = cmd.get("gripper_target", (cmd.get("payload") or [None])[-1])
        expected = source["gripper_target"]
        if (not isinstance(target, (int, float)) or not math.isfinite(target)
                or not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or np.float32(target) != np.float32(expected)):
            return "candidate gripper target differs from fixed disclosed target"
    for source_key, command_key in (("expected_position", "position_target"),
                                    ("expected_rotation", "orientation_target"),
                                    ("expected_payload", "payload")):
        if source_key in source and not np.allclose(source[source_key], cmd.get(command_key), atol=1e-12, rtol=0):
            return "indexed candidate differs from disclosed " + source_key
    if source["kind"] == "known_chunk" and source["remaining_controls"] > 5:
        return "known chunk exceeds five disclosed controls"
    return None


def _make_scope(row, cmd, context, phase, config):
    source = row.get("source")
    base = dict(context=context, phase=phase, gripper=cmd["gripper"],
                gripper_target=cmd.get("gripper_target", (cmd.get("payload") or [None])[-1]),
                first_sequence=row["sequence"], grasp=row.get("grasp_rotation"))
    if source is None:
        count = row.get("remaining_controls", 1)
        if type(count) is not int or count < 1:
            raise ValueError("stationary completion needs a finite positive bound")
        return {**base, "source_id": None, "last_sequence": row["sequence"] + count - 1,
                "position_path": [row["eef_position"]], "rotation_anchor": row["eef_rotation"],
                "orientation_radius_rad": config.direction_tolerance_rad,
                "candidate_stationary": True}
    radius = source.get("orientation_radius_rad", config.direction_tolerance_rad)
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= math.pi:
        raise ValueError("finite disclosed SO(3) radius required")
    _rot(source.get("rotation_anchor", row["eef_rotation"]))
    for point in source["position_path"]:
        _vec(point)
    count = source["remaining_controls"]
    payloads = source.get("payloads")
    if source["kind"] == "known_chunk" and (not isinstance(payloads, list) or len(payloads) != count):
        raise ValueError("known chunk needs every exact disclosed payload")
    return {**base, "source_id": source["id"], "kind": source["kind"],
            "first_index": source["index"], "last_index": source["index"] + count - 1,
            "last_sequence": row["sequence"] + count - 1,
            "position_path": source["position_path"],
            "rotation_anchor": source.get("rotation_anchor", row["eef_rotation"]),
            "orientation_radius_rad": radius, "payloads": payloads}


def _scope_reason(scope, row, cmd, context, phase, config):
    """A reason for expiry, or None.  No production coverage helper is called."""
    if scope["context"] != context:
        return "new primitive/round/binding context"
    if scope["phase"] != phase:
        return "new holding or gripper phase"
    if not scope["first_sequence"] <= row["sequence"] <= scope["last_sequence"]:
        return "finite sequence window exhausted"
    target = cmd.get("gripper_target", (cmd.get("payload") or [None])[-1])
    if cmd["gripper"] != scope["gripper"] or target != scope["gripper_target"]:
        return "opening/closing target changed"
    source = row.get("source")
    if scope["source_id"] is None:
        if source is not None:
            return "new disclosed source"
        if (np.linalg.norm(_vec(cmd["position_target"]) - _vec(row["eef_position"])) > 1e-12
                or np.linalg.norm(_vec(cmd["rotation_delta"])) > 1e-12):
            return "stationary permission cannot authorize new motion"
    else:
        if source is None or source.get("id") != scope["source_id"] or source.get("kind") != scope["kind"]:
            return "source identity changed"
        if not scope["first_index"] <= source["index"] <= scope["last_index"]:
            return "disclosed source index exhausted"
        if source["index"] - scope["first_index"] != row["sequence"] - scope["first_sequence"]:
            return "finite source index/executed sequence lockstep broken"
        if scope.get("payloads") is not None:
            offset = source["index"] - scope["first_index"]
            if not np.array_equal(np.asarray(scope["payloads"][offset], dtype=np.float32),
                                  np.asarray(cmd.get("payload"), dtype=np.float32)):
                return "known chunk payload changed"
    if scope.get("kind") == "known_chunk":
        # The actual reviewed chunk contract is its disclosed AABB, not the
        # narrower polyline tube used by prepared analytic translation.
        points = np.asarray(scope["position_path"], dtype=float)
        low = points.min(axis=0) - config.position_tolerance_m
        high = points.max(axis=0) + config.position_tolerance_m
        outside = any(np.any(_vec(v) < low-1e-12) or np.any(_vec(v) > high+1e-12)
                      for v in (row["eef_position"], cmd["position_target"]))
    else:
        outside = any(_line_distance(_vec(v), scope["position_path"]) > config.position_tolerance_m + 1e-12
                      for v in (row["eef_position"], cmd["position_target"]))
    if outside:
        return "public/candidate position outside reviewed path"
    anchor = _rot(scope["rotation_anchor"])
    if any(_angle(_rot(v), anchor) > scope["orientation_radius_rad"] + 1e-10
           for v in (row["eef_rotation"], cmd["orientation_target"])):
        return "public/candidate orientation outside reviewed range"
    return None


def evaluate_trace(rows: Iterable[dict], *, config=OracleConfig(), initial_last_gripper=None):
    """Compute review positions from public event/scope facts, never actual calls."""
    result = OracleResult()
    scopes = {}
    cancelled_contexts = set()
    known_tilted = False
    gripper_cycle = 0
    if initial_last_gripper not in {None, "OPEN", "CLOSE"}:
        raise ValueError("initial last gripper must be an explicit actual command")
    last_gripper = initial_last_gripper
    prior_sequence = -1
    previous_context = None
    for i, row in enumerate(rows):
        result.rows_checked += 1
        context = _context(row)
        position = dict(row_index=i, sequence=row["sequence"], primitive=row["primitive"],
                        episode=context[0], round_id=context[1])
        def stop(reason):
            result.technical_stops.append({**position, "reason": reason})
            cancelled_contexts.add(context)
            scopes.clear()
            if row.get("dispatched"):
                result.violations.append({**position, "reason": "technical-stop candidate consumed"})
        if context in cancelled_contexts:
            if row.get("dispatched"):
                result.violations.append({**position, "reason": "cancelled primitive tail consumed"})
            continue
        if row["sequence"] < prior_sequence:
            stop("public executed sequence regressed")
            continue
        prior_sequence = row["sequence"]
        if context != previous_context:
            scopes.clear()
            previous_context = context
        if row.get("owner_cancelled"):
            stop("original owner cancelled; late PASS cannot dispatch")
            continue
        cmd = row["candidate"]
        if cmd.get("gripper") not in {"OPEN", "CLOSE", "HOLD"}:
            stop("candidate gripper semantics unavailable")
            continue
        try:
            here, target = _vec(row["eef_position"]), _vec(cmd["position_target"])
            rotation, target_rotation = _rot(row["eef_rotation"]), _rot(cmd["orientation_target"])
            delta = _vec(cmd["rotation_delta"])
            problem = _source_problem(row.get("source"), cmd, row["sequence"])
        except (TypeError, ValueError, KeyError):
            stop("required public state/candidate/source unavailable")
            continue
        if problem:
            stop(problem)
            continue
        source = row.get("source")
        if source is not None and any(
                permit["source_id"] == source["id"]
                and source["index"] - permit["first_index"] != row["sequence"] - permit["first_sequence"]
                for permit in scopes.values()):
            stop("finite source index/executed sequence lockstep broken")
            continue
        holding = row.get("holding_before", "UNKNOWN")
        if holding not in {"EMPTY", "HELD", "UNKNOWN"}:
            stop("explicit public holding phase invalid")
            continue
        phase = (row.get("holding_generation", 0), gripper_cycle)
        events, reasons = [], {}
        opening = cmd["gripper"] == "OPEN" or (cmd["gripper"] == "HOLD" and last_gripper == "OPEN")
        rotating = bool(np.linalg.norm(delta) > 1e-12)
        if cmd["gripper"] == "CLOSE":
            scopes.pop(RELEASE, None)
        if holding == "EMPTY":
            scopes.clear()
            known_tilted = False
        else:
            gap = row.get("aperture_m")
            gap_valid = isinstance(gap, (float, int)) and math.isfinite(gap) and gap >= 0
            if opening:
                if not gap_valid:
                    stop("opening needs valid public aperture")
                    continue
                reason = ("first relevant release in this context" if RELEASE not in scopes else
                          _scope_reason(scopes[RELEASE], row, cmd, context, phase, config))
                if reason:
                    events.append(RELEASE)
                    reasons[RELEASE] = reason
            if not row.get("reliable_non_container", False):
                grasp = row.get("grasp_rotation")
                current = peak = None
                recovery = False
                if grasp is not None:
                    try:
                        current, end, peak, monotone = _tilt_and_segment(rotation, _rot(grasp), delta)
                    except (TypeError, ValueError):
                        stop("relevant grasp pose basis invalid")
                        continue
                    if known_tilted and current < config.rearm_deg:
                        scopes.pop(POUR, None)
                        known_tilted = False
                    known_tilted = known_tilted or current >= config.onset_deg
                    recovery = (rotating and np.linalg.norm(target - here) <= 1e-12
                                and monotone and end < current)
                relevant = rotating or known_tilted or (current is not None and current >= config.onset_deg)
                if relevant:
                    if not gap_valid:
                        stop("relevant rotation needs valid public aperture")
                        continue
                    if grasp is None and holding != "UNKNOWN":
                        stop("relevant rotation lacks reliable public grasp pose basis")
                        continue
                    # Explicit UNKNOWN baseline is the current confidence contract:
                    # semantic uncertainty is reviewed, never defaulted to empty.
                    hit = grasp is None or (peak >= config.onset_deg and not recovery)
                    if hit:
                        reason = ("first relevant pour/onset in this context" if POUR not in scopes else
                                  _scope_reason(scopes[POUR], row, cmd, context, phase, config))
                        if POUR in scopes and scopes[POUR]["grasp"] != grasp:
                            reason = "actual grasp baseline changed"
                        if reason:
                            events.append(POUR)
                            reasons[POUR] = reason
        if events:
            result.reviews.append({**position, "events": events, "reasons": reasons})
            decision = row.get("review_decision")
            if row.get("cancelled_during_review"):
                stop("original owner cancelled during review; late PASS cannot dispatch")
                continue
            if decision == "BLOCK":
                cancelled_contexts.add(context)
                scopes.clear()
                if row.get("dispatched"):
                    result.violations.append({**position, "reason": "BLOCK candidate consumed"})
                continue
            if decision != "PASS":
                stop("review response missing/invalid; no implicit PASS")
                continue
            try:
                scope = _make_scope(row, cmd, context, phase, config)
            except (KeyError, TypeError, ValueError):
                stop("disclosed continuation malformed")
                continue
            for event in events:
                scopes[event] = scope
        if row.get("dispatched") is False:
            # A non-consumed candidate cannot establish a new gripper phase.
            continue
        if cmd["gripper"] == "CLOSE" and last_gripper != "CLOSE":
            gripper_cycle += 1
        if cmd["gripper"] != "HOLD":
            last_gripper = cmd["gripper"]
    return result


def compare_reviews(expected: OracleResult | Iterable[dict], actual: Iterable[dict]):
    """Compare exact position + event-set multiplicities, including duplicates."""
    from collections import Counter
    predicted = expected.reviews if isinstance(expected, OracleResult) else list(expected)
    def key(row):
        return (row.get("episode", "demo"), row.get("round_id", 1), row["primitive"],
                row["sequence"], tuple(sorted(row["events"])))
    want, seen = Counter(map(key, predicted)), Counter(map(key, actual))
    missing, extra = want - seen, seen - want
    return dict(equal=not missing and not extra, zero_missed=not missing, zero_extra=not extra,
                expected_calls=sum(want.values()), actual_calls=sum(seen.values()),
                missed=[dict(position=list(k), count=n) for k, n in sorted(missing.items())],
                extra=[dict(position=list(k), count=n) for k, n in sorted(extra.items())])
