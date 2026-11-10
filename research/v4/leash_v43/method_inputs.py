"""Section 6/7 input isolation for the user-supplied LEASH v4.3 revision.

This module has no capture, reviewer, registry execution, or review cache. It
projects an already acquired public packet; the dispatcher remains responsible
for fresh capture and withholding on validation errors. Ordinary planner
observations are deliberately unchanged. Policy approval is separate from
approval of the method specification; draft use must be explicit in mocks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, IntEnum
from hashlib import sha256
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


class InputValidationError(ValueError):
    """Invalid evidence/output: callers must withhold, not assume permission."""


class GateDisabledError(InputValidationError):
    pass


class InvalidReviewResponse(InputValidationError):
    """An actual reviewer reply failed the output contract, not its transport."""


def is_model_response_failure(error):
    return (isinstance(error, InvalidReviewResponse)
            or isinstance(error.__cause__, InvalidReviewResponse))


class MethodName(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    OURS = "Ours"


class Gate(IntEnum):
    TASK = 1
    ACTION = 2
    COMMIT = 3


@dataclass(frozen=True)
class RegistryConfig:
    """One shared input contract, not a second implementation of the registry."""

    version: str = "leash-v4.3"
    events: tuple[str, ...] = ("HELD_OBJECT_RELEASE", "POUR_ONSET")


SHARED_REGISTRY_CONFIG = RegistryConfig()


@dataclass(frozen=True)
class MethodProfile:
    method: MethodName
    policy_in_planner: bool
    gates: tuple[Gate, ...]
    registry: RegistryConfig | None
    reviewer_views: tuple[str, ...]
    history_mode: str
    lease_mode: str
    repair_constraints: bool

    @property
    def registry_enabled(self) -> bool:
        return self.registry is not None

    @property
    def task_gate(self) -> bool:
        return Gate.TASK in self.gates

    @property
    def action_gate(self) -> bool:
        return Gate.ACTION in self.gates

    @property
    def commit_gate(self) -> bool:
        return Gate.COMMIT in self.gates


METHOD_PROFILES: Mapping[MethodName, MethodProfile] = MappingProxyType({
    MethodName.B0: MethodProfile(MethodName.B0, False, (), None, (), "none", "none", False),
    MethodName.B1: MethodProfile(MethodName.B1, True, (), None, (), "none", "none", False),
    MethodName.B2: MethodProfile(MethodName.B2, True, tuple(Gate), SHARED_REGISTRY_CONFIG,
                                 (), "text", "text", True),
    MethodName.B3: MethodProfile(MethodName.B3, True, (Gate.COMMIT,), SHARED_REGISTRY_CONFIG,
                                 ("agentview",), "none", "none", False),
    MethodName.OURS: MethodProfile(MethodName.OURS, True, tuple(Gate), SHARED_REGISTRY_CONFIG,
                                   ("agentview", "robot0_eye_in_hand"), "public", "public", True),
})


@dataclass(frozen=True)
class PolicySource:
    text: str
    sha256: str
    approved: bool


@dataclass(frozen=True)
class PublicFrame:
    """RGB/depth are buffers or file references, never synthesized descriptions.

    The projector checks array shapes when buffers expose them. It does not
    decode image paths or certify that declared acquisition versions are fresh.
    Only the two public camera names are accepted, including history keyframes.
    """

    view: str
    rgb: Any
    depth: Any
    K: Any
    T: Any
    evidence_version: str
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class TextHistory:
    instruction: str
    action_summaries: tuple[str, ...] = ()
    finish_summary: str | None = None


@dataclass(frozen=True)
class PublicHistory:
    instruction: str
    action_summaries: tuple[str, ...] = ()
    finish_summary: str | None = None
    keyframes: tuple[PublicFrame, ...] = ()


@dataclass(frozen=True)
class EventContext:
    trusted_instruction: str = ""
    proposed_action: str | Mapping[str, Any] = ""
    event_description: str | tuple[str, ...] = ""
    public_entity_names: tuple[str, ...] | None = None
    localized_public_entity_names: tuple[str, ...] | None = None
    holding_observations: str | None = None
    candidate_geometry: str | None = None


@dataclass(frozen=True)
class RobotState:
    """Current necessary proprioception, without R_grasp or command history."""

    eef_pos: Any = None
    eef_rotation: Any = None
    joint_positions: Any = None
    joint_velocities: Any = None
    gripper_aperture_m: float | None = None
    holding_estimate: str | None = None
    initial_occupancy: Mapping[str, Any] | None = None
    holding_proprio: str | None = None
    holding_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class NoGoRegion:
    name: str
    lower_xyz: Any
    upper_xyz: Any


@dataclass(frozen=True)
class ReviewLease:
    """A current method's validated binding, not a shared reviewer result.

    `valid` must come from the existing lease lifecycle. This projector neither
    extends its lifetime nor certifies the reviewer's semantic judgments. B2's
    bindings/regions are text-only; public geometric regions are Ours-only.
    """

    owner_method: MethodName | str
    valid: bool
    mode: str
    binding_version: str
    entities: tuple[str, ...] = ()
    no_go_regions: tuple[str | NoGoRegion, ...] = ()
    commit_conditions: tuple[str, ...] = ()
    task_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommitLeaseInput:
    """Structural seam for the registry's own LeaseContext.

    The last two fields are known internal trigger metadata, never reviewer
    evidence. They are accepted only so an existing context need not be copied
    into a second cache. Only owner/version/conditions/valid are projected.
    """

    owner: MethodName | str
    version: str
    conditions: tuple[str, ...]
    valid: bool
    holding_generation: int | None = None
    reliable_non_container: bool = False


@dataclass(frozen=True)
class ReviewInputRequest:
    method: MethodName | str
    gate: Gate | int
    current_workspace: PublicFrame | Mapping[str, Any] | None = None
    current_wrist: PublicFrame | Mapping[str, Any] | None = None
    history: tuple[PublicHistory, ...] = ()
    text_history: tuple[TextHistory, ...] = ()
    event_context: EventContext | Mapping[str, Any] | None = None
    robot_state: RobotState | Mapping[str, Any] | None = None
    lease: ReviewLease | CommitLeaseInput | Mapping[str, Any] | None = None
    evidence_version: str | None = None
    events: tuple[str, ...] = ()


def _record(value: Any, cls: type, where: str) -> Any:
    if type(value) is cls:
        return value
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{where} must be {cls.__name__} or a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise InputValidationError(f"unknown {where} fields: {sorted(unknown, key=str)}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise InputValidationError(f"invalid {where} schema") from exc


def _method(value: MethodName | str) -> MethodName:
    try:
        return MethodName(value)
    except (ValueError, TypeError) as exc:
        raise InputValidationError("unknown method") from exc


def _gate(value: Gate | int) -> Gate:
    if type(value) not in (int, Gate):
        raise InputValidationError("gate must be 1, 2, or 3")
    try:
        return Gate(value)
    except ValueError as exc:
        raise InputValidationError("unknown gate") from exc


def get_method_profile(method: MethodName | str) -> MethodProfile:
    return METHOD_PROFILES[_method(method)]


def _text(value: Any, where: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise InputValidationError(f"{where} must be {'nonempty ' if nonempty else ''}text")
    return value


def _sequence(value: Any, where: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise InputValidationError(f"{where} must be a tuple or list")
    return tuple(value)


def _texts(value: Any, where: str) -> tuple[str, ...]:
    return tuple(_text(v, where) for v in _sequence(value, where))


def _vector(value: Any, size: int | None, where: str) -> tuple[float, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise InputValidationError(f"{where} must be a numeric vector") from exc
    if size is not None and len(items) != size:
        raise InputValidationError(f"invalid {where} size")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in items):
        raise InputValidationError(f"{where} must contain finite numbers")
    return tuple(float(v) for v in items)


def _matrix(value: Any, size: int, where: str) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise InputValidationError(f"{where} must be a calibration matrix") from exc
    if len(rows) != size:
        raise InputValidationError(f"invalid {where} size")
    return tuple(_vector(row, size, where) for row in rows)


def _image(value: Any, shape: tuple[int, ...], where: str) -> Any:
    if isinstance(value, (str, bytes)):
        if not value:
            raise InputValidationError(f"missing {where}")
    elif getattr(value, "shape", None) is not None:
        if tuple(value.shape) != shape:
            raise InputValidationError(f"invalid {where} shape")
        dtype = getattr(value, "dtype", None)
        if (where == "RGB" and (getattr(dtype, "kind", None) != "u" or getattr(dtype, "itemsize", None) != 1)
                or where == "depth" and getattr(dtype, "kind", None) != "f"):
            raise InputValidationError(f"invalid {where} buffer type")
    else:
        raise InputValidationError(f"{where} must be an image buffer or reference")
    return deepcopy(value)


def _frame(value: Any, view: str | None = None, version: str | None = None) -> dict:
    frame = _record(value, PublicFrame, "public frame")
    if frame.view not in ("agentview", "robot0_eye_in_hand") or (view and frame.view != view):
        raise InputValidationError("invalid public camera")
    if type(frame.width) is not int or type(frame.height) is not int or (frame.width, frame.height) != (640, 480):
        raise InputValidationError("public frames must be 640x480")
    acquired = _text(frame.evidence_version, "frame evidence_version", nonempty=True)
    if version is not None and acquired != version:
        raise InputValidationError("frame is not bound to current evidence_version")
    return {"view": frame.view, "rgb": _image(frame.rgb, (480, 640, 3), "RGB"),
            "depth": _image(frame.depth, (480, 640), "depth"),
            "K": _matrix(frame.K, 3, "K"), "T": _matrix(frame.T, 4, "T"),
            "width": 640, "height": 480, "evidence_version": acquired}


def _robot(value: Any) -> dict:
    robot = _record(value, RobotState, "robot_state")
    result = {}
    for key, size in (("eef_pos", 3), ("joint_positions", None), ("joint_velocities", None)):
        raw = getattr(robot, key)
        if raw is not None:
            result[key] = _vector(raw, size, key)
    if robot.eef_rotation is not None:
        result["eef_rotation"] = _matrix(robot.eef_rotation, 3, "eef_rotation")
    if robot.gripper_aperture_m is not None:
        result["gripper_aperture_m"] = _vector((robot.gripper_aperture_m,), 1, "gripper_aperture_m")[0]
    if robot.holding_estimate is not None:
        if robot.holding_estimate not in ("EMPTY", "HELD", "UNKNOWN"):
            raise InputValidationError("invalid current holding_estimate")
        result["holding_estimate"] = robot.holding_estimate
    if robot.initial_occupancy is not None:
        from .initial_occupancy import validate_initial_fact
        result['initial_occupancy'] = validate_initial_fact(robot.initial_occupancy)
    if robot.holding_proprio is not None or robot.holding_evidence is not None:
        from .visual_holding import enabled
        if not enabled() or robot.holding_proprio not in ('EMPTY','UNKNOWN','HELD'):
            raise InputValidationError('visual holding development input is not enabled')
        result['holding_proprio'] = robot.holding_proprio
        if robot.holding_evidence is not None:
            result['holding_evidence'] = deepcopy(robot.holding_evidence)
    return result


def _action(value: Any) -> str | dict:
    if isinstance(value, str):
        return value
    keys = {"gripper", "translation_m", "rotation_world_axis_angle_rad"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise InputValidationError("proposed_action must be text or a decoded public command")
    if value["gripper"] not in ("OPEN", "CLOSE", "HOLD"):
        raise InputValidationError("invalid decoded gripper description")
    return {"gripper": value["gripper"],
            "translation_m": _vector(value["translation_m"], 3, "translation_m"),
            "rotation_world_axis_angle_rad": _vector(value["rotation_world_axis_angle_rad"], 3,
                                                     "rotation_world_axis_angle_rad")}


def _context(value: Any, method: MethodName, gate: Gate) -> dict:
    context = _record(value, EventContext, "event_context")
    description = context.event_description
    result = {"trusted_instruction": _text(context.trusted_instruction, "trusted_instruction"),
            "proposed_action": _action(context.proposed_action),
            "event_description": (_text(description, "event_description") if isinstance(description, str)
                                  else _texts(description, "event_description"))}
    if context.public_entity_names is not None or context.localized_public_entity_names is not None:
        if method != MethodName.OURS or gate != Gate.ACTION:
            raise InputValidationError("public entity declarations belong only to Ours gate 2")
        names = _texts(context.public_entity_names, "public_entity_names")
        localized = _texts(context.localized_public_entity_names, "localized_public_entity_names")
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise InputValidationError("public entity names must be distinct nonempty strings")
        if len(set(localized)) != len(localized) or not set(localized) <= set(names):
            raise InputValidationError("localized public names must be a distinct declared subset")
        result.update(public_entity_names=names, localized_public_entity_names=localized)
    if context.holding_observations is not None:
        from .visual_holding import enabled
        if not enabled() or method != MethodName.OURS or gate not in (Gate.ACTION,Gate.COMMIT):
            raise InputValidationError('holding observations belong to opt-in Ours G2/G3')
        result['holding_observations'] = _text(context.holding_observations,'holding_observations')
    if context.candidate_geometry is not None:
        import os, json
        if (os.environ.get('SAFEVLA_G2_CANDIDATE_GEOMETRY') != 'g2-candidate-evidence-dev1-20261005'
                or method != MethodName.OURS or gate != Gate.ACTION):
            raise InputValidationError('candidate geometry belongs only to opt-in Ours G2')
        text = _text(context.candidate_geometry, 'candidate_geometry')
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get('version') != 'g2-candidate-evidence-dev1-20261005':
                raise ValueError('candidate geometry version/record missing')
            json.dumps(data, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise InputValidationError('invalid candidate geometry JSON') from exc
        result['candidate_geometry'] = text
    return result


def _history(value: Any, *, public: bool) -> tuple[dict, ...]:
    result = []
    for item in _sequence(value, "history"):
        row = _record(item, PublicHistory if public else TextHistory, "history entry")
        out = {"instruction": _text(row.instruction, "historical instruction"),
               "action_summaries": _texts(row.action_summaries, "action_summaries")}
        if row.finish_summary is not None:
            out["finish_summary"] = _text(row.finish_summary, "finish_summary")
        if public:
            out["keyframes"] = tuple(_frame(f) for f in _sequence(row.keyframes, "keyframes"))
        result.append(out)
    return tuple(result)


def _lease(value: Any, method: MethodName) -> dict:
    if is_dataclass(value) and type(value) is not ReviewLease:
        # Reject extra fields before reading them. Known internal trigger fields
        # are not even fetched; only the four semantic fields are projected.
        names = {f.name for f in fields(value)}
        semantic = {"owner", "version", "conditions", "valid"}
        if not semantic <= names or names - {f.name for f in fields(CommitLeaseInput)}:
            raise InputValidationError("unknown or missing commit lease fields")
        value = {name: getattr(value, name) for name in semantic}
    if isinstance(value, Mapping) and "owner" in value:
        context = _record(value, CommitLeaseInput, "commit lease")
        if _method(context.owner) != method or context.valid is not True:
            raise InputValidationError("lease must be valid and owned by this method")
        return {"owner": method.value, "version": _text(context.version, "lease version", nonempty=True),
                "conditions": _texts(context.conditions, "lease conditions"), "valid": True}
    lease = _record(value, ReviewLease, "lease")
    if _method(lease.owner_method) != method or lease.valid is not True:
        raise InputValidationError("lease must be valid and owned by this method")
    mode = "text" if method == MethodName.B2 else "public"
    if lease.mode != mode:
        raise InputValidationError("lease evidence mode differs from this method")
    regions = []
    for region in _sequence(lease.no_go_regions, "no_go_regions"):
        if isinstance(region, str):
            regions.append(region)
        elif method == MethodName.OURS:
            box = _record(region, NoGoRegion, "no_go_region")
            lo, hi = _vector(box.lower_xyz, 3, "lower_xyz"), _vector(box.upper_xyz, 3, "upper_xyz")
            if any(a > b for a, b in zip(lo, hi)):
                raise InputValidationError("inverted no-go region")
            regions.append({"name": _text(box.name, "region name"), "lower_xyz": lo, "upper_xyz": hi})
        else:
            raise InputValidationError("B2 lease regions must be text-only")
    return {"owner_method": method.value, "valid": True, "mode": mode,
            "binding_version": _text(lease.binding_version, "binding_version", nonempty=True),
            "entities": _texts(lease.entities, "entities"), "no_go_regions": tuple(regions),
            "commit_conditions": _texts(lease.commit_conditions, "commit_conditions"),
            "task_constraints": _texts(lease.task_constraints, "task_constraints")}


def _policy_text(policy: Any, allow_draft_for_mock: bool) -> str:
    source = _record(policy, PolicySource, "policy")
    text = _text(source.text, "policy text", nonempty=True)
    if source.sha256 != sha256(text.encode("utf-8")).hexdigest():
        raise InputValidationError("policy content hash mismatch")
    if type(source.approved) is not bool or type(allow_draft_for_mock) is not bool:
        raise InputValidationError("policy approval/mock switch must be boolean")
    if not source.approved and not allow_draft_for_mock:
        raise InputValidationError("policy draft is not approved for real review")
    return text


def planner_policy(method: MethodName | str, policy: PolicySource | None = None,
                   *, allow_draft_for_mock: bool = False) -> str | None:
    """Return the identical authoritative policy text, or no injection for B0."""
    if not get_method_profile(method).policy_in_planner:
        return None
    return _policy_text(policy, allow_draft_for_mock)


def planner_observation(method: MethodName | str, observation: Any) -> Any:
    """Identity operation: normal planner observations are common, including B0."""
    get_method_profile(method)
    return observation


def _request(value: Any) -> tuple[ReviewInputRequest, MethodName, Gate]:
    request = _record(value, ReviewInputRequest, "review request")
    method, gate = _method(request.method), _gate(request.gate)
    if gate not in METHOD_PROFILES[method].gates:
        raise GateDisabledError(f"{method.value} does not enable gate {gate.value}")
    return request, method, gate


def _events(value: Any) -> tuple[str, ...]:
    events = _texts(value, "events")
    if len(set(events)) != len(events) or any(e not in SHARED_REGISTRY_CONFIG.events for e in events):
        raise InputValidationError("invalid or duplicate registered event")
    return events


def project_review_input(request: ReviewInputRequest | Mapping[str, Any],
                         policy: PolicySource | Mapping[str, Any], *,
                         allow_draft_for_mock: bool = False) -> dict:
    """Build a fresh method-specific packet using only explicitly allowed fields.

    Excluded *known* branches are not inspected or converted: B2 images/public
    history and B3 wrist/history/text history/lease cannot leak through str(), a
    shared cache, or fallback. Unknown fields in consumed records are rejected.
    B3's lack of a lease means lease-dependent facts remain unknown, not true.
    """
    req, method, gate = _request(request)
    version = _text(req.evidence_version, "evidence_version", nonempty=True)
    events = _events(req.events)
    if gate == Gate.COMMIT and not events:
        raise InputValidationError("commit review requires all matched events")
    if gate != Gate.COMMIT and events:
        raise InputValidationError("matched events belong to gate 3")
    out = {"method": method.value, "gate": gate.value,
           "policy": _policy_text(policy, allow_draft_for_mock),
           "evidence_version": version, "events": events,
           "event_context": _context(req.event_context, method, gate),
           "robot_state": _robot(req.robot_state)}
    if method == MethodName.B2:
        out["text_history"] = _history(req.text_history, public=False)
    else:
        out["current_workspace"] = _frame(req.current_workspace, "agentview", version)
        if method == MethodName.OURS:
            out["current_wrist"] = _frame(req.current_wrist, "robot0_eye_in_hand", version)
            out["history"] = _history(req.history, public=True)
    if method in (MethodName.B2, MethodName.OURS) and req.lease is not None:
        out["lease"] = _lease(req.lease, method)
    return out


def validate_review_response(method: MethodName | str, gate: Gate | int,
                             response: Mapping[str, Any], *,
                             request: ReviewInputRequest | Mapping[str, Any] | None = None,
                             expected_events: tuple[str, ...] | None = None,
                             evidence_version: str | None = None) -> dict:
    """Strict schema and context binding; errors must be handled fail-closed.

    No semantic safety is inferred. In particular, a valid PASS is not a task
    success. Gate 3 requires an exact evidence version and coverage of every
    matched event; B3 can return only PASS/BLOCK, reason, and these bindings.
    A dispatcher can bind via the original `request`, or via explicit
    `expected_events`/`evidence_version` extracted from its projected packet.
    The returned verdict key is `result`; transport-specific naming is applied
    only after validation.
    """
    name, boundary = _method(method), _gate(gate)
    if boundary not in METHOD_PROFILES[name].gates:
        raise GateDisabledError("review response for a disabled gate")
    # Check our request bindings before attributing any error to the reply.
    if boundary == Gate.COMMIT:
        if request is not None:
            req, req_method, req_gate = _request(request)
            if req_method != name or req_gate != boundary:
                raise InputValidationError("response method/gate differs from request")
            if ((expected_events is not None and _events(expected_events) != _events(req.events))
                    or (evidence_version is not None and evidence_version != req.evidence_version)):
                raise InputValidationError("conflicting request bindings")
            expected_events, evidence_version = req.events, req.evidence_version
        expected_events = _events(expected_events)
        evidence_version = _text(evidence_version, "request evidence_version", nonempty=True)
        if not expected_events:
            raise InputValidationError("commit request has no matched events")
    try:
        return _validate_response(name, boundary, response, expected_events, evidence_version)
    except (InputValidationError, TypeError, ValueError) as exc:
        raise InvalidReviewResponse(str(exc)) from exc


def _validate_response(name, boundary, response, expected_events, evidence_version):
    if not isinstance(response, Mapping):
        raise InputValidationError("review response must be a mapping")
    allowed = {"result", "reason"}
    import os
    bounded = os.environ.get('SAFEVLA_BOUNDED_OCCUPANCY') == 'unified-bounded-occupancy-dev1-20261006'
    if bounded:
        allowed |= {'ask_kind'} if boundary == Gate.TASK else {'deny_basis','missing_evidence','assumptions_confirmed','unsupported_deny'}
    if boundary == Gate.COMMIT:
        allowed |= {"events", "evidence_version"}
    else:
        allowed |= {"constraints", "lease"}
    from .visual_holding import enabled, validate_assessment
    if name == MethodName.OURS and boundary == Gate.ACTION and enabled():
        allowed.add('holding_assessment')
    if set(response) - allowed or not {"result", "reason"} <= set(response):
        raise InputValidationError("unknown or missing review response fields")
    options = {Gate.TASK: ("PROCEED", "REFUSE", "ASK"),
               Gate.ACTION: ("APPROVE", "DENY"), Gate.COMMIT: ("PASS", "BLOCK")}
    if response["result"] not in options[boundary]:
        raise InputValidationError("invalid review result for this gate")
    out = {"result": response["result"], "reason": _text(response["reason"], "reason", nonempty=True)}
    if bounded:
        if boundary == Gate.TASK:
            kind=response.get('ask_kind')
            if kind not in (None,'fact','authorization'): raise InputValidationError('invalid ask_kind')
            out['ask_kind']=kind
        else:
            basis=response.get('deny_basis')
            if basis not in (None,'observed_hazard','task_inconsistency','assumption_refuted','missing_evidence'):
                raise InputValidationError('invalid deny_basis')
            missing=response.get('missing_evidence')
            if missing is not None:
                if not isinstance(missing,Mapping) or set(missing)!={'field','current_hazard','observation_or_alternative'}:
                    raise InputValidationError('invalid missing_evidence')
                missing={k:_text(v,k) for k,v in missing.items()}
            out.update(deny_basis=basis,missing_evidence=missing,
                assumptions_confirmed=_texts(response.get('assumptions_confirmed',()),'assumptions_confirmed'))
            out['unsupported_deny']=(response['result'] in ('DENY','BLOCK') and
                (basis is None or (basis=='missing_evidence' and
                    (missing is None or not all(v.strip() for v in missing.values())))))
    if boundary == Gate.COMMIT:
        expected = expected_events
        observed = _events(response.get("events"))
        version = evidence_version
        if not expected or set(observed) != set(expected) or response.get("evidence_version") != version:
            raise InputValidationError("commit response does not cover current events/evidence")
        out.update(events=observed, evidence_version=version)
    else:
        if "constraints" in response:
            out["constraints"] = _texts(response["constraints"], "constraints")
        if "lease" in response:
            out["lease"] = _lease(response["lease"], name)
        if 'holding_assessment' in response:
            out['holding_assessment'] = validate_assessment(response['holding_assessment'])
    return out
