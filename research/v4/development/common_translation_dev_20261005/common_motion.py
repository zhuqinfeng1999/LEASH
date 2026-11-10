"""Opt-in common analytic translation, using public EEF state and RPent dispatch.

This module installs no controller mode and does not qualify a motion error
profile. G2 and the caller must share one MotionDescription. An unqualified
profile is usable for the explicitly authorized engineering calibration only.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


CONTROL_VERSION = "common-translation-r0-line-dev1-20261005"
REFERENCE_RULE = "advance each control by bounded monotone line fraction; hold endpoint"


class UnsupportedMotion(ValueError):
    pass


class MotionStartChanged(ValueError):
    """The approved public starting state no longer describes this candidate."""


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return a


def _rotation(value):
    r = np.asarray(value, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
        raise ValueError("rotation must be a proper finite SO(3) matrix")
    return r


def _rows(a):
    return tuple(tuple(float(x) for x in row) for row in a)


def _sign(value):
    v = float(value)
    if not np.isfinite(v):
        raise ValueError("gripper command must be finite")
    return int(np.sign(v))


@dataclass(frozen=True)
class OSCMapping:
    """Immutable copy of the installed delta/achieved control configuration."""
    input_min: tuple[float, ...]
    input_max: tuple[float, ...]
    output_min: tuple[float, ...]
    output_max: tuple[float, ...]
    reference_rotation: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        for name in ("input_min", "input_max", "output_min", "output_max"):
            object.__setattr__(self, name, tuple(_vector(getattr(self, name), 6, name)))
        object.__setattr__(self, "reference_rotation", _rows(_rotation(self.reference_rotation)))
        if (not np.all(np.array(self.input_max) > self.input_min)
                or not np.all(np.array(self.output_max) > self.output_min)):
            raise ValueError("OSC scaling intervals must be strictly increasing")

    @classmethod
    def from_semantics(cls, semantics):
        # Caller supplies the existing OSCControlSemantics.from_scene result.
        # This module never reads controller/simulator/object handles.
        return cls(*(tuple(np.asarray(getattr(semantics, k), float)) for k in
                     ("input_min", "input_max", "output_min", "output_max")),
                   _rows(semantics.reference_rotation))

    def encode(self, translation_world, rotation_world):
        q = np.asarray(self.reference_rotation)
        physical = np.r_[q.T @ _vector(translation_world, 3, "translation"),
                         q.T @ _vector(rotation_world, 3, "rotation delta")]
        lo, hi = np.array(self.input_min), np.array(self.input_max)
        out_lo, out_hi = np.array(self.output_min), np.array(self.output_max)
        scale = (out_hi - out_lo) / (hi - lo)
        a = (physical - (out_hi + out_lo) / 2) / scale + (hi + lo) / 2
        return np.clip(a, lo, hi)

    def decode(self, action):
        lo, hi = np.array(self.input_min), np.array(self.input_max)
        a = np.clip(_vector(action, 6, "OSC action"), lo, hi)
        out_lo, out_hi = np.array(self.output_min), np.array(self.output_max)
        physical = (a - (hi + lo) / 2) * (out_hi - out_lo) / (hi - lo) + (out_hi + out_lo) / 2
        q = np.asarray(self.reference_rotation)
        return q @ physical[:3], q @ physical[3:]

    def translation_gain(self):
        # Preserve RPent's step_clip/action_scale normalized command meaning
        # under the currently installed symmetric/isotropic translation scale.
        lo, hi = np.array(self.input_min[:3]), np.array(self.input_max[:3])
        out_lo, out_hi = np.array(self.output_min[:3]), np.array(self.output_max[:3])
        if (not np.allclose(lo, -1, atol=0, rtol=0)
                or not np.allclose(hi, 1, atol=0, rtol=0)
                or not np.allclose(out_lo, -out_hi, atol=1e-12, rtol=0)
                or not np.allclose(out_hi, out_hi[0], atol=1e-12, rtol=0)
                or out_hi[0] <= 0):
            raise UnsupportedMotion("translation profile requires the installed symmetric isotropic RPent scale")
        return float(out_hi[0])


@dataclass(frozen=True)
class PublicPose:
    position: tuple[float, float, float]
    rotation: tuple[tuple[float, ...], ...]
    sequence: int | None = None
    time_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(_vector(self.position, 3, "public EEF position")))
        object.__setattr__(self, "rotation", _rows(_rotation(self.rotation)))
        if self.sequence is not None:
            if isinstance(self.sequence, bool) or not isinstance(self.sequence, (int, np.integer)):
                raise ValueError("public sequence must be an integer")
            object.__setattr__(self, "sequence", int(self.sequence))
        if self.time_s is not None and not np.isfinite(self.time_s):
            raise ValueError("public time must be finite")
        if self.time_s is not None:
            object.__setattr__(self, "time_s", float(self.time_s))


def public_pose(primitives, reader: Callable[[], Mapping] | None = None):
    """One read of permitted world EEF position and body-world xyzw quaternion."""
    raw = reader() if reader is not None else primitives.env.raw_obs()
    quat = _vector(raw["robot0_eef_quat"], 4, "public body-world xyzw quaternion")
    if not .9 < np.linalg.norm(quat) < 1.1:
        raise ValueError("public quaternion is not normalized")
    return PublicPose(tuple(_vector(raw["robot0_eef_pos"], 3, "public EEF position")),
                      _rows(Rotation.from_quat(quat).as_matrix()),
                      raw.get("sequence"), raw.get("time_s"))


@dataclass(frozen=True)
class MotionDescription:
    start: PublicPose
    goal: tuple[float, float, float]
    mapping: OSCMapping
    max_steps: int
    gripper: float
    previous_gripper_command: float
    step_clip: float
    tol: float
    action_scale: float
    rotation_step_clip: float
    control_dt_s: float
    error_profile: str
    version: str = CONTROL_VERSION
    path_mode: str = "world monotone line segment"
    attitude_mode: str = "fixed primitive-start body-world R0; SO(3) feedback"
    reference_rule: str = REFERENCE_RULE

    def __post_init__(self):
        object.__setattr__(self, "goal", tuple(float(v) for v in _vector(self.goal, 3, "goal")))
        if not isinstance(self.start, PublicPose) or not isinstance(self.mapping, OSCMapping):
            raise ValueError("motion requires immutable public pose and OSC mapping")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if _sign(self.gripper) != _sign(self.previous_gripper_command):
            raise UnsupportedMotion("gripper target changed")
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.step_clip, self.tol, self.action_scale, self.rotation_step_clip, self.control_dt_s)):
            raise ValueError("invalid motion limits")
        self.mapping.translation_gain()

    @property
    def physical_step_limit(self):
        return self.mapping.translation_gain() * min(1., self.step_clip / self.action_scale)

    @property
    def fraction_step(self):
        d = np.array(self.goal) - self.start.position
        base_d = np.array(self.mapping.reference_rotation).T @ d
        # Both world feedback and actual base controls retain component limits.
        extent = max(float(np.max(np.abs(d))), float(np.max(np.abs(base_d))))
        return 1. if extent == 0 else min(1., self.physical_step_limit / extent)

    def reference(self, controls):
        s = min(1., max(0, int(controls)) * self.fraction_step)
        return s, np.array(self.start.position) + s * (np.array(self.goal) - self.start.position)

    def as_dict(self):
        return asdict(self)


def prepare_motion(primitives, xyz, *, semantics, previous_gripper_command,
                   max_steps=80, gripper=-1., step_clip=.025, tol=.012,
                   action_scale=.05, target_yaw=None, yaw_step_clip=.10,
                   control_dt_s=.05, error_profile="unqualified-engineering",
                   public_reader=None, controller_version=CONTROL_VERSION):
    if controller_version not in (CONTROL_VERSION,'common-translation-persistent-world-reference-dev2-20261005'):
        raise UnsupportedMotion('Unknown common analytic controller variant')
    if target_yaw is not None:
        raise UnsupportedMotion("explicit target_yaw retains the original RPent path")
    if previous_gripper_command is None or _sign(gripper) != _sign(previous_gripper_command):
        raise UnsupportedMotion("fixed-gripper target requires an actual executed public command of the same sign")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError("xyz must be a JSON array of three numbers")
    # Preserve the actual RPent float32 target, shared by inspection and runner.
    goal = _vector(np.asarray([float(v) for v in xyz], dtype=np.float32), 3, "goal")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    values = [step_clip, tol, action_scale, yaw_step_clip, control_dt_s]
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError("motion timing, clipping, scale and tolerance must be positive finite values")
    if not isinstance(error_profile, str) or not error_profile:
        raise ValueError("error_profile must identify the applicable frozen profile or engineering calibration")
    mapping = semantics if isinstance(semantics, OSCMapping) else OSCMapping.from_semantics(semantics)
    mapping.translation_gain()  # Unsupported scaling cannot inherit a profile.
    return MotionDescription(public_pose(primitives, public_reader), tuple(goal), mapping,
                             max_steps, float(gripper), float(previous_gripper_command),
                             *map(float, values), error_profile, version=controller_version)


def _same_start(description, current):
    # No physics should run during review. Do not use endpoint tol as freshness.
    return current == description.start


def _errors(description, pose, reference):
    p, a, b = np.array(pose.position), np.array(description.start.position), np.array(description.goal)
    d = b - a
    s = 0. if np.dot(d, d) == 0 else np.clip(np.dot(p - a, d) / np.dot(d, d), 0., 1.)
    angle = Rotation.from_matrix(np.array(description.start.rotation) @ np.array(pose.rotation).T).magnitude()
    return dict(line_error_m=float(np.linalg.norm(p - (a + s * d))),
                reference_error_m=float(np.linalg.norm(p - reference)),
                rotation_error_rad=float(angle))


def execute_motion(primitives, description, *, trace=None, public_reader=None):
    """Execute this exact prepared reference exclusively through _step_env.

    trace is an existing caller-owned ordinary list. Rows include each public
    pre/post state and setpoint; an exception is propagated, not converted into
    a successful action. No error profile is treated as a watchdog or permit.
    """
    if description.version not in (CONTROL_VERSION,'common-translation-persistent-world-reference-dev2-20261005'):
        raise UnsupportedMotion("motion control version does not match this runner")
    if (description.version != CONTROL_VERSION and
            getattr(primitives,'_persistent_world_reference_descriptor',None) is not description):
        raise UnsupportedMotion('Persistent reference descriptor requires its actual instance-local controller context')
    pose = public_pose(primitives, public_reader)
    if not _same_start(description, pose):
        raise MotionStartChanged("public start changed after preparation; recheck the candidate")
    rows = trace if trace is not None else []
    begin = len(rows)
    goal, r0 = np.array(description.goal), np.array(description.start.rotation)
    controls, checks = 0, 0
    for step in range(description.max_steps):
        checks += 1
        distance = float(np.linalg.norm(goal - pose.position))
        if distance < description.tol:
            break  # Original endpoint stopping criterion, not a path error bound.
        fraction, reference = description.reference(controls + 1)
        world_delta = np.clip(reference - pose.position,
                              -description.physical_step_limit, description.physical_step_limit)
        q = np.array(description.mapping.reference_rotation)
        base_delta = np.clip(q.T @ world_delta,
                             -description.physical_step_limit, description.physical_step_limit)
        world_delta = q @ base_delta
        rv = Rotation.from_matrix(r0 @ np.array(pose.rotation).T).as_rotvec()
        angle = float(np.linalg.norm(rv))
        if angle > description.rotation_step_clip:
            rv *= description.rotation_step_clip / angle
        action = np.empty(7, dtype=np.float32)
        action[:6] = description.mapping.encode(world_delta, rv)
        action[6] = description.gripper
        commanded_d, commanded_rv = description.mapping.decode(action[:6])
        row = dict(control=step, reference_fraction=fraction,
                   reference_position=reference.tolist(), reference_rotation=r0.tolist(),
                   public_before=asdict(pose), action=action.tolist(),
                   controller_target_position=(np.array(pose.position) + commanded_d).tolist(),
                   controller_target_rotation=(Rotation.from_rotvec(commanded_rv).as_matrix() @ np.array(pose.rotation)).tolist(),
                   step_returned=False)
        rows.append(row)
        try:
            # Real RPent cancellation/Owner/Consumer/Commit interception stays here.
            primitives._step_env(action)
        except BaseException as exc:
            row["exception_type"] = type(exc).__name__
            # A raised _step_env does not prove whether an underlying send occurred.
            row["dispatch_outcome"] = "consult existing dispatcher record"
            try:
                after = public_pose(primitives, public_reader)
                row["public_after"] = asdict(after)
                row.update(_errors(description, after, reference))
            except Exception:
                row["public_after"] = None
            raise
        controls += 1
        row["step_returned"] = True
        pose = public_pose(primitives, public_reader)
        row["public_after"] = asdict(pose)
        row.update(_errors(description, pose, reference))
        if primitives.env.terminated or primitives.env.truncated:
            break
    final = public_pose(primitives, public_reader)
    errors = [_errors(description, description.start, np.array(description.start.position)),
              _errors(description, final, description.reference(controls)[1])]
    errors.extend(r for r in rows[begin:] if "line_error_m" in r)
    return dict(name="move_to", target_xyz=list(description.goal),
                final_eef_pos=[round(float(x), 4) for x in final.position],
                final_dist_m=round(float(np.linalg.norm(goal - final.position)), 4),
                steps_used=checks, controls_executed=controls, max_steps=description.max_steps,
                terminated=primitives.env.terminated, truncated=primitives.env.truncated,
                common_motion=dict(description=description.as_dict(),
                    final_rotation=[list(r) for r in final.rotation],
                    max_line_error_m=max(r["line_error_m"] for r in errors),
                    max_reference_error_m=max(r["reference_error_m"] for r in errors),
                    max_rotation_error_rad=max(r["rotation_error_rad"] for r in errors)))


@contextmanager
def common_translation_scope(primitives, *, semantics, previous_gripper_command,
                             prepared=None, trace=None, public_reader=None,
                             control_dt_s=.05, error_profile="unqualified-engineering",
                             enabled=False):
    """Instance-local opt-in; install before toolkit captures its bound handler.

    For Ours, pass the very MotionDescription G2 checked in prepared. For B0,
    preparation is performed from its own current public state, with no LEASH
    conclusions. Explicit yaw and changed/unknown gripper targets use the
    original method and receive no refined-motion profile. This context never
    changes OSC state; finally restores even the absence of an instance override.
    """
    if not enabled:
        yield primitives
        return
    if getattr(primitives, "_common_translation_scope_active", False):
        raise RuntimeError("nested common translation installation is not supported")
    original = primitives.move_to
    had_override = "move_to" in primitives.__dict__
    previous_override = primitives.__dict__.get("move_to")

    def move_to(self, xyz, *, max_steps=80, gripper=-1., step_clip=.025,
                tol=.012, action_scale=.05, target_yaw=None, yaw_step_clip=.10):
        prior = previous_gripper_command() if callable(previous_gripper_command) else previous_gripper_command
        params = dict(max_steps=max_steps, gripper=gripper, step_clip=step_clip,
                      tol=tol, action_scale=action_scale, target_yaw=target_yaw,
                      yaw_step_clip=yaw_step_clip)
        if target_yaw is not None or prior is None or _sign(gripper) != _sign(prior):
            if prepared is not None:
                raise UnsupportedMotion("prepared translation cannot authorize a changed tool/gripper operation")
            return original(xyz, **params)
        if prepared is not None:
            expected = (tuple(np.asarray(xyz, dtype=np.float32)), max_steps, float(gripper),
                        float(prior), float(step_clip), float(tol), float(action_scale), float(yaw_step_clip))
            actual = (prepared.goal, prepared.max_steps, prepared.gripper, prepared.previous_gripper_command,
                      prepared.step_clip, prepared.tol, prepared.action_scale, prepared.rotation_step_clip)
            mapping = semantics if isinstance(semantics, OSCMapping) else OSCMapping.from_semantics(semantics)
            if (expected != actual or mapping != prepared.mapping
                    or control_dt_s != prepared.control_dt_s or error_profile != prepared.error_profile):
                raise UnsupportedMotion("runner parameters/configuration differ from the inspected motion description")
            description = prepared
        else:
            description = prepare_motion(self, xyz, semantics=semantics,
                previous_gripper_command=prior, control_dt_s=control_dt_s,
                error_profile=error_profile, public_reader=public_reader, **params)
        return execute_motion(self, description, trace=trace, public_reader=public_reader)

    primitives._common_translation_scope_active = True
    primitives.move_to = MethodType(move_to, primitives)
    try:
        yield primitives
    finally:
        if had_override:
            primitives.move_to = previous_override
        else:
            del primitives.move_to
        del primitives._common_translation_scope_active
