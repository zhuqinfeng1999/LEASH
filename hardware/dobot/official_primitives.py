"""Reuse the official real-robot methods across RPent's two API versions.

The receiving checkout and frozen LEASH checkout have incompatible ToolResult
types. Only the result container and @tool registration are bridged; the
official primitive method bodies are compiled unchanged from their source.
"""

import ast
import json
from hashlib import sha256
from pathlib import Path
from types import FunctionType
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "upstream/RPent-hardware/robots/franka/tools.py"
)
OFFICIAL_ACTIONS = frozenset(
    {"move_delta", "rotate_delta", "open_gripper", "close_gripper", "vla_grasp"}
)


class DataResult:
    def __init__(self, *, data):
        self.data = data


def load_official_primitives():
    raw = SOURCE.read_bytes()
    receipt = json.loads(
        (Path(__file__).parent / "official_reference.json").read_text(encoding="utf-8")
    )
    if sha256(raw).hexdigest() != receipt["source_sha256"]:
        raise ValueError(
            "Official primitive source changed; regenerate/review the paired schemas before use"
        )
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "coerce_vec3":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "FrankaPrimitives":
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
            selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": np, "ToolResult": DataResult}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    cls = namespace["FrankaPrimitives"]
    cls.source_receipt = dict(
        path=str(SOURCE),
        sha256=sha256(source.encode()).hexdigest(),
        adaptation="tool decorators and result container only; primitive method bodies unchanged",
    )
    return cls


class OfficialEnv:
    def __init__(self, native_env):
        self.native = native_env

    def move_delta(self, delta_xyz):
        return self.native.step_physical(np.r_[delta_xyz, [0.0, 0.0, 0.0, 0.0]])

    def rotate_delta(self, delta_rpy):
        # Official real-robot API accepts base-frame XYZ Euler increments;
        # the single SI outlet accepts the equivalent world rotation vector.
        return self.native.step_physical(
            np.r_[
                [0.0, 0.0, 0.0], Rotation.from_euler("xyz", delta_rpy).as_rotvec(), 0.0
            ]
        )

    def set_gripper(self, *, open):
        return self.native.step_physical(np.r_[np.zeros(6), -1.0 if open else 1.0])

    def get_observation(self):
        return self.native.observe()

    def get_robot_state(self):
        return self.native.adapter.public_robot()

    def chunk_step(self, actions):
        obs, *_ = self.native.chunk_step(actions)
        return dict(observation=obs, terminated=False, truncated=False)

    def reset(self):
        return self.get_observation()


class OfficialModel:
    def __init__(self, policy):
        self.policy = policy

    def predict(self, observation, *, options):
        actions, _ = self.policy.predict_action_batch(
            observation, mode=options.get("mode", "eval")
        )
        return actions


def registered_boundary_class(reference):
    """Register hardware names and reject invalid proposals before admission.

    Its original global membership set only knows simulation action names.
    A private globals copy avoids modifying that set or any frozen module.
    Pure argument errors must not poison the owner as infrastructure faults;
    valid calls still use the unchanged frozen admission and dispatch body.
    """
    original = reference.execute_tool
    globals_copy = dict(original.__globals__)
    globals_copy["BUDGETED_ACTIONS"] = (
        globals_copy["BUDGETED_ACTIONS"] | OFFICIAL_ACTIONS
    )
    method = FunctionType(
        original.__code__,
        globals_copy,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    method.__kwdefaults__ = original.__kwdefaults__

    def execute_tool(self, name, input_dict):
        from copy import deepcopy
        from research.v3.bridge.b0 import _tool_result
        from research.v4.leash_v43.tool_boundary import InvalidPrimitiveArguments
        # Do not reopen a failed owner, denied goal, closed round or exhausted
        # denial budget. The original body retains all those decisions.
        if (name in globals_copy['BUDGETED_ACTIONS'] and self.round_open
                and not self.system_failure and self.core.may_call(name)
                and self.denials < self.max_denials):
            try:
                # This callback only validates the proposal and calculates a
                # bound; no preparation, review, dispatch or counter changes.
                self.completion_controls(name, deepcopy(input_dict))
            except InvalidPrimitiveArguments as exc:
                hint = ''
                bounds = getattr(self.base, 'bounds', None)
                if name == 'hold' and bounds is not None:
                    hint = f' hold requires only seconds, finite and in (0,{bounds.max_hold_seconds:g}].'
                self.logs.append(dict(kind='invalid_primitive_arguments', name=name,
                    error=str(exc), candidate_dispatched=False))
                return _tool_result(name, dict(error=f'Invalid primitive arguments: {exc}.{hint}',
                    invalid_arguments=True, system_failure=False,
                    candidate_dispatched=False, recovery_allowed=True))
            except Exception:
                # Non-argument validation faults retain the original fail-closed
                # classification below. Nothing has been dispatched here.
                pass
        return method(self, name, input_dict)

    execute_tool.__wrapped__ = method
    return type(
        "RegisteredHardwareToolBoundary", (reference,), {"execute_tool": execute_tool}
    )
