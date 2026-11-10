"""Coordinate the existing RPent toolkit and v4.3 per-control dispatcher.

Place this wrapper inside OwnerThreadToolkit, around the existing budgeted
toolkit. Callbacks consume the existing public workspace/lease lifecycle; this
module supplies no perception, task-specific geometry, executor or motion.
"""
from __future__ import annotations
from research.v4.leash_v43.run_manifest import source_custody_enabled

from dataclasses import dataclass
from hashlib import sha256
import ast
import inspect
import math
import textwrap
from types import FunctionType, MethodType
from typing import Any, Callable, Mapping

from research.v3.bridge.b0 import BUDGETED_ACTIONS, _tool_result
from .method_inputs import (METHOD_PROFILES, MethodName, PolicySource,
                            validate_review_response, is_model_response_failure, InvalidReviewResponse)
from .registry import LeaseContext
from .run_manifest import record_gate
import time
import os
from .public_access import public_adapter, public_toolkit


@dataclass(frozen=True)
class ResolvedPrimitive:
    """Metadata from the caller's existing public proposal resolver."""
    binding_version: str
    proposal: Any
    entities: tuple[str, ...] = ()
    geometry_unknown: bool = False
    lease_involved: bool = False


@dataclass(frozen=True)
class GeometryDecision:
    approved: bool
    reason: str
    system_failure: bool = False


class InvalidPrimitiveArguments(ValueError):
    """A proposed tool call violates an existing primitive parameter bound."""


class OriginalControlBounds:
    """Read original primitive defaults, without executing any primitive.

    remaining_controls is the caller's *physical* remaining cap, not the
    48/144 primitive-call budget. VLA chunk size must come from the selected
    existing model configuration; an absent size is a fault, not infinity.
    """
    def __init__(self, primitives, *, remaining_controls: Callable[[], int],
                 control_timestep: float, vla_chunk_controls: int | None = None,
                 max_hold_seconds: float = 10.):
        self.primitives = primitives
        self.remaining_controls = remaining_controls
        self.control_timestep = control_timestep
        self.vla_chunk_controls = vla_chunk_controls
        self.max_hold_seconds = max_hold_seconds

    @staticmethod
    def _positive_int(value, field):
        if type(value) is not int or value < 1:
            raise ValueError(f"{field} must be an existing finite positive integer")
        return value

    @staticmethod
    def _argument_positive_int(value, field):
        if type(value) is not int or value < 1:
            raise InvalidPrimitiveArguments(f"{field} must be a positive integer")
        return value

    def primitive_controls(self, name: str, parameters: Mapping[str, Any]) -> int:
        """Validate proposed arguments separately from runtime configuration."""
        if name == "hold":
            seconds = parameters.get("seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or not 0 < seconds <= self.max_hold_seconds):
                raise InvalidPrimitiveArguments("hold requires its original finite positive duration")
            if set(parameters) != {"seconds"}:
                raise InvalidPrimitiveArguments("unknown hold parameters")
            if not math.isfinite(self.control_timestep) or self.control_timestep <= 0:
                raise ValueError("original control timestep unavailable")
            controls = math.ceil(seconds / self.control_timestep)
        else:
            if name not in BUDGETED_ACTIONS:
                raise ValueError("no original robot primitive completion bound")
            signature = inspect.signature(getattr(self.primitives, name))
            try:
                bound = signature.bind(**parameters)
            except TypeError as exc:
                raise InvalidPrimitiveArguments(str(exc)) from exc
            bound.apply_defaults()
            if name in {"pi0_pick", "pi0_doubled"}:
                controls = self._argument_positive_int(bound.arguments["max_chunks"], "max_chunks")
                controls *= self._positive_int(self.vla_chunk_controls, "selected VLA chunk size")
            else:
                key = "steps" if name == "set_gripper" else "max_steps"
                controls = self._argument_positive_int(bound.arguments[key], key)
        return controls

    def __call__(self, name: str, parameters: Mapping[str, Any]) -> int:
        controls = self.primitive_controls(name, parameters)
        remaining = self._positive_int(self.remaining_controls(), "remaining control cap")
        return min(controls, remaining)


class V43ToolBoundary:
    """Thin admission wrapper; all physical execution remains in base.

    task_review/action_review are caller-supplied projection callbacks: before
    real transport they MUST call method_inputs.project_review_input for this
    method/gate. Their local arguments below are not a worker wire packet.
    They return the method_inputs strict response schema, validated again here.
    resolver(name, parameters, workspace) returns ResolvedPrimitive.
    geometry_check(resolved, workspace, validated_lease) returns GeometryDecision.
    completion_bound derives the finite original bound (OriginalControlBounds).
    lease_is_current delegates validity/expiry to the existing own lifecycle.
    Callbacks are synchronous on the existing scene owner. No callback here
    may read an evaluator, assisted latch/weld or private object pose.
    """
    def __init__(self, base, adapter, *, method: str, episode: str,
                 workspace: Callable, resolver: Callable,
                 task_review: Callable, action_review: Callable,
                 geometry_check: Callable, completion_bound: Callable,
                 result_factory=_tool_result, public_transition: Callable | None = None,
                 lease_is_current: Callable | None = None,
                 observation_tools=(), robot_actions=BUDGETED_ACTIONS):
        self.base, self.adapter = public_toolkit(base), public_adapter(adapter)
        self.method = MethodName(method)
        self.profile = METHOD_PROFILES[self.method]
        if self.profile.action_gate and not callable(lease_is_current):
            raise ValueError("existing own lease lifecycle validator required")
        self.lease_is_current = lease_is_current
        self.episode = episode
        self.workspace, self.resolver = workspace, resolver
        self.task_review, self.action_review = task_review, action_review
        self.geometry_check, self.completion_bound = geometry_check, completion_bound
        self.result_factory = result_factory
        self.public_transition = public_transition or self._estimator_transition
        self.robot_actions = frozenset(robot_actions)
        self.observation_tools = frozenset(observation_tools)
        self.round_id = None
        self.round_open = False
        self.task_approved = False
        self.instruction = ""
        self.lease = None
        self.first_action = True
        self.after_transition = False
        self.denials = 0
        self.logs = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _estimator_transition(self):
        estimator = self.adapter.estimator  # Public-qpos/consumed-command estimator only.
        return estimator.kind, estimator.generation

    def _feedback(self, name, reason, *, gate, system_failure=False, **extra):
        row = {"error": reason, "status": "system_failure" if system_failure else "intercepted",
               "system_failure": system_failure, "gate": gate, **extra}
        self.logs.append({"tool": name, **row})
        record_gate(gate=gate, trigger=name, candidate=getattr(self.adapter.dispatcher, "primitive_id", 0) + 1,
                    epoch=self.round_id, decision=extra.get("decision", row["status"]), reason=reason,
                    reviewer_called=extra.get("reviewer_called", False), dispatched=False, invalidated=True)
        return self.result_factory(name, row)

    def _invalidate(self):
        if self.profile.registry_enabled:
            self.adapter.invalidate_binding()

    def _finish(self, status, message):
        # The common budgeted toolkit owns finish and the round's terminal record.
        result = self.base.execute_tool("finish", {"status": status, "message": message})
        if status not in {"done", "refused", "asked"} or not isinstance(message, str):
            return result
        self.round_open = False
        self.task_approved = False
        self.lease = None
        self._invalidate()
        return result

    def begin_round(self, instruction: str, *, round_id: int):
        if (self.round_open or type(round_id) is not int or round_id < 0
                or (self.round_id is not None and round_id <= self.round_id)
                or not isinstance(instruction, str) or not instruction.strip()):
            return self._feedback("begin_round", "round lifecycle/instruction invalid", gate=1,
                                  system_failure=True)
        try:
            self.base.begin_round()
        except Exception as exc:
            return self._feedback("begin_round", f"base round lifecycle fault: {exc}", gate=1,
                                  system_failure=True)
        self.round_id, self.instruction = round_id, instruction
        self.round_open, self.task_approved = True, not self.profile.task_gate
        self.lease = None
        self.first_action, self.after_transition, self.denials = True, False, 0
        self._invalidate()
        if not self.profile.task_gate:
            return self.result_factory("begin_round", {"result": "PROCEED", "reviewer_called": False})
        try:
            review_started = time.monotonic()
            decision = validate_review_response(self.method, 1, self.task_review(
                instruction=instruction, workspace=self.workspace(),
                episode=self.episode, round_id=round_id))
            self.logs.append({"gate": 1, **decision})
            record_gate(gate=1, trigger="new_instruction", candidate=None, epoch=round_id,
                        decision=decision["result"], reason=decision["reason"], reviewer_called=True,
                        elapsed_s=time.monotonic() - review_started,
                        invalidated=decision["result"] != "PROCEED")
        except Exception as exc:
            if is_model_response_failure(exc):
                self._feedback("begin_round", f"invalid task reviewer response: {exc}", gate=1,
                               root_cause="model_response_failure", reviewer_called=True)
                return self._finish("refused", f"Task reviewer did not produce a valid admission: {exc}")
            return self._feedback("begin_round", f"task review fault: {type(exc).__name__}: {exc}",
                                  gate=1, system_failure=True)
        if decision["result"] in {"REFUSE", "ASK"}:
            return self._finish("refused" if decision["result"] == "REFUSE" else "asked",
                                decision["reason"])
        self.task_approved = True
        self.lease = decision.get("lease")
        return self.result_factory("begin_round", decision)

    def _current_lease(self, binding_version):
        lease = self.lease
        if not isinstance(lease, Mapping) or lease.get("valid") is not True:
            return False
        owner = lease.get("owner_method", lease.get("owner"))
        version = lease.get("binding_version", lease.get("version"))
        return (owner == self.method.value and version == binding_version
                and self.lease_is_current(lease) is True)

    def _commit_lease(self):
        lease = self.lease
        if lease is None:
            return None
        return LeaseContext(owner=self.method.value,
                            version=lease.get("binding_version", lease.get("version")),
                            conditions=tuple(lease.get("commit_conditions", lease.get("conditions", ()))),
                            valid=True)

    def _deny(self, name, reason, *, system_failure=False):
        self.lease = None
        self._invalidate()
        if system_failure:
            return self._feedback(name, reason, gate=2, system_failure=True)
        self.denials += 1
        if self.denials >= 5:
            local_scope = os.environ.get('SAFEVLA_POLICY_VERSION') == 'stage_scoped_dev3'
            message = (f"Current candidate {name} was withheld: {reason} Five action DENY responses "
                "exhausted the finite proposal budget; the prior goal admission is not a claim that "
                "this candidate is safe or that the goal is forbidden. Further current evidence or "
                "a revised proposal is needed.") if local_scope else f"{reason} Five dangerous action DENY responses exhausted the round retry budget."
            self.logs.append({"gate": 2, "status": "denial_budget_exhausted", "reason": reason,
                              "denials": self.denials, "system_failure": False})
            return self._finish("asked" if local_scope else "refused", message)
        return self._feedback(name, reason, gate=2, decision="DENY", denials=self.denials,
            goal_judgment="prior task admission retained; current candidate withheld",
            candidate_parameters=getattr(self,"_current_candidate_parameters",None))

    def execute_tool(self, name: str, input_dict: dict):
        if name == "finish":
            if not isinstance(input_dict, dict):
                return self._feedback(name, "finish arguments must be an object", gate=1,
                                      system_failure=True)
            return self._finish(input_dict.get("status"), input_dict.get("message"))
        if not self.round_open or not self.task_approved:
            return self._feedback(name, "round has no valid task admission", gate=1,
                                  system_failure=True)
        if name in self.observation_tools:
            return self.base.execute_tool(name, input_dict)
        if name not in self.robot_actions or not isinstance(input_dict, dict):
            return self._deny(name, "unregistered robot primitive/arguments", system_failure=True)
        if not self.profile.registry_enabled:
            # B0/B1 use the original unwrapped R2 adapter and original execution.
            return self.base.execute_tool(name, input_dict)
        parameters = dict(input_dict)
        self._current_candidate_parameters = parameters.copy()
        try:
            workspace = self.workspace() if self.profile.action_gate else None
            resolved = self.resolver(name, parameters, workspace)
            if (not isinstance(resolved, ResolvedPrimitive) or not resolved.binding_version
                    or type(resolved.geometry_unknown) is not bool
                    or type(resolved.lease_involved) is not bool):
                raise ValueError("public resolver schema invalid")
            controls = self.completion_bound(name, parameters)
            if type(controls) is not int or controls < 1:
                raise ValueError("original finite completion bound unavailable")
            if self.profile.action_gate:
                lease_entities = set((self.lease or {}).get("entities", ()))
                needs_review = (self.first_action or self.after_transition or resolved.geometry_unknown
                                or resolved.lease_involved or bool(lease_entities.intersection(resolved.entities))
                                or not self._current_lease(resolved.binding_version))
                if needs_review:
                    review_started = time.monotonic()
                    checked = validate_review_response(self.method, 2, self.action_review(
                        name=name, parameters=parameters, resolved=resolved, workspace=workspace,
                        lease=self.lease, instruction=self.instruction,
                        episode=self.episode, round_id=self.round_id))
                    self.logs.append({"gate": 2, "tool": name, "reviewer_called": True, **checked})
                    record_gate(gate=2, trigger=name, candidate=self.adapter.dispatcher.primitive_id + 1,
                                epoch=self.round_id, decision=checked["result"], reason=checked["reason"],
                                reviewer_called=True, elapsed_s=time.monotonic() - review_started,
                                invalidated=checked["result"] == "DENY")
                    if checked["result"] == "DENY":
                        return self._deny(name, checked["reason"])
                    self.lease = checked.get("lease")
                    refresh = getattr(self, 'refresh_resolved_after_review', None)
                    if refresh is not None:
                        refreshed = refresh(name,parameters,workspace)
                        if refreshed.binding_version != resolved.binding_version:
                            raise InvalidReviewResponse('state refresh changed the reviewed binding acquisition')
                        resolved = refreshed
                if not self._current_lease(resolved.binding_version):
                    raise InvalidReviewResponse("action APPROVE lacks a current validated own lease")
                geometry = self.geometry_check(resolved, workspace, self.lease)
                if (not isinstance(geometry, GeometryDecision) or type(geometry.approved) is not bool
                        or type(geometry.system_failure) is not bool or not geometry.reason
                        or (geometry.approved and geometry.system_failure)):
                    raise ValueError("public geometry callback schema invalid")
                if not geometry.approved:
                    return self._deny(name, geometry.reason, system_failure=geometry.system_failure)
            transition_before = self.public_transition()
        except Exception as exc:
            if isinstance(exc, InvalidPrimitiveArguments):
                self.lease = None
                self._invalidate()
                self.first_action = True
                return self._feedback(name, f"invalid primitive arguments: {exc}", gate=2,
                                      root_cause="model_argument_failure", reviewer_called=False,
                                      decision="INVALID")
            if is_model_response_failure(exc):
                self.lease = None
                self._invalidate()
                self.first_action = True
                return self._feedback(name, f"invalid action reviewer response: {exc}", gate=2,
                                      root_cause="model_response_failure", reviewer_called=True)
            detail = str(exc)
            fault = getattr(exc, 'row', None)
            if isinstance(fault, dict) and fault.get('failure_kind') == 'public_reader_technical_fault':
                detail += f" ({fault.get('stage')}: {fault.get('error_type')}: {fault.get('error')})"
            return self._deny(name, f"action admission fault: {type(exc).__name__}: {detail}",
                              system_failure=True)

        error = None
        result = None
        outcome = None
        try:
            with self.adapter.primitive_scope(name, parameters=parameters, completion_controls=controls,
                                              episode=self.episode, round_id=self.round_id,
                                              binding_version=resolved.binding_version,
                                              lease=self._commit_lease()):
                try:
                    result = self.base.execute_tool(name, parameters)
                finally:
                    # RPent catches generic handler exceptions. Inspect the actual
                    # dispatcher while this primitive is still owned, even then.
                    outcome = self.adapter.dispatcher.last_outcome
        except Exception as exc:
            error = exc
        self.first_action = False
        try:
            self.after_transition = self.public_transition() != transition_before
        except Exception as exc:
            error = error or exc
        if outcome is not None:
            self.lease = None
            self.first_action = True  # A new proposal must obtain fresh gate-2 admission.
            self._invalidate()
            return self._feedback(name, outcome["reason"], gate=3,
                                  system_failure=outcome.get("system_failure") is True,
                                  root_cause=outcome.get("root_cause"),
                                  reviewer_called=outcome.get("reviewer_called",
                                      outcome.get("root_cause") == "model_response_failure"),
                                  commit_outcome=dict(outcome))
        if error is not None:
            return self._deny(name, f"primitive execution fault: {type(error).__name__}: {error}",
                              system_failure=True)
        return result


def inject_planner_policy(planner, *, method: str, approved_policy: PolicySource | None,
                          expected_policy_sha256: str):
    """Bind exact approved policy at the SDK thread/start trusted field.

    B0 is a strict identity operation. For both actual FrozenLocalCodex
    (_turn_options) and dashboard session (_options), callers must use
    planner_turn_options at Thread.turn: installed SDK does not accept the
    thread-only base_instructions keyword there. This function starts no client.
    """
    profile = METHOD_PROFILES[MethodName(method)]
    if not profile.policy_in_planner:
        return planner
    if (type(approved_policy) is not PolicySource or approved_policy.approved is not True
            or not isinstance(approved_policy.text, str) or not approved_policy.text.strip()
            or not isinstance(expected_policy_sha256, str) or approved_policy.sha256 != expected_policy_sha256
            or sha256(approved_policy.text.encode()).hexdigest() != expected_policy_sha256):
        raise ValueError("exact approved PolicySource/hash required for B1+")
    attributes = [key for key in ("_turn_options", "_options") if hasattr(planner, key)]
    if not attributes or any(type(getattr(planner, key)) is not dict for key in attributes):
        raise ValueError("actual trusted SDK option dictionary unavailable")
    if any(getattr(planner, key).get("base_instructions", approved_policy.text)
           != approved_policy.text for key in attributes):
        raise ValueError("existing trusted instructions differ; do not silently overwrite")
    for key in attributes:
        getattr(planner, key)["base_instructions"] = approved_policy.text
    return planner


def planner_turn_options(planner, *, asynchronous=False):
    """Keep original turn kwargs; policy persists on its already-created thread."""
    options = getattr(planner, "_options" if asynchronous else "_turn_options")
    return {key: value for key, value in options.items() if key != "base_instructions"}


def _filtered_turn_function(function, *, option_attribute, expected_source_sha256):
    """One exact original call-site rewrite; never replace planner.solve."""
    source = textwrap.dedent(inspect.getsource(function))
    if source_custody_enabled() and sha256(source.encode()).hexdigest() != expected_source_sha256:
        raise ValueError("original planner function source hash changed")
    if function.__closure__:
        raise ValueError("unexpected closed-over planner function; cannot rebind by source")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError("unexpected original planner function shape")
    rewritten = 0
    for node in ast.walk(nodes[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if (node.func.attr == "turn" and keyword.arg is None
                    and isinstance(value, ast.Attribute) and value.attr == option_attribute
                    and isinstance(value.value, ast.Name) and value.value.id == "self"):
                keyword.value = ast.Call(func=ast.Name(id="_v43_turn_options", ctx=ast.Load()),
                                         args=[ast.Name(id="self", ctx=ast.Load())],
                                         keywords=[ast.keyword(arg="asynchronous",
                                                               value=ast.Constant(option_attribute == "_options"))])
                rewritten += 1
    if rewritten != 1:
        raise ValueError("expected exactly one original Thread.turn options expansion")
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    namespace = dict(function.__globals__)
    namespace["_v43_turn_options"] = planner_turn_options
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    selected = namespace[function.__name__]
    selected.__defaults__, selected.__kwdefaults__ = function.__defaults__, function.__kwdefaults__
    selected.__annotations__, selected.__qualname__ = function.__annotations__, function.__qualname__
    selected.__v43_original_sha256__ = expected_source_sha256
    return selected


def bind_planner_policy_transport(planner, *, method, approved_policy, expected_policy_sha256,
                                  run_session_sha256, dashboard_session_class=None,
                                  submit_sha256=None, solve_dashboard_sha256=None):
    """Opt-in instance binding for the actual synchronous and Async SDK seams.

    Recompile only the single **options expansion at Thread.turn. thread_start
    retains base_instructions. Async uses an isolated subclass and the original
    _solve_dashboard code object with one rebound class global; upstream classes
    and defaults stay unchanged. No client, thread, model or service is started.
    """
    if not METHOD_PROFILES[MethodName(method)].policy_in_planner:
        return planner
    if "_run_session" in vars(planner) or "_solve_dashboard" in vars(planner):
        raise ValueError("planner instance already rebound; require a fresh selected instance")
    run = planner._run_session.__func__
    selected_run = _filtered_turn_function(run, option_attribute="_turn_options",
                                           expected_source_sha256=run_session_sha256)
    selected_dashboard = None
    if hasattr(planner, "_solve_dashboard"):
        if dashboard_session_class is None or (source_custody_enabled() and (submit_sha256 is None or solve_dashboard_sha256 is None)):
            raise ValueError("actual Async dashboard seam must also be source-bound")
        solve = planner._solve_dashboard.__func__
        source = textwrap.dedent(inspect.getsource(solve))
        if source_custody_enabled() and (sha256(source.encode()).hexdigest() != solve_dashboard_sha256
                or solve.__globals__.get("_CodexDashboardSession") is not dashboard_session_class):
            raise ValueError("original Async constructor source/global changed")
        selected_submit = _filtered_turn_function(dashboard_session_class.submit,
                                                  option_attribute="_options",
                                                  expected_source_sha256=submit_sha256)
        session = type("V43PolicyDashboardSession", (dashboard_session_class,),
                       {"submit": selected_submit, "__module__": __name__})
        globals_copy = dict(solve.__globals__)
        globals_copy["_CodexDashboardSession"] = session
        selected_dashboard = FunctionType(solve.__code__, globals_copy, solve.__name__, solve.__defaults__,
                                         solve.__closure__)
        selected_dashboard.__kwdefaults__ = solve.__kwdefaults__
    inject_planner_policy(planner, method=method, approved_policy=approved_policy,
                          expected_policy_sha256=expected_policy_sha256)
    planner._run_session = MethodType(selected_run, planner)
    if selected_dashboard is not None:
        planner._solve_dashboard = MethodType(selected_dashboard, planner)
    return planner
