"""P102 public admission callbacks; physical execution stays in RPent."""
from __future__ import annotations
from research.v4.leash_v43.run_manifest import source_custody_enabled

import ast
from contextlib import suppress
from dataclasses import asdict, replace
from hashlib import sha256
import inspect
import json
import math
import textwrap
import time
from types import FunctionType, SimpleNamespace

import numpy as np

from research.v4.leash_v43.adapter import V43PublicEnvAdapter
from research.v4.leash_v43.method_inputs import (
    PolicySource, PublicFrame, ReviewInputRequest, project_review_input,
    validate_review_response, InvalidReviewResponse,
)
from research.v4.leash_v43.owner_binding import bind_owner_abort
from research.v4.leash_v43.registry import GripperCalibration
from research.v4.leash_v43.reviewer_transport import ReviewerConfig, ReviewerSession
from research.v4.leash_v43.tool_boundary import (
    GeometryDecision, OriginalControlBounds, ResolvedPrimitive, V43ToolBoundary,
    bind_planner_policy_transport,
)
from research.v4.accelerated.p3_gated_toolkit import (
    OBSERVATION_OR_FILE_TOOLS, proposal_for_p3,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    PublicBox, PublicWorkspace, _segment_hits_box,
)


def admit_run(method, policy, calibration):
    if method == "B0":
        return
    if method != "Ours":
        raise ValueError("only B0/Ours own P102 consumer is selected")
    if (type(policy) is not PolicySource or policy.approved is not True
            or not policy.text.strip() or sha256(policy.text.encode()).hexdigest() != policy.sha256):
        raise ValueError("exact approved policy is required before service or Scene startup")
    if type(calibration) is not GripperCalibration or calibration.closed_gap_proves_empty:
        raise ValueError("measured own gripper calibration required; assisted hard-stop closure cannot prove empty")


def make_reviewer_adapter(scene, *, instruction, model, episode, output, policy, calibration, on_step):
    admit_run("Ours", policy, calibration)
    config = ReviewerConfig(episode_id=episode, model_id=model, reasoning_effort="high",
                            policy_text=policy.text, log_path=output / "reviewer.jsonl",
                            startup_timeout_s=30, enable_live_model=True,
                            policy_stage="development", policy_reviewed=True)
    reviewer = ReviewerSession(config, policy=policy)
    try:
        reviewer.start()
        adapter = V43PublicEnvAdapter(scene, method="Ours", instruction=instruction,
                                     reviewer=reviewer, calibration=calibration, policy=policy,
                                     grasp_assumption="unverified", on_sim_step=on_step)
    except BaseException:
        # The caller's tuple assignment has not received the worker yet.
        # Preserve the original fault while closing this local resource.
        with suppress(Exception):
            reviewer.close()
        raise
    return reviewer, adapter


def _public_workspace(provider, packet, eef, boxes):
    # The common packet may contain a declared planner latch. It is never read
    # here, by the new holding estimator, or included in reviewer projection.
    provider.current_packet = packet
    kind = provider.adapter.estimator.kind.value.lower()
    robot = SimpleNamespace(hold_estimate=kind)
    return PublicWorkspace(SimpleNamespace(robot=robot), eef, boxes)


def public_provider(components, scene, adapter, sam_client):
    """Reuse exact acquisition/SAM code, replacing only its evidence suffix."""
    original = components.PublicWorkspaceFactory
    source = textwrap.dedent(inspect.getsource(original.__call__))
    tree = ast.parse(source)
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != (
            "PublicWorkspace(self.adapter.capture_commit_evidence(), eef, tuple(boxes))"):
        raise RuntimeError("exact own public acquisition return seam changed")
    returns[0].value = ast.parse("_public_workspace(self, packet, eef, tuple(boxes))", mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__call__.__globals__)
    from research.v4.leash_v43.visual_holding import enabled
    if enabled():
        from research.v4.accelerated.public_geometry import PublicGeometryInvalid
        from research.v4.bridge.full_corner_lite import invalid_view
        tree.body[0].body.insert(0, ast.parse('self._invalid_current_entities = set()').body[0])
        for node in ast.walk(tree):
            if isinstance(node,ast.Try):
                node.handlers.insert(0,ast.parse('''try:
    pass
except PublicGeometryInvalid as exc:
    invalid_view(self,packet,view,noun,PROMPTS[noun],exc)
''').body[0].handlers[0])
            if isinstance(node,ast.If) and ast.unparse(node.test)=='detected':
                node.test=ast.parse('detected and noun not in self._invalid_current_entities',mode='eval').body
        namespace.update(PublicGeometryInvalid=PublicGeometryInvalid,invalid_view=invalid_view)
    namespace["_public_workspace"] = _public_workspace
    ast.fix_missing_locations(tree)
    exec(compile(tree, original.__call__.__code__.co_filename, "exec"), namespace)
    cls = type("OwnP102V43PublicWorkspace", (original,), {
        "__call__": namespace["__call__"],
        "public_entity_names": tuple(components.bindings.PROMPTS),
    })
    return cls(scene, adapter, sam_client)


def _lease_digest(lease):
    return sha256(json.dumps(lease, sort_keys=True).encode()).hexdigest()


class Consumer:
    """One coordinator over existing public geometry and V43ToolBoundary."""
    def __init__(self, base, adapter, *, components, scene, sam_client, reviewer,
                 policy, instruction, episode, primitives, vla_chunk_controls=None, provider_factory=None):
        from research.v4.leash_v43.public_access import public_adapter, public_toolkit, public_scene, public_primitives
        self.base, self.adapter, self.components = public_toolkit(base), public_adapter(adapter), components
        self.scene, self.reviewer, self.policy = public_scene(scene), reviewer, policy
        self.instruction, self.episode = instruction, episode
        factory = public_provider if provider_factory is None else provider_factory
        self.provider = factory(components, self.scene, self.adapter, sam_client)
        self.primitives = public_primitives(primitives)
        self.vla_chunk_controls = vla_chunk_controls
        self.proposals, self.capabilities = [], []
        self.lease_receipt = None
        self.binding_version = None
        self.acquire = 0
        self._acquisition_command = None
        self.boundary = V43ToolBoundary(
            self.base, self.adapter, method="Ours", episode=episode, workspace=self.workspace,
            resolver=self.resolve, task_review=self.task_review, action_review=self.action_review,
            geometry_check=self.geometry, completion_bound=self.completion_bound,
            lease_is_current=self.lease_is_current, observation_tools=OBSERVATION_OR_FILE_TOOLS)
        from research.v4.leash_v43.visual_holding import enabled
        if enabled():
            self.boundary.refresh_resolved_after_review = self.resolve

        original_execute = self.boundary.execute_tool
        def execute(name, parameters):
            previous = self._acquisition_command
            self._acquisition_command = name
            try:
                return original_execute(name, parameters)
            finally:
                self._acquisition_command = previous
        self.boundary.execute_tool = execute

    def workspace(self):
        self.acquire += 1
        if self.boundary.task_approved and (self._acquisition_command not in {"release", "set_gripper", "hold"}
                or getattr(self, '_contact_geometry_required', False)):
            public = self.provider()
            packet = self.provider.current_packet
        else:
            packet = self.scene.capture_agent()
            eef = tuple(float(x) for x in packet["robot_state"]["eef_pos"])
            public = _public_workspace(self.provider, packet, eef, ())
        if set(packet.get("cameras", {})) != {"agentview", "robot0_eye_in_hand"}:
            raise ValueError("exact current two public views required")
        frame_id = packet.get("frame_id")
        if any(c.get("frame_id") != frame_id for c in packet["cameras"].values()):
            raise ValueError("mixed public acquisitions")
        # Holding/binding lifecycle and fresh visual acquisition are separate.
        # A new image alone does not expire a validated semantic lease.
        version = f"{self.episode}:1:holding{self.adapter.estimator.generation}"
        acquired = f"{version}:frame{frame_id}:capture{self.acquire}"
        self.binding_version = version
        workspace = SimpleNamespace(public=public, packet=packet, version=version, acquisition=acquired)
        from research.v4.leash_v43.visual_holding import enabled
        if enabled():
            observations = self.scene.public_grasp_observations()
            if observations is None:
                raise ValueError('visual holding requires common public observation boundary')
            # Fixed calibrated wrist channel for temporal numeric comparison;
            # original full two-view union remains separate payload geometry.
            from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox
            current_names = {b.name for b in public.visible_entities}
            motion = []
            for row in self.provider.log:
                if (row.get('frame_id') == frame_id and row.get('camera') == 'robot0_eye_in_hand'
                        and row.get('error') is None and row.get('name') in current_names
                        and 'box_lower' in row and 'box_upper' in row):
                    motion.append(PublicBox(row['name'],tuple(row['box_lower']),tuple(row['box_upper'])))
            observations.observe(packet, motion, payload_boxes=public.visible_entities)
            workspace.grasp_observations = observations
        return workspace

    def request(self, gate, workspace, *, action="", lease=None):
        frames = {}
        for name, camera in workspace.packet["cameras"].items():
            frames[name] = PublicFrame(name, camera["rgb"], camera["depth_m"],
                                       camera["intrinsics"], camera["extrinsics"],
                                       workspace.acquisition)
        state = workspace.packet["robot_state"]
        from scipy.spatial.transform import Rotation
        robot = {"eef_pos": tuple(state["eef_pos"]),
                 "eef_rotation": Rotation.from_quat(state["eef_quat"]).as_matrix().tolist(),
                 "gripper_aperture_m": float(np.abs(state["gripper_qpos"]).sum()),
                 "holding_estimate": self.adapter.estimator.kind.value}
        if self.adapter.estimator.initial_occupancy is not None:
            robot['initial_occupancy'] = self.adapter.estimator.initial_occupancy
        request = ReviewInputRequest(
            method="Ours", gate=gate, current_workspace=frames["agentview"],
            current_wrist=frames["robot0_eye_in_hand"],
            event_context={"trusted_instruction": self.instruction,
                           "proposed_action": action,
                           "event_description": (
                               f"Current semantic binding_version: {workspace.version}. "
                               "Any returned lease must use that version; evidence_version identifies the fresh acquisition.")},
            robot_state=robot, lease=lease, evidence_version=workspace.acquisition)
        from research.v4.leash_v43.visual_holding import enabled
        if enabled() and gate in (2,3):
            observations = workspace.grasp_observations
            robot['holding_proprio'] = self.adapter.estimator.holding_proprio.value
            robot['holding_evidence'] = self.adapter.estimator.visual_record
            request = replace(request,robot_state=robot,history=observations.history(),
                event_context={**request.event_context,
                    'holding_observations': json.dumps(observations.metadata(),allow_nan=False)})
        names = getattr(self.provider, "public_entity_names", None)
        if names is None:
            names = getattr(self.provider, "public_prompts", None)
        if gate == 2 and names is not None:
            from research.v4.bridge.full_corner_lite import declare_public_names
            request = replace(request, event_context=declare_public_names(
                request.event_context, workspace, names))
        # Method projection happens here BEFORE the existing real transport.
        projected = project_review_input(request, self.policy)
        projected["deadline_monotonic_ns"] = time.monotonic_ns() + 180_000_000_000
        return projected

    def task_review(self, *, workspace, **kwargs):
        return self.reviewer.review(self.request(1, workspace))

    def action_review(self, *, name, parameters, workspace, lease, **kwargs):
        current = lease if lease is not None and self.lease_is_current(lease) else None
        response = self.reviewer.review(self.request(
            2, workspace, action=f"{name}({json.dumps(parameters, sort_keys=True)})", lease=current))
        checked = validate_review_response("Ours", 2, response)
        from research.v4.leash_v43.visual_holding import enabled
        if enabled() and 'holding_assessment' in checked:
            self.adapter.estimator.accept_visual(workspace.grasp_observations,checked['holding_assessment'])
            workspace.reviewed_binding_version = workspace.version
            # Same acquisition and candidate: state fusion precedes geometry,
            # without another capture, lookup, model call or approval rewrite.
            workspace.public = replace(workspace.public,evidence=SimpleNamespace(robot=SimpleNamespace(
                hold_estimate=self.adapter.estimator.kind.value.lower())))
        if checked["result"] == "APPROVE" and "lease" in checked:
            if "owner_method" not in checked["lease"]:
                raise InvalidReviewResponse("Action review requires the full owned public lease, not a compact commit lease")
            self.lease_receipt = _lease_digest(checked["lease"])
        return checked

    def lease_is_current(self, lease):
        return bool(self.boundary.round_open and self.lease_receipt is not None
                    and lease.get("owner_method") == "Ours"
                    and lease.get("binding_version") == self.binding_version
                    and _lease_digest(lease) == self.lease_receipt)

    def resolve(self, name, parameters, workspace):
        public = workspace.public
        kind = self.adapter.estimator.kind.value.lower()
        visual = getattr(self.adapter.estimator,'visual_record',None)
        if kind == 'held' and visual is not None:
            if visual['full_extent_supported'] and visual['relative_corners'] is not None:
                from scipy.spatial.transform import Rotation
                r = Rotation.from_quat(np.asarray(workspace.packet['robot_state']['eef_quat'])).as_matrix()
                xyz = np.asarray(public.eef_xyz)
                points = np.asarray(visual['relative_corners']) @ r.T + xyz
                u = visual['relative_uncertainty_m']
                associated = PublicBox(visual['entity'],tuple(points.min(axis=0)-u),tuple(points.max(axis=0)+u))
                public = replace(public,visible_entities=tuple(b for b in public.visible_entities
                    if b.name != visual['entity'])+(associated,))
                workspace.public = public
            else:
                public = replace(public,visible_entities=tuple(b for b in public.visible_entities
                    if b.name != self.components.binding.source))
                workspace.public = public
        box = public.box(self.components.binding.source)
        sweep = self.components.bindings.ObservedPayloadSweep(public) if box is not None else None
        # Use the real parser. UNKNOWN remains UNKNOWN in estimator/reviewer;
        # a possibly-held visible payload receives its conservative envelope.
        parser_public = public
        if kind == "held" and sweep is None:
            parser_public = replace(public, evidence=SimpleNamespace(
                robot=SimpleNamespace(hold_estimate="unknown")))
        proposal = proposal_for_p3(name, parameters, parser_public, self.provider.locate_vla_target,
                                   held_sweep=sweep)
        if kind != "empty" and sweep is not None:
            proposal = replace(proposal, operated_entity=self.components.binding.source,
                               gripper_radius_m=sweep.gripper_radius_m,
                               held_object_radius_m=sweep.held_extra_radius_m)
        missing = kind != "empty" and sweep is None
        proposal = replace(proposal, geometry_ambiguous=proposal.geometry_ambiguous or missing or kind == "unknown")
        self.proposals.append({"name": name, "arguments": parameters, "proposal": asdict(proposal),
                               "holding_source": "public_visual_and_executed_records" if visual else "executed_records_and_public_proprio"})
        return ResolvedPrimitive(workspace.version, proposal,
                                 entities=tuple(x for x in (proposal.operated_entity,) if x),
                                 geometry_unknown=proposal.geometry_ambiguous)

    def geometry(self, resolved, workspace, lease):
        proposal = resolved.proposal
        public = workspace.public
        # Pure release/closure admission has no route or commit-condition check.
        if proposal.name in {"release", "set_gripper", "hold"} and not proposal.motion_waypoints:
            return GeometryDecision(True, "validated own call; physical commits remain at the full-vector outlet")
        if (self.adapter.estimator.kind.value.lower() != "empty"
                and public.box(self.components.binding.source) is None):
            self.capabilities.append({"kind": "missing_public_payload_envelope", "system_failure": False})
            return GeometryDecision(False, "Current public payload envelope is unavailable")
        path = proposal.motion_waypoints
        if proposal.name == "vla_act":
            path = (public.eef_xyz, proposal.vla_target_xyz) if proposal.vla_target_xyz else ()
        if not path or any(p is None for p in path):
            self.capabilities.append({"kind": "missing_public_motion_target", "system_failure": False})
            return GeometryDecision(False, "Allowed public motion target is unavailable")
        radius = proposal.gripper_radius_m + proposal.held_object_radius_m
        for region in lease.get("no_go_regions", ()):
            if isinstance(region, str):
                box = public.box(region)
                if box is None:
                    self.capabilities.append({"kind": "missing_public_lease_entity", "system_failure": False})
                    return GeometryDecision(False, "Reviewer-bound public entity is not localized")
            else:
                box = PublicBox(region["name"], tuple(region["lower_xyz"]), tuple(region["upper_xyz"]))
            expanded = box.expanded(radius)
            if any(_segment_hits_box(a, b, expanded) for a, b in zip(path, path[1:])):
                return GeometryDecision(False, "Public payload/gripper path intersects a validated no-go region")
            if proposal.placement_xyz is not None and expanded.contains(proposal.placement_xyz):
                return GeometryDecision(False, "Public placement intersects a validated no-go region")
        return GeometryDecision(True, "Existing public segment/AABB geometry passed")

    def completion_bound(self, name, parameters):
        dt = self.scene.control_timestep
        # There is no new episode control budget. Each original finite loop is
        # its own completion bound; the configured primitive/turn/wall caps remain.
        bounds = OriginalControlBounds(self.primitives, remaining_controls=lambda: count,
                                       control_timestep=dt, vla_chunk_controls=self.vla_chunk_controls)
        # Validate the proposal before treating its loop length as the cap.
        # Invalid model arguments do not mean the physical runtime is broken.
        count = bounds.primitive_controls(name, parameters)
        return count

    def task_report(self):
        return {"logs": [dict(row) for row in self.boundary.logs if row.get("gate") == 1]}

    def task_fault(self):
        return any(row.get("gate") == 1 and row.get("system_failure") for row in self.boundary.logs)

    def action_report(self):
        return {"logs": [dict(row) for row in self.boundary.logs if row.get("gate") in (2, 3)],
                "proposals": self.proposals, "public_capability_uncertainties": self.capabilities,
                "perception": self.provider.log}


def commit_report(adapter):
    return {"logs": [dict(row) for row in adapter.dispatcher.logs],
            "initial_occupancy": adapter.estimator.initial_occupancy,
            "last_outcome": adapter.dispatcher.last_outcome,
            "holding_proprio": adapter.estimator.holding_proprio.value,
            "holding_estimate": adapter.estimator.kind.value,
            "visual_holding": adapter.estimator.visual_record,
            "visual_holding_updates": adapter.estimator.visual_updates,
            "holding_source": "public_visual_and_executed_records" if adapter.estimator.visual_record else "executed_records_and_public_proprio",
            "grasp_assumption": adapter.estimator.grasp_assumption}


def run_planner(original_run, *, policy, **kwargs):
    """Same original loop CodeType, one explicit planner-constructor binding."""
    if source_custody_enabled():
        from .binding import planner_pins
        expected = planner_pins()
    else:
        expected = dict.fromkeys(("run_session", "submit", "solve_dashboard"))
    original_builder = original_run.__globals__["build_b0_planner"]

    def approved_builder(**builder_kwargs):
        planner = original_builder(**builder_kwargs)
        module = inspect.getmodule(planner._run_session.__func__)
        return bind_planner_policy_transport(
            planner, method="Ours", approved_policy=policy, expected_policy_sha256=policy.sha256,
            run_session_sha256=expected["run_session"],
            dashboard_session_class=module._CodexDashboardSession,
            submit_sha256=expected["submit"], solve_dashboard_sha256=expected["solve_dashboard"])

    namespace = dict(original_run.__globals__)
    namespace["build_b0_planner"] = approved_builder
    selected = FunctionType(original_run.__code__, namespace, original_run.__name__, original_run.__defaults__)
    selected.__kwdefaults__ = original_run.__kwdefaults__
    return selected(**kwargs)
