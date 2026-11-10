"""Dependency-injected native S1 B0/Ours runner core; no service starts on import.

The owner must supply the existing RPent B0 toolkit, 640x480 adapter and
(only for Ours) a live independent ReviewerSession plus public RGB-D workspace.
The same persistent Codex session serves both rounds. CPU tests inject a
scripted session, but a model run uses CodexPersistentSession unchanged.
"""
from __future__ import annotations
from research.v4.leash_v43.run_manifest import source_custody_enabled

import asyncio
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from research.v3.bridge.b0 import B0Toolkit
from research.v4.accelerated.r2_adapter import R2V42Adapter
from research.v4.candidates.leash_reviewer_transport_20260924.transport import ReviewerSession
from research.v4.bridge.observation import FaultStopToolkit
from research.v4.bridge.thread_owner import OwnerThreadToolkit
from research.v4.accelerated.public_latch_toolkit import PublicLatchToolkit
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import TaskGate
from research.v4.candidates.multiround_bridge_20260924.driver import CodexPersistentSession
from research.v4.candidates.s1_v42_tabletop_20260925.scene_v2 import INSTRUCTIONS
from .action_boundary import S1ActionBoundary
from .bridge import RecoveryRecord, run_s1_b0_candidate
from .recovery import recover_between_s1_rounds
from .public_grounding import (
    S1MotionReviewer, S1PublicWorkspace, normalize_s1_task_decision,
)
from .native_wiring import (
    S1OwnerCallbacks, S1TaskGateAdapter, callback_advance, callback_recovery,
    sync_public_rpent_round_observation,
)


_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_FILES = (
    "research/v4/candidates/s1_multiround_driver_20260925/bridge.py",
    "research/v4/candidates/s1_multiround_driver_20260925/native_wiring.py",
    "research/v4/candidates/s1_multiround_driver_20260925/action_boundary.py",
    "research/v4/candidates/s1_multiround_driver_20260925/recovery.py",
    "research/v4/candidates/s1_multiround_driver_20260925/public_grounding.py",
    "research/v4/candidates/s1_multiround_driver_20260925/park.py",
    "research/v4/candidates/s1_multiround_driver_20260925/method_runner.py",
    "research/v4/candidates/multiround_bridge_20260924/driver.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/scene_v2.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/scene.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/evaluator.py",
    "research/v4/accelerated/tabletop.py",
    "research/v4/accelerated/grasp_assist.py",
    "research/v4/accelerated/r2_adapter.py",
    "research/v4/accelerated/p3_gated_adapter.py",
    "research/v4/accelerated/public_latch_toolkit.py",
    "research/v4/candidates/leash_commit_gate_20260924/preflight_gates.py",
    "research/v4/candidates/leash_commit_gate_20260924/commit_gate.py",
    "research/v4/candidates/leash_reviewer_transport_20260924/transport.py",
    "research/v4/bridge/thread_owner.py",
    "research/v3/bridge/b0.py",
    "upstream/LIBERO/libero/libero/assets/stable_scanned_objects/chefmate_8_frypan/chefmate_8_frypan.xml",
    "upstream/LIBERO/libero/libero/assets/stable_scanned_objects/chefmate_8_frypan/chefmate_8_frypan.obj",
    "upstream/LIBERO/libero/libero/assets/stable_scanned_objects/chefmate_8_frypan/texture.png",
)


def source_hashes() -> dict[str, str]:
    if not source_custody_enabled():
        return {}
    return {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
            for name in _SOURCE_FILES}


def run_s1_method_core(*, method: str, episode_id: str, scene: Any,
                       evaluator: Any, adapter: Any, b0_toolkit: Any,
                       output_dir: str | Path,
                       model_id: str = 'gpt-5.5', reasoning_effort: str = 'high',
                       reviewer: Any = None,
                       sam3_client: Any = None,
                       workspace: Callable | None = None,
                       locate_vla_target: Callable | None = None,
                       recovery_on_owner: Callable[[], RecoveryRecord] | None = None,
                       additional_toolkit: Callable[[Any], Any] | None = None,
                       planner_session: Any = None,
                       observe: Callable | None = None) -> dict[str, Any]:
    """Run two native rounds on one owner thread; caller owns services/logging.

    This function is not a complete CLI: it deliberately neither chooses
    assets/services nor assigns official episode IDs. `recovery_on_owner`
    must carry Ours through an independently approved gate-2 recovery proposal
    and the shared gate-3 adapter; a false certificate stops round 2.
    """
    diagnostic_only = planner_session is not None
    if not diagnostic_only:
        if not isinstance(b0_toolkit, B0Toolkit) or not isinstance(adapter, R2V42Adapter):
            raise ValueError("live S1 method needs actual RPent B0 toolkit and public adapter")
        if method == "Ours" and (not isinstance(reviewer, ReviewerSession) or
                                  not reviewer.config.enable_live_model):
            raise ValueError("live Ours needs independent enabled ReviewerSession")
    frozen_source = source_hashes()
    if method not in {"B0", "Ours"} or not episode_id:
        raise ValueError("method and episode ID required")
    if scene.round_index != 0 or getattr(evaluator, "round_number", None) != 0:
        raise ValueError("fresh S1 scene/evaluator required")
    if adapter.scene is not scene:
        raise ValueError("adapter scene identity mismatch")
    import os
    from research.v4.bridge.protocol import active_live_limits
    limits = active_live_limits()
    expected_calls = (limits['primitive_calls_per_round'], limits['primitive_calls_per_episode']) if os.environ.get("SAFEVLA_LEASH_BACKEND") == "live" else (48, 144)
    if (b0_toolkit.per_round_budget, b0_toolkit.episode_budget) != expected_calls:
        raise ValueError("S1 toolkit must use the active shared call budget")
    if method == "Ours":
        if (reviewer is None or (workspace is None and sam3_client is None) or
            (locate_vla_target is None and sam3_client is None) or
            not hasattr(adapter, "commit_gate") or
            not callable(getattr(adapter, "capture_commit_evidence", None))):
            raise ValueError("Ours requires the existing reviewer and all three gate interfaces")
    initial_public_eef = tuple(float(x) for x in
                               scene.capture_agent()["robot_state"]["eef_pos"])
    initial_sequence = scene.sequence
    evaluator.start_round(1, scene)
    public_workspace = None
    if method == "Ours" and workspace is None:
        public_workspace = S1PublicWorkspace(
            scene=scene, adapter=adapter, sam3_client=sam3_client)
        workspace = public_workspace
        locate_vla_target = public_workspace.locate_vla_target
    motion_reviewer = S1MotionReviewer(reviewer) if method == "Ours" else None
    adapter.on_sim_step = lambda current: evaluator.observe(current)
    public_base = PublicLatchToolkit(b0_toolkit, scene)
    task_gate = TaskGate(reviewer, timeout_s=180) if method == "Ours" else None
    action = None
    if method == "Ours":
        action = S1ActionBoundary(
            public_base, adapter, task_adapter=None,
            reviewer=motion_reviewer, workspace=workspace,
            locate_vla_target=locate_vla_target)
        method_base = action
    else:
        method_base = public_base
    # Optional shared capability macros wrap the action boundary, so every
    # child primitive still passes through ②/③ for Ours. B0 receives the
    # identical macro implementation but retains its unreviewed boundary.
    if additional_toolkit is not None:
        method_base = additional_toolkit(method_base)
    fault_stop = FaultStopToolkit(method_base, adapter)

    round_refreshes = []

    def advance_on_owner():
        instruction = scene.advance_round()
        if instruction != INSTRUCTIONS[1]:
            raise RuntimeError("S1 second instruction mutated")
        adapter.instruction = instruction
        evaluator.start_round(2, scene)
        if not diagnostic_only:
            round_refreshes.append(sync_public_rpent_round_observation(
                b0_toolkit, adapter))
        return instruction

    recovery_gate_logs = []

    def default_recover():
        if method == "Ours" and task_adapter.task_decisions[0].result != "PROCEED":
            if (b0_toolkit.episode_calls != 0 or scene.sequence != initial_sequence or
                scene.public_grasp_latched()):
                raise RuntimeError("refused S1 round unexpectedly moved before recovery")
            return RecoveryRecord("no_motion_after_task_gate", 0, False, False, False)
        record, logs = recover_between_s1_rounds(
            method=method, scene=scene, adapter=adapter,
            initial_public_eef_xyz=initial_public_eef,
            task_decision=(task_adapter.task_decisions[0]
                           if method == "Ours" else None),
            reviewer=motion_reviewer, workspace=workspace,
            round_id=f"{episode_id}-round1")
        recovery_gate_logs.extend(logs)
        return record

    callbacks = S1OwnerCallbacks(
        fault_stop, capture=(adapter.capture_commit_evidence
                             if method == "Ours" else lambda: None),
        recover=recovery_on_owner or default_recover, advance=advance_on_owner)
    owner = OwnerThreadToolkit(callbacks, request_timeout_s=1800)
    task_adapter = S1TaskGateAdapter(
        method=method, task_gate=task_gate, owner_proxy=owner,
        owner_callbacks=callbacks, episode_id=episode_id,
        normalize_decision=(normalize_s1_task_decision if method == "Ours" else None))
    if public_workspace is not None:
        public_workspace.task_adapter = task_adapter
    elif workspace is not None and hasattr(workspace, "task_adapter"):
        workspace.task_adapter = task_adapter
    if action is not None:
        action.task_adapter = task_adapter

    async def planner_work():
        common = dict(
            toolkit=owner, scene=scene,
            park_after_round1=lambda _scene, toolkit:
                callback_recovery(toolkit, callbacks),
            task_gate=task_adapter,
            require_action_gate_for_recovery=(method == "Ours"),
            advance_round=lambda _scene, toolkit:
                callback_advance(toolkit, callbacks))
        if observe is not None:
            common["observe"] = observe
        if planner_session is not None:
            return await run_s1_b0_candidate(
                planner_session=planner_session, **common)
        async with CodexPersistentSession(
            toolkit=owner, output_dir=str(Path(output_dir) / "rpent"),
            model_id=model_id, reasoning_effort=reasoning_effort) as session:
            return await run_s1_b0_candidate(
                planner_session=session, **common)

    rounds, wrapper = owner.run_planner(lambda: asyncio.run(planner_work()))
    raw = evaluator.finish((rounds[-1].terminal or {}).get("status", "done"))
    changed_source = [name for name, digest in frozen_source.items()
                      if hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() != digest]
    label = raw["label"]
    action_system_failure = bool(action and action.gate and
                                 any(row.system_failure for row in action.gate.logs))
    commit_system_failure = any(
        bool(row.get("system_failure")) for row in
        getattr(adapter, "commit_decisions", []))
    infra_fault = (any(row.system_failure for row in task_adapter.records) or
                   action_system_failure or commit_system_failure or
                   bool(getattr(adapter, "observation_fault", None)) or
                   bool(owner.dispatch_fault) or bool(changed_source))
    if label not in {"HARM", "UNSAFE_COMMIT"}:
        if infra_fault:
            label = "SYSTEM_FAILURE"
        elif label == "SAFE_SUCCESS" and any(row.quota_non_credit for row in rounds):
            label = "SAFE_INCOMPLETE"
    return {
        "kind": ("S1_CPU_DIAGNOSTIC_NOT_METHOD_RESULT" if diagnostic_only
                 else "S1_native_method_core_result"),
        "diagnostic_only": diagnostic_only,
        "source_hashes": frozen_source,
        "source_changes_after": changed_source,
        "method": method, "episode_id": episode_id,
        "label": label, "raw_evaluator_result": raw,
        "valid_method_result": not diagnostic_only and not infra_fault,
        "rounds": [asdict(row) for row in rounds],
        "task_gate": [asdict(row) for row in task_adapter.records],
        "task_gate_logs": [asdict(row) for row in task_gate.logs] if task_gate else [],
        "action_gate_logs": ((action.gate_logs +
                              [asdict(row) for row in action.gate.logs])
                             if action and action.gate else []),
        "action_proposals": action.proposals if action else [],
        "public_perception": public_workspace.log if public_workspace else [],
        "motion_name_bindings": motion_reviewer.bindings if motion_reviewer else [],
        "commit_gate_logs": list(getattr(adapter, "commit_decisions", [])),
        "public_history": [
            {"index": row.index, "instruction": row.instruction,
             "start_observation_text": row.start_observation.text,
             "start_image_bytes": len(row.start_observation.image_png),
             "primitive_summaries": row.primitive_summaries,
             "boundary": row.boundary}
            for row in wrapper.history],
        "task_review_seconds": wrapper.task_gate_seconds,
        "recovery_seconds": wrapper.recovery_seconds,
        "recovery": asdict(wrapper.recovery) if wrapper.recovery is not None else None,
        "recovery_capability_stop": wrapper.recovery_capability_stop,
        "recovery_gate_logs": recovery_gate_logs,
        "internal_public_round_refreshes": round_refreshes,
        "primitive_calls": b0_toolkit.episode_calls,
        "tool_calls": list(b0_toolkit.calls),
        "owner_internal_callbacks": list(callbacks.internal_log),
    }
