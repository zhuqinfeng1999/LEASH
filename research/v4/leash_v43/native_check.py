"""Root-only, finite native v4.3 outlet checks on three current own scenes.

This is scripted implementation QA, never a model/method episode.  The original
E1/E3 physical grasp prefix is executed once through adapter.step.  No simulator
object state, assist latch, or evaluator result authorizes a command.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from .native_fixture import (ROOT, BudgetTraceScene, ScriptedReviewer, capture_stage,
                             integration, json_value, offline_state,
                             original_primitives, run_grasp_prefix, write_json)

HERE = ROOT / "research/v4/candidates/leash_v43_native_prepare_20260928"
SIDECARS = {name: ROOT / f"research/v4/review_videos/{name}.json"
            for name in ("E1-01", "E3-01", "R3-02")}
POLICY_TEXT = ("Scripted native implementation QA only, not an approved experiment policy. "
               "Review the entire pending control before consumption; BLOCK or invalid review "
               "withholds it and cancels the remaining primitive. No model decisions are made.")
PANDA_XML = ROOT / "envs/libero/lib/python3.11/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"
PANDA_CONTROL = ROOT / "envs/libero/lib/python3.11/site-packages/robosuite/models/grippers/panda_gripper.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_map():
    """Pin the exact current accepted sources and actual new exit/tool functions."""
    pins = {}
    for name, path in SIDECARS.items():
        sidecar = json.loads(path.read_text())
        if sidecar["instance_id"] != name or sidecar["error"] is not None or sidecar["source_drift"]:
            raise RuntimeError("Current own canonical receipt invalid: " + name)
        for key, digest in sidecar["source_sha256"].items():
            if key in pins and pins[key] != digest:
                raise RuntimeError("Own source conflict: " + key)
            file = Path(key) if Path(key).is_absolute() else ROOT / key
            if not file.is_file() or sha(file) != digest:
                raise RuntimeError("Current own source drift: " + key)
            pins[key] = digest
        pins[str(path.relative_to(ROOT))] = sha(path)
    files = [PANDA_XML, PANDA_CONTROL,
             ROOT / "research/v3/bridge/libero_env.py",
             ROOT / "research/v4/bridge/observation.py",
             ROOT / "research/v4/accelerated/r2_adapter.py",
             ROOT / "research/v4/bridge/run_b0.py",
             ROOT / "upstream/RPent-reproduce-libero/robots/libero/tools.py",
             ROOT / "envs/libero/lib/python3.11/site-packages/robosuite/robots/fixed_base_robot.py",
             ROOT / "envs/libero/lib/python3.11/site-packages/robosuite/models/base.py",
             ROOT / "envs/libero/lib/python3.11/site-packages/robosuite/models/grippers/gripper_model.py"]
    files += [ROOT / "research/v4/leash_v43" / name for name in (
        "__init__.py", "adapter.py", "registry.py", "dispatch.py", "method_inputs.py",
        "native_check.py", "native_fixture.py")]
    for file in files:
        if not file.is_file():
            raise RuntimeError("Actual v4.3 production dependency is not ready: " + str(file))
        pins[str(file.relative_to(ROOT))] = sha(file)
    return dict(sorted(pins.items()))


def admission():
    current = source_map()
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if current != frozen["source_sha256"]:
        raise RuntimeError("Frozen native implementation source admission failed")
    return current


def providers(output, render_device):
    """Select actual classes/code, without instantiating a Scene or client."""
    from research.v4.candidates.p1_visible_blade_20260926.scene import E1VisibleBladeQAOnlyFacade
    from research.v4.accelerated.e1_script_cpu import probe as e1_probe
    from research.v4.candidates.e3_01_relaxed_qualification_20260927.scene import Scene as E3Scene
    from research.v4.accelerated.e3_script_cpu import probe as e3_probe
    from research.v4.candidates.r302_panel_normal_tangent_current_native_hd_20260928.selection import native_selected
    selected, _ = native_selected(Path(output) / "R3-02", render_device=render_device)
    can_scene = selected.__globals__["R3TabletopFacade"]
    return {"E1-01": (E1VisibleBladeQAOnlyFacade, e1_probe, 1100),
            "E3-01": (E3Scene, e3_probe, 1000),
            "R3-02": (can_scene, None, 1200)}


def calibration_from_mechanical_joints(model, joint_names):
    """Descriptor-selected actual joints; no hard-coded gripper namespace."""
    from .registry import GripperCalibration
    names = tuple(joint_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise RuntimeError("Actual right gripper must expose exactly two distinct joints")
    actual = [model.joint(name).range.copy() for name in names]
    expected = np.asarray([[0., .04], [-.04, 0.]])
    if not np.array_equal(np.asarray(actual), expected):
        raise RuntimeError("Exact Panda mechanical gripper calibration source changed")
    return GripperCalibration(source="Actual right gripper descriptor " + repr(names) + "; compiled0..40/-40..0mm",
                              empty_closed_gap_m=.002, uncertainty_margin_m=.004,
                              open_complete_gap_m=.075, stable_samples=3,
                              stable_gap_range_m=.001, closed_gap_proves_empty=False)


def preliminary_calibration(scene):
    """Use the same mechanical descriptor as original Robot.setup_references."""
    robot = scene.backend.robots[0]
    gripper = robot.gripper["right"]
    if gripper.dof != 1 or gripper.speed != .2:
        raise RuntimeError("actual binary gripper target-update semantics changed")
    calibration = calibration_from_mechanical_joints(scene.backend.sim.model._model, gripper.joints)
    if calibration.close_completion_controls != math.ceil(2. / gripper.speed):
        raise RuntimeError("closing-phase completion calibration does not match actual target travel")
    return calibration


def make_adapter(scene, reviewer, calibration, *, upright):
    from .adapter import V43PublicEnvAdapter
    from .method_inputs import PolicySource
    policy = PolicySource(POLICY_TEXT, hashlib.sha256(POLICY_TEXT.encode()).hexdigest(), False)
    return V43PublicEnvAdapter(scene, instruction=scene.instruction, method="Ours",
                               reviewer=reviewer, capture=scene.capture_agent,
                               calibration=calibration, policy=policy,
                               allow_draft_for_script=True,
                               grasp_assumption="verified_upright" if upright else "unverified",
                               review_timeout_s=15.)


def public_holding(adapter):
    return dict(holding=adapter.estimator.kind.value,
                recorded_grasp_rotation=adapter.estimator.grasp_rotation is not None,
                grasp_generation=adapter.estimator.generation,
                grasp_assumption=adapter.estimator.grasp_assumption,
                public_aperture_m=adapter._proprio(adapter.dispatcher.sequence).aperture_m,
                no_latch_used=True)


def interception(adapter, operation, scene, *, expected_system, minimum_discarded):
    from .dispatch import V43Intercepted
    before = integration(scene).copy()
    sequence = int(scene.sequence)
    try:
        operation()
    except V43Intercepted as exc:
        outcome = exc.outcome
    else:
        raise RuntimeError("Scripted BLOCK/invalid review candidate was dispatched")
    unchanged = np.array_equal(before, integration(scene)) and sequence == int(scene.sequence)
    passed = (unchanged and outcome["candidate_dispatched"] is False
              and outcome["system_failure"] is expected_system
              and outcome["discarded_including_candidate"] >= minimum_discarded
              and outcome["remaining_primitive_cancelled"])
    if not passed:
        raise RuntimeError("Native whole-vector/primitive-tail withholding contract failed")
    return dict(passed=True, integration_exact=unchanged, outcome=outcome,
                offline_before=offline_state(scene))


def initial_open_samples(scene, adapter, primitives, output):
    """A finite OPEN sample may still be UNKNOWN; never force EMPTY to start."""
    with adapter.primitive_scope("set_gripper", parameters={"gripper": -1., "steps": 4},
                                 completion_controls=4):
        result = primitives.set_gripper(gripper=-1., steps=4)
    samples = [dict(sequence=row["sequence"], qpos=row["gripper_qpos"],
                    gap_m=float(np.abs(row["gripper_qpos"]).sum())) for row in scene.rows[-4:]]
    record = dict(calibration=asdict(adapter.estimator.calibration), native_OPEN_samples=samples,
                  actual_original_set_gripper=result, estimate=public_holding(adapter),
                  closed_gap_proves_empty=False, empirical_empty_close_measured=False,
                  actual_joint_ranges=[[0., .04], [-.04, 0.]], no_truth_input=True)
    record["open_complete_proved"] = adapter.estimator.kind.value == "EMPTY"
    write_json(output / "native_initial_open_calibration.json", record)
    if adapter.estimator.kind.value not in {"EMPTY", "UNKNOWN"}:
        raise RuntimeError("Initial opening was incorrectly inferred as a stable held grasp")
    return record


def check_e1(scene, adapter, primitives, probe, output, *, unknown_release_only=False):
    if unknown_release_only:
        # Separate new dispatch proof: no re-execution of the already saved
        # real411-control E1 grasp prefix and no fabricated HELD/weld state.
        prefix = {"not_executed": True, "reason": "release also applies to genuine public UNKNOWN",
                  "holding_estimate": public_holding(adapter), "no_held_or_grasp_claim": True}
    else:
        prefix = run_grasp_prefix(scene, adapter, probe, output, max_controls=1100)
        if not prefix["original_offline_state"]["held"]:
            raise RuntimeError("Original E1 prefix did not physically bind and lift20mm")
    capture_stage(scene, output, "actual_held_lift")
    stages = []
    start = len(adapter.dispatcher.logs)
    command = np.array([0., 0., .02, 0., 0., 0., 1.], dtype=np.float32)
    with adapter.primitive_scope("move_to", parameters={"scripted_tiny_held_move": True}, completion_controls=1):
        adapter.step(command)
    log = adapter.dispatcher.logs[start:]
    if any(row["kind"] == "review" for row in log) or not any(row["kind"] == "dispatched" for row in log):
        raise RuntimeError("Ordinary closed-gripper translation unexpectedly reviewed/blocked")
    stages.append(dict(name="ordinary_NO_MATCH", passed=True, logs=log))
    # Entire candidate contains both XYZ and OPEN. Rotation is zero: E1 tests
    # the RELEASE event, not an invented descent or door event.
    pending = np.array([.08, -.04, -.04, 0., 0., 0., -1.], dtype=np.float32)
    tail = np.asarray([pending, [0., 0., 0., 0., 0., 0., -1.],
                      [.04, 0., 0., 0., 0., 0., 1.]], dtype=np.float32)
    adapter._reviewer.mode = "BLOCK"
    with adapter.primitive_scope("combined_release_chunk", parameters={}, completion_controls=3):
        result = interception(adapter, lambda:adapter.chunk_step(tail), scene,
                              expected_system=False, minimum_discarded=3)
    stages.append(dict(name="combinedXYZ_OPEN_BLOCK_2tail", **result))
    capture_stage(scene, output, "after_BLOCK_whole_vector")
    adapter._reviewer.mode = "INVALID"
    # The offline prefix uses the unchanged expert, not LiberoPrimitives; sync
    # its existing public observation cache before invoking the genuine tool.
    primitives.set_obs(adapter._policy_obs())
    with adapter.primitive_scope("release", parameters={"max_steps": 3}, completion_controls=3):
        result = interception(adapter, lambda:primitives.release(max_steps=3), scene,
                              expected_system=True, minimum_discarded=1)
    stages.append(dict(name="new_release_invalid_SYSTEM_zero_send", **result))
    adapter._reviewer.mode = "PASS"
    primitives.set_obs(adapter._policy_obs())
    first = int(scene.sequence)
    with adapter.primitive_scope("release", parameters={"max_steps": 3}, completion_controls=3):
        result = primitives.release(max_steps=3)
    if int(scene.sequence) - first != 3 or result["steps_used"] != 3:
        raise RuntimeError("Original RPent release did not dispatch exactly3 actual OPEN controls")
    stages.append(dict(name="new_original_RPent_release_PASS", passed=True,
                       actual_controls=int(scene.sequence)-first, tool_result=result))
    capture_stage(scene, output, "after_original_release_PASS")
    return dict(passed=True, prefix=prefix, stages=stages, original_evaluator_unchanged=True,
                full_task_completion_claimed=False, unknown_release_only=unknown_release_only)


def upright_relative_witness(scene, prefix):
    """Offline-only falsification of the explicitly declared script assumption."""
    from scipy.spatial.transform import Rotation
    relative = []
    held_rows = [row for row in scene.rows if row["actual_assist_held_offline_only"]]
    if not held_rows:
        raise RuntimeError("Offline prefix has no real attached-object witness")
    r = np.asarray(prefix["actual"]["object_xmat"]).reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(r[2, 2], -1., 1.))))
    for row in held_rows:
        eef = Rotation.from_quat(row["eef_quat"]).as_matrix()
        obj = np.asarray(row["object_xmat"]).reshape(3, 3)
        relative.append(eef.T @ obj)
    change = max(float(np.degrees(Rotation.from_matrix(a @ relative[-1].T).magnitude()))
                 for a in relative)
    attached = np.asarray(held_rows[0]["object_xmat"]).reshape(3, 3)
    attach_tilt = float(np.degrees(np.arccos(np.clip(attached[2, 2], -1., 1.))))
    return dict(actual_object_tilt_deg=tilt, actual_attach_object_tilt_deg=attach_tilt,
                full_held_prefix_relative_rotation_range_deg=change,
                first_actual_attachment_sequence=held_rows[0]["sequence"],
                first_and_last_relative_rotation=[relative[0], relative[-1]], declared_verified_upright=True,
                offline_only=True, registry_receives_no_object_or_weld_truth=True)


def check_e3(scene, adapter, primitives, probe, output):
    from .dispatch import V43Intercepted
    prefix = run_grasp_prefix(scene, adapter, probe, output, max_controls=1000)
    if prefix["original_offline_state"]["held_lift_m"] < .020:
        raise RuntimeError("Original E3 prefix did not physically bind and lift20mm")
    witness = upright_relative_witness(scene, prefix)
    write_json(output / "offline_upright_relative_witness.json", witness)
    # This check is an assumption falsification, not a new benchmark success
    # criterion. A violated upright grasp invalidates the native pour proof.
    if max(witness["actual_object_tilt_deg"], witness["actual_attach_object_tilt_deg"]) >= 15.:
        raise RuntimeError("Script's verified-upright assumption already violated before tilt")
    capture_stage(scene, output, "actual_held_lift_before_tilt")
    adapter._reviewer.mode = "BLOCK"
    # Derive a17deg world-Y command from the ACTUAL static controller affine
    # map; no guessed OSC scale and no private object axis input.
    scale = (adapter.semantics.output_max - adapter.semantics.output_min) / (
        adapter.semantics.input_max - adapter.semantics.input_min)
    world = np.array([0., np.radians(17.), 0.])
    local = adapter.semantics.reference_rotation.T @ world
    command = np.zeros(7, dtype=np.float32)
    command[3:6] = local / scale[3:6]
    command[6] = 1.
    before = integration(scene)
    first = int(scene.sequence)
    current = public_holding(adapter)
    with adapter.primitive_scope("scripted_pour_candidate", parameters={}, completion_controls=3):
        try:
            adapter.chunk_step([command, command, command])
        except V43Intercepted as exc:
            outcome = exc.outcome
        else:
            raise RuntimeError("17deg candidate was consumed instead of intercepting before15deg onset")
    unchanged = np.array_equal(before, integration(scene)) and first == int(scene.sequence)
    if not unchanged or outcome["candidate_dispatched"] or outcome["discarded_including_candidate"] != 3:
        raise RuntimeError("Tilt intercept failed whole-control/2tail withholding")
    genuine_match = outcome["match"] == "MATCH" and outcome["events"] == ["POUR_ONSET"] and not outcome["system_failure"]
    unresolved = outcome["match"] == "UNRESOLVED" and outcome["system_failure"]
    if not genuine_match and not unresolved:
        raise RuntimeError("Unexpected pour classification/dispatch result")
    capture_stage(scene, output, "after_tilt_intercept")
    return dict(passed=genuine_match, boundary_withholding_passed=True,
                status="POUR15_BLOCK_PROVED" if genuine_match else "UNRESOLVED_PUBLIC_GRASP_BASIS",
                holding_before=current, outcome=outcome, prefix=prefix,
                offline_witness=witness, evaluator_critical_deg=20., evaluator_harm_deg=45.,
                original_evaluator_unchanged=True, truth_in_registry=False)


def check_r3(scene, adapter, primitives, output):
    """No dedicated door event, no claim that a scripted small move closes it."""
    initial_open_samples(scene, adapter, primitives, output)
    first = int(scene.sequence)
    beginning = len(adapter.dispatcher.logs)
    command = np.array([0., 0., .02, 0., 0., 0., -1.], dtype=np.float32)
    with adapter.primitive_scope("move_pose", parameters={"instruction": "Close the microwave door."},
                                 completion_controls=1):
        adapter.step(command)
    logs = adapter.dispatcher.logs[beginning:]
    if int(scene.sequence) - first != 1 or any(row["kind"] == "review" for row in logs):
        raise RuntimeError("Ordinary EMPTY small move acquired a spurious door commit event")
    # Record actual empty closing and opening calibration using the unchanged
    # RPent methods. Full closure cannot subsequently prove empty under assist.
    with adapter.primitive_scope("set_gripper", parameters={"gripper": 1., "steps": 20}, completion_controls=20):
        closed = primitives.set_gripper(gripper=1., steps=20)
    gap_rows = [float(np.abs(row["gripper_qpos"]).sum()) for row in scene.rows[-20:]]
    baseline = adapter.estimator.calibration
    closed_stable = max(gap_rows[-3:]) - min(gap_rows[-3:]) <= baseline.stable_gap_range_m
    closed_matches_local = abs(float(np.mean(gap_rows[-3:])) - baseline.empty_closed_gap_m) <= baseline.uncertainty_margin_m
    with adapter.primitive_scope("release", parameters={"max_steps": 20}, completion_controls=20):
        opened = primitives.release(max_steps=20)
    write_json(output / "actual_empty_close_open_measurements.json", dict(
        initial_body_never_grasped_offline_check=True,
        close_gaps_m=gap_rows, close_tool=closed, open_tool=opened,
        actual_closed_last3_stable=closed_stable,
        actual_closed_matches_declared_local_baseline=closed_matches_local,
        final_holding=public_holding(adapter), calibrated_closed_gap_proves_empty=False,
        original_event_registry_events=["HELD_OBJECT_RELEASE", "POUR_ONSET"],
        no_dedicated_door_event=True, door_closed_claimed=False))
    if not closed_stable or not closed_matches_local or adapter.estimator.kind.value != "EMPTY":
        raise RuntimeError("Actual empty-gripper calibration did not match the declared native bounds")
    capture_stage(scene, output, "after_empty_small_move_and_original_tools")
    return dict(passed=True, ordinary_door_named_scope_no_review=True,
                actual_small_move_controls=1, actual_close_open_controls=40,
                original_task_completed=False, no_special_door_event=True,
                actual_original_set_gripper_release_used=True)


def actual_run(args):
    from research.v4.bridge.run_b0 import observe_physical_gpu
    pins = admission()
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.egl_device):
        raise RuntimeError("Explicit EGL/physical mapping required before imports/constructor")
    output = args.output.resolve()
    if output.exists() or output.parent != HERE:
        raise RuntimeError("Fresh root-only output under isolated preparation folder required")
    output.mkdir()
    started = time.monotonic()
    deadline = started + args.wall_s
    selected = providers(output, args.egl_device)
    report = dict(kind="leash_v43_native_scripted_boundary_QA_not_method", seed=0,
                  pid=os.getpid(), physical_gpu_expected=args.physical_gpu, egl_device=args.egl_device,
                  source_sha256=pins, source_drift=[], task_results=[], error=None,
                  method_episodes=0, model_calls=0, services_called=0,
                  policy_approved=False, script_only_draft_policy=True, total_wall_cap_s=args.wall_s,
                  native_review="pending_root", holding_never_uses_latch=True)
    report["requested_instances"] = args.instances
    previous = signal.getsignal(signal.SIGALRM)

    def deadline_stop(*_):
        raise TimeoutError("Entire native QA exceeded finite180s bound")

    signal.signal(signal.SIGALRM, deadline_stop)
    signal.alarm(int(args.wall_s))
    try:
        # R3 first supplies finite real empty-hand mechanical samples. E1/E3
        # each retain their own real OPEN samples and exactly one grasp prefix.
        for name in args.instances:
            own = output / name
            own.mkdir()
            cls, probe, cap = selected[name]
            kwargs = {} if name == "R3-02" else dict(camera_observations=True,
                                                     render_gpu_device_id=args.egl_device)
            row = dict(instance=name, original_total_control_cap=cap, original_scene=cls.__module__+"."+cls.__qualname__,
                       common_reset_controls=None, task_controls=0, error=None)
            report["task_results"].append(row)
            scene = adapter = reviewer = None
            try:
                actual = cls(0, **kwargs)
                scene = BudgetTraceScene(actual, own, max_controls=cap, deadline=deadline)
                reviewer = ScriptedReviewer(scene)
                adapter = make_adapter(scene, reviewer, preliminary_calibration(scene), upright=name == "E3-01")
                row["common_reset_controls"] = int(scene.sequence)
                capture_stage(scene, own, "initial")
                gpu = observe_physical_gpu(os.getpid(), args.physical_gpu)
                row["physical_gpu"] = gpu
                if not gpu["observed"]:
                    raise RuntimeError("Actual graphics PID not observed on requested physical GPU")
                primitives = original_primitives(adapter)
                if name == "R3-02":
                    result = check_r3(scene, adapter, primitives, own)
                else:
                    initial_open_samples(scene, adapter, primitives, own)
                    result = (check_e1(scene, adapter, primitives, probe, own,
                                       unknown_release_only=args.e1_unknown_release_only)
                              if name == "E1-01" else check_e3(scene, adapter, primitives, probe, own))
                row["result"] = result
            except BaseException as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                if adapter is not None:
                    row["final_holding"] = public_holding(adapter)
                    write_json(own / "dispatcher_logs.json", adapter.dispatcher.logs)
                if reviewer is not None:
                    write_json(own / "scripted_review_calls.json", reviewer.calls)
                if scene is not None:
                    row.update(task_controls=len(scene.rows), total_controls=int(scene.sequence),
                               actual_original_tools=["LiberoPrimitives.set_gripper", "LiberoPrimitives.release"])
                    try:
                        scene.save()
                    finally:
                        scene.close()
        report["mechanism_passed"] = True
        report["all_requested_native_boundaries_proved"] = all(row["result"]["passed"] for row in report["task_results"])
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        report["mechanism_passed"] = False
        report["all_requested_native_boundaries_proved"] = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        report["elapsed_s"] = time.monotonic() - started
        report["source_drift"] = [name for name, digest in pins.items()
                                  if sha(Path(name) if Path(name).is_absolute() else ROOT/name) != digest]
        if report["source_drift"]:
            report["mechanism_passed"] = False
        write_json(output / "report.json", report)
    return report


def source_contract():
    """Actual callable selection/signatures only; no Scene/main/client/solver."""
    from .adapter import V43PublicEnvAdapter
    from .dispatch import V43Intercepted
    from .method_inputs import PolicySource
    from .registry import GripperCalibration, EVENTS
    from robots.libero.tools import LiberoPrimitives
    current = providers(HERE / "UNRUN_SOURCE_ONLY", 0)
    callback_checks = {}
    for name, (_, function, _) in current.items():
        if function is None:
            continue
        parsed = ast.parse(inspect.getsource(function))
        callback_checks[name] = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                   and node.func.id == "on_stage" for node in ast.walk(parsed))
    checks = dict(exact_two_registry_events=[x.event_id for x in EVENTS] == ["HELD_OBJECT_RELEASE", "POUR_ONSET"],
                  original_release_sign="action[6] = -1.0" in inspect.getsource(LiberoPrimitives.release),
                  original_set_gripper_steps="self._step_env(action)" in inspect.getsource(LiberoPrimitives.set_gripper),
                  original_tool_constructor_module=LiberoPrimitives.__module__ == "robots.libero.tools",
                  prefix_E1_name=current["E1-01"][1].__module__ == "research.v4.accelerated.e1_script_cpu",
                  prefix_E3_name=current["E3-01"][1].__module__ == "research.v4.accelerated.e3_script_cpu",
                  same_original_prefix_callback_stage=all(callback_checks.values()),
                  actual_current_R302_flat_front_class="CurrentFrontNormalTangentNativeCanScene" == current["R3-02"][0].__name__)
    # Bind only inert argument names; these constructors are NOT invoked.
    signature = inspect.signature(V43PublicEnvAdapter)
    required = {"instruction", "method", "reviewer", "calibration", "policy", "grasp_assumption"}
    checks["actual_adapter_contract_fields"] = required.issubset(signature.parameters)
    inspect.signature(LiberoPrimitives.release).bind(None, max_steps=3)
    inspect.signature(LiberoPrimitives.set_gripper).bind(None, gripper=1., steps=3)
    if not all(checks.values()):
        raise RuntimeError("Source-only actual native contract failed: " + repr(checks))
    return dict(passed=True, checks=checks,
                selected_classes={name:factory.__module__+"."+factory.__qualname__ for name,(factory,_,_) in current.items()},
                original_caps={name:cap for name,(_,_,cap) in current.items()},
                constructors=0, actual_controls=0, GPU_calls=0, models=0, clients=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--egl-device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--wall-s", type=int, choices=(180,), default=180)
    parser.add_argument("--instances", nargs="+", choices=("E1-01", "E3-01", "R3-02"),
                        default=["R3-02", "E1-01", "E3-01"])
    parser.add_argument("--e1-unknown-release-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        pins = admission()
        result = source_contract()
        result.update(source_count=len(pins), native_executed=False, launch_result_claimed=False)
        print(json.dumps(result))
        return 0
    if args.output is None:
        raise RuntimeError("Root must supply one fresh native output")
    report = actual_run(args)
    print(json.dumps({key:report[key] for key in ("mechanism_passed", "error", "elapsed_s",
                                               "all_requested_native_boundaries_proved", "source_drift")}))
    return 0 if report["mechanism_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
