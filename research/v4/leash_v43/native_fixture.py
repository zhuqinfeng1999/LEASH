"""Small offline instruments for existing real scenes; no synthetic scene.

Private object/equality/contact records are written only to the offline trace.
They are never passed to the v4.3 adapter, registry, or scripted reviewer.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ("agentview", "robot0_eye_in_hand")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_value) + "\n")


def integration(scene):
    sim = scene.backend.sim
    model, data = sim.model._model, sim.data._data
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, result, kind)
    return result


def offline_state(scene):
    """Actual authoritative state, explicitly offline and evaluator-only."""
    sim = scene.backend.sim
    model, data = sim.model._model, sim.data._data
    body = model.body(scene.backend.objects_dict[scene.item_name].root_body).id
    force = np.zeros(6)
    contacts = []
    for i, contact in enumerate(data.contact[:data.ncon]):
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append(dict(geom1=model.geom(int(contact.geom1)).name,
                             geom2=model.geom(int(contact.geom2)).name,
                             dist_m=float(contact.dist), force=force.copy().tolist()))
    return dict(time_s=float(data.time), sequence=int(scene.sequence),
                qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                eef_pos=np.asarray(scene.raw["robot0_eef_pos"]).copy(),
                eef_quat=np.asarray(scene.raw["robot0_eef_quat"]).copy(),
                gripper_qpos=np.asarray(scene.raw["robot0_gripper_qpos"]).copy(),
                object_xpos=data.xpos[body].copy(), object_xmat=data.xmat[body].copy(),
                actual_assist_held_offline_only=bool(scene.grasp_assist.held),
                equality_active=data.eq_active.copy(), equality_data=model.eq_data.copy(),
                contacts=contacts, offline_only=True)


def lockstep_signature(scene):
    """Reviewer wait must consume no control, even when a verdict is PASS."""
    return dict(time_s=float(scene.backend.sim.data.time), sequence=int(scene.sequence),
                integration_sha256=hashlib.sha256(integration(scene).tobytes()).hexdigest(),
                public_qpos=np.asarray(scene.raw["robot0_gripper_qpos"]).copy().tolist())


class ScriptedReviewer:
    """External scripted feedback only. It never reads private object truth."""
    def __init__(self, scene):
        self.scene = scene
        self.mode = "PASS"
        self.calls = []

    def __call__(self, request):
        before = lockstep_signature(self.scene)
        response = dict(decision=self.mode, reason="Scripted native outlet fixture: " + self.mode,
                        events=list(request["events"]),
                        evidence_version=request["evidence_version"])
        if self.mode == "INVALID":
            response = dict(decision="PASS", reason="Deliberately invalid missing evidence version")
        after = lockstep_signature(self.scene)
        row = dict(mode=self.mode, events=list(request["events"]),
                   evidence_version=request["evidence_version"],
                   request_keys=sorted(request), before=before, after=after,
                   lockstep_exact=before == after, model_called=False)
        self.calls.append(row)
        if before != after:
            raise RuntimeError("Scripted reviewer advanced real state")
        return response


class BudgetTraceScene:
    """Delegate every command to the same original scene, with a finite sink."""
    def __init__(self, scene, output, *, max_controls, deadline):
        self.actual = scene
        self.output = Path(output)
        self.max_controls = max_controls
        self.deadline = deadline
        self.rows = []
        self.prefix_controls = 0
        self.initial = offline_state(scene)
        self.initial_sequence = int(scene.sequence)

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def step(self, action):
        if time.monotonic() >= self.deadline or int(self.sequence) >= self.max_controls:
            raise TimeoutError("Original scene control cap /180s aggregate wall bound")
        answer = self.actual.step(action)
        row = offline_state(self.actual)
        row["action"] = np.asarray(action).copy()
        self.rows.append(row)
        return answer

    def save(self):
        write_json(self.output / "actual_control_trace.json", dict(
            initial=self.initial, rows=self.rows, original_total_control_cap=self.max_controls,
            common_reset_controls=self.initial_sequence,
            task_controls=len(self.rows), offline_only=True))

    def close(self):
        self.actual.close()


class PrefixReady(RuntimeError):
    """Stop the original offline probe once its unchanged lift returns."""


class PrefixThroughAdapter:
    """Original offline script controls enter the real v4.3 outlet one by one."""
    def __init__(self, scene, adapter):
        self.scene = scene
        self.adapter = adapter
        self.probe_requested_close = False

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self, action):
        return self.adapter.step(action)

    def close(self):
        # The outer fixture owns the single original Scene.  Probe.finally may
        # ask to close at the prefix sentinel; it must remain alive for tests.
        self.probe_requested_close = True


def run_grasp_prefix(scene, adapter, probe, output, *, max_controls):
    wrapped = PrefixThroughAdapter(scene, adapter)
    record = {}

    def stage(name, _scene, original_state):
        if name != "lift":
            return
        record.update(stage=name, original_offline_state=original_state,
                      sequence=int(scene.sequence), actual=offline_state(scene),
                      evaluator_result_claimed=False)
        # Original probes catch RuntimeError and initialize their final report.
        # A BaseException sentinel bypassed that and left result uninitialized.
        raise PrefixReady("intentional_v43_lift_prefix_stop_not_task_result")

    with adapter.primitive_scope("scripted_grasp_prefix", parameters={},
                                 completion_controls=max_controls):
        try:
            returned = probe("safe", 0, scene=wrapped, on_stage=stage)
        except PrefixReady:
            pass
        else:
            write_json(Path(output) / "intentionally_stopped_original_probe_raw.json", returned)
            if (not record or returned.get("error") not in {
                    "intentional_v43_lift_prefix_stop_not_task_result",
                    "PrefixReady('intentional_v43_lift_prefix_stop_not_task_result')"}):
                raise RuntimeError("Original script did not reach its actual lift callback")
            record["original_probe_raw_result"] = returned.get("result")
            record["original_probe_raw_accepted"] = returned.get("accepted")
            record["intentional_fixture_stop"] = True
    if not record or not wrapped.probe_requested_close:
        raise RuntimeError("Original prefix / lifecycle sentinel did not execute")
    record["original_probe_close_deferred_until_outer_finally"] = True
    write_json(Path(output) / "original_prefix_receipt.json", record)
    return record


def capture_stage(scene, output, name):
    """Same physical state, original public640 plus named offline640 view."""
    output = Path(output)
    before = integration(scene)
    packet = scene.capture_agent()
    if not np.array_equal(before, integration(scene)):
        raise RuntimeError("Public acquisition changed authoritative integration")
    if set(packet["cameras"]) != set(PUBLIC):
        raise RuntimeError("Only original two public cameras are permitted")
    sim = scene.backend.sim
    model = sim.model._model
    cameras = {key:np.asarray(getattr(model, key)).copy()
               for key in ("cam_pos", "cam_quat", "cam_fovy")}
    fields = ("eef_pos", "eef_quat", "gripper_qpos")
    proprio = {key:np.asarray(packet["robot_state"][key]).copy() for key in fields}
    for view, values in packet["cameras"].items():
        rgb = np.asarray(values["rgb"])
        depth = np.asarray(values["depth_m"])
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError("Actual native640 RGB-D dimensions changed")
        if not np.isfinite(depth).all() or not np.all(depth > 0):
            raise ValueError("Actual native640 metric depth invalid")
        Image.fromarray(rgb).save(output / f"{name}_{view}.png")
        np.savez_compressed(output / f"{name}_{view}_rgbd.npz", rgb=rgb, depth_m=depth,
                            intrinsics=values["intrinsics"], extrinsics=values["extrinsics"],
                            frame_id=values["frame_id"])
    # No free camera, fallback, resizing, or transparency.
    image = sim.render(width=640, height=480, camera_name=scene.spectator_view)[::-1]
    if image.shape != (480, 640, 3):
        raise ValueError("Actual named spectator640 missing")
    Image.fromarray(np.ascontiguousarray(image)).save(output / f"{name}_spectator.png")
    fresh = scene.capture_agent()
    rgb_checks = {}
    calibration_exact = True
    for view, values in packet["cameras"].items():
        other = fresh["cameras"][view]
        delta = np.abs(np.asarray(values["rgb"]).astype(np.int16)
                       - np.asarray(other["rgb"]).astype(np.int16))
        changed, maximum = int(np.count_nonzero(delta)), int(delta.max())
        rgb_checks[view] = dict(changed_channels=changed, max_delta=maximum,
                               passed=changed <= 20 and maximum <= 1)
        calibration_exact = calibration_exact and all(np.array_equal(values[key], other[key])
                                                      for key in ("depth_m", "intrinsics", "extrinsics"))
    row = dict(stage=name, sequence=int(scene.sequence),
               integration_exact=np.array_equal(before, integration(scene)),
               camera_exact=all(np.array_equal(value, getattr(model, key)) for key, value in cameras.items()),
               depth_K_T_exact=bool(calibration_exact),
               proprio_exact=all(np.array_equal(value, fresh["robot_state"][key]) for key, value in proprio.items()),
               RGB_checks=rgb_checks, public_views=list(PUBLIC), public_resolution=[640, 480],
               spectator_online=False, spectator_named_camera=scene.spectator_view)
    write_json(output / f"{name}_capture_guard.json", row)
    if not all(row[key] for key in ("integration_exact", "camera_exact", "depth_K_T_exact", "proprio_exact")) \
            or not all(check["passed"] for check in rgb_checks.values()):
        raise RuntimeError("Existing native rendering/public calibration guard rejected capture")
    return row


class ForbiddenUnusedIO:
    def __getattr__(self, name):
        raise RuntimeError("Native fixture forbids SDK/SAM/VLA/service I/O: " + name)


def original_primitives(adapter):
    from robots.libero.tools import LiberoPrimitives
    primitives = LiberoPrimitives(env=adapter, model=ForbiddenUnusedIO(),
                                 sam3_client=ForbiddenUnusedIO(), check_cancelled=lambda:None)
    primitives.reset()
    return primitives
