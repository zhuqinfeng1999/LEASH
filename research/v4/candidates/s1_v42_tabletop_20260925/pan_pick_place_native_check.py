"""One native physical pan-pick diagnostic; no planner or method score."""
from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from research.v4.bridge.run_b0 import ROOT, bind_rpent_checkout
from research.v4.accelerated.r2_adapter import R2V42Adapter
from research.v4.accelerated.public_latch_toolkit import PublicLatchToolkit
from research.v3.bridge.b0 import make_b0_toolkit
from .pan_place_public import PanPickPlaceToolkit
from .scene_v5_stable_pan300 import S1StablePan300Facade
from .evaluator import S1Evaluator


OUT = ROOT / "research/v4/candidates/s1_v42_tabletop_20260925/S1PANPICKPLACE_STABLE300_SEED0_GPU1"
SOURCE = (
    "research/v4/candidates/s1_v42_tabletop_20260925/pan_pick_place_native_check.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/pan_pick_public.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/pan_place_public.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/scene_v3_clear_handle.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/scene_v5_stable_pan300.py",
    "research/v4/candidates/s1_v42_tabletop_20260925/development_pan_300g.xml",
    "research/v4/accelerated/grasp_assist.py",
    "research/v4/accelerated/r2_adapter.py",
    "research/v4/accelerated/public_latch_toolkit.py",
)


def main():
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != "1":
        raise RuntimeError("physical GPU1/EGL1 required")
    OUT.mkdir(exist_ok=False)
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in SOURCE}
    bind_rpent_checkout()
    started = time.monotonic()
    scene = base = None
    report = {"kind": "S1 common public pan-pick native physical diagnostic, not method",
              "seed": 0, "source_sha256": before, "simulator_pid": os.getpid(),
              "result": None, "error": None}
    prior = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("240s wall")))
    signal.alarm(240)
    try:
        scene = S1StablePan300Facade(0, camera_observations=True,
                                    render_gpu_device_id=1)
        pmon = subprocess.check_output(
            ["nvidia-smi", "pmon", "-i", "1", "-c", "2"], text=True)
        (OUT / "gpu1_pmon.txt").write_text(pmon)
        report["physical_gpu1_pid_seen"] = any(
            parts[:2] == ["1", str(os.getpid())]
            for parts in (line.split() for line in pmon.splitlines()))
        if not report["physical_gpu1_pid_seen"]:
            raise RuntimeError("native simulator PID not seen on physical GPU1")
        adapter = R2V42Adapter(scene, instruction=scene.instruction)
        evaluator = S1Evaluator()
        evaluator.start_round(1, scene)
        adapter.on_sim_step = lambda current: evaluator.observe(current)
        base = make_b0_toolkit(
            primitives_kwargs={"env": adapter, "model": None, "sam3_client": None},
            hold=adapter.hold, output_dir=OUT / "rpent")
        base.per_round_budget = 48
        base.episode_budget = 144
        public = PublicLatchToolkit(base, scene)
        private_stage_trace = []
        def offline_trace(event):
            if event.get("event") != "pan_pick_child":
                return
            sim = scene.backend.sim
            zone = scene.grasp_assist._entries[0][2]
            private_stage_trace.append({
                "stage": event.get("stage"),
                "private_zone_xyz": sim.data.site_xpos[zone].tolist(),
                "public_eef_xyz": event.get("public_eef_xyz_m"),
            })
        macro = PanPickPlaceToolkit(public, public_packet=scene.capture_agent,
                                    log_event=offline_trace)
        pickup = macro.execute_tool("pick", {"query": "pan handle"})
        placement = macro.execute_tool("place", {"query": "wooden board"}) if pickup.result.get("success") else None
        report["pickup_result"] = pickup.result
        report["placement_result"] = placement.result if placement else None
        report["result"] = placement.result if placement else pickup.result
        report["macro_events"] = macro.events
        report["offline_private_stage_trace"] = private_stage_trace
        report["primitive_calls"] = base.episode_calls
        report["assist_events"] = list(scene.assist_events)
        report["public_grasp_latched"] = scene.capture_agent()["robot_state"]["grasp_latched"]
        # Offline mechanics forensic only. Never supplied to the estimator,
        # macro, planner, or reviewer.
        sim = scene.backend.sim
        assist = scene.grasp_assist
        zone = assist._entries[0][2]
        report["offline_private_grasp_forensic"] = {
            "zone_xyz": sim.data.site_xpos[zone].tolist(),
            "center_xyz": sim.data.site_xpos[assist._center].tolist(),
            "left_pad_xyz": sim.data.site_xpos[assist._left].tolist(),
            "right_pad_xyz": sim.data.site_xpos[assist._right].tolist(),
            "eef_xyz": scene.raw["robot0_eef_pos"].tolist(),
        }
        report["offline_round1_evidence"] = evaluator.placement.evidence().__dict__
        report["accepted"] = bool(pickup.result.get("success") and placement and
                                  placement.result.get("success") and
                                  report["public_grasp_latched"] is False and
                                  evaluator.placement.evidence().complete and
                                  base.episode_calls in (10, 11) and
                                  any(e.get("kind") == "attached" for e in scene.assist_events) and
                                  any(e.get("kind") == "detached" for e in scene.assist_events))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["accepted"] = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
        for obj in (base, scene):
            if obj is not None:
                with suppress(Exception): obj.close()
        report["elapsed_wall_s"] = round(time.monotonic() - started, 3)
        report["source_drift"] = [name for name, digest in before.items()
                                  if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"accepted": report["accepted"],
                          "error": report["error"],
                          "result": report["result"],
                          "elapsed_wall_s": report["elapsed_wall_s"]}), flush=True)
    return 0 if report["accepted"] and not report["source_drift"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
