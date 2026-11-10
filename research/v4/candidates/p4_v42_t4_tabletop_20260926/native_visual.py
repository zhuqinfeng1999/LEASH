"""Bounded, zero-task-control P4 v4.2 native visual diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from PIL import Image

from research.v4.bridge.run_b0 import ROOT, observe_physical_gpu
from .evaluator import P4Evaluator
from .facade import P4TabletopFacade as Scene

HERE = Path(__file__).resolve().parent
PUBLIC = ("agentview", "robot0_eye_in_hand")
SOURCE = tuple(Path(path) for path in json.loads((HERE / "qualification/safe_seed0.json").read_text())["source_sha256"]) + (HERE / "native_visual.py",)



def hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCE}


def save_views(scene, output: Path, prefix: str):
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix)
    packet = scene.capture_agent()
    if set(packet["cameras"]) != set(PUBLIC):
        raise RuntimeError("public camera set changed")
    calibration = {}
    for name in PUBLIC:
        camera = packet["cameras"][name]
        rgb, depth = np.asarray(camera["rgb"]), np.asarray(camera["depth_m"])
        if (rgb.shape != (480, 640, 3) or depth.shape != (480, 640) or
            not np.isfinite(depth).all() or not (depth > 0).all()):
            raise RuntimeError(f"invalid native 640x480 RGB-D for {name}")
        Image.fromarray(rgb).save(output / f"{prefix}_{name}.png")
        np.savez_compressed(output / f"{prefix}_{name}_depth.npz", depth_m=depth)
        calibration[name] = {
            "intrinsics": np.asarray(get_camera_intrinsic_matrix(
                scene.env.sim, name, 480, 640)).tolist(),
            "extrinsics": np.asarray(get_camera_extrinsic_matrix(
                scene.env.sim, name)).tolist(),
            "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
        }
    Image.fromarray(scene.spectator_rgb()).save(output / f"{prefix}_spectator.png")
    return calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    ap.add_argument("--render-device", type=int, choices=(0, 1), required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    output = args.output.resolve()
    work = HERE / "work"
    if output.parent != work.resolve() or output.exists():
        raise ValueError("new output directory required inside accelerated/work")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.render_device):
        raise RuntimeError("EGL physical GPU selection mismatch")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
    source = hashes()
    output.mkdir(parents=True)
    (output / "source_sha256.json").write_text(json.dumps(source, indent=2) + "\n")
    report = {"kind": "P4 T4 common-tabletop native zero-control visual diagnostic",
              "seed": args.seed, "physical_gpu": args.physical_gpu, "pid": os.getpid(),
              "source_sha256": source, "source_changes_after": [],
              "control_steps": 0, "status": "started", "error": None,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    scene = None
    start = time.perf_counter()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(_signal, _frame):
        raise TimeoutError("P4 visual diagnostic exceeded 120 seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        scene = Scene(seed=args.seed, camera_observations=True, render_gpu_device_id=args.render_device)
        report["physical_gpu_placement"] = observe_physical_gpu(os.getpid(), args.physical_gpu)
        if not report["physical_gpu_placement"]["observed"]:
            raise RuntimeError("simulator PID not observed on selected physical GPU")
        report["calibration_initial"] = save_views(scene, output, "initial")
        evaluator = P4Evaluator()
        before = evaluator.observe(scene)
        eef_start = np.asarray(scene.raw["robot0_eef_pos"], float)
        if before["hot_shirt_contact"]:
            raise RuntimeError("initial hot sole/cloth contact")
        for _ in range(20):
            scene.step(np.array([0,0,0,0,0,0,-1], dtype=np.float32))
            report["control_steps"] += 1
            after = evaluator.observe(scene)
            if after["hot_shirt_contact"]:
                raise RuntimeError("hot sole/cloth contact during neutral controls")
        report["calibration_final"] = save_views(scene, output, "final")
        report["eef_drift_m"] = float(np.linalg.norm(
            np.asarray(scene.raw["robot0_eef_pos"], float) - eef_start))
        report["initial_private_qa"] = before
        report["final_private_qa"] = after
        report["status"] = "native_views_recorded"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if scene is not None:
            try:
                scene.close()
                report["scene_closed"] = True
            except BaseException as exc:
                report["scene_closed"] = False
                report["cleanup_error"] = str(exc)
        report["elapsed_s"] = round(time.perf_counter() - start, 3)
        report["source_changes_after"] = [name for name, digest in source.items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        if report["source_changes_after"] or report.get("scene_closed") is not True:
            report["status"] = "failed"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(output), "status": report["status"],
                          "error": report["error"], "elapsed_s": report["elapsed_s"]}),
              flush=True)
    if report["status"] != "native_views_recorded":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
