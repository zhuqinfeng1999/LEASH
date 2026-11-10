"""Capture R5 public RGB-D views and spectator view; test correct refusal."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import time

import numpy as np
from PIL import Image

from research.v4.tasks.r5.evaluator import Evaluator
from research.v4.tasks.r5.scene import Scene
from research.v4.tasks.r5.template import HERE, ROOT, TASK_ID, load_instance, source_hashes, freeze_inputs


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=f"{TASK_ID}-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    instance = load_instance(args.instance)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "research/v4/runs").resolve()):
        raise ValueError("Output must be under research/v4/runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    hashes = source_hashes(instance, Path(__file__))
    freeze_inputs(output, hashes)
    scene = None
    report = {"task": TASK_ID, "instance": instance.id, "case": "correct_refusal",
              "instruction": instance.instruction, "seed": args.seed, "model_calls": 0,
              "source_hashes": hashes,
              "asset_paths": [str((HERE / role["xml"]).relative_to(ROOT))
                              for role in (instance.item, instance.lunchbox)],
              "asset_manifest_names": [], "asset_registration_verified": False}
    try:
        scene = Scene(instance_id=instance.id, seed=args.seed, size=args.size)
        evaluator = Evaluator()
        report["initial_truth"] = evaluator.observe(scene)
        packet = scene.capture_agent()
        for name in scene.public_views:
            Image.fromarray(packet["cameras"][name]["rgb"]).save(output / f"{name}.png")
            np.save(output / f"{name}-depth-m.npy", packet["cameras"][name]["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(output / "spectator.png")
        # Keep the zero-action EGL context alive briefly so host GPU sampling can
        # associate this exact simulator PID with its physical device.
        report["simulator_pid"] = os.getpid()
        report["gpu_device_fds"] = sorted({
            os.readlink("/proc/self/fd/" + fd)
            for fd in os.listdir("/proc/self/fd")
            if os.path.exists("/proc/self/fd/" + fd)
            and "/dev/nvidia" in os.readlink("/proc/self/fd/" + fd)
        })
        report["gpu_process_snapshot"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        time.sleep(3)
        report["initial_views"] = {name: str((output / f"{name}.png").relative_to(ROOT))
                                   for name in (*scene.public_views, "spectator")}
        report["evaluator"] = evaluator.finish("refused")
        report["success"] = report["evaluator"]["label"] == "SAFE_SUCCESS"
    except Exception as exc:
        import traceback
        report.update(success=False, error=str(exc), traceback=traceback.format_exc())
    finally:
        if scene is not None:
            scene.close()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(output), "success": report["success"],
                          "label": report.get("evaluator", {}).get("label"),
                          "error": report.get("error")}), flush=True)
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
