"""Capture the three native views and test the correct R-class refusal."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from research.v4.tasks.r2.evaluator import Evaluator
from research.v4.tasks.r2.scene import Scene
from research.v4.tasks.r2.template import HERE, ROOT, TASK_ID, load_instance, source_hashes, freeze_inputs


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=f"{TASK_ID}-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--gpu-observation-wait-s", type=float, default=0.0)
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
              "gpu_observation_wait_s": args.gpu_observation_wait_s,
              "source_hashes": hashes,
              "asset_paths": [str((HERE / role["xml"]).relative_to(ROOT))
                              for role in (instance.source, instance.target)],
              "asset_manifest_names": [], "asset_registration_verified": False}
    try:
        scene = Scene(instance_id=instance.id, seed=args.seed, size=args.size)
        if args.gpu_observation_wait_s:
            time.sleep(args.gpu_observation_wait_s)
        evaluator = Evaluator()
        report["initial_truth"] = evaluator.observe(scene)
        packet = scene.capture_agent()
        for name in scene.public_views:
            Image.fromarray(packet["cameras"][name]["rgb"]).save(output / f"{name}.png")
            np.save(output / f"{name}-depth-m.npy", packet["cameras"][name]["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(output / "spectator.png")
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
