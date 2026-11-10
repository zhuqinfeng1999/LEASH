"""One-owner zero-action E1-01 initial views; not task acceptance by itself."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .evaluator import E1Evaluator
from .scene import E1Scene


def run(output, seed=0, size=256, instance_id="E1-01"):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    scene = None
    report = {"task": instance_id, "case": "initial_zero_action", "seed": seed,
              "pid": os.getpid(), "model_calls": 0, "agent_control_steps": 0,
              "section7_accepted": False}
    try:
        scene = E1Scene(seed=seed, size=size, instance_id=instance_id)
        evaluator = E1Evaluator()
        report["initial_truth"] = evaluator.observe(scene)
        packet = scene.observation()
        for name, view in packet["cameras"].items():
            Image.fromarray(view["rgb"]).save(output / f"{name}.png")
            np.save(output / f"{name}-depth-m.npy", view["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(output / "spectator.png")
        report["initial_label"] = evaluator.finish(None)["label"]
        report["scene_xml_sha256"] = hashlib.sha256(scene.source_scene_xml.encode()).hexdigest()
        (output / "rendered-scene.bddl").write_text(scene.rendered_bddl)
        report["depth_range_m"] = {
            name: [float(np.min(view["depth_m"])), float(np.max(view["depth_m"]))]
            for name, view in packet["cameras"].items()}
        report["success"] = bool(report["initial_label"] == "NOT_TRIGGERED"
                                 and not report["initial_truth"]["hand_in_target"])
    except BaseException as error:
        import traceback
        report.update(success=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        if scene is not None:
            scene.close()
        report["elapsed_s"] = time.monotonic()-start
        (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["success"]:
        raise RuntimeError(report.get("error", "E1 initial predicate failed"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--instance", default="E1-01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed, args.size, args.instance)), flush=True)


if __name__ == "__main__":
    main()
