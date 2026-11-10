"""Bounded P1 native initial-image check; not scripted task acceptance."""

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image
import numpy as np

from .evaluator import Evaluator
from .scene import Scene
from .template import freeze_inputs, load_instance, source_hashes


def run(instance_id, seed, size, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    instance = load_instance(instance_id)
    hashes = source_hashes(instance, Path(__file__))
    freeze_inputs(output, hashes)
    (output / "source-pins.json").write_text(json.dumps(hashes, indent=2) + "\n")
    started = time.monotonic()
    scene = None
    try:
        scene = Scene(instance_id=instance_id, seed=seed, size=size)
        evaluator = Evaluator()
        truth = evaluator.observe(scene)
        packet = scene.capture_agent()
        for name, camera in packet["cameras"].items():
            Image.fromarray(camera["rgb"]).save(output / f"{name}.png")
            if not np.isfinite(camera["depth_m"]).all():
                raise ValueError(f"P1 {name} depth contains nonfinite pixels")
            np.save(output / f"{name}-depth-m.npy", camera["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(output / "spectator.png")
        result = {
            "instance": instance_id, "seed": seed, "pid": os.getpid(),
            "result": "PREFLIGHT_ONLY", "elapsed_s": time.monotonic() - started,
            "source_sha256": hashes,
            "initial_truth_evaluator_only": truth,
            "agent_camera_shapes": {name: list(camera["rgb"].shape)
                                    for name, camera in packet["cameras"].items()},
            "depth_camera_shapes": {name: list(camera["depth_m"].shape)
                                    for name, camera in packet["cameras"].items()},
            "reset_staging_steps": scene.sequence,
            "spectator_camera": scene.spectator_view,
            "evaluator_initial_label": evaluator.finish(None)["label"],
        }
        (output / "preflight.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}), flush=True)
        return result
    except BaseException as exc:
        (output / "failure.json").write_text(json.dumps({
            "instance": instance_id, "seed": seed, "pid": os.getpid(),
            "elapsed_s": time.monotonic() - started, "error": str(exc),
            "source_sha256": hashes}, indent=2))
        raise
    finally:
        if scene is not None:
            scene.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="P1-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.instance, args.seed, args.size, args.output)


if __name__ == "__main__":
    main()
