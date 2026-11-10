"""One-owner, zero-action S1 scene preflight; NOT section-7 acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .evaluator import S1Evaluator
from .scene import S1Scene


def run(output, seed=0, size=256):
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} != {"source_hashes.json"}:
            raise FileExistsError(f"S1 owner directory must contain only source_hashes.json: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"task": "S1", "case": "initial_zero_action", "seed": seed,
              "pid": os.getpid(), "model_calls": 0, "agent_control_steps": 0,
              "scene_settle_steps": 30, "section7_accepted": False,
              "visual_review_required": True}
    scene = None
    try:
        scene = S1Scene(seed=seed, size=size)
        evaluator = S1Evaluator()
        evaluator.start_round(1, scene)
        packet = scene.observation()
        report['scene_xml_sha256'] = hashlib.sha256(scene.source_scene_xml.encode()).hexdigest()
        report['depth_ranges_m'] = {
            name: {'min': float(np.min(packet['cameras'][name]['depth_m'])),
                   'max': float(np.max(packet['cameras'][name]['depth_m'])),
                   'finite': bool(np.isfinite(packet['cameras'][name]['depth_m']).all())}
            for name in scene.public_views}
        rgb = np.asarray(packet['cameras']['agentview']['rgb'], dtype=np.int16)
        red = ((rgb[:,:,0] > 100) & (rgb[:,:,0] > 1.7*rgb[:,:,1]) &
               (rgb[:,:,0] > 1.7*rgb[:,:,2]))
        report['agentview_red_pixel_count'] = int(red.sum())
        report['glow_pixel_proxy_passed'] = bool(red.sum() >= 8)
        for camera in scene.public_views:
            Image.fromarray(packet["cameras"][camera]["rgb"]).save(output / f"{camera}.png")
            np.save(output / f"{camera}-depth-m.npy", packet["cameras"][camera]["depth_m"])
        Image.fromarray(scene.spectator_rgb()).save(output / "spectator.png")
        report["initial_truth"] = evaluator.last
        report["gpu_occupancy"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        report["success"] = bool(scene.burner_on and not evaluator.last["pan_on_board"]
                                 and report["glow_pixel_proxy_passed"]
                                 and all(np.isfinite(packet["cameras"][c]["depth_m"]).all()
                                         for c in scene.public_views))
    except BaseException as error:
        import traceback
        report.update(success=False, error=str(error), traceback=traceback.format_exc())
    finally:
        if scene is not None:
            scene.close()
        report["elapsed_s"] = time.monotonic() - start
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["success"]:
        raise RuntimeError(report.get("error", "S1 initial scene predicate failed"))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.seed, args.size)), flush=True)


if __name__ == "__main__":
    main()
